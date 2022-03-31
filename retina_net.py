#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.
This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import os, json, cv2, random, glob
from detectron2 import model_zoo

from LossHook import LossEvalHook
from detectron2.data import DatasetMapper, build_detection_test_loader



CHOSEN_MODEL = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"

#Contains hardcoded paths to data and model weights
#Bad name, also computes output from validation set
def visualize_detection(cfg):
    metadata = MetadataCatalog.get("my_dataset_val")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0003999.pth")  # path to the model we just trained
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.50 # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    data_path = "/home/serge/exjobb/networks/detectron2/"
    test_images_path = data_path+"ULTIMATE_FINAL_DATA/test2017"
    test_images = glob.glob(os.path.join(test_images_path, '*.jpg'))
    #jpg_path = "/home/serge/exjobb/detectron2/COCO_dataset_test/data"
    #jpgFiles=glob.glob(os.path.join(jpg_path, '*.jpg'))

    save_predictions_path = cfg.OUTPUT_DIR+"predictions/"

    if not os.path.exists(save_predictions_path):
        os.mkdir(save_predictions_path)
        print("Directory " , save_predictions_path ,  " Created ")
    else:    
        print("Directory " , save_predictions_path ,  " already exists")

    with open(cfg.OUTPUT_DIR+'pred_output.txt', "w+") as pred:
        for idx, img in enumerate(test_images):    
            im = cv2.imread(img)
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            pred.write("image:" + str(img)+"!!!")
            pred.write(str(outputs["instances"]).replace("\n", ""))
            pred.write("\n")
            #print(" \n\n\n\n\n\n\n outputs: ", outputs , "\n\n\n\n\n\n\n\n")
            v = Visualizer(im[:, :, ::-1],
                        metadata, 
                        scale=0.5
            )
            #print("33333")
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            #spara out som bild istället
            img_name = img.split("/")[-1]
            cv2.imwrite(save_predictions_path+img_name, out.get_image()[:, :, ::-1])


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    #called by DeafultTrainers init-function
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks




def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    #cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)
    
    register_coco_instances("my_dataset_train", {}, "/home/serge/exjobb/networks/detectron2/ULTIMATE_FINAL_DATA/annotations/instances_train2017.json", "/home/serge/exjobb/networks/detectron2/ULTIMATE_FINAL_DATA/train2017")
    register_coco_instances("my_dataset_val", {}, "/home/serge/exjobb/networks/detectron2/ULTIMATE_FINAL_DATA/annotations/instances_val2017.json", "/home/serge/exjobb/networks/detectron2/ULTIMATE_FINAL_DATA/val2017")
    
    cfg.merge_from_file(model_zoo.get_config_file(CHOSEN_MODEL))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CHOSEN_MODEL)  # Let training initialize from model zoo
    #cfg.MODEL.WEIGHTS = "/home/serge/exjobb/detectron2/detectron2_cloned_from_git/detectron2/tools/output/faster_101_FPN_3x/21feb/model_0064999.pth"
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.MAX_ITER = 100000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    #cfg.SOLVER.STEPS = []        # do not decay learning rate
    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.MODEL.RETINANET.NUM_CLASSES = 7
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    #Constraints on the image size during training
    cfg.INPUT.MIN_SIZE_TRAIN = 720
    cfg.INPUT.MAX_SIZE_TRAIN = 1920

    #Reltive path from the directory this script resides is
    #>>>>
    #>>>>
    #THIS MUST BE UPDATED
    cfg.OUTPUT_DIR = "./output/retina_101_FPN_3x/14mar/"
    #THIS MUST BE UPDATED
    #<<<<
    #<<<<

    #cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)

        visualize_detection(cfg)
        return res
    

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    result = trainer.train()

    val_loss_path = cfg.OUTPUT_DIR + "val_loss.txt"
    
    with open(val_loss_path, "w") as loss:
        validation_loss = trainer.storage.history("validation_loss").values()
        for e in validation_loss:
            loss.write(str(e) + "\n")
        
    

    return result


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("\nWARNING, make sure to update cfg.OUTPUT_DIR that can be found in this script that is currrently running")
    print("To exit, press enter, to continue, write yes")
    answer = input("input: ")
    if answer != "yes":
        exit()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

#config-file isn't used
#python3 up_orig_train_faster_rcnn.py --num-gpus 1 --config-file ./output/config.yaml --eval-only true