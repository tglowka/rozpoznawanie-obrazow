import os
import pyrootutils
import pytorch_lightning as pl
from test.face_age_model_tester import FaceAgeModelTester
from test.face_age_transforms import FaceAgeTransforms
from commons.face_age_module import Nets
import argparse


pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


def parse_args():
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-c", "--checkpoints_dir", help="Directory of checkpoints to load"
    )
    argParser.add_argument("-n", "--net_type", help="Net type, either EFF or SIMPLE")
    argParser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help='Flag to run in debug mode, either "True" or "False". Debug mode reads data from the "data/face_age_dataset_debug"',
    )
    argParser.add_argument(
        "-t",
        "--transforms",
        action="store_true",
        help="Flag to run for each transform returned from FaceAgeTransformers.create_transforms method",
    )

    args = argParser.parse_args()

    debug = args.debug
    dir = args.checkpoints_dir
    net_type = args.net_type

    if args.transforms:
        transforms_list = FaceAgeTransforms().create_transforms()
    else:
        transforms_list = []

    if net_type == "SIMPLE":
        net = Nets.SimpleConvNet_224x224
    elif net_type == "EFF":
        net = Nets.PretrainedEfficientNet
    else:
        raise ValueError("Unknown arg net")

    ckpt_paths = [os.path.join(dir, x) for x in os.listdir(dir)]

    return (net, ckpt_paths, debug, transforms_list)


def main():
    (net, ckpt_paths, debug, transforms_list) = parse_args()

    pl.seed_everything(42)

    if transforms_list:
        results = []
        for transform, transform_name in transforms_list:
            result = FaceAgeModelTester(
                net=net,
                ckpt_paths=ckpt_paths,
                transforms_list=[transform],
                debug=debug,
            ).predict()
            results.append((result, transform_name))

        for result, transform_name in results:
            print(f"Transform name: {transform_name}. Result: {result}")

    else:
        result = FaceAgeModelTester(
            net=net,
            ckpt_paths=ckpt_paths,
            transforms_list=[],
            debug=debug,
        ).predict()
        print("Result: ", result)


if __name__ == "__main__":
    main()
