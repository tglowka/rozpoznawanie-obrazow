import argparse
import pyrootutils
import pytorch_lightning as pl
from train.face_age_model_fitter import FaceAgeModelFitter
from commons.face_age_module import Nets

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


def parse_args():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-e", "--epochs", type=int, help="Epochs count")
    argParser.add_argument("-f", "--folds", type=int, help="Folds count")
    argParser.add_argument("-n", "--net_type", help="Net type, either EFF or SIMPLE")
    argParser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        help="Learning rate value of the Adam optimizer",
    )
    argParser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help='Flag to run in debug mode. Debug mode reads data from the "data/face_age_dataset_debug"',
    )

    args = argParser.parse_args()

    epochs = args.epochs
    folds = args.folds
    debug = args.debug
    learning_rate = args.learning_rate
    net_type = args.net_type
    if net_type == "SIMPLE":
        net = Nets.SimpleConvNet_224x224
    elif net_type == "EFF":
        net = Nets.PretrainedEfficientNet
    else:
        raise ValueError("Unknown arg net")

    return (epochs, folds, net, debug, learning_rate)


def main():
    (epochs, folds, net, debug, learning_rate) = parse_args()

    pl.seed_everything(42)

    fitter = FaceAgeModelFitter(
        folds=folds,
        net=net,
        epoch=epochs,
        transformer_list=[],
        debug=debug,
        learning_rate=learning_rate,
    )

    fitter.fit()


if __name__ == "__main__":
    main()
