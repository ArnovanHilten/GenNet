import sys
import os
import warnings
warnings.filterwarnings('ignore')
import argparse
sys.path.insert(1, os.path.dirname(os.getcwd()) + "/utils/")
from utils.Create_plots import plot
from utils.Train_network import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "GenNet: Interpretable neural networks for phenotype prediction.",
                                     epilog="Check the wiki on github.com/arnovanhilten/gennet/ for more info")
    subparsers = parser.add_subparsers(help = "GenNet main options", dest="mode")
    parser_convert = subparsers.add_parser("convert", help = "Convert genotype data to hdf5")
    parser_convert.add_argument(
        "path",
        type=str,
        help="path to the data"
    )
    parser_train = subparsers.add_parser("train", help="Trains the network")
    parser_train.add_argument(
        "path",
        type=str,
        help="path to the data"
    )
    parser_train.add_argument(
        "ID",
        type=int,
        help="ID of the experiment"
    )
    parser_train.add_argument(
        "-problem_type",
        default='classification', type=str,
        choices=['classification','regression'],
        help="Type of problem, choices are: classification or regression"
    )
    parser_train.add_argument(
        "-wpc",
        type=float,
        metavar="weight positive class",
        default=1,
        help="Hyperparameter:weight of the positive class"
    )
    parser_train.add_argument(
        "-lr", '--learning_rate',
        type=float,
        metavar="learning rate",
        default=0.001,
        help="Hyperparameter: learning rate of the optimizer"
    )
    parser_train.add_argument(
        "-bs", '--batch_size',
        type=int,
        metavar="batch size",
        default=32,
        help='Hyperparameter: batch size'
    )
    parser_train.add_argument(
        "-epochs",
        type=int,
        metavar="number of epochs",
        default=100,
        help='Hyperparameter: batch size'
    )
    parser_train.add_argument(
        "-L1",
        metavar="",
        type=float,
        default=0.01,
        help='Hyperparameter: value for the L1 regularization pentalty similar as in lasso, enforces sparsity'
    )

    parser_plot = subparsers.add_parser("plot", help="Generate plots from a trained network")
    parser_plot.add_argument(
        "ID",
        type=int,
        help="ID of the experiment"
    )
    parser_plot.add_argument(
        "-type",
        type=str,
        choices=['layer_weight', 'circos'],
    )
    parser_plot.add_argument(
        "-layer_n",
        type=int,
        help="Number of the to be plotted layer",
        metavar="Layer_number:"
    )

    args = parser.parse_args()

    if args.mode == 'train':
        train(datapath=args.path, jobid=args.ID, wpc=args.wpc, lr_opt=args.learning_rate, batch_size=args.batch_size,
         epochs = args.epochs, l1_value=args.L1, problem_type = args.problem_type)
    elif args.mode == "plot":
        plot(jobid=args.ID, type=args.type, layer=args.layer_n)
    elif args.mode == 'convert':
        NotImplemented
        exit()
