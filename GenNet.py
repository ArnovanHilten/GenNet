import os
import sys
import warnings

warnings.filterwarnings('ignore')
import argparse

sys.path.insert(1, os.path.dirname(os.getcwd()) + "/GenNet_utils/")


def main():
    args = ArgumentParser().parse_cmd_args()

    if args.mode == 'train':
        if args.problem_type == "classification":
            args.regression = False
        elif args.problem_type == "regression":
            args.regression = True
        else:
            print('something went wrong invalid problem type', args.problem_type)
        from GenNet_utils.Train_network import train_model
        train_model(args)
        
    elif args.mode == "plot":
        from GenNet_utils.Create_plots import plot
        plot(args)
    if args.mode == 'convert':
        from GenNet_utils.Convert import convert
        convert(args)
    if args.mode == "topology":
        from GenNet_utils.Topology import topology
        topology(args)
    if args.mode == "interpret":
        from GenNet_utils.Interpret import interpret
        interpret(args)


class ArgumentParser():
    """Argumentparser"""

    def __init__(self):
        parser = argparse.ArgumentParser(description="GenNet: Interpretable neural networks for phenotype prediction.",
                                         epilog="Check the wiki on github.com/arnovanhilten/gennet/ for more info")

        subparsers = parser.add_subparsers(help="GenNet main options", dest="mode")

        parser_convert = subparsers.add_parser("convert", help="Convert genotype data to hdf5")
        self.make_parser_covert(parser_convert)

        parser_train = subparsers.add_parser("train", help="Trains the network")
        self.make_parser_train(parser_train)

        parser_plot = subparsers.add_parser("plot", help="Generate plots from a trained network")
        self.make_parser_plot(parser_plot)

        parser_topology = subparsers.add_parser("topology", help="Create standard topology files")
        self.make_parser_topology(parser_topology)

        parser_interpret = subparsers.add_parser("interpret", help="Post-hoc interpretation analysis on the network")
        self.make_parser_interpret(parser_interpret)

        self.parser = parser

    def parse_cmd_args(self):
        args = self.parser.parse_args()
        return args

    def make_parser_covert(self, parser_convert):
        parser_convert.add_argument(
            "-g", "--genotype",
            nargs='+',
            type=str,
            help="Path/paths to genotype data folder")
        parser_convert.add_argument(
            '-study_name',
            type=str,
            required=True,
            nargs='+',
            help=' Name for saved genotype data, without ext')
        parser_convert.add_argument(
            '-variants',
            type=str,
            help="Path to file with row numbers of variants to include, if none is "
                 "given all variants will be used",
            default=None)
        parser_convert.add_argument(
            "-o", "--out",
            type=str,
            default=os.getcwd() + '/processed_data/',
            help="Path for saving the results, default ./processed_data")
        parser_convert.add_argument(
            '-ID',
            action='store_true',
            default=False,
            help='Flag to convert minimac data to genotype per subject files first (default '
                 'False)')
        parser_convert.add_argument(
            '-vcf',
            action='store_true',
            default=False,
            help='Flag for VCF data to convert')
        parser_convert.add_argument(
            '-tcm',
            type=int,
            default=500000000,
            help='Modifier for chunk size during TRANSPOSING make it lower if you run out of '
                 'memory during transposing')
        parser_convert.add_argument(
            '-step',
            type=str,
            default='all',
            choices=['all', 'hase_convert', 'merge', 'impute_missing', 'exclude', 'transpose',
                     'merge_transpose', 'checksum'],
            help='Modifier to choose step to do')
        parser_convert.add_argument(
            '-n_jobs',
            type=int,
            default=1,
            help='Choose jobs > 1 for multiple job submission on a cluster')
        parser_convert.add_argument(
            '-comp_level',
            type=int,
            default=1,
            help='How compressed should the data be? Between 1-9. 1 \
                                    for low compression, 9 is highest compression')
        return parser_convert

    def make_parser_train(self, parser_train):
        parser_train.add_argument(
            "-path",
            type=str,
            help="Path to the data. Subject file, npz masks/topology and/or genotype.h5",
            required=True)
        parser_train.add_argument(
            "-ID",
            type=int,
            help="Number of the experiment",
            required=True)
        parser_train.add_argument(
            "-genotype_path",
            type=str,
            help="Path to genotype data if the location is not the same as given in -path",
            default="undefined")
        parser_train.add_argument(
            "-problem_type",
            default='classification', type=str,
            choices=['classification', 'regression'],
            help="Type of problem, choices are: classification or regression")
        parser_train.add_argument(
            "-wpc",
            type=float,
            metavar="weight positive class",
            default=1,
            help="Hyperparameter:weight of the positive class")
        parser_train.add_argument(
            "-lr", '--learning_rate',
            type=float,
            metavar="learning rate",
            default=0.001,
            help="Hyperparameter: learning rate of the optimizer")
        parser_train.add_argument(
            "-bs", '--batch_size',
            type=int,
            metavar="batch size",
            default=32,
            help='Hyperparameter: batch size')
        parser_train.add_argument(
            "-epochs",
            type=int,
            metavar="number of epochs",
            default=1000,
            help='Hyperparameter: batch size')
        parser_train.add_argument(
            "-workers",
            type=int,
            metavar="number of workers for multiprocessing",
            default=1,
            help='Speed-up: number of workers (CPU cores) for multiprocessing. Can cause memory-leaks in some tensorflow versions')
        parser_train.add_argument(
            "-L1",
            metavar="",
            type=float,
            default=0.01,
            help='Hyperparameter: value for the L1 regularization pentalty similar as in lasso, enforces sparsity')
        parser_train.add_argument(
            "-L1_act",
            metavar="",
            type=float,
            default=0.01,
            help='Hyperparameter: value for the L1 regularization on the activation, enforces sparse activations')
        parser_train.add_argument(
            "-network_name",
            type=str,
            help="Name of the network",
            default="undefined")
        parser_train.add_argument(
            "-filters",
            type=int,
            metavar="number of filters for the gene layer",
            default=2,
            help='Hyperparameter: number of filters for the gene layer')
        parser_train.add_argument(
            "-mixed_precision",
            action='store_true',
            default=False,
            help='Flag for mixed precision to save memory (can reduce performance)')
        parser_train.add_argument(
            "-suffix",
            metavar="extra_info",
            type=str,
            default='',
            help='Add extra suffix for easier identification of the folder')
        parser_train.add_argument(
            "-out",
            metavar="outfolder",
            type=str,
            default='undefined',
            help='Use this argument to change the output directory')
        parser_train.add_argument(
            "-mask_order",
            metavar="mask_order",
            nargs='+',
            default=[],
            help='Use this to define the order of the mask if they should not be ordered by size. '
                 'list masks by full name and in order. (e.g. --mask_order SNP_gene_mask mask_gene_local'
                 ' mask_local_mid mask_mid_global)')
        parser_train.add_argument(
            "-epoch_size",
            metavar="epoch_size",
            type=int,
            default=None,
            help='Use this argument to shorten an epoch if an epoch takes to long.'
                 'Epoch_size will be the new epoch size. Epochs will be shuffled after all data has been seen')
        parser_train.add_argument(
            "-patience",
            metavar="patience",
            type=int,
            default=50,
            help='Number of epochs with no improvement after which training will be stopped.')
        parser_train.add_argument(
            "-resume",
            action='store_true',
            default=False,
            help='Flag for resuming training with existing weights (if they exist)')
        parser_train.add_argument(
            "-onehot",
            action='store_true',
            default=False,
            help='Flag for one hot encoding as a first layer in the network')        
        parser_train.add_argument(
            "-init_linear",
            action='store_true',
            default=False,
            help='initialize the one-hot encoding for the neural network with a linear assumption')
        return parser_train

    def make_parser_plot(self, parser_plot):
        parser_plot.add_argument(
            "-ID",
            type=int,
            help="ID of the experiment",
            required=True)
        parser_plot.add_argument(
            "-type",
            type=str,
            choices=['layer_weight', 'sunburst', 'manhattan_relative_importance'],
            required=True)
        parser_plot.add_argument(
            "-layer_n",
            type=int,
            help="Only used for layer weight: Number of the to be plotted layer",
            metavar="Layer_number:",
            default=0)
        parser_plot.add_argument(
            "-out",
            metavar="outfolder",
            type=str,
            default='undefined',
            help='Use this argument to change the output directory')
        parser_plot.add_argument(
            "-suffix",
            metavar="extra_info",
            type=str,
            default='',
            help='Add extra suffix if you used this in training')
        return parser_plot

    def make_parser_topology(self, parser_topology):
        parser_topology.add_argument(
            "-type",
            default='create_annovar_input', type=str,
            choices=['create_annovar_input', 'create_gene_network', 'create_pathway_KEGG', 'create_GTEx_network'],
            help="Create annovar input, create network topology from annovar output")
        parser_topology.add_argument(
            "-path",
            type=str,
            required=True,
            help="Path to the input data. For create_annovar_input this is the folder containing hase: genotype, "
                 "probes and individuals ")
        parser_topology.add_argument(
            '-study_name',
            type=str,
            required=True,
            help='Study name used in Convert. Name of the files in the genotype individuals and probe folders')
        parser_topology.add_argument(
            "-out",
            type=str,
            help="Path. Location of the results, default to ./processed_data/",
            default=os.getcwd() + '/processed_data/')
        return parser_topology



    def make_parser_interpret(self, parser_topology):
        parser_topology.add_argument(
            "-type",
            default='get_weight_scores', type=str,
            choices=['get_weight_scores', 'NID', 'RLIPP', 'DFIM',"PathExplain","DeepExplain"],
            help="choose interpretation method, choice")
        parser_topology.add_argument(
            "-resultpath",
            type=str,
            required=True,
            help="Path to the folder with the trained network (resultfolder) ")
        parser_topology.add_argument(
            '-layer',
            type=int,
            required=False,
            help='Select a layer for interpretation only necessary for NID')
        parser_topology.add_argument(
            '-num_eval',
            type=int,
            required=False,
            default = 100,
            help='Select the number of SNPs to eval in DFIM')
        parser_topology.add_argument(
            '-start_rank',
            type=int,
            required=False,
            default = 0,
            help='Multiprocessing, start from Nth ranked important variant')
        parser_topology.add_argument(
            '-end_rank',
            type=int,
            required=False,
            default = 0,
            help='Multiprocessing, stop at Nth ranked important SNP')
        parser_topology.add_argument(
            '-num_sample_pat',
            type=int,
            required=False,
            default = 1000,
            help='Select a number of patients to sample for DFIM')
        return parser_topology


if __name__ == '__main__':
    main()
