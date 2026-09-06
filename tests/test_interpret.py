import pytest

from GenNet_utils.Interpret import interpret


class InterpretArgs:
    def __init__(self, resultpath, type_name, genotype_path="examples/example_classification/"):
        self.resultpath = resultpath
        self.type = type_name
        self.layer = None
        self.genotype_path = genotype_path
        self.path = genotype_path
        self.datapath = genotype_path
        self.onehot = 0
        self.num_eval = 10
        self.num_sample_pat = 10


def test_interpret_rejects_unknown_type():
    args = InterpretArgs(resultpath="/tmp/gennet-missing/", type_name="not_a_real_method")
    with pytest.raises(SystemExit):
        interpret(args)
