import shutil
from pathlib import Path

from GenNet_utils.Convert import convert


class ConvertArgs:
    def __init__(self, genotype, out, study_name="toy_data"):
        self.mode = "convert"
        self.genotype = [str(genotype)]
        self.study_name = [study_name]
        self.out = str(out)
        self.outfolder = str(out) + "/"
        self.step = "all"
        self.vcf = False
        self.variants = None
        self.tcm = 500000000
        self.n_jobs = 1
        self.comp_level = 1
        self.id = None


def test_convert_plink2_to_tmp(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    src = repo / "examples" / "plink2"
    work = tmp_path / "plink"
    out = tmp_path / "out"
    shutil.copytree(src, work)
    out.mkdir()
    convert(ConvertArgs(work, out))
    assert any(out.rglob("*.h5")) or any(work.rglob("*.h5"))
