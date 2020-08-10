import os
import sys
from utils.hase.config import basedir, CONVERTER_SPLIT_SIZE, PYTHON_PATH
os.environ['HASEDIR'] = basedir
if PYTHON_PATH is not None:
    for i in PYTHON_PATH: sys.path.insert(0, i)
from utils.hase.hdgwas.tools import Timer, check_converter
from utils.hase.hdgwas.converter import GenotypePLINK, GenotypeMINIMAC, GenotypeVCF
from utils.hase.hdgwas.data import Reader


def hase():
    R = Reader('genotype')

    R.start(args.genotype[0], vcf=args.vcf)

    with Timer() as t:
        if R.format == 'PLINK':
            G = GenotypePLINK(args.study_name[0], reader=R)
            G.split_size = CONVERTER_SPLIT_SIZE
            G.plink2hdf5(out=args.out)

        elif R.format == 'MINIMAC':
            G = GenotypeMINIMAC(args.study_name[0], reader=R)
            if args.cluster == 'y':
                G.cluster = True
            G.split_size = CONVERTER_SPLIT_SIZE
            G.MACH2hdf5(args.out, id=args.id)

        elif R.format == 'VCF':
            G = GenotypeVCF(args.study_name[0], reader=R)
            if args.cluster == 'y':
                G.cluster = True
            G.split_size = CONVERTER_SPLIT_SIZE
            G.VCF2hdf5(args.out)
        else:
            raise ValueError('Genotype data should be in PLINK/MINIMAC/VCF format and alone in folder')

    check_converter(args.out, args.study_name[0])
    print(('Time to convert all data: {} sec'.format(t.secs)))