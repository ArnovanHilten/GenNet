import sys
import os
import h5py
import glob
import tables
import tqdm
import numpy as np
import pandas as pd
from utils.hase.config import basedir, CONVERTER_SPLIT_SIZE, PYTHON_PATH
os.environ['HASEDIR'] = basedir
if PYTHON_PATH is not None:
    for i in PYTHON_PATH: sys.path.insert(0, i)
from utils.hase.hdgwas.tools import Timer, check_converter
from utils.hase.hdgwas.converter import GenotypePLINK, GenotypeMINIMAC, GenotypeVCF
from utils.hase.hdgwas.data import Reader


def hase_convert(args):
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
    return
def merge_hdf5_hase(args):
    filepath_hase = args.outfolder + '/genotype/{}_' + args.study_name + '.h5'
    g = h5py.File(filepath_hase.format(1), 'r')['genotype']
    num_pat = g.shape[1]
    number_of_files = len(glob.glob(args.outfolder + "/genotype/*.h5"))
    print('number of files ', number_of_files)

    f = tables.open_file(args.outfolder + args.study_name + '_genotype.h5', mode='w')
    atom = tables.Int8Col()
    filter_zlib = tables.Filters(complib='zlib', complevel=1)
    f.create_earray(f.root, 'data', atom, (0, num_pat), filters=filter_zlib)
    f.close()

    print("\n merge all files...")
    f = tables.open_file(args.outfolder + args.study_name + '_genotype.h5', mode='a')
    for i in tqdm.tqdm(range(number_of_files)):
        gen_tmp = h5py.File(filepath_hase.format(i), 'r')['genotype']
        f.root.data.append(np.array(np.round(gen_tmp[:, :]), dtype=np.int))
    f.close()


def impute_hase_hdf5_no_chunk(args):
    t = tables.open_file(args.outfolder + args.study_name + '_genotype.h5', mode='r')
    print('merged shape =', t.root.data.shape)
    num_SNPS = t.root.data.shape[0]
    num_pat = t.root.data.shape[1]

    hdf5_name = args.study_name + '_genotype_imputed.h5'
    p = pd.read_hdf(args.outfolder + '/probes/' + args.study_name + ".h5")
    print('probe shape =', p.shape)

    print("\n Impute...")
    f = tables.open_file(args.outfolder + args.study_name + '_genotype_imputed.h5', mode='w')
    atom = tables.Int8Col()

    filter_zlib = tables.Filters(complib='zlib', complevel=1)
    f.create_earray(f.root, 'data', atom, (0, num_pat), filters=filter_zlib)
    f.close()

    stdSNPs = np.zeros(num_SNPS)
    f = tables.open_file(args.outfolder + args.study_name + '_genotype_imputed.h5', mode='a')

    for i in tqdm.tqdm(range(t.root.data.shape[0])):
        d = t.root.data[i, :].astype("float32")
        m = np.where(d == 9)
        d[m] = np.nan
        d[m] = np.nanmean(d)
        d = d[np.newaxis, :]
        f.root.data.append(np.round(d).astype(np.int8))
        stdSNPs[i] = np.std(d)
    f.close()
    t.close()

    np.save(args.outfolder + args.study_name + '_std.npy', stdSNPs)
    return hdf5_name

def impute_hase_hdf5(args):
    t = tables.open_file(args.outfolder + args.study_name + '_genotype.h5', mode='r')
    print('merged shape =', t.root.data.shape)
    num_SNPS = t.root.data.shape[0]
    num_pat = t.root.data.shape[1]

    hdf5_name = args.study_name + '_genotype_imputed.h5'
    p = pd.read_hdf(args.outfolder + '/probes/' + args.study_name + ".h5")
    print('probe shape =', p.shape)

    print("\n Impute...")
    f = tables.open_file(args.outfolder + args.study_name + '_genotype_imputed.h5', mode='w')
    atom = tables.Int8Col()

    filter_zlib = tables.Filters(complib='zlib', complevel=1)
    f.create_earray(f.root, 'data', atom, (0, num_pat), filters=filter_zlib)
    f.close()

    stdSNPs = np.zeros(num_SNPS)
    f = tables.open_file(args.outfolder + args.study_name + '_genotype_imputed.h5', mode='a')

    chunk = args.tcm // num_SNPS
    chunk = int(np.clip(chunk, 1, num_pat))
    print(chunk)

    for part in tqdm.tqdm(range(int(np.ceil(num_SNPS / chunk) + 1))):
        begins = part * chunk
        tills = min(((part + 1) * chunk), num_SNPS)
        d = t.root.data[begins:tills, :].astype("float32")
        d[d == 9] = np.nan
        a = np.where(np.isnan(d), np.ma.array(d, mask=np.isnan(d)).mean(axis=1)[:, np.newaxis], d)
        stdSNPs[begins:tills] = np.std(a, axis=1)
        f.root.data.append(np.round(d).astype(np.int8))
    f.close()
    t.close()

    np.save(args.outfolder + args.study_name + '_std.npy', stdSNPs)
    return hdf5_name

def exclude_variants(args):
    print("Selecting the variants..")
    t = tables.open_file(args.outfolder + args.study_name + '_genotype_imputed.h5', mode='r')
    data = t.root.data
    num_pat = data.shape[1]
    num_variants = data.shape[0]

    used_indices = pd.read_csv(args.variants, header = None)



    hdf5_name = args.study_name + '_genotype_used.h5'

    if len(used_indices) == num_variants:
        used_indices = used_indices.index.values[used_indices.values.flatten()]
        f = tables.open_file(args.outfolder + args.study_name + '_genotype_used.h5', mode='w')
        f.create_earray(f.root, 'data', tables.IntCol(), (0, num_pat), expectedrows=len(used_indices),
                        filters=tables.Filters(complib='zlib', complevel=1))
        f.close()

        f = tables.open_file(args.outfolder + args.study_name + '_genotype_used.h5', mode='a')
        for feat in tqdm.tqdm(used_indices):
            a = data[feat, :]
            a = np.reshape(a, (1, -1))
            f.root.data.append(a)
        f.close()
        t.close()
        return hdf5_name
    else:
        print("Something wrong with the included_snps file.")
        print("Expected " + str(num_variants) + "but got " + str(len(used_indices)))
        print('Used indices looks like:', used_indices)
        exit()

def transpose_genotype(args, hdf_name):

    t = tables.open_file(args.outfolder +  hdf_name, mode='r')
    data = t.root.data
    num_pat = data.shape[1]
    num_feat = data.shape[0]
    chunk = args.tcm // num_feat
    chunk = int(np.clip(chunk, 1, num_pat))
    f = tables.open_file(args.outfolder + '/genotype.h5', mode='w')
    f.create_earray(f.root, 'data', tables.IntCol(), (0, num_feat), expectedrows=num_pat,
                              filters=tables.Filters(complib='zlib', complevel=1))
    f.close()

    f = tables.open_file(args.outfolder + '/genotype.h5', mode='a')

    for pat in tqdm.tqdm(range(int(np.ceil(num_pat / chunk) + 1)) ):
        begins = pat * chunk
        tills = min(((pat + 1) * chunk), num_pat)
        a = np.array(data[:, begins:tills], dtype=int)
        a = a.T
        f.root.data.append(a)
    f.close()
    t.close()
    print("Completed", args.study_name)



def convert(args):

    # hase_convert(args)
    if type(args.out) is list:
        args.outfolder = args.out[0]
    else:
        args.outfolder = args.out

    if type(args.study_name) is list:
        args.study_name = args.study_name[0]
    else:
        args.study_name = args.study_name

    # merge_hdf5_hase(args)
    # hdf5_name = impute_hase_hdf5(args)


    if args.variants is None:
        pass
    else:
        hdf5_name =exclude_variants(args)

    transpose_genotype(args, hdf_name=hdf5_name)

    


