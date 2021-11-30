import glob
import os
import sys
import argparse
import h5py
import numpy as np
import pandas as pd
import tables
import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from GenNet_utils.Utility_functions import query_yes_no
from GenNet_utils.hase.config import basedir, CONVERTER_SPLIT_SIZE, PYTHON_PATH

os.environ['HASEDIR'] = basedir
if PYTHON_PATH is not None:
    for i in PYTHON_PATH: sys.path.insert(0, i)
from GenNet_utils.hase.hdgwas.tools import Timer, check_converter
from GenNet_utils.hase.hdgwas.converter import GenotypePLINK, GenotypeMINIMAC, GenotypeVCF
from GenNet_utils.hase.hdgwas.data import Reader


def hase_convert(args):
    if (os.path.exists(args.outfolder + '/probes/')) and (os.path.exists(args.outfolder + '/genotype/')) and (
            os.path.exists(args.outfolder + '/individuals/')):
        print("The folders: probes, genotype and individuals already exist. Data seems already in HASE format. Delete "
              "the folders if the files are not converted properly. Continuing with the current files:")
        return
    else:
        print('using', args.outfolder)

    R = Reader('genotype')

    R.start(args.genotype[0], vcf=args.vcf)

    with Timer() as t:
        if R.format == 'PLINK':
            G = GenotypePLINK(args.study_name[0], reader=R)
            G.split_size = CONVERTER_SPLIT_SIZE
            G.plink2hdf5(out=args.out)

        elif R.format == 'MINIMAC':
            G = GenotypeMINIMAC(args.study_name[0], reader=R)
            G.split_size = CONVERTER_SPLIT_SIZE
            G.MACH2hdf5(args.out, id=args.id)

        elif R.format == 'VCF':
            G = GenotypeVCF(args.study_name[0], reader=R)
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
    filter_zlib = tables.Filters(complib='zlib', complevel=args.comp_level)
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

    filter_zlib = tables.Filters(complib='zlib', complevel=args.comp_level)
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

    filter_zlib = tables.Filters(complib='zlib', complevel=args.comp_level)
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

    used_indices = pd.read_csv(args.variants, header=None)

    hdf5_name = args.study_name + '_genotype_used.h5'

    if len(used_indices) == num_variants:
        used_indices = used_indices.index.values[used_indices.values.flatten()]
        f = tables.open_file(args.outfolder + args.study_name + '_genotype_used.h5', mode='w')
        f.create_earray(f.root, 'data', tables.IntCol(), (0, num_pat), expectedrows=len(used_indices),
                        filters=tables.Filters(complib='zlib', complevel=args.comp_level))
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


def transpose_genotype(args):
    hdf5_name = '/' + args.study_name + '_genotype_used.h5'
    if (os.path.exists(args.outfolder + hdf5_name)):
        t = tables.open_file(args.outfolder + hdf5_name, mode='r')
    else:
        print('using', args.outfolder + args.study_name + '_genotype_imputed.h5')
        t = tables.open_file(args.outfolder + args.study_name + '_genotype_imputed.h5', mode='r')

    data = t.root.data
    num_pat = data.shape[1]
    num_feat = data.shape[0]
    chunk = args.tcm // num_feat
    chunk = int(np.clip(chunk, 1, num_pat))
    print("chuncksize =", chunk)

    f = tables.open_file(args.outfolder + '/genotype.h5', mode='w')
    f.create_earray(f.root, 'data', tables.IntCol(), (0, num_feat), expectedrows=num_pat,
                    filters=tables.Filters(complib='zlib', complevel=args.comp_level))
    f.close()

    f = tables.open_file(args.outfolder + '/genotype.h5', mode='a')

    for pat in tqdm.tqdm(range(int(np.ceil(num_pat / chunk) + 1))):
        begins = pat * chunk
        tills = min(((pat + 1) * chunk), num_pat)
        a = np.array(data[:, begins:tills], dtype=np.int8)
        a = a.T
        f.root.data.append(a)
    f.close()
    t.close()
    print("Completed", args.study_name)


def transpose_genotype_scheduler(args):
    local_run = False

    hdf5_name = '/' + args.study_name + '_genotype_used.h5'
    if (os.path.exists(args.outfolder + hdf5_name)):
        t = tables.open_file(args.outfolder + hdf5_name, mode='r')
    else:
        print('using', args.outfolder + args.study_name + '_genotype_imputed.h5')
        t = tables.open_file(args.outfolder + args.study_name + '_genotype_imputed.h5', mode='r')

    data = t.root.data
    num_pat = data.shape[1]
    t.close()
    print("n_jobs", args.n_jobs)
    jobchunk = int(np.ceil(num_pat / args.n_jobs))

    print("____________________________________________________________________")
    print('Submitting' + str(args.n_jobs))
    print('Please make sure this file has the correct settings for your cluster')
    print("____________________________________________________________________")
    with open('./GenNet_utils/submit_SLURM_job.sh', 'r') as f:
        print(f.read())
    print("____________________________________________________________________")

    if query_yes_no(question="Does the file contain the right settings?"):
        for job_n in (range(int(np.ceil(num_pat / jobchunk)))):
            begins = job_n * jobchunk
            tills = min(((job_n + 1) * jobchunk), num_pat)
            if local_run:
                # transpose_genotype_job(begins, tills, job_n, args.study_name, args.outfolder, args.tcm)
                str_sbatch = 'nohup python ./GenNet_utils/Convert.py -job_begins ' + str(begins) + ' -job_tills ' + str(
                    tills) + ' -job_n ' + str(job_n) + ' -study_name ' + str(args.study_name) + ' -outfolder ' + str(
                    args.outfolder) + ' -tcm ' + str(args.tcm) + " & "
                print(str_sbatch)
                os.system(str_sbatch)
            else:
                str_sbatch = 'sbatch ./GenNet_utils/submit_SLURM_job.sh ' + str(begins) + ' ' + str(
                    tills) + ' ' + str(job_n) + ' ' + str(args.study_name) + ' ' + str(
                    args.outfolder) + ' ' + str(args.tcm)
                print(str_sbatch)
                os.system(str_sbatch)
        print("all jobs submitted please run GenNet convert -step merge_transpose next")
    else:
        print("Please change the script")
        transpose_genotype_scheduler(args)


def transpose_genotype_job(job_begins, job_tills, job_n, study_name, outfolder, tcm):
    print("job_n:", job_n, 'job_begins:', job_begins, 'job_tills:', job_tills)
    hdf5_name = '/' + study_name + '_genotype_used.h5'
    if (os.path.exists(outfolder + hdf5_name)):
        t = tables.open_file(outfolder + hdf5_name, mode='r')
    else:
        print('using', outfolder + study_name + '_genotype_imputed.h5')
        t = tables.open_file(outfolder + study_name + '_genotype_imputed.h5', mode='r')

    data = t.root.data
    num_pat = data.shape[1]
    num_feat = data.shape[0]
    chunk = tcm // num_feat
    chunk = int(np.clip(chunk, 1, num_pat))
    print("chuncksize =", chunk)

    f = tables.open_file(outfolder + '/genotype_' + str(job_n) + '.h5', mode='w')
    f.create_earray(f.root, 'data', tables.IntCol(), (0, num_feat), expectedrows=num_pat,
                    filters=tables.Filters(complib='zlib', complevel=args.comp_level))
    f.close()
    n_in_job = job_tills - job_begins
    f = tables.open_file(outfolder + '/genotype_' + str(job_n) + '.h5', mode='a')

    for subjects in tqdm.tqdm(range(int(np.ceil(n_in_job / chunk) + 1))):
        begins = job_begins + subjects * chunk
        tills = min((job_begins + (subjects + 1) * chunk), job_tills)
        a = np.array(data[:, begins:tills], dtype=np.int8)
        a = a.T
        f.root.data.append(a)
    f.close()
    t.close()
    print("Completed", job_n)


def merge_transpose(args):
    hdf5_name = '/' + args.study_name + '_genotype_used.h5'
    if (os.path.exists(args.outfolder + hdf5_name)):
        t = tables.open_file(args.outfolder + hdf5_name, mode='r')
    else:
        print('using', args.outfolder + args.study_name + '_genotype_imputed.h5')
        t = tables.open_file(args.outfolder + args.study_name + '_genotype_imputed.h5', mode='r')

    num_pat = t.root.data.shape[1]
    num_feat = t.root.data.shape[0]
    chunk = args.tcm // num_feat
    chunk = int(np.clip(chunk, 1, num_pat))
    t.close()

    number_of_files = len(glob.glob(args.outfolder + "/genotype_*.h5"))

    if number_of_files == args.n_jobs:
        print('number of files ', number_of_files)
    else:
        print("WARNING!", 'number_of_files', number_of_files, 'args.n_jobs', args.n_jobs)
        print("Continueing to merge with n_jobs, merging:", args.n_jobs, "files")

    f = tables.open_file(args.outfolder + '/genotype.h5', mode='w')
    f.create_earray(f.root, 'data', tables.IntCol(), (0, num_feat), expectedrows=num_pat,
                    filters=tables.Filters(complib='zlib', complevel=args.comp_level))
    f.close()

    f = tables.open_file(args.outfolder + '/genotype.h5', mode='a')

    gen_tmp = tables.open_file(args.outfolder + '/genotype_' + str(0) + '.h5', mode='r')
    filesize = gen_tmp.root.data.shape[0]
    gen_tmp.close()
    print("\n merge all files...")
    if chunk > filesize:
        print("chunking is not necessary")
        for job_n in tqdm.tqdm(range(args.n_jobs)):
            gen_tmp = tables.open_file(args.outfolder + '/genotype_' + str(job_n) + '.h5', mode='r')
            f.root.data.append(np.array(np.round(gen_tmp.root.data[:, :]), dtype=np.int))
            gen_tmp.close()
        f.close()
    else:
        print("per chunk", chunk, "subjects")
        for job_n in tqdm.tqdm(range(args.n_jobs)):
            gen_tmp = tables.open_file(args.outfolder + '/genotype_' + str(job_n) + '.h5', mode='r')
            for chunckblock in range(int(np.ceil(gen_tmp.root.data.shape[0] / chunk))):
                begins = chunckblock * chunk
                tills = min(((chunckblock + 1) * chunk), gen_tmp.root.data.shape[0])
                f.root.data.append(np.array(np.round(gen_tmp.root.data[begins:tills, :]), dtype=np.int))
            gen_tmp.close()
        f.close()
    print("completed")
    f = tables.open_file(args.outfolder + '/genotype.h5', mode='r')
    finalshape = f.root.data.shape
    f.close()
    print("Shape merged file", finalshape)


def exclude_variants_probes(args):
    if args.variants is None:
        return

    used_indices = pd.read_csv(args.variants, header=None)
    used_indices = used_indices.index.values[used_indices.values.flatten()]
    probes = pd.read_hdf(args.outfolder + '/probes/' + args.study_name + '.h5', mode="r")
    print("Probes shape", probes.shape)
    print("Selecting variants..")
    probes = probes.iloc[used_indices]
    print("Probes shape", probes.shape)
    probes.to_hdf(args.outfolder + '/probes/' + args.study_name + '_selected.h5', key='probes', format='table',
                  data_columns=True, append=True,
                  complib='zlib', complevel=args.comp_level, min_itemsize=45)


def getsum_(path_file):
    h5file = tables.open_file(path_file, mode="r")
    rowsum = 0
    for i in range(h5file.root.data.shape[0]):
        rowsum += np.sum(h5file.root.data[i, :])
    return rowsum


def get_checksum(path_file1, path_file2):
    sumfile1 = getsum_(path_file1)
    sumfile2 = getsum_(path_file2)

    print(sumfile1)
    print(sumfile2)
    assert sumfile1 == sumfile2


def select_first_arg_out(args):
    if type(args.out) is list:
        args.outfolder = args.out[0] + '/'
    else:
        args.outfolder = args.out + '/'


def select_first_arg_study(args):
    if type(args.study_name) is list:
        args.study_name = args.study_name[0]
    else:
        args.study_name = args.study_name


def convert(args):
    if args.step == "all":
        # 1. hase
        select_first_arg_out(args)
        hase_convert(args)
        # 2. merge
        select_first_arg_study(args)
        merge_hdf5_hase(args)
        # 3. impute
        impute_hase_hdf5(args)
        # 4. exclude variants
        exclude_variants_probes(args)
        # 5. transpose
        transpose_genotype(args)


    elif args.step == "hase_convert":
        select_first_arg_out(args)
        hase_convert(args)
    elif args.step == "merge":
        select_first_arg_out(args)
        select_first_arg_study(args)
        merge_hdf5_hase(args)
    elif args.step == "impute":
        select_first_arg_out(args)
        select_first_arg_study(args)
        impute_hase_hdf5(args)
    elif args.step == "exclude":
        select_first_arg_out(args)
        select_first_arg_study(args)
        exclude_variants_probes(args)
    elif ((args.step == "transpose") & (args.n_jobs == 1)):
        select_first_arg_out(args)
        select_first_arg_study(args)
        transpose_genotype(args)
    elif ((args.step == "transpose") & (args.n_jobs > 1)):
        select_first_arg_out(args)
        select_first_arg_study(args)
        transpose_genotype_scheduler(args)
    elif ((args.step == "merge_transpose")):
        select_first_arg_out(args)
        select_first_arg_study(args)
        merge_transpose(args)
    elif ((args.step == "checksum")):
        print("Checksum does not count up if you exclude variants")
        select_first_arg_out(args)
        select_first_arg_study(args)
        get_checksum(path_file1=args.outfolder + '/genotype.h5',
                     path_file2=args.outfolder + args.study_name + '_genotype.h5')

    else:
        print('invalid parameters')
        exit()


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "-job_begins",
        type=int,
    )
    CLI.add_argument(
        "-job_tills",
        type=int,
    )
    CLI.add_argument(
        "-job_n",
        type=int,
    )
    CLI.add_argument(
        "-study_name",
        type=str,
        default=32,
    )
    CLI.add_argument(
        "-outfolder",
        type=str
    )
    CLI.add_argument(
        "-tcm",
        type=int,
    )
    arg = CLI.parse_args()
    transpose_genotype_job(job_begins=arg.job_begins,
                           job_tills=arg.job_tills,
                           job_n=arg.job_n,
                           study_name=arg.study_name,
                           outfolder=arg.outfolder,
                           tcm=arg.tcm)
