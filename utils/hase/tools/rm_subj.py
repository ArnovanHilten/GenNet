import os
import h5py
import pandas as pd
import numpy as np
import argparse
import tables

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to remove subjects from HASE hdf5 files')
    parser.add_argument("-g", required=True, type=str, help="path/paths to genotype data folder")
    parser.add_argument('-study_name', type=str, required=True, default=None, help=' Study names')
    parser.add_argument('-exclude_ids', type=str, default=None,
                        help='Table with IDs to exclude from data. Should have ID column')
    args = parser.parse_args()
    print(args)

    if args.exclude_ids is not None:
        df = pd.DataFrame.from_csv(args.snp_id_inc, index_col=None)
        print(df.head())
        if 'ID' not in df.columns:
            raise ValueError('{} table does not have ID or columns'.format(args.exclude_ids))

        df['ID'] = df.ID.astype(str)

        df_ids = pd.read_hdf(os.path.join(args.g, 'individuals', args.study_name + '.h5'), 'individuals')
        df_ids['individual'] = df_ids.individual.astype(str)

        info_index = df_ids.individual.isin(df.ID)
        remove_index = np.where(info_index == True)[0]
        keep_index = np.where(info_index == False)[0]

        if len(remove_index) == 0:
            print('There is no ids to remove!')
            exit(0)
        if len(keep_index) == len(df_ids.individual):
            print("Need to remove everybody!!! ")
            exit(0)

        individuals = df_ids.individual[~remove_index]
        chunk = pd.DataFrame.from_dict({"individual": individuals})
        chunk.to_hdf(os.path.join(args.g, 'individuals', args.study_name + '.h5'), key='individuals', format='table',
                     min_itemsize=25, complib='zlib', complevel=9)

        for g_file in os.listdir(os.path.join(args.g, 'genotype')):
            print(g_file)

            data = h5py.File(os.path.join(args.g, 'genotype', g_file), 'r')['genotype'][...]
            data = data[:, keep_index]

            h5_gen_file = tables.open_file(
                os.path.join(args.g, 'genotype', g_file), 'w', title=args.study_name)

            atom = tables.Float16Atom()
            genotype = h5_gen_file.create_carray(h5_gen_file.root, 'genotype', atom,
                                                 (data.shape),
                                                 title='Genotype',
                                                 filters=tables.Filters(complevel=9, complib='zlib'))
            genotype[:] = data
            h5_gen_file.close()



