import os

import numpy as np
import pandas as pd


def Create_Annovar_input(args):
    hasepath = args.path
    studyname = args.study_name
    savepath = args.out

    probes = pd.read_hdf(hasepath + '/probes/' + studyname + '.h5')
    print(probes.shape)
    num_probes = probes.shape[0]
    probes.head()

    if os.path.exists(hasepath + '/probes/' + studyname + '_hash_table.csv.gz'):
        hashtable = pd.read_csv(hasepath + '/probes/' + studyname + '_hash_table.csv.gz', compression="gzip", sep='\t')
    else:
        hashtable = pd.read_csv(hasepath + '/probes/' + studyname + '_hash_table.csv', sep='\t')

    hashtable['allele1'] = hashtable['keys']
    unhashed_probes = probes.merge(hashtable, on='allele1', how="left")
    unhashed_probes = unhashed_probes.drop(columns=["keys", "allele1"])
    unhashed_probes = unhashed_probes.rename(columns={'allele': 'allele1'})

    # reload hashtable for other allele

    if os.path.exists(hasepath + '/probes/' + studyname + '_hash_table.csv.gz'):
        hashtable = pd.read_csv(hasepath + '/probes/' + studyname + '_hash_table.csv.gz', compression="gzip", sep='\t')
    else:
        hashtable = pd.read_csv(hasepath + '/probes/' + studyname + '_hash_table.csv', sep='\t')

    hashtable['allele2'] = hashtable['keys']
    unhashed_probes = unhashed_probes.merge(hashtable, on='allele2', how="left")
    unhashed_probes = unhashed_probes.drop(columns=["keys", "allele2"])
    unhashed_probes = unhashed_probes.rename(columns={'allele': 'allele2'})

    # clean
    annovar_input = unhashed_probes.drop(columns=["ID", "distance"])
    annovar_input["bp2"] = annovar_input["bp"]
    annovar_input["index_col"] = annovar_input.index
    annovar_input = annovar_input[['CHR', 'bp', "bp2", "allele1", "allele2", "index_col"]]

    print(annovar_input.shape)
    if args.variants is None:
        pass
    else:
        maxvalue = annovar_input['index_col'].max()
        used_indices = pd.read_csv(args.variants, header=None)
        used_indices = used_indices.index.values[used_indices.values.flatten()]
        annovar_input = annovar_input.loc[annovar_input['index_col'].isin(used_indices)]
        annovar_input['index_col'] = np.arange(len(annovar_input))     # after splitting out the unused variants the numbering needs to be reset to match the genotype matrix
        print(annovar_input['index_col'].max(), maxvalue)
        if annovar_input['index_col'].max() < maxvalue:
            np.save(os.getcwd() + "/processed_data/network_input_shape.npy", int(maxvalue))
            print("Since the shape is now smaller than the file size the original shape is saved:",maxvalue )
    print('Number of variants', annovar_input.shape)

    annovar_input_path = savepath + '/annovar_input_' + studyname + '.csv'
    annovar_input.to_csv(annovar_input_path, sep="\t", index=False, header=False)
    annovar_input.head()
    print('\n')
    print('Annovar input files ready \n')
    print("Install annovar: https://doc-openbio.readthedocs.io/projects/annovar/en/latest/user-guide/download/")
    print("Navigate to annovar, e.g cd /home/charlesdarwin/annovar/")
    print("Update annovar:\n perl annotate_variation.pl -buildver hg19 -downdb -webfrom annovar refGene humandb/")
    print("Run:\n perl annotate_variation.pl -geneanno -dbtype refGene -buildver hg19 " + str(
        savepath) + "/annovar_input_" + str(studyname) + ".csv humandb --outfile " + str(savepath) + "/" + str(
        studyname) + "_RefGene")
    print('\n')
    print(
        'After obtaining the Annovar annotations, run topology create_gene_network to get the topology file for the SNPs-gene-output network:')


def Create_gene_network_topology(args):
    datapath = args.path + '/'
    studyname = args.study_name
    savepath = args.out + '/'

    print(args.study_name)

    gene_annotation = pd.read_csv(datapath + str(studyname) + "_RefGene.variant_function", sep='\t', header=None)
    gene_annotation.columns = ['into/exonic', 'gene', 'chr', 'bps', 'bpe', "mutation1", "mutation2", 'index_col']
    gene_annotation['gene'] = gene_annotation['gene'].str.replace(r"\,.*", "")
    # gene_annotation['dist'] = gene_annotation['gene'].str.extract(r"(?<=dist\=)(.*)(?=\))")
    gene_annotation['gene'] = gene_annotation['gene'].str.replace(r"\(.*\)", "")
    gene_annotation['gene'] = gene_annotation['gene'].str.replace(r"\(.*", "")
    gene_annotation['gene'] = gene_annotation['gene'].str.replace(r"\;.*", "")
    gene_annotation = gene_annotation[(gene_annotation['gene'] != "NONE")]
    gene_annotation = gene_annotation.dropna()

    gene_list = gene_annotation.drop_duplicates("gene")
    gene_list = gene_list.sort_values(by=["chr", "bps"], ascending=[True, True])
    gene_list["gene_id"] = np.arange(len(gene_list))
    gene_list = gene_list[["gene", "gene_id"]]

    gene_annotation = gene_annotation.merge(gene_list, on="gene")
    gene_annotation = gene_annotation.sort_values(by="index_col", ascending=True)

    gene_annotation = gene_annotation.assign(
        chrbp='chr' + gene_annotation.chr.astype(str) + ':' + gene_annotation.bps.astype(str))
    gene_annotation.to_csv(savepath + "/gene_network_description.csv")

    topology = gene_annotation[["chr", "index_col", "chrbp", "gene_id", "gene"]]
    topology.columns = ['chr', 'layer0_node', 'layer0_name', 'layer1_node', 'layer1_name']
    topology.to_csv(savepath + "/topology.csv")

    print('Topology file saved:', savepath + "/topology.csv")


def topology(args):
    if args.type == 'create_annovar_input':
        Create_Annovar_input(args)
    elif args.type == 'create_gene_network':
        Create_gene_network_topology(args)
    else:
        print("invalid type:", args.type)
        exit()
