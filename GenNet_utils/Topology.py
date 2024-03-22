import os
import numpy as np
import pandas as pd
import scipy
import scipy.sparse


def Create_Annovar_input(args):
    hasepath = args.path
    studyname = args.study_name
    savepath = args.out

    if os.path.exists(hasepath + '/probes/' + studyname + '_selected.h5'):
        probes = pd.read_hdf(hasepath + '/probes/' + studyname + '_selected.h5', mode="r")
    else:
        probes = pd.read_hdf(hasepath + '/probes/' + studyname + '.h5', mode="r")
        print(probes.shape)

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

    print('Number of variants', annovar_input.shape)

    annovar_input_path = savepath + '/annovar_input_' + studyname + '.csv'
    annovar_input.to_csv(annovar_input_path, sep="\t", index=False, header=False)

    print('\n')
    print('Annovar input files ready \n')
    print("Install annovar: https://annovar.openbioinformatics.org/en/latest/user-guide/download/")
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
    print(topology['index_col'].max())
    topology.columns = ['chr', 'layer0_node', 'layer0_name', 'layer1_node', 'layer1_name']

    topology.to_csv(savepath + "/topology.csv")

    print('Topology file saved:', savepath + "/topology.csv")

    # Additionally create a mask for creating larger networks
    data = np.ones(gene_annotation.shape[0], bool)
    coord = (gene_annotation["index_col"].values, gene_annotation["gene_id"].values)
    SNP_gene_matrix = scipy.sparse.coo_matrix(((data), coord),
                                              shape=(gene_annotation["index_col"].max() + 1,
                                                     gene_annotation["gene_id"].max() + 1))
    scipy.sparse.save_npz(savepath + '/SNP_gene_mask', SNP_gene_matrix)
    print("Alternatively you can choose to use the .npz mask (building blocks for deeper networks)",
          savepath + '/SNP_gene_mask', 'The mask has shape:', SNP_gene_matrix.shape)

    return gene_annotation


def Create_gene_to_pathway_KEGG(args):
    gene_overview = Create_gene_network_topology(args)
    savepath = args.out + '/'

    pathway_overview_higher_levels = pd.read_csv('resources/pathways/pathway_overview_KEGG.csv').drop(
        "local_id", axis=1)

    CPDB_pathway_overview = pd.read_csv('resources/pathways/CPDB_pathways_genes.tab', sep='\t')

    pathway_overview_source = CPDB_pathway_overview[CPDB_pathway_overview["source"] == "KEGG"].copy()
    pathway_overview_source["hsaid"] = pd.to_numeric(pathway_overview_source['external_id'].str.replace("path:hsa", ""),
                                                     errors='coerce')
    pathway_overview_source = pathway_overview_source.merge(pathway_overview_higher_levels, on="hsaid")
    pathway_overview_source["local_id"] = np.arange(len(pathway_overview_source))
    pathway_overview_source["num_genes"] = pathway_overview_source["hgnc_symbol_ids"].str.split(",").str.len()

    coordinate = np.empty(shape=(0, 2), dtype=int)
    for pathnum in pathway_overview_source["local_id"]:
        gene_ids_in_pathway = np.array(gene_overview[gene_overview["gene"].isin(
            set(pathway_overview_source["hgnc_symbol_ids"].iloc[pathnum].split(",")))]["gene_id"], dtype=int)
        current_coordinates = np.ones((len(gene_ids_in_pathway), 2), dtype=int) * pathnum
        current_coordinates[:, 0] = gene_ids_in_pathway
        # print(pathnum, current_coordinates.shape[0], pathway_overview_source["num_genes"].iloc[pathnum])
        coordinate = np.append(coordinate, current_coordinates, axis=0)

    data = np.ones(len(coordinate), np.bool)
    mask_gene_local = scipy.sparse.coo_matrix(((data), (coordinate[:, 0], coordinate[:, 1])), shape=(
        gene_overview["gene_id"].max() + 1, pathway_overview_source["local_id"].max() + 1))
    scipy.sparse.save_npz(savepath + 'mask_gene_local', mask_gene_local)
    print("mask_gene_local saved in ", savepath, "with shape", mask_gene_local.shape, "and", len(coordinate),
          "elements")

    HSA_overview_mid_unique = pathway_overview_source.drop_duplicates("local_id")

    mask_pathway_mid = scipy.sparse.coo_matrix(
        (np.ones(len(HSA_overview_mid_unique), np.bool),
         (HSA_overview_mid_unique['local_id'], HSA_overview_mid_unique['mid_id'].values)),
        shape=(HSA_overview_mid_unique['local_id'].max() + 1, HSA_overview_mid_unique['mid_id'].max() + 1),
    )
    scipy.sparse.save_npz(savepath + 'mask_local_mid', mask_pathway_mid)
    print("mask_local_mid saved in ", savepath, "with shape", mask_pathway_mid.shape, "and",
          len(HSA_overview_mid_unique), "elements")

    HSA_overview_global_unique = pathway_overview_source.drop_duplicates("mid_id")

    mask_pathway_global = scipy.sparse.coo_matrix(
        (np.ones(len(HSA_overview_global_unique), np.bool),
         (HSA_overview_global_unique['mid_id'], HSA_overview_global_unique['global_id'].values)),
        shape=(HSA_overview_global_unique['mid_id'].max() + 1, HSA_overview_global_unique['global_id'].max() + 1),
    )
    scipy.sparse.save_npz(savepath + 'mask_mid_global', mask_pathway_global)
    print("mask_mid_global saved in ", savepath, "with shape", mask_pathway_global.shape,
          "and", len(HSA_overview_global_unique), "elements")


def Create_gene_to_GTEx(args):
    print("This will be implemented in the next version")
    raise NotImplementedError


def topology(args):
    if args.type == 'create_annovar_input':
        Create_Annovar_input(args)
    elif args.type == 'create_gene_network':
        Create_gene_network_topology(args)
    elif args.type == 'create_pathway_KEGG':
        Create_gene_to_pathway_KEGG(args)
    elif args.type == 'create_GTEx_network':
        Create_gene_to_GTEx(args)
    else:
        print("invalid type:", args.type)
        exit()
