import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from GenNet_utils.Utility_functions import query_yes_no, get_paths


def cicos_plot(resultpath, importance_csv, plot_weights=True, plot_arrows=False):
    print("in progress...")
    colormap = ['#7dcfe2', '#4b78b5', 'darkgrey', 'dimgray'] * 1000

    skip_first = False
    if len(importance_csv) > 50000:
        if query_yes_no(
                "Layer 0: There are going to be " + str(len(importance_csv)) + " objects plotted, are you sure?"):
            skip_first = False
        else:
            skip_first = True

    nodes = importance_csv.filter(like="node_layer")
    importance_csv["node_layer_" + str(len(nodes.columns))] = np.zeros(len(importance_csv))

    # colors = []
    weights = np.array([])
    G = nx.DiGraph()
    for i in range(len(nodes.columns)):
        if (skip_first & (i == 0)):
            pass
        else:
            importance_csv['node_layer_' + str(i + 1)] = importance_csv['node_layer_' + str(i + 1)] + importance_csv[
                'node_layer_' + str(i)].max() + 1
            cur_importance_csv = importance_csv[
                ['node_layer_' + str(i), 'node_layer_' + str(i + 1), 'weights_' + str(i)]].drop_duplicates()
            coord = list(
                cur_importance_csv[['node_layer_' + str(i), 'node_layer_' + str(i + 1)]].itertuples(index=False,
                                                                                                    name=None))
            G.add_edges_from(coord, )
            weights = np.append(weights, cur_importance_csv['weights_' + str(i)].values)
            # colors = colors + [colormap[i]]*cur_importance_csv.shape[0]

    G = nx.relabel.convert_node_labels_to_integers(G)
    plt.figure(figsize=(8, 8))
    plt.grid("off")
    pos = nx.nx_pydot.graphviz_layout(G, prog="twopi", root=importance_csv['node_layer_' + str(i + 1)].max())
    if plot_weights:
        nx.draw_networkx(G, pos=pos, with_labels=True, arrows=plot_arrows, width=weights)
    else:
        nx.draw_networkx(G, pos=pos, with_labels=True, arrows=plot_arrows)
    plt.savefig(resultpath + "network_plot.png", format="PNG")


def plot_layer_weight(resultpath, importance_csv, layer=0, num_annotated=10):
    csv_file = importance_csv.copy()
    if int(layer) < csv_file.filter(like="weights").shape[1] - 1:
        pass
    elif int(layer) == csv_file.filter(like="weights").shape[1] - 1:
        csv_file["node_layer_" + str(csv_file.filter(like="weights").shape[1])] = 0
        csv_file["layer" + str(csv_file.filter(like="weights").shape[1]) + '_name'] = "end_node"
    else:
        print("error cant plot that many layers, there are not that many layers")
        sys.exit()

    if 'chr' in csv_file.columns:
        columns = ["node_layer_" + str(layer), "node_layer_" + str(layer + 1), "weights_" + str(layer),
         'layer' + str(layer) + '_name', 'layer' + str(layer + 1) + '_name', 'chr']
    else:
        columns = ["node_layer_" + str(layer), "node_layer_" + str(layer + 1), "weights_" + str(layer),
                   'layer' + str(layer) + '_name', 'layer' + str(layer + 1) + '_name']
    csv_file = csv_file[columns]
    csv_file = csv_file.drop_duplicates()

    csv_file = csv_file.sort_values(by="node_layer_" + str(layer), ascending=True)
    csv_file["pos"] = np.arange(len(csv_file))
    weights = abs(csv_file["weights_" + str(layer)].values)
    weights = weights / max(weights)
    csv_file["plot_weights"] = weights


    plt.figure(figsize=(20, 10))


    gene_middle = []
    if "chr" in csv_file.columns:
        color_end = np.sort(csv_file.groupby("chr")["pos"].max().values)
        print('coloring per chromosome')
        color_end = np.insert(color_end, 0, 0)
        for i in range(len(color_end) - 1):
            gene_middle.append((color_end[i] + color_end[i + 1]) / 2)
    else:
        color_end = np.sort(csv_file.groupby("node_layer_" + str(layer + 1))["pos"].max().values)
        color_end = np.insert(color_end, 0, 0)
        print("no chr information continuing by coloring per group in node_layer_1")

    colormap = ['#7dcfe2', '#4b78b5', 'darkgrey', 'dimgray'] * len(color_end)


    if len(weights) < 500:

        sns.set(style="whitegrid")
        sns.set_color_codes("pastel")

        f, ax = plt.subplots(figsize=(6, 10))

        sns.barplot(x="plot_weights", y='layer' + str(layer) + '_name', data=csv_file,
                    label="Total")

        ax.set(ylabel="Layer node",
               xlabel="Normalized Weights")
        new_labels = [label for label in ax.get_yticklabels()]
        fontsize = int(np.round((-8 / 90) * len(new_labels) + 13.888))
        ax.set_yticklabels(new_labels, fontsize=fontsize)
        sns.despine(left=True, bottom=True)
        plt.savefig(resultpath + "manhattan_weights_" + str(layer) + ".png", bbox_inches='tight')

    else:
        for i in range(len(color_end) - 1):
            plt.scatter(csv_file['pos'].iloc[color_end[i]:color_end[i + 1]],
                        csv_file["plot_weights"].iloc[color_end[i]:color_end[i + 1]], c=colormap[i])

        plt.ylim(bottom=0, top=1.2)
        plt.xlim(0, len(weights) + int(len(weights) / 100))
        plt.title("Trained Network Weights", size=36)
        if len(gene_middle) > 1:
            plt.xticks(gene_middle, np.arange(len(gene_middle)) + 1, size=16)
            plt.xlabel("Chromosome", size=18)
        else:
            plt.xlabel("Chromosome position", size=18)
        plt.ylabel("Weights", size=18)

        gene5_overview = csv_file.sort_values("plot_weights", ascending=False).head(num_annotated)

        if len(gene5_overview) < num_annotated:
            num_annotated = len(gene5_overview)
        for i in range(num_annotated):
            print(gene5_overview['layer' + str(layer) + '_name'].iloc[i],
                         gene5_overview["pos"].iloc[i], gene5_overview["plot_weights"].iloc[i])
            plt.annotate(gene5_overview['layer' + str(layer) + '_name'].iloc[i],
                         (gene5_overview["pos"].iloc[i], gene5_overview["plot_weights"].iloc[i]),
                         xytext=(gene5_overview["pos"].iloc[i],
                                 gene5_overview["plot_weights"].iloc[i]), size=16)

        plt.gca().spines['right'].set_color('none')
        plt.gca().spines['top'].set_color('none')

        plt.savefig(resultpath + "manhattan_weights_" + str(layer) + ".png", bbox_inches='tight', pad_inches=0)


def manhattan_importance(resultpath, importance_csv, num_annotated=10):
    csv_file = importance_csv.copy()
    plt.figure(figsize=(20, 10))

    gene_middle = []

    csv_file = csv_file.sort_values(by="node_layer_0", ascending=True)
    csv_file["pos"] = np.arange(len(csv_file))

    if "chr" in csv_file.columns:
        color_end = np.sort(csv_file.groupby("chr")["pos"].max().values)
        print('coloring per chromosome')
        color_end = np.insert(color_end, 0, 0)
        for i in range(len(color_end) - 1):
            gene_middle.append((color_end[i] + color_end[i + 1]) / 2)
    else:
        color_end = np.sort(csv_file.groupby("node_layer_1")["pos"].max().values)
        color_end = np.insert(color_end, 0, 0)
        print("no chr information continuing by coloring per group in node_layer_1")

    weights = abs(csv_file["raw_importance"])
    weights = weights / max(weights)
    csv_file["plot_weights"] = weights


    print(len(color_end), "color groups")
    colormap = ['#7dcfe2', '#4b78b5', 'darkgrey', 'dimgray'] * len(color_end)

    for i in range(len(color_end) - 1):
        plt.scatter(csv_file['pos'].iloc[color_end[i]:color_end[i + 1]],
                    csv_file["plot_weights"].iloc[color_end[i]:color_end[i + 1]], c=colormap[i])

    plt.ylim(bottom=0, top=1.2)
    plt.xlim(0, len(weights) + int(len(weights) / 100))
    plt.title("Raw Importance Manhattan", size=36)
    if len(gene_middle) > 1:
        plt.xticks(gene_middle, np.arange(len(gene_middle)) + 1, size=16)
        plt.xlabel("Chromosome", size=18)
    else:
        plt.xlabel("Chromosome position", size=18)
    plt.ylabel("Weights", size=18)


    offset = len(csv_file) / 200
    offset = np.clip(offset, 0.1, 100)

    gene5_overview = csv_file.sort_values("plot_weights", ascending=False).head(num_annotated)

    if len(gene5_overview) < num_annotated:
        num_annotated = len(gene5_overview)
    for i in range(num_annotated):
        plt.annotate(gene5_overview['layer0_name'].iloc[i],
                     (gene5_overview["pos"].iloc[i], gene5_overview["plot_weights"].iloc[i]),
                     xytext=(gene5_overview["pos"].iloc[i] + offset,
                             gene5_overview["plot_weights"].iloc[i]), size=16)

    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')

    plt.savefig(resultpath + "Path_importance.png", bbox_inches='tight', pad_inches=0)
    plt.show()


def plot(args):
    folder, resultpath = get_paths(args.ID)
    importance_csv = pd.read_csv(resultpath + "/connection_weights.csv", index_col=0)
    print(resultpath)
    layer = args.layer_n
    if args.type == "layer_weight":
        plot_layer_weight(resultpath, importance_csv, layer=layer, num_annotated=10)
    elif args.type == "circos":
        cicos_plot(resultpath=resultpath, importance_csv=importance_csv, plot_weights=False, plot_arrows=True)
    elif args.type == "raw_importance":
        manhattan_importance(resultpath=resultpath, importance_csv=importance_csv)
    else:
        print("invalid type:", args.type)
        exit()
