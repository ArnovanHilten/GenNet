import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from GenNet_utils.Utility_functions import get_paths


def sunburst_plot(resultpath, importance_csv, num_layers=3, plot_threshold=0.01, add_end_node=True):
    csv_file = importance_csv.copy()

    number_of_weights = csv_file.filter(like="weights").shape[1]

    if add_end_node:
        csv_file["node_layer_" + str(number_of_weights)] = 0
        csv_file["layer" + str(number_of_weights) + '_name'] = "Prediction"

    plot_layer_names = []
    for i in range(number_of_weights, number_of_weights - num_layers, -1):
        plot_layer_names.append("layer" + str(i) + '_name')

    first_number_layer_to_plot = i

    csv_file["percentage_0"] = csv_file.filter(like="weights").prod(axis=1)
    csv_file["percentage_0"] = abs(csv_file["percentage_0"]) / abs(csv_file["percentage_0"]).sum() * 100
    print(csv_file["percentage_" + str(0)].sum())

    for i in range(1, first_number_layer_to_plot + 1):
        csv_file["percentage_" + str(i)] = csv_file.groupby("layer" + str(i) + '_name')[
            "percentage_" + str(i - 1)].transform('sum')

    print(csv_file["percentage_" + str(first_number_layer_to_plot)].sum())
    csv_file = csv_file.drop_duplicates("layer" + str(first_number_layer_to_plot) + '_name')
    print(csv_file["percentage_" + str(first_number_layer_to_plot)].sum())

    print(csv_file.shape)

    csv_file = csv_file[csv_file["percentage_" + str(first_number_layer_to_plot)] > plot_threshold]

    print(csv_file.shape)
    print(csv_file["percentage_" + str(first_number_layer_to_plot)].sum())
    print("start plotting")
    fig = px.sunburst(csv_file,
                      path=plot_layer_names,
                      values="percentage_" + str(first_number_layer_to_plot),
                      width=1000, height=1000,
                      template="presentation",
                      color_discrete_sequence=px.colors.qualitative.G10
                      )

    fig.write_image(resultpath + "sunburst.png")
    fig.write_html(resultpath + "sunburst.html")
    print(resultpath)
    print("done")


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
        csv_file = csv_file[columns]
        csv_file = csv_file.drop_duplicates()

        csv_file = csv_file.sort_values(by=['chr', 'node_layer_' + str(layer)], ascending=True)
    else:
        columns = ["node_layer_" + str(layer), "node_layer_" + str(layer + 1), "weights_" + str(layer),
                   'layer' + str(layer) + '_name', 'layer' + str(layer + 1) + '_name']
        csv_file = csv_file[columns]
        csv_file = csv_file.drop_duplicates()

        csv_file = csv_file.sort_values(by=['node_layer_' + str(layer)], ascending=True)

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
        plt.title("Network Weights layer " + str(layer), size=36)
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

        plt.savefig(resultpath + "Weights_layer_" + str(layer) + ".png", bbox_inches='tight', pad_inches=0)


def manhattan_relative_importance(resultpath, importance_csv, num_annotated=10):
    csv_file = importance_csv.copy()
    plt.figure(figsize=(20, 10))

    gene_middle = []


    if "chr" in csv_file.columns:
        csv_file = csv_file.sort_values(by=['chr', 'node_layer_0'], ascending=True)
        csv_file["pos"] = np.arange(len(csv_file))
        color_end = np.sort(csv_file.groupby("chr")["pos"].max().values)
        print('coloring per chromosome')
        color_end = np.insert(color_end, 0, 0)
        for i in range(len(color_end) - 1):
            gene_middle.append((color_end[i] + color_end[i + 1]) / 2)
    else:
        csv_file = csv_file.sort_values(by=['node_layer_0'], ascending=True)
        csv_file["pos"] = np.arange(len(csv_file))
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
    plt.title("Relative importance of all SNPs", size=36)
    if len(gene_middle) > 1:
        plt.xticks(gene_middle, np.arange(len(gene_middle)) + 1, size=16)
        plt.xlabel("Chromosome", size=18)
    else:
        plt.xlabel("Chromosome position", size=18)
    plt.ylabel("Relative importance", size=18)

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

    plt.savefig(resultpath + "Manhattan_relative_importance_SNPs.png", bbox_inches='tight', pad_inches=0)
    plt.show()


def plot(args):
    folder, resultpath = get_paths(args.ID)
    importance_csv = pd.read_csv(resultpath + "/connection_weights.csv", index_col=0)
    print(resultpath)
    layer = args.layer_n
    if args.type == "layer_weight":
        plot_layer_weight(resultpath, importance_csv, layer=layer, num_annotated=10)
    elif args.type == "sunburst":
        sunburst_plot(resultpath=resultpath, importance_csv=importance_csv)
    elif args.type == "manhattan_relative_importance":
        manhattan_relative_importance(resultpath=resultpath, importance_csv=importance_csv)
    else:
        print("invalid type:", args.type)
        exit()
