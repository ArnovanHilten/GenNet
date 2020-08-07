import sys
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from nxviz import CircosPlot
import networkx as nx
from utils.utils import query_yes_no, get_paths
import seaborn as sns

def plot(jobid, type, layer):
    folder, resultpath = get_paths(jobid)
    importance_csv = pd.read_csv(resultpath + "/connection_weights.csv", index_col = 0 )

    if type == "layer_weight":
        plot_layer_weight(resultpath, importance_csv, layer=layer, num_annotated=10)
    elif type == "circos":
        cicos_plot(resultpath=resultpath, importance_csv=importance_csv, plot_weights=False, plot_arrows=True )
    else:
        print("invalid type:", type)
        exit()

def cicos_plot(resultpath, importance_csv, plot_weights = True, plot_arrows=False):
    print("in progress...")
    colormap = ['#7dcfe2','#4b78b5','darkgrey','dimgray']*25


    skip_first = False
    if len(importance_csv) > 50000:
        if query_yes_no("Layer 0: There are going to be "+str(len(importance_csv))+" objects plotted, are you sure?"):
            skip_first = False
        else:
            skip_first = True



    nodes = importance_csv.filter(like="node_layer")
    importance_csv["node_layer_" + str(len(nodes.columns))] = np.zeros(len(importance_csv))

    # colors = []
    weights = np.array([])
    G = nx.DiGraph()
    for i in range(len(nodes.columns)):
        if (skip_first & (i==0)):
            pass
        else:
            importance_csv['node_layer_'+str(i+1)] = importance_csv['node_layer_'+str(i+1)]  +  importance_csv['node_layer_'+str(i)].max()+1
            cur_importance_csv = importance_csv[['node_layer_'+str(i), 'node_layer_'+str(i+1), 'weights_' + str(i)]].drop_duplicates()
            coord= list(cur_importance_csv[['node_layer_'+str(i), 'node_layer_'+str(i+1)]].itertuples(index=False, name=None))
            G.add_edges_from(coord, )
            weights = np.append(weights,cur_importance_csv['weights_' + str(i)].values)
            # colors = colors + [colormap[i]]*cur_importance_csv.shape[0]

    plt.figure(figsize=(8, 8))
    pos=nx.nx_pydot.graphviz_layout(G,prog="twopi",root=importance_csv['node_layer_'+str(i+1)].max())
    if plot_weights:
        nx.draw_networkx(G, pos=pos, with_labels=True,arrows=plot_arrows, width=weights)
    else:
        nx.draw_networkx(G, pos=pos, with_labels=True, arrows=plot_arrows)
    plt.show()
    plt.savefig(resultpath + "network_plot.png", format="PNG")


def plot_layer_weight(resultpath, importance_csv, layer = 0, num_annotated = 10 ):

    csv_file = importance_csv.copy()
    if int(layer) < csv_file.filter(like="weights").shape[1]-1:
        pass
    elif int(layer) == csv_file.filter(like="weights").shape[1]-1:
        csv_file["node_layer_"+str(csv_file.filter(like="weights").shape[1])] = 0
        csv_file["layer" + str(csv_file.filter(like="weights").shape[1]) + '_name'] = "end_node"
    else:
        print("error cant plot that many layers, there are not that many layers")
        sys.exit()


    plt.figure(figsize=(20, 10))
    colormap = ['#7dcfe2', '#4b78b5', 'darkgrey', 'dimgray'] * 25
    color_end = np.sort(csv_file.groupby("node_layer_"+str(layer+1))["node_layer_"+str(layer)].max().values)
    color_end = np.insert(color_end, 0, 0)

    csv_file = csv_file[["node_layer_"+str(layer),"node_layer_"+str(layer+1),"weights_"+str(layer),'layer'+str(layer)+'_name','layer'+str(layer+1)+'_name']]
    csv_file = csv_file.drop_duplicates()

    weights = abs(csv_file["weights_"+str(layer)].values)
    weights = weights/max(weights)
    x = np.arange(len(weights))

    csv_file["pos"] = x
    csv_file["plot_weights"] = weights

    if len(weights) < 500:

        sns.set(style="whitegrid")
        sns.set_color_codes("pastel")

        f, ax = plt.subplots(figsize=(6, 10))

        sns.barplot(x="plot_weights", y='layer'+str(layer)+'_name', data=csv_file,
                    label="Total", color="b")

        ax.set(ylabel="Layer node name",
               xlabel="Normalized Weights")
        sns.despine(left=True, bottom=True)
        plt.savefig(resultpath + "manhattan_weights_" + str(layer) + ".png", bbox_inches='tight', pad_inches=0)

    else:
        for i in range(len(color_end) - 1):
            plt.scatter(x[color_end[i]:color_end[i + 1]], weights[color_end[i]:color_end[i + 1]], c=colormap[i])

        plt.ylim(bottom=0, top=1.2)
        plt.xlim(0, len(weights) + int(len(weights)/100))
        plt.title("Trained Network Weights", size=36)
        plt.xlabel("Node", size=18)
        plt.ylabel("Weights", size=18)


        gene5_overview = csv_file.sort_values("plot_weights", ascending=False).head(num_annotated)

        if len(gene5_overview) < num_annotated:
            num_annotated = len(gene5_overview)
        for i in range(num_annotated):
            plt.annotate(gene5_overview['layer'+str(layer)+'_name'].iloc[i],
                         (gene5_overview["pos"].iloc[i], gene5_overview["plot_weights"].iloc[i]),
                         xytext=(gene5_overview["pos"].iloc[i],
                                 gene5_overview["plot_weights"].iloc[i]), size=16)

        plt.gca().spines['right'].set_color('none')
        plt.gca().spines['top'].set_color('none')

        plt.savefig(resultpath + "manhattan_weights_"+str(layer)+".png", bbox_inches='tight', pad_inches=0)



def manhattan_importance(resultpath, importance_csv, num_annotated = 10 ):
    csv_file = importance_csv.copy()
    plt.figure(figsize=(20, 10))
    colormap = ['#7dcfe2', '#4b78b5', 'darkgrey', 'dimgray'] * 25
    color_end = np.sort(csv_file.groupby("node_layer_1")["node_layer_0"].max().values)
    color_end = np.insert(color_end, 0, 0)

    weights = abs(csv_file["raw_importance"])
    weights = weights/max(weights)
    x = np.arange(len(weights))

    for i in range(len(color_end) - 1):
        plt.scatter(x[color_end[i]:color_end[i + 1]], weights[color_end[i]:color_end[i + 1]], c=colormap[i])

    plt.ylim(bottom=0, top=1.2)
    plt.xlim(0, len(weights) + int(len(weights)/100))
    plt.title("Raw importance for each path", size=36)
    plt.xlabel("Path", size=18)
    plt.ylabel("Weights", size=18)

    csv_file["pos"] = x
    csv_file["plot_weights"] = weights

    gene5_overview = csv_file.sort_values("plot_weights", ascending=False).head(num_annotated)

    if len(gene5_overview) < num_annotated:
        num_annotated = len(gene5_overview)
    for i in range(num_annotated):
        plt.annotate(gene5_overview['layer0_name'].iloc[i],
                     (gene5_overview["pos"].iloc[i], gene5_overview["plot_weights"].iloc[i]),
                     xytext=(gene5_overview["pos"].iloc[i],
                             gene5_overview["plot_weights"].iloc[i]), size=16)

    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')

    plt.savefig(resultpath + "Path_importance.png", bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == '__main__':


    importance_csv = pd.read_csv("/home/avanhilten/PycharmProjects/GenNet/results/GenNet_experiment_1/connection_weights.csv", index_col = 0 )
    resultpath = '/home/avanhilten/PycharmProjects/GenNet/results/GenNet_experiment_1/'
    manhattan_importance(resultpath,importance_csv)
    plot_layer_weight(resultpath, importance_csv, layer=0)
    plot_layer_weight(resultpath, importance_csv, layer=1)
    plot_layer_weight(resultpath, importance_csv, layer=2)
    plot_layer_weight(resultpath, importance_csv, layer=4)