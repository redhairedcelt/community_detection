import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# network tools
import networkx as nx
import igraph as ig
import cdlib
from cdlib import algorithms
from cdlib import viz
from cdlib import NodeClustering
from cdlib import evaluation

#metrics
from sklearn import metrics

#%% define graph name, graph, and ground truth communities
graph_name = "Karate Club"
nx_g = nx.karate_club_graph()
truth = pd.read_csv('karate/karate_truth.csv')

# define the positions here so all the cluster plots in the loop are the same structure
pos = nx.spring_layout(nx_g)

# need to make a nodeclustering object for ground truth
# first get all the unique communities
truth_coms = truth['truth'].unique()
# define an empty set to catch a frozenset of all nodes in each community
communities_set = set()
# iterate through the communities to get the nodes in each community and add them to the set as frozensets
for com in truth_coms:
    com_nodes = truth[truth['truth'] == com].iloc[:, 0].values
    communities_set.add(frozenset(com_nodes))
# make the nodeclustering object
ground_truth_com = NodeClustering(communities=communities_set, graph=nx_g, method_name="ground_truth")

# adjust label pos
label_pos = dict()
for k, v in pos.items():
    label_pos[k] = (v[0], (v[1] + 0.05))
#%%
# plot the original network with the ground truth communities
viz.plot_network_clusters(nx_g, ground_truth_com, pos, figsize=(5, 5))
nx.draw_networkx_labels(nx_g, pos=label_pos)
plt.title(f'Ground Truth of {graph_name}')
plt.show()

# %%
# define algorithms and their names to iterate through
algo_dict = {'louvain': algorithms.louvain,
             'leidan': algorithms.leiden,
             'greed_modularity': algorithms.greedy_modularity,
             'label_prop': algorithms.label_propagation,
             'walktrap': algorithms.walktrap,
             'infomap': algorithms.infomap,
             'eigenvector': algorithms.eigenvector,
             'spinglass': algorithms.spinglass,
             'signficant_communities': algorithms.significance_communities}

algo_dict = {'girvan_newman': algorithms.girvan_newman}

# make a dict to store results about each algo
results_dict = dict()
# make a df with all the nodes.  will capture each model's clustering
df_nodes = pd.DataFrame(list(nx_g.nodes))
df_nodes.columns = ['node']
df_nodes = pd.merge(df_nodes, truth, how='left', left_on='node', right_on='node')


for name, algo in algo_dict.items():
    # run algos to make node_clustering objects
    pred_coms = algo(nx_g, level=1)
    communities = pred_coms.communities

    # need to convert the community groups from list of lists to a dict of lists for ingest to df
    coms_dict = dict()
    for c in range(len(communities)):
        for i in communities[c]:
            coms_dict[i] = [c]

    # make a df with the results of the algo
    df_coms = pd.DataFrame.from_dict(coms_dict).T.reset_index()
    df_coms.columns = ['node', name]
    # merge this results with the df_nodes to keep track of all the nodes' clusters
    df_nodes = pd.merge(df_nodes, df_coms, how='left', left_on='node', right_on='node')

    # merge the results and the ground truth together.
    df_compare = pd.merge(truth, df_coms, how='left', left_on='node', right_on='node')
    # We can then just adjusted mutual info to find similarity score
    ami_score = metrics.adjusted_mutual_info_score(df_compare[name], df_compare['truth'])
    print(f'The AMI for {name} algorithm is {round(ami_score, 3)}.')
    results_dict[name] = {'AMI' : round(ami_score, 3),
                          'pred_coms' : pred_coms,
                          'numb_communities' : len(communities),
                          'truth_communities' : len(df_compare['truth'].unique())}

    # plot the network clusters
    viz.plot_network_clusters(nx_g, pred_coms, pos, figsize=(5, 5))
    nx.draw_networkx_labels(nx_g, pos=label_pos)
    plt.title(f'Clusters for {name} algo of {graph_name}, \n AMI = {round(ami_score, 3)}')
    plt.show()

    # # plot the graph
    # viz.plot_community_graph(nx_g, pred_coms, figsize=(5, 5))
    # plt.title(f'Communities for {name} algo of {graph_name}.')
    # plt.show()

df_results = pd.DataFrame.from_dict(results_dict).T.drop('pred_coms', axis=1)

#%%
coms = [ground_truth_com]
for name, results in results_dict.items():
    coms.append(results['pred_coms'])

viz.plot_sim_matrix(coms,evaluation.adjusted_mutual_information)
plt.show()

viz.plot_com_properties_relation(coms, evaluation.size, evaluation.internal_edge_density)
plt.show()

viz.plot_com_stat(coms, evaluation.internal_edge_density)
plt.show()

#%%


def analyze_communities(pred_coms, pos, label_pos):
    # make a dict to store results about each algo
    results_dict = dict()
    # make a df with all the nodes.  will capture each model's clustering
    df_nodes = pd.DataFrame(list(nx_g.nodes))
    df_nodes.columns = ['node']
    df_nodes = pd.merge(df_nodes, truth, how='left', left_on='node', right_on='node')
    name = pred_coms.method_name
    communities = pred_coms.communities

    # need to convert the community groups from list of lists to a dict of lists for ingest to df
    coms_dict = dict()
    for c in range(len(communities)):
        for i in communities[c]:
            coms_dict[i] = [c]

    # make a df with the results of the algo
    df_results = pd.DataFrame.from_dict(coms_dict).T.reset_index()
    df_results.columns = ['node', name]
    # merge this results with the df_nodes to keep track of all the nodes' clusters
    df_nodes = pd.merge(df_nodes, df_results, how='left', left_on='node', right_on='node')

    # merge the results and the ground truth together.
    df_compare = pd.merge(truth, df_results, how='left', left_on='node', right_on='node')
    # We can then just adjusted mutual info to find similarity score
    ami_score = metrics.adjusted_mutual_info_score(df_compare[name], df_compare['truth'])
    print(f'The AMI for {name} algorithm is {round(ami_score, 3)}.')
    results_dict[name] = {'AMI': round(ami_score, 3),
                          'pred_coms': pred_coms,
                          'numb_communities': len(communities),
                          'truth_communities': len(df_compare['truth'].unique())}

    # plot the network clusters
    viz.plot_network_clusters(nx_g, pred_coms, pos, figsize=(5, 5))
    if label_pos == True:
        nx.draw_networkx_labels(nx_g, pos=label_pos)
    plt.title(f'Clusters for {name} algo of {graph_name}, \n AMI = {round(ami_score, 3)}')
    plt.show()

pred_coms = algorithms.leiden(nx_g)
analyze_communities(pred_coms, pos, label_pos)