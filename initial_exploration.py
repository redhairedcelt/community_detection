import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import random

# network tools
import networkx as nx
import igraph as ig

#cdlib, separate import for algos
import cdlib
from cdlib import algorithms

#metrics
from sklearn import metrics

#%%
def return_principle_comp(graph):
    ## graph needs to be an ig graph
    comps = graph.decompose()
    pc_graph = comps[0]
    print(f'Original graph has {len(graph.vs())} nodes and {len(graph.es())} edges.'
          f'There are {len(comps)} components.  \n'
          f'The principle components has {round(len(pc_graph.vs())/len(graph.vs()),3)} of nodes and {round(len(pc_graph.es())/len(graph.es()),3)} of edges. ')
    return pc_graph

#%%
truth = pd.read_csv('email-Eu-core-department-labels.txt', sep=' ', header=None)
truth.columns = ['node', 'truth']


edge_df = pd.read_csv('email-Eu-core.txt', sep=' ', header=None)
edge_df.columns = ['Source', 'Target']
edge_df.to_csv('emails.csv', index=False)

graph = ig.Graph.TupleList(edge_df.values)
pc_ig = return_principle_comp(graph)
#%%
for node in pc_ig.vs:
    # iterate through the graph vertices, and get the node id for each vertex
    node_id = node['name']
    # the ground truth community is then set as an attr for each vertex by referring back to the truth df,
    # which has been reindexed to the original node id.
    node['truth'] = truth['truth'].loc[node_id]


#%%

g = ig.Graph.Barabasi(n = 20, m = 1)
i = g.community_infomap()
pal = ig.drawing.colors.ClusterColoringPalette(len(i))
g.vs['color'] = pal.get_many(i.membership)
ig.plot(g)
plt.show()

#%%
def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]
comms = partition(list(range(1,42)), 5)

import matplotlib.colors as mcolors

print(list(mcolors.TABLEAU_COLORS))
#%%
import random
random.seed(42)
colors = ['blue', 'green', 'red', 'purple', 'yellow', 'cyan', 'magenta', 'black']
color_dict = dict()
len_comms = len(truth['truth'].unique())
for i in range(0, len_comms):
    color_dict[i] = random.choice(colors)
pc_ig.vs['color'] =[color_dict[attr] for attr in pc_ig.vs["truth"]]

#%%
pal = ig.drawing.colors.ClusterColoringPalette(len(truth['truth'].unique()))
for node in pc_ig.vs:
    comm_id = node['truth']
    node['color'] = pal.get(comm_id)


#%%
layout = pc_ig.layout('fr')
ig.plot(pc_ig, layout=layout)
#pc_ig.write_svg('test.svg')

#%%
from cdlib import viz
pos = nx.spring_layout(pc_ig)  # positions for all nodes
viz.plot_network_clusters(pc_ig, louvain_coms, pos, figsize=(5, 5))
plt.title('Leiden Method')
plt.show()


#%%


G = nx.from_pandas_edgelist(edge_df, source='Source',
                            target='Target',
                            create_using=nx.Graph)
pc_nx_custom = nx.from_edgelist([(names[x[0]], names[x[1]])
                      for names in [pc_ig.vs['name']] # simply a let
                      for x in pc_ig.get_edgelist()], nx.Graph())

#%%
plt.figure(figsize=(10, 10))
edges = G.edges()
pos = nx.spring_layout(G)  # positions for all nodes
# nodes
nx.draw_networkx_nodes(G, pos)
# edges
nx.draw_networkx_edges(G, pos)
plt.axis('off')
plt.title('Full Network Plot')
plt.show()
#%%
g = nx.karate_club_graph()
leiden_coms = algorithms.leiden(pc_ig)



#%%
louvain_coms = algorithms.louvain(pc_ig)
print(louvain_coms.communities)
#%%

pc = return_principle_comp(graph)
pos = nx.spring_layout(graph)
#%%
viz.plot_network_clusters(pc_nx, leiden_coms, pos, figsize=(5, 5))
plt.title('Leiden Method')
plt.show()

viz.plot_network_clusters(pc_nx, louvain_coms, pos, figsize=(5, 5))
plt.title('Louvain Method')
plt.show()
#%%
viz.plot_network_clusters(g, lp_coms, pos, figsize=(5, 5))
plt.title('LPA Method')
plt.show()
viz.plot_network_clusters(g, gm_coms, pos, figsize=(5, 5))
plt.title('GM Method')
plt.show()

#%%
print(cdlib.evaluation.normalized_mutual_information(leiden_coms, lp_coms))

#%%
cdlib.viz.plot_com_stat([leiden_coms, louvain_coms, lp_coms, gm_coms], cdlib.evaluation.scaled_density)
leiden_coms.edges_inside()
plt.show()

#%%
from karateclub import LabelPropagation
model = LabelPropagation()
model.fit(g)
cluster_membership = model.get_memberships()



#%% walkthrough analysis
import importlib
importlib.reload(cdlib)
from cdlib import algorithms
from cdlib import viz
import infomap as imp

graph_name = "Euro University Emails"
# need to read the edges df as a list of values into igraph
graph = ig.Graph.TupleList(edge_df.values)
# most algos need a single connected components.
# Use the igraph version since it reindexes node numbers and is stricter than networkx
pc_ig = return_principle_comp(graph)
# convert the pc back to a nx graph for plotting
# this version keeps the nx nodes as the same from original igraph
pc_nx = nx.from_edgelist([(names[x[0]], names[x[1]])
                      for names in [pc_ig.vs['name']] # simply a let
                      for x in pc_ig.get_edgelist()], nx.Graph())
# this version renumbers the node ids starting at 0.
#pc_nx = cdlib.utils.convert_graph_formats(pc_ig, nx.Graph)

#%%
# algos
algo_dict = {'louvain' : algorithms.louvain,
             'label_prop' : algorithms.label_propagation,
             'walktrap' : algorithms.walktrap,
             'eigenvector': algorithms.eigenvector,
             'spinglass' : algorithms.spinglass,
             'signficant_communities' : algorithms.significance_communities}

results_dict = dict()

for name, algo in algo_dict.items():
    # run algos to make node_clustering objects
    pred_coms = algo(pc_ig)
    communities = pred_coms.communities

    # need to convert the community groups from list of lists to a dict of lists for ingest to df
    coms_dict = dict()
    for c in range(len(communities)):
        for i in communities[c]:
            coms_dict[i] = [c]

    # make a df with the results of this algo
    df_results = pd.DataFrame.from_dict(coms_dict).T.reset_index()
    df_results.columns = ['node','pred']

    # merge the results and the ground truth together.  We can then just adjusted mutual info to find similarity score
    final_results = pd.merge(df_results, truth, how='inner', left_on='node', right_on='node')
    ami_score = metrics.adjusted_mutual_info_score(final_results['pred'], final_results['truth'])
    print(f'The AMI for {name} algorithm is {round(ami_score, 3)}.')
    results_dict[name] = {'AMI' : round(ami_score, 3),
                          'numb_communities' : len(communities),
                          'truth_communities' : len(final_results['truth'])}


    # plot the network clusters
    pos = nx.spring_layout(pc_nx)
    viz.plot_network_clusters(pc_nx, pred_coms, pos, figsize=(5, 5))
    plt.title(f'Clusters for {name} algo of {graph_name}.')
    plt.show()

    # plot the graph
    viz.plot_community_graph(pc_nx, pred_coms, figsize=(5, 5))
    plt.title(f'Communities for {name} algo of {graph_name}.')
    plt.show()


#%%

from cdlib import NodeClustering

g1 = nx.generators.community.LFR_benchmark_graph(1000, 3, 1.5, 0.5, min_community=20, average_degree=5)
g2 = nx.generators.community.LFR_benchmark_graph(1000, 3, 1.5, 0.6, min_community=20, average_degree=5)
g3 = nx.generators.community.LFR_benchmark_graph(1000, 3, 1.5, 0.7, min_community=20, average_degree=5)

names = ["g1", "g2", "g3"]
graphs = [g1, g2, g3]
references = []

# building the NodeClustering ground truth for the graphs
for g in graphs:
    ground_truth = NodeClustering(communities={frozenset(g.nodes[v]['community']) for v in g}, graph=g, method_name="reference")
    references.append(ground_truth)