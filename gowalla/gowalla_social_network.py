import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

import gsta
import gsta_config

import networkx as nx
import igraph as ig

from cdlib import algorithms
from cdlib import viz
from cdlib import NodeClustering
from cdlib import evaluation

#%% read in edgelist
loc_engine = gsta.connect_engine(gsta_config.gowalla_params)
df_edges = pd.read_sql_table('edges', loc_engine)
df_edges.columns = ['Source', 'Target']

graph_name = "Gowalla Social Network"

#%% igraph
g = ig.Graph.TupleList(df_edges.values)

#%% networkx
G = nx.from_pandas_edgelist(df_edges, source='Source',
                            target='Target', create_using=nx.Graph)

# %% define algorithms and their names to iterate through
algo_dict = {'louvain': algorithms.louvain,
             'leidan': algorithms.leiden,
             'greed_modularity': algorithms.greedy_modularity,
             'label_prop': algorithms.label_propagation,
             'walktrap': algorithms.walktrap,
             'infomap': algorithms.infomap,
             'eigenvector': algorithms.eigenvector,
             'spinglass': algorithms.spinglass}

#%% iterate through all the algorithms
# set variables for the iteration loop
# make a dict to store results about each algo
results_dict = dict()
# make a df with all the nodes.  will capture each model's clustering
df_nodes = pd.DataFrame(list(G.nodes))
df_nodes.columns = ['node']

# iterate through the alogrithms
for name, algo in algo_dict.items():
    start = datetime.now()
    # run algos to make node_clustering objects
    pred_coms = algo(G)
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

    lapse = datetime.now() - start
    print(f"{name} algo identified {len(communities)} in {lapse}.")



#%%
df_report = pd.DataFrame([list(G),
                          [len(G[node]) for node in G.nodes()],
                          list(nx.degree_centrality(G).values()),
                          nx.in_degree_centrality(G).values(),
                          nx.out_degree_centrality(G).values(),
                          nx.eigenvector_centrality(G).values(),
                          nx.closeness_centrality(G).values(),
                          nx.betweenness_centrality(G).values()]
                         ).T
df_report.columns = ['Node', 'Targets', 'Degree', 'In-Degree', 'Out-Degree',
                     'Eigenvector', 'Centrality', 'Betweenness']
df_report = (df_report.astype({'Degree':'float', 'In-Degree':'float', 'Out-Degree':'float',
                               'Eigenvector':'float', 'Centrality':'float', 'Targets': 'int',
                               'Betweenness':'float'}).round(3))
df_report.hist()
plt.show()

df_report.boxplot(['Degree', 'In-Degree', 'Out-Degree', 'Eigenvector', 'Centrality',
                  'Betweenness'])
plt.show()

print(df_report.sort_values('Betweenness'))


#%%
# define the positions here so all the cluster plots in the loop are the same structure
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 10))
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos)
plt.axis('off')
plt.title('Full Network Plot')
plt.show()

#%%
df_analysis = pd.read_sql(f"""SELECT 
s.loc_id,
s.position_count, 
s.unique_uids
c.uid, 
count(c.loc_id) as uid_counts
FROM sites as s, checkins as c
WHERE s.loc_id = '420315' and
c.uid = '0';
""", loc_engine)




#%%
conn = gsta.connect_psycopg2(gsta_config.gowalla_params)
c = conn.cursor()
c.execute("""SELECT DISTINCT(uid) FROM trips;""")
uid_list = c.fetchall()
c.close()
conn.close()


#%%
conn = gsta.connect_psycopg2(gsta_config.gowalla_params)
c = conn.cursor()
for uid in uid_list:
    try:
        c.execute(f"""select ST_Length(geography(line))/1000 AS line_length_km
                    from trips
                    where uid = '{uid[0]}';""")

    except: print('Failed:', uid)
conn.close()
#%%
conn = gsta.connect_psycopg2(gsta_config.gowalla_params)
c = conn.cursor()
c.execute("""select ST_Length(geography(line))/1000 AS line_length_km
            from trips
            where uid = '9999';""")
print(c.fetchone())
conn.close()