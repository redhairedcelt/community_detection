import pandas as pd
import matplotlib.pyplot as plt
import gsta
import gsta_config
import networkx as nx


#%% get spatial network for one selector
loc_engine = gsta.connect_engine(gsta_config.gowalla_params)
df = pd.read_sql(f"""select uid, loc_id
                 from checkins
                 where uid = '{0}'
                 order by time""", loc_engine)

#%%
df_edgelist_weighted = (df_edges.groupby(['Source', 'Target'])
                        .count()
                        .rename(columns={'uid': 'weight'})
                        .reset_index())

#%%

G = nx.from_pandas_edgelist(df, source='Source',
                            target='Target', create_using=nx.DiGraph)

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