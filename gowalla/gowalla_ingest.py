import pandas as pd
import matplotlib.pyplot as plt
import gsta
import gsta_config

# %%
conn = gsta.connect_psycopg2(gsta_config.gowalla_params)
loc_engine = gsta.connect_engine(gsta_config.gowalla_params)
conn.close()
# %%
conn = gsta.connect_psycopg2(gsta_config.gowalla_params)
c = conn.cursor()
c.execute("""CREATE TABLE if not exists checkins (
    uid 			text,
    time     		timestamp,
    lat				numeric,
    lon				numeric,
    loc_id          text
);""")
conn.commit()
c.close()
conn.close()

# %% use a generator to read the file in as chunks
generator = pd.read_csv('gowalla/Gowalla_totalCheckins.txt', sep='\t', header=None,
                        names=['uid', 'time', 'lat', 'lon', 'loc_id'], chunksize=100000)
for df in generator:
    df.to_sql(name='checkins', con=loc_engine, if_exists='append', method='multi', index=False)
# %%
conn = gsta.connect_psycopg2(gsta_config.gowalla_params)
c = conn.cursor()
c.execute("""ALTER TABLE checkins
add column geog geography(Point, 4326);""")
conn.commit()
c.execute("""
UPDATE checkins SET geog = ST_SetSRID(
	ST_MakePoint(lon, lat), 4326);""")
conn.commit()
c.close()
conn.close()
#%% make a sites table
def make_sites(new_table_name, source_table, conn):
    c = conn.cursor()
    c.execute(f"""DROP TABLE IF EXISTS {new_table_name};""")
    conn.commit()
    c.execute(f"""CREATE TABLE {new_table_name} AS
    SELECT 
        loc_id,
        position_count,
        unique_uids,
        lat,
        lon,
        geom
        FROM (  
                SELECT pos.loc_id as loc_id,
                COUNT (pos.geom) as position_count,
                COUNT (DISTINCT (pos.uid)) as unique_uids,
                pos.lat as lat,
                pos.lon as lon,
                pos.geom as geom
                FROM {source_table} as pos
                GROUP BY pos.loc_id, pos.lat, pos.lon, pos.geom) 
                AS foo;""")
    conn.commit()
    c.execute(f"""CREATE INDEX if not exists sites_loc_id_idx on {new_table_name} (loc_id);""")
    conn.commit()
    c.close()

conn = gsta.connect_psycopg2(gsta_config.gowalla_params)
make_sites('sites', 'checkins', conn)
conn.close()
#%% make trips table
def make_trips(new_table_name, source_table, conn):
    c = conn.cursor()
    c.execute(f"""DROP TABLE IF EXISTS {new_table_name};""")
    conn.commit()
    c.execute(f"""CREATE TABLE {new_table_name} AS
    SELECT 
        uid,
        position_count,
        first_date,
        last_date,
        last_date - first_date as time_diff,
        line
        FROM (
                SELECT pos.uid as uid,
                COUNT (pos.geog) as position_count,
                ST_MakeLine(pos.geog::geometry ORDER BY pos.time) AS line,
                MIN (pos.time) as first_date,
                MAX (pos.time) as last_date
                FROM {source_table} as pos
                GROUP BY pos.uid) 
                AS foo
        WHERE position_count > 2;""")
    conn.commit()
    c.execute(f"""CREATE INDEX if not exists trips_uid_idx on {new_table_name} (uid);""")
    conn.commit()
    c.close()

conn = gsta.connect_psycopg2(gsta_config.gowalla_params)
make_trips('trips', 'checkins', conn)
conn.close()
# %% make the edges table
conn = gsta.connect_psycopg2(gsta_config.gowalla_params)
c = conn.cursor()
c.execute("""CREATE TABLE if not exists edges (
    source  text,
    target  text
);""")
conn.commit()
c.close()
conn.close()

# %% use a generator to read the edges file in as chunks
generator = pd.read_csv('gowalla/Gowalla_edges.txt', sep='\t', header=None,
                        names=['Source', 'Target'], chunksize=100000)
for df in generator:
    df.to_sql(name='edges', con=loc_engine, if_exists='append', method='multi', index=False)
