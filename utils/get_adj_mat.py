import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkb


                                                                              
def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf

                                    
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

                                               
    for row in distance_df:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

                                                 
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
                                                           
                                                    

                                                                              
    adj_mx[adj_mx < normalized_k] = 0
    return adj_mx


def read_map(dataset, year):
    df = gpd.read_file(
        f"./data/tl_{year}_us_county.zip"
    )
    state_code = {"fl": "12", "tx": "48", "ny": "36", "ca": "06"}
    return df[df["STATEFP"] == state_code[dataset]].sort_values(by="GEOID").reset_index(drop=True)


if __name__ == "__main__":
    for db in ["fl","ny","ca","tx",]:
        gdf=read_map(db,2018)
        save_path = f"./data/safegraph_{db}/adj.npy"
        gdf = gdf.set_geometry("geometry").sort_values(by="GEOID")
        ctr = gdf.centroid.reset_index(drop=True)
        N = len(ctr)
        ph_area = list(range(N))
        distance = []
        for i in ph_area:
            for j in ph_area:
                distance.append([i, j, ctr[i].distance(ctr[j])])
        adj_mx = get_adjacency_matrix(distance_df=distance, sensor_ids=ph_area)
        print(db, f"The shape of Adjacency Matrix: {adj_mx.shape}")
        np.save(save_path, adj_mx)
