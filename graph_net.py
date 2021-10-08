import numpy as np
import pickle

def convert_ds(ds):
    graph_ds = []
    for artifact in ds:
        nb_art = len(artifact['imgs'])
        adj_mat = np.zeros([68*(nb_art + 1), 68*(nb_art + 1)])
        for img_nb in range(nb_art + 1):
            adj_mat[68*img_nb:68*(img_nb+1), 68*img_nb:68*(img_nb+1)] = 1
            for pt_nb in range(68):
                for i in range(nb_art + 1):
                    adj_mat[68*img_nb+pt_nb:68*i+pt_nb] = 1

        graph_ds.append(art_graph)
    return graph_ds


with open('ds_full.pkl', 'rb') as f:
    ds = pickle.load(f)
ds_train, test_val = ds
new_ds_train = [i for i in ds_train if len(i['imgs']) != 0]
graph_ds = convert_ds(new_ds_train)
