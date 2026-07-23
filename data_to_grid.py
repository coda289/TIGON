import faiss
import numpy as np
import pandas as pd 
import torch
import numpy as np
from sklearn.decomposition import PCA
import random

def knn_to_grid(df,survivor,X,k):
    df=df[df['Sequence'].isin(survivor)]
    survivor_index=df.index.to_list()
    X = np.ascontiguousarray(X.astype(np.float32)) 
    d = X.shape[1] 
    index = faiss.IndexFlatL2(d) 
    index.add(X) 
    query_subset = X[survivor_index]
    distances, indices = index.search(query_subset, k)
    sub=X[indices]
    grid=np.mean(sub,axis=1)
    return distances,indices,np.asanyarray(grid)

def all_t_process(time,name,dir,k,n_componets):
    survivors=pd.read_csv(f'{dir}/survivors.csv')
    survivors = survivors.rename(columns={"sequence": "Sequence"})
    survivors = survivors.rename(columns={"count": "Count"})
    survivors=survivors['Sequence'].to_list()
    data=[]
    extra=[]
    pca = PCA(n_components=n_componets)
    for t in time:
        df=pd.read_csv(f'{dir}/{name}{(t):02d}_2d/cleaned.csv')
        X=np.load(f'{dir}/{name}{(t):02d}_2d/descriptor.npz')['descriptor']
        count=np.load(f'{dir}/{name}{(t):02d}_2d/descriptor.npz')['counts']
        X_reduced = pca.fit_transform(X)
        dist,ind,grid=knn_to_grid(df,survivors,X_reduced,k)
        data.append(torch.from_numpy(grid).type(torch.float32).to(device))
        mean=np.mean(count[ind])
        extra.append(mean/np.sum(mean))
    torch.save(data,f'grid{name}.pt')
    torch.save(extra,f'counts{name}.pt')
    return data,extra

if __name__ == '__main__':
    all_t_process([4,5,6,7,8,9,10],'doxycol','all',1250,30)


