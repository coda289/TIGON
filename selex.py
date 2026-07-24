import faiss
import numpy as np
import pandas as pd 
import torch
import numpy as np
from sklearn.decomposition import PCA
import joblib
import os



def Sampling_selex(num_samples,time_all,time_pt,data_train,counts,sigma,device):
    #perturb the  coordinate x with Gaussian noise N (0, sigma*I )
    mu = data_train[time_all[time_pt]]
    count=counts[time_all[time_pt]]
    count = np.log1p(count)
    count = count / count.sum()
    num_gaussian = mu.shape[0] # mu is number_sample * dimension
    dim = mu.shape[1]
    sigma_matrix = sigma * torch.eye(dim)
    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim), sigma_matrix)
    noise_add = m.rsample(torch.Size([num_samples])).type(torch.float32).to(device)
    # check if number of points is <num_samples
    count = torch.as_tensor(count, dtype=torch.float32)
    idx = torch.multinomial(count, num_samples, replacement=True)
    samples = mu[idx] + noise_add

    return samples


def MultimodalGaussian_density_selex(x,time_all,time_pt,data_train,counts,sigma,device):
    """density function for MultimodalGaussian
    """
    mu = data_train[time_all[time_pt]]
    count=counts[time_all[time_pt]]
    count = np.log1p(count)
    num_gaussian = mu.shape[0] # mu is number_sample * dimension
    dim = mu.shape[1]
    sigma_matrix = sigma * torch.eye(dim).type(torch.float32).to(device)
    p_unn = torch.zeros([x.shape[0]]).type(torch.float32).to(device)
    for i in range(num_gaussian):
        m = torch.distributions.multivariate_normal.MultivariateNormal(mu[i,:], sigma_matrix)
        p_unn = p_unn + count[i]*torch.exp(m.log_prob(x)).type(torch.float32).to(device)
    p_n = p_unn/count.sum()
    return p_n

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

def all_t_process(survivors,df,X,count,time,name,dir,k,n_componets):
    #survivors=pd.read_csv(f'{dir}/survivors.csv')
    survivors = survivors.rename(columns={"sequence": "Sequence"})
    survivors = survivors.rename(columns={"count": "Count"})
    survivors=survivors['Sequence'].to_list()
    data=[]
    extra=[]
    pca = PCA(n_components=n_componets)
    for t in time:
        #df=pd.read_csv(f'{dir}/{name}{(t):02d}_2d/cleaned.csv')
        #X=np.load(f'{dir}/{name}{(t):02d}_2d/descriptor.npz' )['descriptor']
        #count=np.load(f'{dir}/{name}{(t):02d}_2d/descriptor.npz')['counts']
        #X_reduced = pca.fit_transform(X)
        dist,ind,grid=knn_to_grid(df,survivors,X,k)
        data.append(grid)
        mean=np.mean(count[ind],axis=1)
        extra.append(mean)
    PCA_data=np.vstack(data)
    pca.fit(PCA_data)
    for i in range(len(time)):
        d=pca.transform(data[i])
        np.savez_compressed(f'{dir}/{name}{(time[i]):02d}_2d/grid.npz',grid=d,count=extra[i])
    joblib.dump(pca, f"{dir}/pca_info.pkl")
    
    return data,extra 

def loaddata_selex(args,device):
    #data=np.load(os.path.join(args.input_dir,(args.dataset+'.npy')),allow_pickle=True)
    data_train=[]
    counts_train=[]
    for t in args.timepoints:
        with np.load(f'{args.input_dir}/{args.dataset}{(t):02d}_2d/grid.npz') as data:
            grid=data['grid'] 
            count=data['count']
        counts_train.append(torch.from_numpy(count).type(torch.float32).to(device))
        data_train.append(torch.from_numpy(grid).type(torch.float32).to(device))
    return data_train,counts_train