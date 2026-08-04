import faiss
import numpy as np
import pandas as pd 
import torch
import numpy as np
from sklearn.decomposition import PCA
import joblib
import os
import math
import time
from concurrent.futures import ProcessPoolExecutor




def Sampling_selex(num_samples,time_all,time_pt,data_train,counts,sigma,device):
    #perturb the  coordinate x with Gaussian noise N (0, sigma*I )
    mu = data_train[time_pt]
    count=counts[time_pt]
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
    mu = data_train[time_pt]
    count=counts[time_pt]
    count = torch.log1p(count)#data
    count = count / count.sum()
    num_gaussian = mu.shape[0] # mu is number_data * dimension
    dim = mu.shape[1]
    dist2 = ((x[:, None, :] - mu[None, :, :]) ** 2).sum(dim=-1)
    dist2=dist2/sigma
    const=dim*(math.log(2*math.pi)+math.log(sigma))
    dist2=torch.exp(-.5*(dist2+const))#samp by data
    prob = dist2 @ count
    return prob

def knn_to_grid(r,df,survivor,top,X):
    top=df[df['Sequence'].isin(top)]
    t=top['Sequence'].to_list()
    grid=[]
    ind=[]
    X = np.asarray(X, dtype=np.float32)
    d = X.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(X)
    k = 100

    for seq in t:
        muts=survivor[survivor['top']==seq]['mut'].to_list()
        sub=df[df['Sequence'].isin(muts)].index.to_list()
        idx = df.index[df["Sequence"] == seq][0]
        #sub.append(idx)
        sub_des = np.asarray(X[sub], dtype=np.float32)
        index = faiss.IndexFlatL2(sub_des.shape[1])
        index.add(sub_des)
        k = min(100, len(sub_des))
        if k>0:
            
            _, nn = index.search(np.asarray(X[idx:idx+1], dtype=np.float32), k)

            # indices into sub_des
            nn = nn[0]

            # corresponding indices into X
            sub_nn = np.array(sub)[nn]

            grid.append(np.mean(X[sub_nn],axis=0))
            ind.append(sub)
    return np.asanyarray(grid),np.asanyarray(ind)






def all_t_edit(time,name,dir,n_componets):
    survivors=pd.read_csv('./all/edit.csv')
    top=survivors['top'].to_list()
    data=[]
    extra=[]
    pca = PCA(n_components=n_componets)
    args = [
        (t, survivors, top, name, dir)
        for t in time
    ]

    # Run each time point in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_time, args))

    # Make sure results are in the same order as `time`
    results = sorted(results, key=lambda x: time.index(x[0]))

    data = [x[1] for x in results]
    extra = [x[2] for x in results]
    PCA_data=np.vstack(data)
    pca.fit(PCA_data)
    for i in range(len(time)):
        d=pca.transform(data[i])
        np.savez_compressed(f'{dir}/{name}{(time[i]):02d}_2d/grid_edit.npz',grid=d,count=extra[i])
    joblib.dump(pca, f"{dir}/pca_info_edit.pkl")
    
    return data,extra 

 
def process_time(args):
    t, survivors, top, name, dir = args

    df = pd.read_csv(
        f"{dir}/{name}{t:02d}_2d/cleaned.csv"
    )

    descriptor = np.load(
        f"{dir}/{name}{t:02d}_2d/descriptor.npz"
    )

    X = descriptor["descriptor"]
    count = descriptor["counts"]

    grid, ind = knn_to_grid(t,
        df,
        survivors,
        top,
        X
    )

    mean = np.array([
        np.mean(count[idx])
        for idx in ind
    ])

    print(f"Finished round {t}")

    return t, grid, mean


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
        with np.load(f'{args.input_dir}/{args.dataset}{(t):02d}_2d/grid_edit.npz') as data:
            grid=data['grid'] 
            count=data['count']
        counts_train.append(torch.from_numpy(count).type(torch.float32).to(device))
        data_train.append(grid)
    grid_t=np.vstack(data_train)
    max=np.sqrt(np.max(grid_t**2))
    new=[]
    for i in range(len(args.timepoints)):
        n=data_train[i]/max
        new.append(torch.from_numpy(n).type(torch.float32).to(device))

    return new,counts_train

if __name__=='__main__':
    all_t_edit([3,4,5,6,7,8,9,10],'doxycol','all',30)