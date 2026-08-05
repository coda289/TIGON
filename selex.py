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
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from utility import *




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

    return new,counts_train,max

def plot_loss(checkpoint,integral_time):
    fig, axs = plt.subplots(2, 2, figsize=(28, 20))

    loss = checkpoint["LOSS"]
    axs[0, 0].plot(loss)
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_xlabel('Iteration')

    tr = checkpoint["TRANS"]
    axs[0, 1].plot(tr)
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_ylabel('Transport Cost')
    axs[0, 1].set_xlabel('Iteration')

    l22 = np.squeeze(checkpoint["L2_2"])
    for i in range(7):
        axs[1, 0].plot(l22[:, i], label=f'Time point {integral_time[0]} to {integral_time[i+1]}')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('L2 Loss')
    axs[1, 0].legend(loc='upper right', fontsize=25)

    l21 = np.squeeze(checkpoint["L2_1"])
    for i in range(7):
        axs[1, 1].plot(l21[:, i], label=f'Time point {integral_time[i]} to {integral_time[i+1]}')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('L2 Loss')
    axs[1, 1].legend(loc='upper right', fontsize=25)

    plt.tight_layout()
    plt.show()


def plot_jac_v(pca,func,z_t,time_pt,title,args,device):
    gene_list=np.arange(1,100)
    g_xt0 = torch.zeros(1, 1).type(torch.float32).to(device)
    logp_diff_xt0 = g_xt0
    # compute the mean of jacobian of v within cells z_t at time (time_pt)
    dim = z_t.shape[1]
    jac = np.zeros((dim,dim))
    for i in range(z_t.shape[0]):
        x_t = z_t[i,:].reshape([1,dim])
        v_xt = func(torch.tensor(time_pt).type(torch.float32).to(device),(x_t,g_xt0, logp_diff_xt0))[0]
        jac = jac+Jacobian(v_xt, x_t).reshape(dim,dim).detach().cpu().numpy()
    jac = jac/z_t.shape[0]

    components = pca.components_   # shape: (n_components, n_genes)

    jac = components.T @ jac @ components
    
    #ax = fig.add_subplot(111)
    plt.tight_layout()
    plt.axis('off')
    plt.margins(0, 0)
    plt.title('Jacobian of velocity')
    sns.heatmap(jac,cmap="vlag",xticklabels=gene_list,yticklabels=gene_list)
    plt.xticks([])  # Remove x-axis tick marks
    plt.yticks([])  # Remove y-axis tick marks
    plt.axis('off')
    #plt.savefig(os.path.join(args.save_dir, title),format="pdf",
    #            pad_inches=0.2, bbox_inches='tight')
    plt.show()

def plot_grad_g(pca,func,z_t,time_pt,title,args,device):
    g_xt0 = torch.zeros(1, 1).type(torch.float32).to(device)
    logp_diff_xt0 = g_xt0
    dim = z_t.shape[1]
    gg = np.zeros((dim,1))
    for i in range(z_t.shape[0]):
        x_t = z_t[i,:].reshape([1,dim])
        g_xt = func(torch.tensor(time_pt).type(torch.float32).to(device),(x_t,g_xt0, logp_diff_xt0))[1]
        gg = gg+torch.autograd.grad(g_xt, x_t, torch.ones_like(g_xt),retain_graph=True, create_graph=True)[0].view(x_t.shape[0], -1).reshape(dim,1).detach().cpu().numpy()
    gg = gg/z_t.shape[0]
    components = pca.components_  # (n_components, n_genes)

    gg = components.T @ gg
    np.save("array.npy", gg)



    n_chunks = 10
    chunk_size = 100

    fig, axes = plt.subplots(
        2, 5,
        figsize=(20, 16),   # adjust as needed
        dpi=200
    )

    axes = axes.flatten()
    list_n=['AA','AC','AG','AT','CC','CG','CT','GG','GT','TT']

    for i, ax in enumerate(axes):
        start = i * chunk_size
        end = (i + 1) * chunk_size

        gg_chunk = gg[start:end]

        sns.heatmap(
            gg_chunk,
            cmap="vlag",
            xticklabels=False,
            yticklabels=False,   # or gene_list[start:end] if you want labels
            cbar=(i == 0),       # only one colorbar
            ax=ax
        )

        ax.set_title(list_n[i], fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    return gg

def plot_growth_v(func,data_train,counts_train,times,device):
    g_xt0 = torch.zeros(1, 1).type(torch.float32).to(device)
    logp_diff_xt0 = g_xt0
    dim = data_train[0].shape[1]
    g_xt = [[] for _ in range(len(times))]
    for t in range(len(times)):
        z_t=data_train[t]
        for i in range(z_t.shape[0]):
            x_t = z_t[i,:].reshape([1,dim])
            g_xt[t].append((func(torch.tensor(times[t]).type(torch.float32).to(device),(x_t,g_xt0, logp_diff_xt0))[1]).detach().cpu().numpy())

    rows = []

    for t, growths in enumerate(g_xt):
        for g in growths:
            rows.append({
                "Round": np.arange(3,11)[t],
                "Growth": float(g)
            })

    df = pd.DataFrame(rows)

    sns.violinplot(x="Round", y="Growth", data=df)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.axvline(x=3.5,label='Counter Selection',color='red',linestyle='--')
    plt.legend(loc='upper right',fontsize=25)
    plt.show()

def plot_3d_vel_change(func,data_train,count_train,train_time,integral_time,args,device):
    viz_samples = 10000
    sigma_a = 0.001

    t_list = []#list(reversed(integral_time))#integral_time #np.linspace(5, 0, viz_timesteps)
    #options.update({'t_eval':t_list})
    
    z_t_samples = []
    z_t_data = []
    v = []
    g = []
    t_list2 = [] 
    odeint_setp = gcd_list([num * 100 for num in integral_time])/100
    integral_time2 = np.arange(integral_time[0], integral_time[-1]+odeint_setp, odeint_setp)
    integral_time2 = np.round_(integral_time2, decimals = 2)
    plot_time = list(reversed(integral_time2))
    sample_time = np.where(np.isin(np.array(plot_time),integral_time))[0]
    sample_time = list(reversed(sample_time))

    with torch.no_grad():
        for i in range(len(integral_time)):

            z_t0 =  data_train[i]

            z_t_data.append(z_t0.cpu().detach().numpy())
            t_list2.append(integral_time[i])
        
        # traj backward
        z_t0 =  data_train[-1]#Sampling_selex(viz_samples, train_time, len(train_time)-1,data_train,count_train,sigma_a,device)
        #z_t0 = z_t0[z_t0[:,2]>1]
        logp_diff_t0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
        g0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
        v_t = func(torch.tensor(integral_time[-1]).type(torch.float32).to(device),(z_t0,g0, logp_diff_t0))[0] #True_v(z_t0)
        g_t = func(torch.tensor(integral_time[-1]).type(torch.float32).to(device),(z_t0,g0, logp_diff_t0))[1]
        
        v.append(v_t.cpu().detach().numpy())
        g.append(g_t.cpu().detach().numpy())
        z_t_samples.append(z_t0.cpu().detach().numpy())
        t_list.append(plot_time[0])
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': None})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-5})
        options.update({'print_neval': False})
        options.update({'neval_max': 1000000})
        options.update({'safety': None})

        options.update({'t0': integral_time[-1]})
        options.update({'t1': 0})
        options.update({'t_eval':plot_time})
        z_t1,_, logp_diff_t1= odesolve(func,y0=(z_t0,g0, logp_diff_t0),options=options)
        for i in range(len(plot_time)-1):
            v_t = func(torch.tensor(plot_time[i+1]).type(torch.float32).to(device),(z_t1[i+1], g0, logp_diff_t1))[0] #True_v(z_t0)
            g_t = func(torch.tensor(plot_time[i+1]).type(torch.float32).to(device),(z_t1[i+1], g0, logp_diff_t1))[1]
            
            z_t_samples.append(z_t1[i+1].cpu().detach().numpy())
            g.append(g_t.cpu().detach().numpy())
            v.append(v_t.cpu().detach().numpy())
            t_list.append(plot_time[i+1])
        
    return z_t_samples, v, t_list


def plot_change(samp,vel,integral_time):
    speed = [np.mean(np.linalg.norm(vel[i], axis=1)) for i in range(len(integral_time))]
    speed_sd = [np.std(np.linalg.norm(vel[i], axis=1)) for i in range(len(integral_time))]
    change=[]
    sd=[]
    for i in range(len(integral_time)):
        dists = np.linalg.norm(samp[i][:, None, :] - samp[i][None, :, :], axis=2)
        pairwise= dists[np.triu_indices(len(samp[i]), k=1)]
        avg_dist = pairwise.mean()
        change.append(avg_dist)
        sd.append(pairwise.std())
    fig, axs = plt.subplots(2, figsize=(28, 20))
    axs[0].errorbar(
        np.arange(3, 11),
        change,
        yerr=sd,
        fmt='o-',
        capsize=4,
        linewidth=2,
        markersize=5
    )

    axs[0].set_xlabel("Round")
    axs[0].set_ylabel("Average Pairwise Distance")

    axs[0].set_xticks(np.arange(3, 11))
    axs[0].grid(axis='y', alpha=0.3)
    axs[0].set_yscale('log')

    axs[1].errorbar(
        np.arange(3, 11),
        speed,
        yerr=(speed_sd),
        fmt='o-',
        capsize=4,
        linewidth=2,
        markersize=5
    )

    axs[1].set_xlabel("Round")
    axs[1].set_ylabel("Average Predicted Speed")

    axs[1].set_xticks(np.arange(3, 11))
    axs[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    all_t_edit([3,4,5,6,7,8,9,10],'doxycol','all',30)