import seaborn as sns

def plot_corr(data, sz=10):    
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(sz,sz))         # Sample figsize in inches
    sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, ax=ax)
    
def plot_corr2(df,size=10):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);