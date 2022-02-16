import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == '__main__':

    dataset = "ml-100k"
    hyper_num = {'ConvNCF': 7, 'GCMC': 6, 'LightGCN': 4, 'LINE': 4, 'NeuMF': 4, 'NGCF': 7, 'SpectralCF': 4, 'DGCF': 9}

    for this_round in range(1):
        addr = dataset + ' hyper_results ' + str(this_round) + ".csv"
        pd_data = pd.read_csv(addr, sep = ',')
        #pd_data.sort_values(by = ['model', 'embedding_size'], ascending = [True, True], inplace = True)
        pd_mean_by_embedding_size = pd_data.groupby(['model','embedding_size'],as_index = False)[['recall@10', 'mrr@10','precision@10']].mean()
        pd_std_by_embedding_size = pd_data.groupby(['model','embedding_size'],as_index = False)[['recall@10', 'mrr@10','precision@10']].agg(np.std)
        pd_mean_by_learning_rate = pd_data.groupby(['model','learning_rate'],as_index = False)[['recall@10', 'mrr@10','precision@10']].mean()
        pd_std_by_learning_rate = pd_data.groupby(['model','learning_rate'],as_index = False)[['recall@10', 'mrr@10','precision@10']].agg(np.std)


        for this_hyper in ['embedding_size', 'learning_rate']:
            my_fig = plt.figure(3, figsize=(18,5))
            subplot_num = 1
            for this_metric in ['recall@10', 'mrr@10', 'precision@10']:
                plt.subplot(1,3,subplot_num)
                for this_model in ['ConvNCF', 'GCMC', 'LightGCN', 'LINE', 'NeuMF', 'NGCF', 'SpectralCF']:
                    if this_hyper == "embedding_size":
                        this_pd_mean_by_embedding_size = pd_mean_by_embedding_size[pd_mean_by_embedding_size['model'] == this_model]
                        this_pd_std_by_embedding_size = pd_std_by_embedding_size[pd_std_by_embedding_size['model'] == this_model]

                        x = np.array(this_pd_mean_by_embedding_size[this_hyper])
                        xi = list(x)
                        y = np.array(this_pd_mean_by_embedding_size[this_metric])
                        error = np.array(this_pd_std_by_embedding_size[this_metric])

                    if this_hyper == "learning_rate":
                        this_pd_mean_by_learning_rate = pd_mean_by_learning_rate[pd_mean_by_learning_rate['model'] == this_model]
                        this_pd_std_by_learning_rate = pd_std_by_learning_rate[pd_std_by_learning_rate['model'] == this_model]

                        x = np.array(this_pd_mean_by_learning_rate[this_hyper])
                        xi = list(x)
                        y = np.array(this_pd_mean_by_learning_rate[this_metric])
                        error = np.array(this_pd_std_by_learning_rate[this_metric])

                    plt.errorbar(x,y,error, label = this_model +" (" + str(hyper_num[this_model]) +")", fmt = 's--', markersize = 2.0, linewidth = 1.0,elinewidth = 0.3, uplims = True, lolims = True, capsize= 1.5)

                if this_hyper == "embedding_size":
                    plt.xticks(xi, x)
                    plt.xlim([0, 170])
                    plt.xlabel('embedding size')
                if this_hyper == "learning_rate":
                    plt.xticks(xi, x, rotation = 90)
                    plt.xlim([-0.001, 0.014])
                    plt.xlabel('learning rate')
                plt.ylabel(this_metric)
                plt.legend(loc='upper right', fontsize = 8)
                title_name =  "Results of " + this_metric + " on " + dataset + " realization " + str(this_round + 1)
                plt.title(title_name)

                subplot_num += 1

            plt.show()
            addr = title_name + '_' +this_hyper + ".pdf"
            pp = PdfPages(addr)
            pp.savefig(my_fig)
            pp.close()