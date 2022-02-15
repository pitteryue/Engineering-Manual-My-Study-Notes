import matplotlib.pyplot as plt
import pandas as pd
import re

if __name__=='__main__':
    this_data = "ml-100k"
    this_one = "embedding_size"
    this_two = 'learning_rate'

    hyper_results = pd.DataFrame(columns=['model', this_one, this_two, 'recall@10', 'mrr@10', 'ndcg@10', 'hit@10', 'precision@10'])
    hyper_results_line = 0

    for num in range(30):
        for this_model in ['BPR', 'ConvNCF', 'CDAE', 'DMF', 'GCMC', 'LightGCN', 'LINE', 'MacridVAE', 'MultiVAE',
                           'MultiDAE', 'NeuMF', 'NGCF', 'SpectralCF']:
            read_addr = this_data + "-hyper-results/" + str(num) + this_model + ".result"
            this_result = []
            group = 1
            line_num = 1

            f = open(read_addr)
            for line in f:
                if line_num == 1 or line_num == 5:
                    line_list = re.split(r'[:,\t \n ]\s*', line)
                    line_list = [element for element in line_list if element != '']
                    if line_num == 1:
                        this_result.append(this_model)
                        posi = 0
                        for ele in line_list:
                            if ele == this_one:
                                this_result.append(line_list[posi + 1])
                            posi += 1
                        posi = 0
                        for ele in line_list:
                            if ele == this_two:
                                this_result.append(line_list[posi + 1])
                            posi += 1
                    if line_num == 5:
                        for posi in [1, 3, 5, 7, 9]:
                            this_result.append(line_list[posi])

                line_num += 1
                if line_num == 7:
                    line_num = 1
                    group += 1
                    hyper_results.loc[hyper_results_line] = this_result
                    hyper_results_line += 1
                    this_result.clear()

            f.close()

        save_addr = this_data + " hyper_results " + str(num) + ".csv"
        hyper_results.to_csv(save_addr, index=False)