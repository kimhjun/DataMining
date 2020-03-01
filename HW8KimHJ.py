import sys
import math
import numpy as np
import pandas as pd
from numpy.linalg import inv, det
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.stats as stats
import itertools

def preprocessing(filename, res_col, sep=' '):
    with open(filename) as input:
        data = input.read()
        row_list = data.split('\n')
        row_list = row_list[:-1]
        row_float_list = []

        for row in row_list:
            elem_list = row.split(sep)
            row_float = map(float, elem_list)
            row_float_list.append(list(row_float))

        responses = []
        inputs = []
        for row_ in row_float_list:
            responses.append(int(row_[int(res_col) - 1]))
            del row_[int(res_col) - 1]
            inputs.append([1] + row_)
    input.close()    
    return responses, inputs

def normalize(input):
    max = []
    min = []
    for i in range(len(input[0])):
        value = (list(map(lambda x: int(x[i]), input)))
        max_ = np.max(value)
        min_ = np.min(value)
        max.append(max_)
        min.append(min_)

    normed_vec = []
    for j in range(len(input)):    
        row_vec = []
        for i in range(1, len(input[0])):
            normed = (input[j][i] - min[i])/(max[i]-min[i])
            row_vec.append(normed)
        normed_vec.append([1] + row_vec)
    return(normed_vec)
    
# get initial value of weight vector
def init(Y, X, w_vector=[1, 30, 200, 300, 500]):
    if len(Y) != len(X):
        raise Exception('Index matching error')
    else:
        num_data = len(Y)
        num_dim = len(X[0])
    return w_vector, num_data, num_dim
    
# get loss function
def get_dev(Y, X, W):
    n_data = len(Y)
    y_est = get_y_est(Y, X, W)
    dev = [y_est[j] - Y[j] for j in range(n_data)]
    return dev
        
def loss_function(Y, X, W):
    dev_list = get_dev(Y, X, W)
    ss = sum(i**2 for i in dev_list)
    
    return ss / (2 * len(Y)) 

def get_grad(Y, X, W):
    n_dim = len(X[0])
    n_data = len(Y)
    dev_list = get_dev(Y, X, W)
    grad = []
    for j in range(n_dim):
        grad_dim = sum(dev_list[i] * X[i][j] for i in range(n_data))
        grad_dim /= n_data
        grad.append(grad_dim)
    return grad
    
def get_new_vec(Y, X, W, grad, lr=0.00005):
    new_vec = [0.0] * 5
    for j in range(5):
        new_vec[j] = W[j] - (lr * grad[j])
    return new_vec

def get_y_est(Y, X, W):
    n_dim = len(X[0])
    n_data = len(Y)
    y_est = []
    for i in range(n_data):
        y_est_ = sum(W[j] * X[i][j] for j in range(n_dim))
        y_est.append(y_est_)
    return y_est

def mse(Y, Y_est):
    num_data = len(Y)
    ssr = sum((Y[i] - Y_est[i]) ** 2 for i in range(num_data))
    mse = ssr / num_data
    return mse

def r2(Y, Y_est):
    num_data = len(Y)
    ssr = sum((Y[i] - Y_est[i]) ** 2 for i in range(num_data))
    mean_Y = sum(Y) / num_data
    sst = sum((Y[i] - mean_Y) ** 2 for i in range(num_data))
    return 1 - ssr/sst


def reg_main():
    filename = input("Enter the data file name: " )
    o_filename = input("Enter the result file name: ")
    res_col = input("Enter the column number of the response variable: ")
    sep = input("Select the data coding format(1 = 'a b c' or 2 = 'a,b,c'):  ")
    
    sep_opt = {'1':' ', '2':','}
    response_list, input_list = preprocessing(filename, res_col, sep_opt[sep])
    input_list = normalize(input_list)
    w_vec, _, _ = init(response_list, input_list, w_vector=[0, 0, 2, 30, 20]) # for running time, sets the initial vector closer enough than zero-vector 
    print("Processing... ")
    new_grad = [10, 10, 10, 10, 10]
    squared = math.sqrt(sum(list(map(lambda x: x**2, new_grad))))
    i = 0
    while(squared > 0.01): 
        new_grad = get_grad(response_list, input_list, w_vec)
        squared = math.sqrt(sum(list(map(lambda x: x**2, new_grad))))
        
        new_vec = get_new_vec(response_list, input_list, w_vec, new_grad, lr=0.001)
        if i % 200000 == 0:
            print("iter: {}, loss: {}".format(i, loss_function(response_list, input_list, new_vec)))
            print("Current L2-norm value of Gradient is {:.2f}".format(squared))
        w_vec = new_vec
        i += 1

    y_estimated = get_y_est(response_list, input_list, w_vec)

    # 출력부
    with open(o_filename, 'w') as outf:
        outf.write("Coefficients\n")
        outf.write("-----------------\n")
        outf.write("Constant: {}\n".format(w_vec[0]))
        for i in range(1, 5):
            outf.write("Beta{}: {}\n".format(i, w_vec[i]))
        outf.write("\n")

        outf.write("ID, Actual, Fitted Values\n")
        outf.write("--------------------------\n")
        for i in range(len(response_list)):
            outf.write("{}, {}, {:.2f}\n".format(i+1, response_list[i], y_estimated[i]))

        outf.write("\n")
        outf.write("Model Summary\n")
        outf.write("--------------\n")
        outf.write("R-square: {:.3f}\n".format(r2(response_list, y_estimated)))
        outf.write("MSE: {:.3f}\n".format(mse(response_list, y_estimated)))
    outf.close()

def generate_components(input):
    dat_label = pd.DataFrame(input.iloc[:,-1]).copy()
    dat_label.columns = ['class']
    n_class = len(dat_label['class'].value_counts())
    dat_pred = pd.DataFrame(input.iloc[:,:-1]).copy()
    freq_vec = np.array([len(dat_pred[dat_label['class'] == i]) for i in range(1, n_class +1)])
    categorized = [dat_pred[dat_label['class'] == i] for i in range(1, n_class + 1)]
    mean_vec = [categorized[i].apply(np.mean) for i in range(0, n_class)]
    cov_vec = [categorized[i].cov() for i in range(0, n_class)]
    return dat_pred, dat_label, mean_vec, cov_vec, freq_vec, n_class

def run_classification(input_pred, mean, cov, freq, n_class, mode='lda', alpha=None, gamma=None):
    prior_vec = freq/sum(freq)
    dev_mat = [input_pred - mean[i] for i in range(n_class)]
    cov_lda = sum([(freq[i] - 1) * cov[i] / (sum(freq) - n_class) for i in range(n_class)])
    if mode == "lda":
        res = [dev_mat[i].apply(lambda x: math.exp(-0.5 * np.matmul(np.matmul(x,  inv(cov_lda)), np.transpose(x))), axis=1) * prior_vec[i] for i in range(n_class)]
    elif mode == "qda":
        res = [dev_mat[i].apply(lambda x: det(cov[i]) ** (-0.5) * math.exp(-0.5 * np.matmul(np.matmul(x, inv(cov[i])), np.transpose(x))), axis=1) * prior_vec[i] for i in range(n_class)]
    elif mode == "rda":
        cov_rda_comp_1 = [alpha * cov[i] for i in range(n_class)]
        sigma = np.mean(np.diagonal(cov_lda)) * np.identity(len(cov_lda))
        cov_rda_comp_2 = gamma * cov_lda + (1 - gamma) * sigma
        cov_rda = [cov_rda_comp_1[i] + (1 - alpha) * cov_rda_comp_2 for i in range(n_class)]
        res = [dev_mat[i].apply(lambda x: det(cov_rda[i]) ** (-0.5) * math.exp(-0.5 * np.matmul(np.matmul(x, inv(cov_rda[i])), np.transpose(x))), axis=1) * prior_vec[i] for i in range(n_class)]
    res_ = pd.DataFrame(np.transpose(res))
    pred_res = res_.idxmax(axis=1)
    pred_res = pred_res.map(lambda x: x+1)
    pred_res = pd.DataFrame(pred_res, columns=['pred_label'])
    return pred_res, res_

def accuracy(model_res, label):
    return sum(model_res == label)/len(label)

def get_ss(conf_mat, method):
    if method == '4':
        accuracy = (conf_mat[0][0] + conf_mat[1][1]) / sum(np.sum(conf_mat))
        sensitivity = conf_mat[1][1]/ sum(conf_mat[1])
        specificity = conf_mat[0][0]/ sum(conf_mat[0])
    elif method == '7':
        accuracy = (conf_mat.loc["Yes", "Yes"] + conf_mat.loc["No", "No"]) / sum(np.sum(conf_mat))
        sensitivity = conf_mat.loc["Yes", "Yes"]/ sum(conf_mat["Yes"])
        specificity = conf_mat.loc["No", "No"]/ sum(conf_mat["No"])
    else:
        accuracy = (conf_mat[1][1] + conf_mat[2][2]) / sum(np.sum(conf_mat))
        sensitivity = conf_mat[2][2]/ sum(conf_mat[2])
        specificity = conf_mat[1][1]/ sum(conf_mat[1])
    return accuracy, sensitivity, specificity

def sigmoid(power):
    return 1 / (np.exp(-power) + 1)

class LogisticRegression():
    def __init__(self, data, mode='train'):
        self.data = data
        self.mode = mode
        self.iter_count = 0
        self.n_class = 2
        #print('Logistic init')
    
    def preprop(self):
        self.resp = self.data.iloc[:, -1]
        self.predictor = self.data.iloc[:, :-1]
        self.nrow = self.predictor.shape[0]
        self.predictor.insert(loc=0, column='zero', value=[1] * self.nrow)
        self.ncol = self.predictor.shape[1]
        resp_map = {1:0, 2:1}
        self.resp = self.resp.map(lambda x: resp_map[x])
        self.weight_vec = np.zeros(self.ncol)
        return self.predictor, self.resp, self.weight_vec
    
    @staticmethod
    def predict(predictor, weight):
        mod = np.dot(predictor, weight)
        pred_prob = sigmoid(mod)
        return pred_prob
    
    def get_loss(self):
        pred_ = self.predict(self.predictor, self.weight_vec)
        factor_1 = np.log(pred_) * self.resp
        factor_2 = np.log(1 - pred_) * (1 - self.resp)
        loss = (factor_1 + factor_2)
        self.loss = loss.mean() * -1
        return self.loss

    def get_grad(self):
        pred_ = self.predict(self.predictor, self.weight_vec)
        grad = np.dot(self.predictor.T, pred_ - self.resp) / self.nrow
        return grad

    def stoch_grad(self, row_num, batch_size):
        pred_ = self.predict(self.predictor, self.weight_vec)
        minibatch = self.predictor.sample(n=batch_size)
        s_grad = np.dot(self.predictor.iloc[minibatch.index, :].T, pred_[minibatch.index] - self.resp[minibatch.index]) /  batch_size
        return s_grad
    
    def train(self, learning_rate=0.001, mode='batch', max_iter=100000, batch_size=28):
        while(self.iter_count < max_iter):
            if mode == 'batch':
                grad = self.get_grad()
                new_vec = self.weight_vec - learning_rate * grad
                self.weight_vec = new_vec
                self.iter_count += 1
            elif mode == 'stochastic':
                for row_num in range(0, self.nrow, batch_size):
                    grad = self.stoch_grad(row_num, batch_size)
                    new_vec = self.weight_vec - learning_rate * grad
                    self.weight_vec = new_vec
                    self.iter_count += 1
                    
            if self.iter_count % 45000 == 0:
                print("grad vector size: {}".format(math.sqrt(sum(grad ** 2))))
                print("epoch: {}, loss: {}".format(self.iter_count, self.get_loss()))    
        return new_vec
    
    def get_result(self, cutoff):
        pred_df = pd.DataFrame(self.predict(self.predictor, self.weight_vec), columns=['pred'])
        pred_df['pred_label'] = pred_df['pred'].map(lambda x : 1 if x >= cutoff else 0)
        pred_df['original_label'] = self.resp
        return pred_df

class NaiveBayesClassifier():
    def __init__(self, data, n_class=2, response_column=13, seed=2000):
        self.data = data
        self.n_class = n_class
        self.response_column = response_column
        self.seed = seed
    
    def split_tr_test(self, test_size):
        self.test_data = self.data.sample(n=test_size, random_state=self.seed)
        self.train_data = self.data.drop(self.test_data.index)
        return self.train_data, self.test_data
    
    @staticmethod
    def preprop(df):
        predictor = df.iloc[:, :-1]
        response = df.iloc[:, -1]
        return predictor, response
    
    def assign_column_type(self, categorical_index=[1, 2, 5, 6, 8, 10, 11, 12]):
        self.categorical_index = categorical_index
        self.numeric_index = [i for i in range(0, 13)]
        for j in self.categorical_index:
            self.numeric_index.remove(j)
        return self.categorical_index, self.numeric_index
    
    def set_probtable(self, predictor, tf=1):
        self.prob_table = {}
        for i in self.categorical_index:
            vc = predictor[self.train_data[self.response_column] == tf][i].value_counts().sort_index()
            try:
                vc = vc.drop(['?'])
            except KeyError:
                pass
            vc = vc/sum(vc)
            self.prob_table[i] = vc

        for j in self.numeric_index:
            mean_val = predictor[self.train_data[self.response_column] == tf][j].mean()
            std_val = predictor[self.train_data[self.response_column] == tf][j].std()
            self.prob_table[j] = stats.norm(mean_val, std_val)
        return self.prob_table
    
    def get_prob(self, data, row_index, true_table, false_table):
        true_prob_vec = np.zeros(len(data.columns))
        false_prob_vec = np.zeros(len(data.columns))

        for col_no in data.columns:
            col_val = data.iloc[row_index, col_no]
            if col_val == '?':
                true_prob_vec[col_no] = 1
                false_prob_vec[col_no] = 1
            else:
                if col_no in self.categorical_index:
                    try:
                        true_prob_vec[col_no] = true_table[col_no][col_val]
                    except KeyError:
                        true_prob_vec[col_no] = 1

                    try:  
                        false_prob_vec[col_no] = false_table[col_no][col_val]
                    except KeyError:
                        false_prob_vec[col_no] = 1    
                else:
                    true_prob_vec[col_no] = true_table[col_no].pdf(col_val)
                    false_prob_vec[col_no] = false_table[col_no].pdf(col_val)

        true_prob = np.prod(true_prob_vec)
        false_prob = np.prod(false_prob_vec)

        prob = true_prob / (true_prob + false_prob)
        return prob
    
    @staticmethod
    def get_result(result, label):
        res_df = pd.DataFrame(result, columns=['pred'])
        res_df['pred_label'] = res_df['pred'].map(lambda x : 1 if x > 0.5 else 2)
        res_df['original_label'] = label.reset_index(drop=True)
        return res_df
        
class OneLevelDecisionTree(NaiveBayesClassifier):
    def set_probtable(self, predictor, tf=1):
        self.prob_table = {}
        for j in self.numeric_index:
            mean_val = predictor[self.train_data[self.response_column] == tf][j].mean()
            std_val = predictor[self.train_data[self.response_column] == tf][j].std()
            self.prob_table[j] = stats.norm(mean_val, std_val)
        return self.prob_table

    def get_majority_list(self, predictor):
        self.major_table = {}
        for i in self.numeric_index:
            fcount = predictor[self.train_data[self.response_column] == 1][i].value_counts().sort_index()
            tcount = predictor[self.train_data[self.response_column] == 2][i].value_counts().sort_index()
            counts = pd.concat([fcount, tcount], axis=1)
            counts.columns = [1, 2]
            counts = counts.fillna(0)
            self.major_table[i] = counts.idxmax(axis=1)
        return self.major_table
    
    def get_partition(self, majority_table, cutoff=7):            
        self.partition_map = {}
        for list_key in majority_table.keys():
            part_list = []
            count_1 = 0
            count_2 = 0
            for k, i in pd.DataFrame(majority_table[list_key], columns=['counts']).iterrows():
                if i['counts'] == 1: 
                    count_1 += 1
                else:
                    count_2 += 1
                if min(count_1, count_2) == cutoff:
                    partition_1 = majority_table[list_key][majority_table[list_key].index < k]
                    partition_2 = majority_table[list_key][majority_table[list_key].index >= k]
                    pivot = k
                    part_list.append(pivot)
                    part_list.append(partition_1)
                    part_list.append(partition_2)
                    break
                
            self.partition_map[list_key] = part_list
        return self.partition_map
    
    @staticmethod
    def get_result(dataset, label, selected_var, response_column, partition_info):
        pivot, part_1, part_2 = partition_info[selected_var]
        part_1_majority = part_1.value_counts().idxmax()
        part_2_majority = part_2.value_counts().idxmax()
        tr_res_df = dataset[[selected_var, response_column]].copy()
        tr_res_df.loc[tr_res_df[selected_var] < pivot, 'pred_label'] = part_1_majority
        tr_res_df.loc[tr_res_df[selected_var] >= pivot, 'pred_label'] = part_2_majority
        tr_res_df['original_label'] = label
        return tr_res_df

class CARTDecisionTree(OneLevelDecisionTree): 
    def cart_split_rule(self):
        col_list = self.train_data.columns[:-1]
        self.root_impurity = self.get_impurity(self.train_data, 'Survived', 'Yes', 'No')
        impurity_array = np.zeros(3)
        part_1_set = []
        part_2_set = []
        for col_i, col in enumerate(col_list):
            candidate = self.train_data[col].value_counts().index
            if len(candidate) == 2:
                part_1 = self.train_data.loc[self.train_data[col] == candidate[0], :]
                part_2 = self.train_data.loc[self.train_data[col] == candidate[1], :]
                part_1_impurity = self.get_impurity(part_1, 'Survived', 'Yes', 'No')
                part_2_impurity = self.get_impurity(part_2, 'Survived', 'Yes', 'No')
                part_1_weight = len(part_1)/len(self.train_data)
                part_2_weight = len(part_2)/len(self.train_data)
                final_impurity = part_1_impurity * part_1_weight + part_2_impurity * part_2_weight
                impurity_array[col_i] = final_impurity
                part_1_set.append(candidate[0].split(' '))
                part_2_set.append(candidate[1].split(' '))
            else:
                part_1_index = []
                part_2_index = []
                for pattern in itertools.product([True,False],repeat=len(candidate)):
                    part_1_index.append([x[1] for x in zip(pattern,candidate) if x[0]])
                    part_2_index.append([x[1] for x in zip(pattern,candidate) if not x[0]])
                part_1_index = part_1_index[1:2**(len(candidate)-1)-1]
                part_2_index = part_2_index[1:2**(len(candidate)-1)-1]
                part_dict = {}
                for i, part_i in enumerate(part_1_index):
                    part_1 = self.train_data.loc[self.train_data[col].isin(part_i), :]
                    part_2 = self.train_data.loc[self.train_data[col].isin(part_2_index[i]), :]
                    part_1_impurity = self.get_impurity(part_1, 'Survived', 'Yes', 'No')
                    part_2_impurity = self.get_impurity(part_2, 'Survived', 'Yes', 'No')
                    part_1_weight = len(part_1)/len(self.train_data)
                    part_2_weight = len(part_2)/len(self.train_data)
                    final_impurity = part_1_impurity * part_1_weight + part_2_impurity * part_2_weight
                    part_dict[i] = final_impurity
                fin_id = min(part_dict, key = lambda x: part_dict.get(x))
                final_impurity = part_dict[min(part_dict, key = lambda x: part_dict.get(x))]
                impurity_array[col_i] = final_impurity
                part_1_set.append(part_1_index[fin_id])
                part_2_set.append(part_2_index[fin_id])
        return impurity_array, part_1_set, part_2_set
        
    @staticmethod
    def get_impurity(df, response_var, class_1, class_2):
        prob_c1 = sum(df[response_var] == class_1) / len(df)
        prob_c2 = sum(df[response_var] == class_2) / len(df)
        return 1 - prob_c1 ** 2 - prob_c2 ** 2
    
    def get_result(self, dataset, label, response_column):
        res_impurity, partition_1, partition_2 = self.cart_split_rule()
        selected_idx = np.argmin(res_impurity)
        res_imp = res_impurity[selected_idx]
        part_1 = dataset.loc[dataset[dataset.columns[selected_idx]].isin(partition_1[selected_idx]), :]
        part_2 = dataset.loc[dataset[dataset.columns[selected_idx]].isin(partition_2[selected_idx]), :]
        part_1_majority = part_1[response_column].value_counts().idxmax()
        part_2_majority = part_2[response_column].value_counts().idxmax()
        tr_res_df = dataset.copy()
        tr_res_df.loc[tr_res_df[tr_res_df.columns[selected_idx]].isin(partition_1[selected_idx]), 'pred_label'] = part_1_majority
        tr_res_df.loc[tr_res_df[tr_res_df.columns[selected_idx]].isin(partition_2[selected_idx]), 'pred_label'] = part_2_majority
        tr_res_df['original_label'] = label
        return tr_res_df

def get_result(input_pred, input_resp, weight):
    pred_df = pd.DataFrame(predict(input_pred, weight), columns=['pred'])
    pred_df['pred_label'] = pred_df['pred'].map(lambda x : 1 if x >= 0.5 else 0)
    pred_df['original_label'] = input_resp
    return pred_df

def get_confmat(pred_res, true, n_class, method='4'):
    if method != '4' and method != '7':
        conf_mat = pd.DataFrame([[sum(pred_res[true == i] == j) for i in range(1, n_class+1)] for j in range(1, n_class + 1)])
    elif method == '7':
        class_label = ["No", "Yes"]
        conf_mat = pd.DataFrame([[sum(pred_res[true == i] == j) for i in class_label] for j in class_label])
    else:
        conf_mat = pd.DataFrame([[sum(pred_res[true == i] == j) for i in range(0, n_class)] for j in range(0, n_class)])
    conf_mat = conf_mat.rename(columns={0:1, 1:2, 2:3, 3:4}, index={0:1, 1:2, 2:3, 3:4})
    conf_mat.index.name = 'Predicted Class'
    conf_mat.columns.name = 'Actual Class'
    return conf_mat

def accuracy(model_res, label):
    return sum(model_res == label)/len(label)

def get_ss(conf_mat, method):
    if method == '4':
        accuracy = (conf_mat[0][0] + conf_mat[1][1]) / sum(np.sum(conf_mat))
        sensitivity = conf_mat[1][1]/ sum(conf_mat[1])
        specificity = conf_mat[0][0]/ sum(conf_mat[0])
    elif method == '6':
        accuracy = (conf_mat.loc["Yes", "Yes"] + conf_mat.loc["No", "No"]) / sum(np.sum(conf_mat))
        sensitivity = conf_mat.loc["Yes", "Yes"]/ sum(conf_mat["Yes"])
        specificity = conf_mat.loc["No", "No"]/ sum(conf_mat["No"])
    else:
        accuracy = (conf_mat[1][1] + conf_mat[2][2]) / sum(np.sum(conf_mat))
        sensitivity = conf_mat[2][2]/ sum(conf_mat[2])
        specificity = conf_mat[1][1]/ sum(conf_mat[1])
    return accuracy, sensitivity, specificity


def get_confmat(pred_res, true, n_class, method='4'):
    if method != '4' and method != '6':
        conf_mat = pd.DataFrame([[sum(pred_res[true == i] == j) for i in range(1, n_class+1)] for j in range(1, n_class + 1)])
    elif method == '6':
        class_label = ["No", "Yes"]
        conf_mat = pd.DataFrame([[sum(pred_res[true == i] == j) for i in class_label] for j in class_label])
    else:
        conf_mat = pd.DataFrame([[sum(pred_res[true == i] == j) for i in range(0, n_class)] for j in range(0, n_class)])
    conf_mat = conf_mat.rename(columns={0:1, 1:2, 2:3, 3:4}, index={0:1, 1:2, 2:3, 3:4})
    conf_mat.index.name = 'Predicted Class'
    conf_mat.columns.name = 'Actual Class'
    return conf_mat

def write_summary_cls(o_file, train_df, test_df, train_label, test_label, n_class, method, selected_var=None, partition_info=None):
    with open(o_file, 'w') as outf:
        if method == '6':
            outf.write("Tree Structure\n")
            outf.write("\tNode 1: {} in {} ({}, {})\n".format(train_df.columns[selected_var], partition_info[0][selected_var], sum(train_label=="Yes"), sum(train_label=="No")))
            p1_yes = sum(train_df.loc[train_df[train_df.columns[selected_var]].isin(partition_info[0][selected_var]), :]["Survived"] == "Yes")
            p1_no = sum(train_df.loc[train_df[train_df.columns[selected_var]].isin(partition_info[0][selected_var]), :]["Survived"] == "No")
            p2_yes = sum(train_df.loc[train_df[train_df.columns[selected_var]].isin(partition_info[1][selected_var]), :]["Survived"] == "Yes")
            p2_no = sum(train_df.loc[train_df[train_df.columns[selected_var]].isin(partition_info[1][selected_var]), :]["Survived"] == "No")
            outf.write("\t\tNode 2: Yes ({}, {})\n".format(p1_yes, p1_no))
            outf.write("\t\tNode 3: No ({}, {})\n".format(p2_yes, p2_no))
            outf.write("\n")
        
        if method == '4': 
            outf.write("ID, Actual Class, Resub pred, Resub Prob\n")
            outf.write("----------------------------\n")
            for i in range(10):
                outf.write("{}, {}, {}, {}\n".format(i+1, train_label[i], train_df['pred_label'][i], train_df['pred'][i]))
            outf.write("(Continue...)\n")

        else:
            outf.write("ID, Actual Class, Resub pred\n")
            outf.write("----------------------------\n")
            for i in range(10):
                outf.write("{}, {}, {}\n".format(i+1, train_label[i], train_df['pred_label'][i]))
            outf.write("(Continue...)\n")
        outf.write("\n")
        outf.write("Confusion Matrix (Resubstitution)\n")
        outf.write("---------------------------------\n")
        train_mat = get_confmat(train_df['pred_label'], train_label, n_class, method)
        
        if method == '4':
            train_mat.index = [0, 1]
            train_mat.columns = [0, 1]
        elif method == '6':
            train_mat.index = ["Yes", "No"]
            train_mat.columns = ["Yes", "No"]
        train_mat.index.name = 'Predicted Class'
        train_mat.columns.name = 'Actual Class'
        outf.write(str(train_mat))
        outf.write("\n")
        outf.write("\n")
        outf.write("Model Summary (Resubstitution)\n")
        outf.write("------------------------------\n")
        outf.write("Overall accuracy: {:.3f}\n".format(accuracy(train_df['pred_label'], train_label)))
        if n_class == 2:
            outf.write("Sensitivity: {:.3f}\n".format(get_ss(train_mat, method)[1]))
            outf.write("Specificity: {:.3f}\n".format(get_ss(train_mat, method)[2]))
        outf.write("\n")
        
        if method == '4':
            outf.write("ID, Actual Class, Test pred, Test Prob\n")
            outf.write("----------------------------\n")
            for i in range(10):
                outf.write("{}, {}, {}, {}\n".format(i+1, test_label[i], test_df['pred_label'][i], test_df['pred'][i]))
            outf.write("(Continue...)\n")
        else:
            outf.write("ID, Actual Class, Test pred\n")
            outf.write("----------------------------\n")
            for i in range(10):
                outf.write("{}, {}, {}\n".format(i+1, test_label[i], test_df['pred_label'][i]))
            outf.write("(Continue...)\n")
        outf.write("\n")
        outf.write("Confusion Matrix (Test)\n")
        outf.write("---------------------------------\n")
        test_mat = get_confmat(test_df['pred_label'], test_label, n_class, method)
        if method == '4': 
            test_mat.index = [0, 1]
            test_mat.columns = [0, 1]
        elif method == '6':
            test_mat.index = ["Yes", "No"]
            test_mat.columns = ["Yes", "No"]
        test_mat.index.name = 'Predicted Class'
        test_mat.columns.name = 'Actual Class'
        outf.write(str(test_mat))
        outf.write("\n")
        outf.write("\n")
        outf.write("Model Summary (Test)\n")
        outf.write("------------------------------\n")
        outf.write("Overall accuracy: {:.3f}\n".format(accuracy(test_df['pred_label'], test_label)))
        if n_class == 2:
            outf.write("Sensitivity: {:.3f}\n".format(get_ss(test_mat, method)[1]))
            outf.write("Specificity: {:.3f}\n".format(get_ss(test_mat, method)[2]))
    outf.close()

def cls_main():
    bon = input("Binary or not?(1 = 'Binary', 2 = 'Not Binary': " )
    if bon == '1':        
        method = input("Choose method for Classification(1 = 'LDA', 2 = 'QDA', 3 = 'RDA', 4 = 'Logistic Regression', 5 = 'Naive Bayes', 6 = '1-level Decesion Tree'):  ")
        method_opt = {'1': 'lda', '2': 'qda', '3': 'rda', '4': 'logistic', '5': 'naivebayes', '6': '1r'}
        if method != '5' and method != '6':
            train_filename = "data/pid.dat"
            test_filename = "data/pidtest.dat"
            sep = input("Select the data coding format(1 = 'a b c' or 2 = 'a,b,c'):  ")
            sep_opt = {'1':' ', '2':','}
            print("file for train: {}, file for test: {}".format(train_filename, test_filename))
        elif method == '6':
            train_filename = "data/titanic.csv"
            sep = input("Select the data coding format(1 = 'a b c' or 2 = 'a,b,c'):  ")
            sep_opt = {'1':' ', '2':','}
        else:
            train_filename = "data/heart.dat"
            sep = input("Select the data coding format(1 = 'a b c' or 2 = 'a,b,c'):  ")
            sep_opt = {'1':' ', '2':','}

    else:
        train_filename = input("Enter the training data file name: " )
        test_filename = input("Enter the test data file name: " )
        sep = input("Select the data coding format(1 = 'a b c' or 2 = 'a,b,c'):  ")
        sep_opt = {'1':' ', '2':','}
        method = input("Choose method for Classification(1 = 'LDA', 2 = 'QDA', 3 = 'RDA'):  ")
        method_opt = {'1': 'lda', '2': 'qda', '3': 'rda'}

    o_filename = input("Enter the result file name: ")
    if method == '6':
        tr_data = pd.read_csv(train_filename, sep=sep_opt[sep], header=0)
    else:
        tr_data = pd.read_csv(train_filename, sep=sep_opt[sep], header=None)
    if method != '5' and method != '6':
        test_data = pd.read_csv(test_filename, sep=sep_opt[sep], header=None)

    if method != '4' and method != '5' and method != '6':
        tr_pred, tr_label, mean_vec, cov_vec, freq_vec, n_class = generate_components(tr_data)
        test_pred, test_label, _, _, _, _ = generate_components(test_data)

        # alpha, gamma 별로 값 저장하는 거 만들기
        if method_opt[method] == 'rda':
            values = np.linspace(start=0, stop=1, num=21)
            box_tr = np.zeros([len(values), len(values)])
            box_test = np.zeros([len(values), len(values)])
            i = 0
            for x in values:
                tr_error = np.array([accuracy(run_classification(tr_pred, mean_vec, cov_vec, freq_vec, n_class, mode='rda', alpha=x, gamma=y)[0]['pred_label'], tr_label['class']) for y in values])
                test_error = np.array([accuracy(run_classification(test_pred, mean_vec, cov_vec, freq_vec, n_class, mode='rda', alpha=x, gamma=y)[0]['pred_label'], test_label['class']) for y in values])
                box_tr[i, :] = tr_error
                box_test[i, :] = test_error
                i += 1
            x_, y_ = np.unravel_index(box_test.argmax(), box_test.shape)
            predicted_label_train, _ = run_classification(tr_pred, mean_vec, cov_vec, freq_vec, n_class, mode='rda', alpha=x_ * 0.05 , gamma=y_ * 0.05)
            predicted_label_test, _ = run_classification(test_pred, mean_vec, cov_vec, freq_vec, n_class, mode=method_opt[method], alpha=x_ * 0.05, gamma=y_ * 0.05)
            
            # Plotting
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            zdata = box_test
            xdata = values
            ydata = values
            X, Y = np.meshgrid(xdata, ydata)
            ax.view_init(20, 270)
            ax.plot_surface(Y, X, zdata, rstride=1, cstride=1,
                            edgecolor='none')
            ax.plot_surface(Y, X, box_tr, rstride=1, cstride=1, alpha=0.4,
                            edgecolor='none', )
            ax.plot([x_ * 0.05], [y_ * 0.05], [np.max(box_test)], markerfacecolor='r', markeredgecolor='w', marker='o', markersize=5, alpha=1)

            ax.set_title('Surface Plot')
            ax.set_xlabel('alpha')
            ax.set_ylabel('gamma')
            fig.savefig('3dplot.png')
        else:
            predicted_label_train, _ = run_classification(tr_pred, mean_vec, cov_vec, freq_vec, n_class, mode=method_opt[method])
            predicted_label_test, _ = run_classification(test_pred, mean_vec, cov_vec, freq_vec, n_class, mode=method_opt[method])

        write_summary_cls(o_filename, predicted_label_train, predicted_label_test, tr_label['class'], test_label['class'], n_class, method)
    
    elif method == '4':
        lr = LogisticRegression(tr_data)
        lr_test = LogisticRegression(test_data)
        pred_df, resp_df, weight_vec = lr.preprop()
        test_pred_df, test_resp_df, _ = lr_test.preprop()
        final_weight = lr.train(learning_rate=0.0005, mode='batch', max_iter=20000)
        lr_test.weight_vec = final_weight
        tr_res_df = lr.get_result(0.6)
        test_res_df = lr_test.get_result(0.6)
        conf_tr = get_confmat(tr_res_df['pred_label'], lr.resp, lr.n_class, '4')
        conf_test = get_confmat(test_res_df['pred_label'], lr_test.resp, lr_test.n_class, '4')
        write_summary_cls(o_filename, tr_res_df, test_res_df, resp_df, test_resp_df, 2, '4')
    
    elif method == '5':
        NBC = NaiveBayesClassifier(tr_data, seed=1270)
        train, test = NBC.split_tr_test(50)
        categorical_index, numeric_index = NBC.assign_column_type()
        tr_pred, tr_resp = NBC.preprop(train)
        test_pred, test_resp = NBC.preprop(test)
        true_table = NBC.set_probtable(tr_pred)
        false_table = NBC.set_probtable(tr_pred, tf=2)
        train_probs = np.array([NBC.get_prob(data=tr_pred, row_index=i, true_table=true_table, false_table=false_table) for i in range(len(tr_pred))])
        test_probs = np.array([NBC.get_prob(data=test_pred, row_index=i, true_table=true_table, false_table=false_table) for i in range(len(test_pred))])
        train_res_df = NBC.get_result(train_probs, tr_resp)
        test_res_df = NBC.get_result(test_probs, test_resp)
        write_summary_cls(o_filename, train_res_df, test_res_df, tr_resp.reset_index(drop=True), test_resp.reset_index(drop=True), 2, '5')

    elif method == '6':
        CDT = CARTDecisionTree(tr_data, seed=1270)
        train, test = CDT.split_tr_test(250)
        tr_pred, tr_resp = CDT.preprop(train)
        test_pred, test_resp = CDT.preprop(test)
        tr_res_df = CDT.get_result(train, tr_resp, "Survived")
        test_res_df = CDT.get_result(test, test_resp, "Survived")
        res_impurity, partition_1, partition_2 = CDT.cart_split_rule()
        selected_idx = np.argmin(res_impurity)
        res_imp = res_impurity[selected_idx]
        part_list = []
        part_list.append(partition_1)
        part_list.append(partition_2)
        write_summary_cls(o_filename, tr_res_df.reset_index(drop=True), test_res_df.reset_index(drop=True), tr_resp.reset_index(drop=True), test_resp.reset_index(drop=True), 2, '6', selected_var=selected_idx, partition_info=part_list)

if __name__== "__main__":
    task_name = input("Select the task (1 = 'Regression' or 2 = 'Classification'): " )
    if int(task_name) == 1:
        reg_main()
    elif int(task_name) == 2:
        cls_main()