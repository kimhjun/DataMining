"""
# R vers.

# enter manual params --------------------------------
datafile <- readline("Enter the data file name: " )
cat("Select the data coding format(1 = 'a b c' or 2 = 'a,b,c'):  ")
fm = scan(n=1, quiet=TRUE)
if(fm==1) {form = ""} else {form = ","}

cat("Select the number of a column of response variable:  ")
cn = scan(n=1, quiet=TRUE)
if(cn==1) {col_no = 1} else if(cn==2) {col_no = 2} else if(cn==3) {col_no = 3} else if(cn==4) {col_no = 4} else if(cn==5) {col_no = 5} else cat("invalid number")

outputfile <- readline("Enter the output file name: ")
data <- read.table(datafile, sep = form)

# func ---------------------------------------------

get_dev <- function(target, input, weight) {
  model <- as.matrix(input) %*% diag(weight)
  dev <- apply(model, 1, sum) - target
  return(dev)
}


loss_func <- function(deviation, weight) {
  ss <- sum(deviation ^ 2)
  #print(ss)
  mse <- ss / (2 * nrow(deviation))
  return(mse)
}

grad_ <- function(deviation, input, weight, lr) {
  grad_ <- t(deviation) %*% as.matrix(input)
  grad_ <- grad_ / (nrow(input))
  new_W <- weight - (lr * grad_)
  return(as.numeric(new_W))
}

grad <- function(deviation, input) {
  grad_ <- t(deviation) %*% as.matrix(input)
  grad_ <- grad_ / (nrow(input))
  return(grad_)
}

train <- function(target, input, weight, lr=0.000105, tol=0.01) {
  iter_ <- 0
  deviation <- get_dev(target, input, weight)
  cat("iter: ", iter_, "\tloss: ", loss_func(deviation, weight), "\n")
  grad_ <- grad(deviation, inputs)
  new_W <- as.numeric(weight - (lr  * grad_))
  iter_ <- iter_ + 1
  
  while(sqrt(sum(grad_ ^ 2)) > tol) {
    
    deviation <- get_dev(target, input, new_W)
    grad_ <- grad(deviation, inputs)
    new_W <- as.numeric(new_W - (lr  * grad_))
    if (iter_ %% 200000 == 0){
      cat("Gradient Vector: ", grad_, "\n")
      cat("Current Weight Vector: ", new_W, "\n")
      cat("iter: ", iter_, "\tloss: ", loss_func(deviation, new_W), "\n")
    }
    
    new_W <- as.numeric(new_W)
    iter_ <- iter_ + 1
  }
  cat("final_iter: ", iter_, "final_loss: ", loss_func(deviation, new_W), "\n")
  cat("Job done")
  return(new_W)
} 


normali <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}


# data 정리---------------------------------------

predictors <- c(1:5)
predictors <- predictors[!predictors == cn]

target <- as.matrix(data[cn])
tmax_ <- max(target)
tmin_ <- min(target)

inputs_ <- data[predictors]
inputs <- apply(inputs_, 2, normali)
max_ <- apply(inputs_, 2, max)
min_ <- apply(inputs_, 2, min)
      
dummy <- rep(1, nrow(target))
inputs <- cbind(dummy, inputs)
w_vec <- c(10, 0, 0, 0, 0)
w_diag <- diag(w_vec)


# main -------------------------------------------------
init_w <- c(0, 90, 1, 23, 0)
conv_w <- train(target, inputs, init_w, lr=0.01, tol=0.001)
y_est <- as.matrix(inputs) %*% conv_w
#y_est <- y_est_*(tmax_ - tmin_) + tmin_
ssr <- sum((target - y_est) ^ 2)
y_avg <- mean(as.matrix(target))
sst <- sum((target - y_avg) ^ 2)

mse <- ssr / nrow(target)
r2 <- 1 - (ssr/sst)

a <- inputs_[10, ]
b <- (conv_w[2:5] - min_) / (max_ - min_)
est <- sum(a*b) + conv_w[1]


# print output -----------------------------------------------------
fileConn <- file(outputfile, 'w')
writeLines(c("Coefficient"), fileConn)
writeLines(c("------------"), fileConn)
for ( i in 1:length(conv_w)){
  if(i == 1){
    writeLines(paste0("Constant: ", round(conv_w[1], 3)), fileConn)
  }
  else {
    writeLines(paste0("Beta", i, ": ", round(conv_w[i], 3)), fileConn)
  }
}
writeLines(c("\n"), fileConn)

writeLines(c("ID, Actual values, Fitted Values"), fileConn)
writeLines(c("---------------------------------"), fileConn)
for (i in 1:length(y_est)){
  writeLines(paste(i, target[i, ], round(y_est[i, ], 3), sep=', '), fileConn)
}
writeLines(c("\n"), fileConn)

writeLines(c("Model Summary"), fileConn)
writeLines(c("-------------"), fileConn)
writeLines(paste("R-sqaure = ", round(r2, 3)), fileConn)
writeLines(paste("MSE = ", round(mse, 3)), fileConn)
close(fileConn)

"""

# Python ver.

import sys
import math
import numpy as np

# raw file to input file : Resource list and Input list 

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
            normed = (input_list[j][i] - min[i])/(max[i]-min[i])
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
        new_vec[j] = w_vec[j] - (lr * grad[j])
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