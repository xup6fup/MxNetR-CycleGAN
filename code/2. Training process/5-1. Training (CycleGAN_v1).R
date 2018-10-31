
# Libraries

library(mxnet)

# Parameters

CTX <- mx.gpu()
Batch_size <- 1
num_show_img <- 1
start.epoch <- 1
#start.epoch <- 13
end.epoch <- 200
n.print <- 500
w_limit <- 0.1
learning_rate <- 1e-4
dis_drop_p <- 0
lambda_cycle_consistency_loss <- 10
lambda_identity_mapping_loss <- 0
model_name <- 'v1'

# Source

source('code/2. Training process/4. Training loop.R')

