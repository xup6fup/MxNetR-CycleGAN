
library(mxnet)

P2M_gen_model <- mx.model.load(prefix = 'model/CycleGAN_v2/P2M_gen_v2', iteration = 28)

arg.params_list <- list()

for (i in 1:length(P2M_gen_model$arg.params)) {
  
  arg.params_list[[i]] <- as.array(P2M_gen_model$arg.params[[i]])
  
}

save(arg.params_list, file = 'P2M_gen_v2.RData')
