
# Source

source('code/2. Training process/2. Model architecture.R')

# Executor

M2P_gen_executor <- mx.simple.bind(symbol = M2P_gen,
                                   M2P_img = c(256, 256, 3, Batch_size),
                                   ctx = CTX, grad.req = "write")

P2M_gen_executor <- mx.simple.bind(symbol = P2M_gen,
                                   P2M_img = c(256, 256, 3, Batch_size),
                                   ctx = CTX, grad.req = "write")

Monet_dis_executor <- mx.simple.bind(symbol = Monet_loss,
                                     Monet_img = c(256, 256, 3, Batch_size), label = c(Batch_size),
                                     ctx = CTX, grad.req = "write")

Photo_dis_executor <- mx.simple.bind(symbol = Photo_loss,
                                     Photo_img = c(256, 256, 3, Batch_size), label = c(Batch_size),
                                     ctx = CTX, grad.req = "write")

cycle_consistency_executor <- mx.simple.bind(symbol = CC_loss,
                                             pred = c(256, 256, 3, Batch_size), label = c(256, 256, 3, Batch_size),
                                             ctx = CTX, grad.req = "write")

identity_mapping_executor <- mx.simple.bind(symbol = IM_loss,
                                            pred = c(256, 256, 3, Batch_size), label = c(256, 256, 3, Batch_size),
                                            ctx = CTX, grad.req = "write")

if (start.epoch == 1) {
  
  # Initial parameters
  
  mx.set.seed(0)
  
  M2P_gen_arg <- mxnet:::mx.model.init.params(symbol = M2P_gen,
                                              input.shape = list(M2P_img = c(256, 256, 3, Batch_size)),
                                              output.shape = NULL,
                                              initializer = mxnet:::mx.init.normal(0.02),
                                              ctx = CTX)
  
  P2M_gen_arg <- mxnet:::mx.model.init.params(symbol = P2M_gen,
                                              input.shape = list(P2M_img = c(256, 256, 3, Batch_size)),
                                              output.shape = NULL,
                                              initializer = mxnet:::mx.init.normal(0.02),
                                              ctx = CTX)
  
  Monet_dis_arg <- mxnet:::mx.model.init.params(symbol = Monet_loss,
                                                input.shape = list(Monet_img = c(256, 256, 3, Batch_size), label = c(Batch_size)),
                                                output.shape = NULL,
                                                initializer = mxnet:::mx.init.normal(0.02),
                                                ctx = CTX)
  
  Photo_dis_arg <- mxnet:::mx.model.init.params(symbol = Photo_loss,
                                                input.shape = list(Photo_img = c(256, 256, 3, Batch_size), label = c(Batch_size)),
                                                output.shape = NULL,
                                                initializer = mxnet:::mx.init.normal(0.02),
                                                ctx = CTX)
  
  # Update parameters
  
  mx.exec.update.arg.arrays(M2P_gen_executor, M2P_gen_arg$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(M2P_gen_executor, M2P_gen_arg$aux.params, match.name = TRUE)
  mx.exec.update.arg.arrays(P2M_gen_executor, P2M_gen_arg$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(P2M_gen_executor, P2M_gen_arg$aux.params, match.name = TRUE)
  mx.exec.update.arg.arrays(Monet_dis_executor, Monet_dis_arg$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(Monet_dis_executor, Monet_dis_arg$aux.params, match.name = TRUE)
  mx.exec.update.arg.arrays(Photo_dis_executor, Photo_dis_arg$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(Photo_dis_executor, Photo_dis_arg$aux.params, match.name = TRUE)
  
} else {
  
  # Load pre-trained models
  
  M2P_gen_model <- mx.model.load(prefix = paste0('model/CycleGAN_', model_name, '/M2P_gen_', model_name), iteration = start.epoch - 1)
  P2M_gen_model <- mx.model.load(prefix = paste0('model/CycleGAN_', model_name, '/P2M_gen_', model_name), iteration = start.epoch - 1)
  Monet_dis_model <- mx.model.load(prefix = paste0('model/CycleGAN_', model_name, '/Monet_dis_', model_name), iteration = start.epoch - 1)
  Photo_dis_model <- mx.model.load(prefix = paste0('model/CycleGAN_', model_name, '/Photo_dis_', model_name), iteration = start.epoch - 1)
  
  # Update parameters
  
  mx.exec.update.arg.arrays(M2P_gen_executor, M2P_gen_model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(M2P_gen_executor, M2P_gen_model$aux.params, match.name = TRUE)
  mx.exec.update.arg.arrays(P2M_gen_executor, P2M_gen_model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(P2M_gen_executor, P2M_gen_model$aux.params, match.name = TRUE)
  mx.exec.update.arg.arrays(Monet_dis_executor, Monet_dis_model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(Monet_dis_executor, Monet_dis_model$aux.params, match.name = TRUE)
  mx.exec.update.arg.arrays(Photo_dis_executor, Photo_dis_model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(Photo_dis_executor, Photo_dis_model$aux.params, match.name = TRUE)
  
}

# Optimizers

M2P_gen_optimizer <- mx.opt.create(name = "adam", learning.rate = learning_rate, beta1 = 0, beta2 = 0.9, wd = 0)
P2M_gen_optimizer <- mx.opt.create(name = "adam", learning.rate = learning_rate, beta1 = 0, beta2 = 0.9, wd = 0)

Monet_dis_optimizer <- mx.opt.create(name = "adam", learning.rate = learning_rate, beta1 = 0, beta2 = 0.9, wd = 0)
Photo_dis_optimizer <- mx.opt.create(name = "adam", learning.rate = learning_rate, beta1 = 0, beta2 = 0.9, wd = 0)

# Updaters

M2P_gen_updater <- mx.opt.get.updater(optimizer = M2P_gen_optimizer, weights = M2P_gen_executor$ref.arg.arrays)
P2M_gen_updater <- mx.opt.get.updater(optimizer = P2M_gen_optimizer, weights = P2M_gen_executor$ref.arg.arrays)
Monet_dis_updater <- mx.opt.get.updater(optimizer = Monet_dis_optimizer, weights = Monet_dis_executor$ref.arg.arrays)
Photo_dis_updater <- mx.opt.get.updater(optimizer = Photo_dis_optimizer, weights = Photo_dis_executor$ref.arg.arrays)
