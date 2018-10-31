
# Libraries

library(mxnet)
library(imager)
library(jpeg)
library(magrittr)

# Parameters

model_name <- 'v2'
iter <- 28

# Load model

M2P_gen_model <- mx.model.load(prefix = paste0('model/CycleGAN_', model_name, '/M2P_gen_', model_name), iteration = iter)
P2M_gen_model <- mx.model.load(prefix = paste0('model/CycleGAN_', model_name, '/P2M_gen_', model_name), iteration = iter)

# Load data

load('monet2photo/train_list.RData')

# Functions

my_predict_1 <- function (model, img) {
  
  dim(img) <- c(dim(img), 1)
  
  M2P_executor <- mx.simple.bind(symbol = model$symbol,
                                 M2P_img = dim(img),
                                 ctx = mx.cpu())
  
  mx.exec.update.arg.arrays(M2P_executor, model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(M2P_executor, model$aux.params, match.name = TRUE)
  
  mx.exec.update.arg.arrays(M2P_executor, arg.arrays = list(M2P_img = mx.nd.array(img)), match.name = TRUE)
  mx.exec.forward(M2P_executor, is.train = FALSE)
  M2P_pred_output <- M2P_executor$ref.outputs[[1]]
  
  return(as.array(M2P_pred_output)[,,,1])
  
}

my_predict_2 <- function (model, img) {
  
  dim(img) <- c(dim(img), 1)
  
  P2M_executor <- mx.simple.bind(symbol = model$symbol,
                                 P2M_img = dim(img),
                                 ctx = mx.cpu())
  
  mx.exec.update.arg.arrays(P2M_executor, model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(P2M_executor, model$aux.params, match.name = TRUE)
  
  mx.exec.update.arg.arrays(P2M_executor, arg.arrays = list(P2M_img = mx.nd.array(img)), match.name = TRUE)
  mx.exec.forward(P2M_executor, is.train = FALSE)
  P2M_pred_output <- P2M_executor$ref.outputs[[1]]
  
  return(as.array(P2M_pred_output)[,,,1])
  
}

Show_img <- function (img) {
  plot(NA, xlim = c(0.04, 0.96), ylim = c(0.96, 0.04), xaxt = "n", yaxt = "n", bty = "n")
  rasterImage(as.raster(img), 0, 1, 1, 0, interpolate = FALSE)
}

# Read images

real_monet <- readJPEG(train_list[[1]][[927]])
real_photo <- readJPEG(train_list[[2]][[6287]])

# Process images

fake_photo <- my_predict_1(model = M2P_gen_model, img = real_monet)
mirror_monet <- my_predict_2(model = P2M_gen_model, img = real_monet)
restored_monet <- my_predict_2(model = P2M_gen_model, img = fake_photo)

fake_monet <- my_predict_2(model = P2M_gen_model, img = real_photo)
mirror_photo <- my_predict_1(model = M2P_gen_model, img = real_photo)
restored_photo <- my_predict_1(model = M2P_gen_model, img = fake_monet)

# Plot

pdf(paste0('CycleGAN_', model_name, ' (epoch ', iter, ').pdf'), height = 6, width = 12)

par(mfrow = c(2, 4), mar = c(0.1, 0.1, 0.1, 0.1))

Show_img(img = real_monet)
text(0.5, 0.1, 'real monet', col = 'blue', cex = 2)
Show_img(img = fake_photo)
text(0.5, 0.1, 'fake photo', col = 'red', cex = 2)
Show_img(img = mirror_monet)
text(0.5, 0.1, 'mirror monet', col = 'blue', cex = 2)
Show_img(img = restored_monet)
text(0.5, 0.1, 'restored monet', col = 'blue', cex = 2)

Show_img(img = real_photo)
text(0.5, 0.9, 'real photo', col = 'red', cex = 2)
Show_img(img = fake_monet)
text(0.5, 0.9, 'fake monet', col = 'blue', cex = 2)
Show_img(img = mirror_photo)
text(0.5, 0.9, 'mirror photo', col = 'red', cex = 2)
Show_img(img = restored_photo)
text(0.5, 0.9, 'restored photo', col = 'red', cex = 2)

dev.off()



