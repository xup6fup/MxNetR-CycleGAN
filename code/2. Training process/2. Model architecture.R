
# https://hardikbansal.github.io/CycleGANBlog/

# Source

source('code/2. Training process/1. Parameters & Iterator.R')

# Modules & symbols

Residual.CONV_Module <- function (indata, num_filters = 128, kernel_size = 3, relu_slope = 0, name = 'g1', stage = 1) {
  
  Conv.1 <- mx.symbol.Convolution(data = indata, kernel = c(kernel_size, kernel_size), stride = c(1, 1),
                                  pad = c((kernel_size - 1)/2, (kernel_size - 1)/2),
                                  no.bias = TRUE, num.filter = num_filters,
                                  name = paste0(name, '_', stage, '_Conv.1'))
  InstNorm.1 <- mx.symbol.InstanceNorm(data = Conv.1, name = paste0(name, '_', stage, '_InstNorm.1'))
  ReLU.1 <- mx.symbol.LeakyReLU(data = InstNorm.1, act.type = 'leaky', slope = relu_slope, name = paste0(name, '_', stage, '_ReLU.1'))
  
  Conv.2 <- mx.symbol.Convolution(data = ReLU.1, kernel = c(kernel_size, kernel_size), stride = c(1, 1),
                                  pad = c((kernel_size - 1)/2, (kernel_size - 1)/2),
                                  no.bias = TRUE, num.filter = num_filters,
                                  name = paste0(name, '_', stage, '_Conv.2'))
  InstNorm.2 <- mx.symbol.InstanceNorm(data = Conv.2, name = paste0(name, '_', stage, '_InstNorm.2'))
  ReLU.2 <- mx.symbol.LeakyReLU(data = InstNorm.2, act.type = 'leaky', slope = relu_slope, name = paste0(name, '_', stage, '_ReLU.2'))
  ResBlock <- mx.symbol.broadcast_plus(lhs = indata, rhs = ReLU.2, name = paste0(name, '_', stage, '_ResBlock'))
  
  return(ResBlock)
  
}

general.CONV_Module <- function (indata, num_filters = 128, kernel_size = 3, stride = 1, pad = 1, relu_slope = 0, drop_p = 0, name = 'g1', stage = 1, normalization = FALSE) {
  
  Drop <- mx.symbol.Dropout(data = indata, p = drop_p, name = paste0(name, '_', stage, '_Drop'))
  
  if (normalization) {
    
    Conv <- mx.symbol.Convolution(data = Drop, kernel = c(kernel_size, kernel_size), stride = c(stride, stride),
                                  pad = c(pad, pad),
                                  no.bias = TRUE, num.filter = num_filters,
                                  name = paste0(name, '_', stage, '_Conv'))
    InstNorm <- mx.symbol.InstanceNorm(data = Conv, name = paste0(name, '_', stage, '_InstNorm'))
    ReLU <- mx.symbol.LeakyReLU(data = InstNorm, act.type = 'leaky', slope = relu_slope, name = paste0(name, '_', stage, '_ReLU'))
    
    return(ReLU)
    
  } else {
    
    Conv <- mx.symbol.Convolution(data = Drop, kernel = c(kernel_size, kernel_size), stride = c(stride, stride),
                                  pad = c(pad, pad),
                                  no.bias = FALSE, num.filter = num_filters,
                                  name = paste0(name, '_', stage, '_Conv'))
    
    return(Conv)
    
  }
  
}

DECONV_Module <- function (indata, updata = NULL, num_filters = 128, relu_slope = 0, name = 'g1', stage = 1) {
  
  DeConv <- mx.symbol.Deconvolution(data = indata, kernel = c(2, 2), stride = c(2, 2),
                                    num_filter = num_filters,
                                    name = paste0(name, '_', stage, '_DeConv'))
  
  InstNorm <- mx.symbol.InstanceNorm(data = DeConv, name = paste0(name, '_', stage, '_InstNorm'))
  ReLU <- mx.symbol.LeakyReLU(data = InstNorm, act.type = 'leaky', slope = relu_slope, name = paste0(name, '_', stage, '_ReLU'))

  if (is.null(updata)) {
    return(ReLU)
  } else {
    DenBlock <- mx.symbol.concat(data = list(updata, ReLU), num.args = 2, dim = 1, name = paste0(name, '_', stage, '_DenBlock'))
    return(DenBlock)
  }
  
}

Generator_symbol <- function (name = 'g1') {
  
  g_img <- mx.symbol.Variable(paste0(name, '_img'))
  g_1 <- general.CONV_Module(indata = g_img, num_filters = 32, kernel_size = 7, stride = 1, pad = 3, relu_slope = 0, drop_p = 0, name = name, stage = 1, normalization = TRUE)
  g_2 <- general.CONV_Module(indata = g_1, num_filters = 64, kernel_size = 3, stride = 2, pad = 1, relu_slope = 0, drop_p = 0, name = name, stage = 2, normalization = TRUE)
  g_3 <- general.CONV_Module(indata = g_2, num_filters = 128, kernel_size = 3, stride = 2, pad = 1, relu_slope = 0, drop_p = 0, name = name, stage = 3, normalization = TRUE)
  g_4 <- Residual.CONV_Module(indata = g_3, num_filters = 128, kernel_size = 3, relu_slope = 0, name = name, stage = 4)
  g_5 <- Residual.CONV_Module(indata = g_4, num_filters = 128, kernel_size = 3, relu_slope = 0, name = name, stage = 5)
  g_6 <- Residual.CONV_Module(indata = g_5, num_filters = 128, kernel_size = 3, relu_slope = 0, name = name, stage = 6)
  g_7 <- Residual.CONV_Module(indata = g_6, num_filters = 128, kernel_size = 3, relu_slope = 0, name = name, stage = 7)
  g_8 <- Residual.CONV_Module(indata = g_7, num_filters = 128, kernel_size = 3, relu_slope = 0, name = name, stage = 8)
  g_9 <- Residual.CONV_Module(indata = g_8, num_filters = 128, kernel_size = 3, relu_slope = 0, name = name, stage = 9)
  g_10 <- Residual.CONV_Module(indata = g_9, num_filters = 128, kernel_size = 3, relu_slope = 0, name = name, stage = 10)
  g_11 <- Residual.CONV_Module(indata = g_10, num_filters = 128, kernel_size = 3, relu_slope = 0, name = name, stage = 11)
  g_12 <- Residual.CONV_Module(indata = g_11, num_filters = 128, kernel_size = 3, relu_slope = 0, name = name, stage = 12)
  g_13 <- DECONV_Module(indata = g_12, updata = g_2, num_filters = 64, relu_slope = 0, name = name, stage = 13)
  g_14 <- DECONV_Module(indata = g_13, updata = g_1, num_filters = 32, relu_slope = 0, name = name, stage = 14)
  g_15 <- general.CONV_Module(indata = g_14, num_filters = 3, kernel_size = 7, stride = 1, pad = 3, relu_slope = 0, drop_p = 0, name = name, stage = 15, normalization = FALSE)
  g_pred <- mx.symbol.Activation(data = g_15, act_type = "sigmoid", name = paste0(name, '_pred'))
  
  return(g_pred)
  
}

Discriminator_symbol <- function (name = 'd1', drop_p = 0) {
  
  d_img <- mx.symbol.Variable(paste0(name, '_img'))
  d_1 <- general.CONV_Module(indata = d_img, num_filters = 32, kernel_size = 4, stride = 2, pad = 0, relu_slope = 0.2, drop_p = drop_p, name = name, stage = 1, normalization = TRUE)
  d_2 <- general.CONV_Module(indata = d_1, num_filters = 64, kernel_size = 4, stride = 2, pad = 0, relu_slope = 0.2, drop_p = drop_p, name = name, stage = 2, normalization = TRUE)
  d_3 <- general.CONV_Module(indata = d_2, num_filters = 128, kernel_size = 4, stride = 2, pad = 0, relu_slope = 0.2, drop_p = drop_p, name = name, stage = 3, normalization = TRUE)
  d_4 <- general.CONV_Module(indata = d_3, num_filters = 256, kernel_size = 4, stride = 2, pad = 0, relu_slope = 0.2, drop_p = drop_p, name = name, stage = 4, normalization = TRUE)
  d_5 <- general.CONV_Module(indata = d_4, num_filters = 1, kernel_size = 1, stride = 1, pad = 0, relu_slope = 0, drop_p = drop_p, name = name, stage = 5, normalization = FALSE)
  d_pred <- mx.symbol.mean(data = d_5, axis = 1:3, keepdims = FALSE, name = paste0(name, '_pred'))
  
  return(d_pred)
  
}

adversarial_loss <- function (pred, label, lambda = 1) {
  
  loss_pos <-  mx.symbol.broadcast_mul(pred, label)
  loss_neg <-  mx.symbol.broadcast_mul(pred, 1 - label)
  loss_mean <- mx.symbol.mean(loss_neg - loss_pos)
  weighted_loss_mean <- loss_mean * lambda
  adversarial_loss <- mx.symbol.MakeLoss(weighted_loss_mean)
  
  return(adversarial_loss)
  
}

cycle_consistency_loss <- function (pred, label, lambda = 10) {
  
  diff_pred_label <- mx.symbol.broadcast_minus(lhs = pred, rhs = label)
  abs_diff_pred_label <- mx.symbol.abs(data = diff_pred_label)
  mean_loss <- mx.symbol.mean(data = abs_diff_pred_label, axis = 0:3, keepdims = FALSE)
  weighted_mean_loss <- mean_loss * lambda
  cycle_consistency_loss <- mx.symbol.MakeLoss(weighted_mean_loss)
  
  return(cycle_consistency_loss)
  
}

identity_mapping_loss <- function (pred, label, lambda = 5) {
  
  diff_pred_label <- mx.symbol.broadcast_minus(lhs = pred, rhs = label)
  abs_diff_pred_label <- mx.symbol.abs(data = diff_pred_label)
  mean_loss <- mx.symbol.mean(data = abs_diff_pred_label, axis = 0:3, keepdims = FALSE)
  weighted_mean_loss <- mean_loss * lambda
  cycle_consistency_loss <- mx.symbol.MakeLoss(weighted_mean_loss)
  
  return(cycle_consistency_loss)
  
}

# Generator-1 (Monet to Photo)

M2P_gen <- Generator_symbol(name = 'M2P')

# Generator-2 (Photo to Monet)

P2M_gen <- Generator_symbol(name = 'P2M')

# Discriminator-1 (Monet)

Monet_dis <- Discriminator_symbol(name = 'Monet', drop_p = dis_drop_p)

# Discriminator-2 (Photo)

Photo_dis <- Discriminator_symbol(name = 'Photo', drop_p = dis_drop_p)

# adversarial loss-1 (Monet)

label <- mx.symbol.Variable('label')
Monet_loss <- adversarial_loss(pred = Monet_dis, label = label, lambda = 1)

# adversarial loss-2 (Photo)

label <- mx.symbol.Variable('label')
Photo_loss <- adversarial_loss(pred = Photo_dis, label = label, lambda = 1)

# cycle consistency loss

pred <- mx.symbol.Variable('pred')
label <- mx.symbol.Variable('label')
CC_loss <- cycle_consistency_loss(pred = pred, label = label, lambda = lambda_cycle_consistency_loss)

# identity mapping loss

pred <- mx.symbol.Variable('pred')
label <- mx.symbol.Variable('label')
IM_loss <- identity_mapping_loss(pred = pred, label = label, lambda = lambda_identity_mapping_loss)
