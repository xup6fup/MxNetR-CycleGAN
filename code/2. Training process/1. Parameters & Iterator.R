
# Libraries

library(mxnet)
library(imager)
library(jpeg)
library(magrittr)

# Check directories 

if (!dir.exists('model')) {
  dir.create('model')
}

if (!dir.exists('result')) {
  dir.create('result')
}

if (!dir.exists(paste0('model/CycleGAN_', model_name))) {
  dir.create(paste0('model/CycleGAN_', model_name))
}

if (!dir.exists(paste0('result/CycleGAN_', model_name))) {
  dir.create(paste0('result/CycleGAN_', model_name))
}

# Load data

load('monet2photo/train_list.RData')

# Iterator function

my_iterator_core <- function (batch_size) {
  
  batch <-  0
  batch_per_epoch <- floor(length(train_list[[2]])/batch_size)
  ids.1 <- rep(1:length(train_list[[1]]), ceiling(length(train_list[[2]])/length(train_list[[1]])))
  ids.1 <- ids.1[1:length(train_list[[2]])]
  ids.2 <- 1:length(train_list[[2]])
  
  reset <- function() {batch <<- 0}
  
  iter.next <- function() {
    
    batch <<- batch + 1
    if (batch > batch_per_epoch) {return(FALSE)} else {return(TRUE)}
    
  }
  
  value <- function() {
    
    idx <- 1:batch_size + (batch - 1) * batch_size
    
    idx.1 <- ids.1[idx]
    idx.2 <- ids.2[idx]
    
    img_array.1 <- array(0, dim = c(256, 256, 3, batch_size))
    img_array.2 <- array(0, dim = c(256, 256, 3, batch_size))
    
    for (i in 1:batch_size) {
      
      img_array.1[,,,i] <- readJPEG(train_list[[1]][[idx.1[i]]])
      img_array.2[,,,i] <- readJPEG(train_list[[2]][[idx.2[i]]])
      
    }
    
    img_array.1 <- mx.nd.array(img_array.1)
    img_array.2 <- mx.nd.array(img_array.2)
    
    return(list(monet = img_array.1, photo = img_array.2))
    
  }
  
  return(list(reset = reset, iter.next = iter.next, value = value, batch_size = batch_size, batch = batch))
  
}

my_iterator_func <- setRefClass("Custom_Iter",
                                fields = c("iter", "batch_size"),
                                contains = "Rcpp_MXArrayDataIter",
                                methods = list(
                                  initialize = function(iter, batch_size = 16){
                                    .self$iter <- my_iterator_core(batch_size = batch_size)
                                    .self
                                  },
                                  value = function(){
                                    .self$iter$value()
                                  },
                                  iter.next = function(){
                                    .self$iter$iter.next()
                                  },
                                  reset = function(){
                                    .self$iter$reset()
                                  },
                                  finalize=function(){
                                  }
                                )
)

# Build an iterator

my_iter <- my_iterator_func(iter = NULL, batch_size = Batch_size)

#my_iter$reset()
#my_iter$iter.next()
#test_data <- my_iter$value()
#imageShow(as.array(test_data[[1]])[,,,2])
