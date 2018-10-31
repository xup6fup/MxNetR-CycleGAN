
library(mxnet)
library(OpenImageR)
library(png)

photo <- readPNG('images/input_photo.png')
resize_photo <- resizeImage(image = photo,
                            width = round(dim(photo)[2]/4) * 4,
                            height = round(dim(photo)[1]/4) * 4,
                            method = "bilinear")

P2M_gen_model <- mx.model.load(prefix = 'well trained model/P2M_gen_v2', iteration = 0)

my_predict <- function (model, img) {
  
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
  par(mai = rep(0, 4))
  plot(NA, xlim = c(0.04, 0.96), ylim = c(0.96, 0.04), xaxt = "n", yaxt = "n", bty = "n")
  rasterImage(as.raster(img), 0, 1, 1, 0, interpolate = FALSE)
}

monet_img <- my_predict(model = P2M_gen_model, img = resize_photo)

Show_img(monet_img)
