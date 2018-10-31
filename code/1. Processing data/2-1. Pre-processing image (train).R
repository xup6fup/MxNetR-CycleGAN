
library(jpeg)
library(OpenImageR)

file_path <- 'monet2photo'

train_list <- list()

train_monet_path <- paste0(file_path, '/trainA')
files <- list.files(train_monet_path, full.names = TRUE)

train_list[[1]] <- list()

pb <- txtProgressBar(max = length(files), style = 3)

for (i in 1:length(files)) {
  img <- readJPEG(files[i])
  if (all.equal(dim(img), c(256, 256, 3)) == TRUE) {
    train_list[[1]][[i]] <- readBin(con = files[i], what = 'raw', n = file.size(files[i]))
  } else {
    resized_img <- resizeImage(img, width = 256, height = 256, method = 'bilinear')
    train_list[[1]][[i]] <- writeJPEG(resized_img)
  }
  setTxtProgressBar(pb, i)
}

close(pb)

train_photo_path <- paste0(file_path, '/trainB')
files <- list.files(train_photo_path, full.names = TRUE)

train_list[[2]] <- list()

pb <- txtProgressBar(max = length(files), style = 3)

for (i in 1:length(files)) {
  img <- readJPEG(files[i])
  if (all.equal(dim(img), c(256, 256, 3)) == TRUE) {
    train_list[[2]][[i]] <- readBin(con = files[i], what = 'raw', n = file.size(files[i]))
  } else {
    resized_img <- resizeImage(img, width = 256, height = 256, method = 'bilinear')
    train_list[[2]][[i]] <- writeJPEG(resized_img)
  }
  setTxtProgressBar(pb, i)
}

close(pb)

names(train_list) <- c('monet', 'photo')

save(train_list, file = paste0(file_path, '/train_list.RData'))
