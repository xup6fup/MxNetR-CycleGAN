
library(jpeg)
library(OpenImageR)

file_path <- 'monet2photo'

test_list <- list()

test_monet_path <- paste0(file_path, '/testA')
files <- list.files(test_monet_path, full.names = TRUE)

test_list[[1]] <- list()

pb <- txtProgressBar(max = length(files), style = 3)

for (i in 1:length(files)) {
  img <- readJPEG(files[i])
  if (all.equal(dim(img), c(256, 256, 3)) == TRUE) {
    test_list[[1]][[i]] <- readBin(con = files[i], what = 'raw', n = file.size(files[i]))
  } else {
    resized_img <- resizeImage(img, width = 256, height = 256, method = 'bilinear')
    test_list[[1]][[i]] <- writeJPEG(resized_img)
  }
  setTxtProgressBar(pb, i)
}

close(pb)

test_photo_path <- paste0(file_path, '/testB')
files <- list.files(test_photo_path, full.names = TRUE)

test_list[[2]] <- list()

pb <- txtProgressBar(max = length(files), style = 3)

for (i in 1:length(files)) {
  img <- readJPEG(files[i])
  if (all.equal(dim(img), c(256, 256, 3)) == TRUE) {
    test_list[[2]][[i]] <- readBin(con = files[i], what = 'raw', n = file.size(files[i]))
  } else {
    resized_img <- resizeImage(img, width = 256, height = 256, method = 'bilinear')
    test_list[[2]][[i]] <- writeJPEG(resized_img)
  }
  setTxtProgressBar(pb, i)
}

close(pb)

names(test_list) <- c('monet', 'photo')

save(test_list, file = paste0(file_path, '/test_list.RData'))
