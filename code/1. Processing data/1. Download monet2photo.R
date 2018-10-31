
URL <- 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip'
zip_path <- 'monet2photo.zip'
file_path <- 'monet2photo'

download.file(url = URL, destfile = zip_path, method = 'wget', quiet = TRUE)

unzip(zipfile = zip_path, files = file_path)