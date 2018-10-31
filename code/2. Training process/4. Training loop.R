
# Source

source('code/2. Training process/3. Optimizer, Executor & Updater.R')

# Show image function

Show_img <- function (img) {
  plot(NA, xlim = c(0.04, 0.96), ylim = c(0.96, 0.04), xaxt = "n", yaxt = "n", bty = "n")
  rasterImage(as.raster(img), 0, 1, 1, 0, interpolate = FALSE)
}

# Start to train

if (start.epoch == 1) {
  
  logger <- list(Monet_adversarial_loss.gen = NULL,
                 Monet_adversarial_loss.fake = NULL,
                 Monet_adversarial_loss.real = NULL,
                 Photo_adversarial_loss.gen = NULL,
                 Photo_adversarial_loss.fake = NULL,
                 Photo_adversarial_loss.real = NULL,
                 Monet_cycle_consistency_loss = NULL,
                 Photo_cycle_consistency_loss = NULL,
                 Monet_identity_mapping_loss = NULL,
                 Photo_identity_mapping_loss = NULL)
  
} else {
  
  load(paste0('result/CycleGAN_', model_name, '_logger.RData'))
  
}

for (j in start.epoch:end.epoch) {
  
  current_batch <- 0
  t0 <- Sys.time()
  my_iter$reset()
  
  batch_logger <- list(Monet_adversarial_loss.gen = NULL,
                       Monet_adversarial_loss.fake = NULL,
                       Monet_adversarial_loss.real = NULL,
                       Photo_adversarial_loss.gen = NULL,
                       Photo_adversarial_loss.fake = NULL,
                       Photo_adversarial_loss.real = NULL,
                       Monet_cycle_consistency_loss = NULL,
                       Photo_cycle_consistency_loss = NULL,
                       Monet_identity_mapping_loss = NULL,
                       Photo_identity_mapping_loss = NULL)
  
  while (my_iter$iter.next()) {
    
    my_values <- my_iter$value()
    
    ##################################
    #                                #
    # Cycle consistency loss (Part1) #
    #                                #
    ##################################
    
    # Generator-1 forward (real Monet to fake Photo)
    
    mx.exec.update.arg.arrays(M2P_gen_executor, arg.arrays = list(M2P_img = my_values[['monet']]), match.name = TRUE)
    mx.exec.forward(M2P_gen_executor, is.train = TRUE)
    fake.Photo_output <- M2P_gen_executor$ref.outputs[[1]]
    fake.Photo_img <- as.array(fake.Photo_output)
     
    # Generator-2 forward (fake Photo to restored Monet)
    
    mx.exec.update.arg.arrays(P2M_gen_executor, arg.arrays = list(P2M_img = fake.Photo_output), match.name = TRUE)
    mx.exec.forward(P2M_gen_executor, is.train = TRUE)
    restored.Monet_output <- P2M_gen_executor$ref.outputs[[1]]
    restored.Monet_img <- as.array(restored.Monet_output)

    # Cycle consistency loss (Monet)
    
    mx.exec.update.arg.arrays(cycle_consistency_executor, arg.arrays = list(pred = restored.Monet_output, label = my_values[['monet']]), match.name = TRUE)
    mx.exec.forward(cycle_consistency_executor, is.train = TRUE)
    mx.exec.backward(cycle_consistency_executor)
    
    batch_logger$Monet_cycle_consistency_loss <- c(batch_logger$Monet_cycle_consistency_loss, as.array(cycle_consistency_executor$ref.outputs[[1]]))
    
    # Generator-2 backward
    
    P2M_grads <- cycle_consistency_executor$ref.grad.arrays[['pred']]
    mx.exec.backward(P2M_gen_executor, out_grads = P2M_grads)
    P2M_gen_update_args <- P2M_gen_updater(weight = P2M_gen_executor$ref.arg.arrays, grad = P2M_gen_executor$ref.grad.arrays)
    mx.exec.update.arg.arrays(P2M_gen_executor, P2M_gen_update_args, skip.null = TRUE)
    
    # Generator-1 backward
    
    M2P_grads <- P2M_gen_executor$ref.grad.arrays[['P2M_img']]
    mx.exec.backward(M2P_gen_executor, out_grads = M2P_grads)
    M2P_gen_update_args <- M2P_gen_updater(weight = M2P_gen_executor$ref.arg.arrays, grad = M2P_gen_executor$ref.grad.arrays)
    mx.exec.update.arg.arrays(M2P_gen_executor, M2P_gen_update_args, skip.null = TRUE)
    
    #################################
    #                               #
    # Identity mapping loss (Part1) #
    #                               #
    #################################
    
    # Generator-1 forward (real Photo to fake Photo)
    
    mx.exec.update.arg.arrays(M2P_gen_executor, arg.arrays = list(M2P_img = my_values[['photo']]), match.name = TRUE)
    mx.exec.forward(M2P_gen_executor, is.train = TRUE)
    mirror.Photo_output <- M2P_gen_executor$ref.outputs[[1]]
    mirror.Photo_img <- as.array(mirror.Photo_output)

    # Identity mapping loss (Photo)
    
    mx.exec.update.arg.arrays(identity_mapping_executor, arg.arrays = list(pred = mirror.Photo_output, label = my_values[['photo']]), match.name = TRUE)
    mx.exec.forward(identity_mapping_executor, is.train = TRUE)
    mx.exec.backward(identity_mapping_executor)
    
    batch_logger$Photo_identity_mapping_loss <- c(batch_logger$Photo_identity_mapping_loss, as.array(identity_mapping_executor$ref.outputs[[1]]))
    
    # Generator-1 backward
    
    M2P_grads <- identity_mapping_executor$ref.grad.arrays[['pred']]
    mx.exec.backward(M2P_gen_executor, out_grads = M2P_grads)
    M2P_gen_update_args <- M2P_gen_updater(weight = M2P_gen_executor$ref.arg.arrays, grad = M2P_gen_executor$ref.grad.arrays)
    mx.exec.update.arg.arrays(M2P_gen_executor, M2P_gen_update_args, skip.null = TRUE)
    
    ############################
    #                          #
    # Adversarial loss (Part1) #
    #                          #
    ############################
    
    # Generator-1 forward (real Monet to fake Photo)
    
    mx.exec.update.arg.arrays(M2P_gen_executor, arg.arrays = list(M2P_img = my_values[['monet']]), match.name = TRUE)
    mx.exec.forward(M2P_gen_executor, is.train = TRUE)
    fake.Photo_output <- M2P_gen_executor$ref.outputs[[1]]
    
    # Discriminator-2 fake (Photo)
    
    mx.exec.update.arg.arrays(Photo_dis_executor, arg.arrays = list(Photo_img = fake.Photo_output, label = mx.nd.array(rep(1, Batch_size))), match.name = TRUE)
    mx.exec.forward(Photo_dis_executor, is.train = TRUE)
    mx.exec.backward(Photo_dis_executor)
    Photo_dis_update_args <- Photo_dis_updater(weight = Photo_dis_executor$ref.arg.arrays, grad = Photo_dis_executor$ref.grad.arrays)
    mx.exec.update.arg.arrays(Photo_dis_executor, Photo_dis_update_args, skip.null = TRUE)
    
    batch_logger$Photo_adversarial_loss.fake <- c(batch_logger$Photo_adversarial_loss.fake, as.array(Photo_dis_executor$ref.outputs[[1]]))
    
    # Discriminator-2 real (Photo)
    
    mx.exec.update.arg.arrays(Photo_dis_executor, arg.arrays = list(Photo_img = my_values[['photo']], label = mx.nd.array(rep(0, Batch_size))), match.name = TRUE)
    mx.exec.forward(Photo_dis_executor, is.train = TRUE)
    mx.exec.backward(Photo_dis_executor)
    Photo_dis_update_args <- Photo_dis_updater(weight = Photo_dis_executor$ref.arg.arrays, grad = Photo_dis_executor$ref.grad.arrays)
    mx.exec.update.arg.arrays(Photo_dis_executor, Photo_dis_update_args, skip.null = TRUE)
    
    batch_logger$Photo_adversarial_loss.real <- c(batch_logger$Photo_adversarial_loss.real, as.array(Photo_dis_executor$ref.outputs[[1]]))
    
    # Adversarial loss (Photo)
    
    mx.exec.update.arg.arrays(Photo_dis_executor, arg.arrays = list(Photo_img = fake.Photo_output, label = mx.nd.array(rep(0, Batch_size))), match.name = TRUE)
    mx.exec.forward(Photo_dis_executor, is.train = TRUE)
    mx.exec.backward(Photo_dis_executor)
    
    batch_logger$Photo_adversarial_loss.gen <- c(batch_logger$Photo_adversarial_loss.gen, as.array(Photo_dis_executor$ref.outputs[[1]]))
    
    # Generator-1 backward
    
    M2P_grads <- Photo_dis_executor$ref.grad.arrays[['Photo_img']]
    mx.exec.backward(M2P_gen_executor, out_grads = M2P_grads)
    M2P_gen_update_args <- M2P_gen_updater(weight = M2P_gen_executor$ref.arg.arrays, grad = M2P_gen_executor$ref.grad.arrays)
    mx.exec.update.arg.arrays(M2P_gen_executor, M2P_gen_update_args, skip.null = TRUE)
    
    # Weight clipping (Discriminator-2)
    
    if (!is.null(w_limit)) {
      
      dis_weight_names <- grep('weight', names(Photo_dis_executor$ref.arg.arrays), value = TRUE)
      
      for (k in dis_weight_names) {
        
        current_dis_weight <- Photo_dis_executor$ref.arg.arrays[[k]] %>% as.array()
        current_dis_weight_list <- current_dis_weight %>% mx.nd.array() %>%
          mx.nd.broadcast.minimum(., mx.nd.array(w_limit)) %>%
          mx.nd.broadcast.maximum(., mx.nd.array(-w_limit)) %>%
          list()
        names(current_dis_weight_list) <- k
        mx.exec.update.arg.arrays(Photo_dis_executor, arg.arrays = current_dis_weight_list, match.name = TRUE)
        
      }
      
    }
    
    ##################################
    #                                #
    # Cycle consistency loss (Part2) #
    #                                #
    ##################################
    
    # Generator-2 forward (real Photo to fake Monet)
    
    mx.exec.update.arg.arrays(P2M_gen_executor, arg.arrays = list(P2M_img = my_values[['photo']]), match.name = TRUE)
    mx.exec.forward(P2M_gen_executor, is.train = TRUE)
    fake.Monet_output <- P2M_gen_executor$ref.outputs[[1]]
    fake.Monet_img <- as.array(fake.Monet_output)

    # Generator-1 forward (fake Monet to restored Photo)
    
    mx.exec.update.arg.arrays(M2P_gen_executor, arg.arrays = list(M2P_img = fake.Monet_output), match.name = TRUE)
    mx.exec.forward(M2P_gen_executor, is.train = TRUE)
    restored.Photo_output <- M2P_gen_executor$ref.outputs[[1]]
    restored.Photo_img <- as.array(restored.Photo_output)

    # Cycle consistency loss (Photo)
    
    mx.exec.update.arg.arrays(cycle_consistency_executor, arg.arrays = list(pred = restored.Photo_output, label = my_values[['photo']]), match.name = TRUE)
    mx.exec.forward(cycle_consistency_executor, is.train = TRUE)
    mx.exec.backward(cycle_consistency_executor)
    
    batch_logger$Photo_cycle_consistency_loss <- c(batch_logger$Photo_cycle_consistency_loss, as.array(cycle_consistency_executor$ref.outputs[[1]]))
    
    # Generator-1 backward
    
    M2P_grads <- cycle_consistency_executor$ref.grad.arrays[['pred']]
    mx.exec.backward(M2P_gen_executor, out_grads = M2P_grads)
    M2P_gen_update_args <- M2P_gen_updater(weight = M2P_gen_executor$ref.arg.arrays, grad = M2P_gen_executor$ref.grad.arrays)
    mx.exec.update.arg.arrays(M2P_gen_executor, M2P_gen_update_args, skip.null = TRUE)
    
    # Generator-2 backward
    
    P2M_grads <- M2P_gen_executor$ref.grad.arrays[['M2P_img']]
    mx.exec.backward(P2M_gen_executor, out_grads = P2M_grads)
    P2M_gen_update_args <- P2M_gen_updater(weight = P2M_gen_executor$ref.arg.arrays, grad = P2M_gen_executor$ref.grad.arrays)
    mx.exec.update.arg.arrays(P2M_gen_executor, P2M_gen_update_args, skip.null = TRUE)
    
    #################################
    #                               #
    # Identity mapping loss (Part2) #
    #                               #
    #################################
    
    # Generator-2 forward (real Monet to fake Monet)
    
    mx.exec.update.arg.arrays(P2M_gen_executor, arg.arrays = list(P2M_img = my_values[['monet']]), match.name = TRUE)
    mx.exec.forward(P2M_gen_executor, is.train = TRUE)
    mirror.Monet_output <- P2M_gen_executor$ref.outputs[[1]]
    mirror.Monet_img <- as.array(mirror.Monet_output)

    # Identity mapping loss (Monet)
    
    mx.exec.update.arg.arrays(identity_mapping_executor, arg.arrays = list(pred = mirror.Monet_output, label = my_values[['monet']]), match.name = TRUE)
    mx.exec.forward(identity_mapping_executor, is.train = TRUE)
    mx.exec.backward(identity_mapping_executor)
    
    batch_logger$Monet_identity_mapping_loss <- c(batch_logger$Monet_identity_mapping_loss, as.array(identity_mapping_executor$ref.outputs[[1]]))
    
    # Generator-2 backward
    
    P2M_grads <- identity_mapping_executor$ref.grad.arrays[['pred']]
    mx.exec.backward(P2M_gen_executor, out_grads = P2M_grads)
    P2M_gen_update_args <- P2M_gen_updater(weight = P2M_gen_executor$ref.arg.arrays, grad = P2M_gen_executor$ref.grad.arrays)
    mx.exec.update.arg.arrays(P2M_gen_executor, P2M_gen_update_args, skip.null = TRUE)
    
    ############################
    #                          #
    # Adversarial loss (Part2) #
    #                          #
    ############################
    
    # Generator-2 forward (real Photo to fake Monet)
    
    mx.exec.update.arg.arrays(P2M_gen_executor, arg.arrays = list(P2M_img = my_values[['photo']]), match.name = TRUE)
    mx.exec.forward(P2M_gen_executor, is.train = TRUE)
    fake.Monet_output <- P2M_gen_executor$ref.outputs[[1]]
    
    # Discriminator-1 fake (Monet)
    
    mx.exec.update.arg.arrays(Monet_dis_executor, arg.arrays = list(Monet_img = fake.Monet_output, label = mx.nd.array(rep(1, Batch_size))), match.name = TRUE)
    mx.exec.forward(Monet_dis_executor, is.train = TRUE)
    mx.exec.backward(Monet_dis_executor)
    Monet_dis_update_args <- Monet_dis_updater(weight = Monet_dis_executor$ref.arg.arrays, grad = Monet_dis_executor$ref.grad.arrays)
    mx.exec.update.arg.arrays(Monet_dis_executor, Monet_dis_update_args, skip.null = TRUE)
    
    batch_logger$Monet_adversarial_loss.fake <- c(batch_logger$Monet_adversarial_loss.fake, as.array(Monet_dis_executor$ref.outputs[[1]]))
    
    # Discriminator-1 real (Monet)
    
    mx.exec.update.arg.arrays(Monet_dis_executor, arg.arrays = list(Monet_img = my_values[['monet']], label = mx.nd.array(rep(0, Batch_size))), match.name = TRUE)
    mx.exec.forward(Monet_dis_executor, is.train = TRUE)
    mx.exec.backward(Monet_dis_executor)
    Monet_dis_update_args <- Monet_dis_updater(weight = Monet_dis_executor$ref.arg.arrays, grad = Monet_dis_executor$ref.grad.arrays)
    mx.exec.update.arg.arrays(Monet_dis_executor, Monet_dis_update_args, skip.null = TRUE)
    
    batch_logger$Monet_adversarial_loss.real <- c(batch_logger$Monet_adversarial_loss.real, as.array(Monet_dis_executor$ref.outputs[[1]]))
    
    # Adversarial loss (Monet)
    
    mx.exec.update.arg.arrays(Monet_dis_executor, arg.arrays = list(Monet_img = fake.Monet_output, label = mx.nd.array(rep(0, Batch_size))), match.name = TRUE)
    mx.exec.forward(Monet_dis_executor, is.train = TRUE)
    mx.exec.backward(Monet_dis_executor)
    
    batch_logger$Monet_adversarial_loss.gen <- c(batch_logger$Monet_adversarial_loss.gen, as.array(Monet_dis_executor$ref.outputs[[1]]))
    
    # Generator-2 backward
    
    P2M_grads <- Monet_dis_executor$ref.grad.arrays[['Monet_img']]
    mx.exec.backward(P2M_gen_executor, out_grads = P2M_grads)
    P2M_gen_update_args <- P2M_gen_updater(weight = P2M_gen_executor$ref.arg.arrays, grad = P2M_gen_executor$ref.grad.arrays)
    mx.exec.update.arg.arrays(P2M_gen_executor, P2M_gen_update_args, skip.null = TRUE)
    
    # Weight clipping (Discriminator-1)
    
    if (!is.null(w_limit)) {
      
      dis_weight_names <- grep('weight', names(Monet_dis_executor$ref.arg.arrays), value = TRUE)
      
      for (k in dis_weight_names) {
        
        current_dis_weight <- Monet_dis_executor$ref.arg.arrays[[k]] %>% as.array()
        current_dis_weight_list <- current_dis_weight %>% mx.nd.array() %>%
          mx.nd.broadcast.minimum(., mx.nd.array(w_limit)) %>%
          mx.nd.broadcast.maximum(., mx.nd.array(-w_limit)) %>%
          list()
        names(current_dis_weight_list) <- k
        mx.exec.update.arg.arrays(Monet_dis_executor, arg.arrays = current_dis_weight_list, match.name = TRUE)
        
      }
      
    }
    
    ############################
    #                          #
    # Show current performance #
    #                          #
    ############################
    
    if (current_batch %% n.print == 0) {
      
      # Show current images
      
      par(mfrow = c(num_show_img * 2, 4), mar = c(0.1, 0.1, 0.1, 0.1))
      
      for (i in 1:num_show_img) {
        Show_img(img = as.array(my_values[['monet']])[,,,i])
        if (i == 1) {text(0.5, 0.1, 'real monet', col = 'blue', cex = 2)}
        Show_img(img = as.array(fake.Photo_img)[,,,i])
        if (i == 1) {text(0.5, 0.1, 'fake photo', col = 'red', cex = 2)}
        Show_img(img = as.array(mirror.Monet_img)[,,,i])
        if (i == 1) {text(0.5, 0.1, 'mirror monet', col = 'blue', cex = 2)}
        Show_img(img = as.array(restored.Monet_img)[,,,i])
        if (i == 1) {text(0.5, 0.1, 'restored monet', col = 'blue', cex = 2)}
      }
      
      for (i in 1:num_show_img) {
        Show_img(img = as.array(my_values[['photo']])[,,,i])
        if (i == num_show_img) {text(0.5, 0.9, 'real photo', col = 'red', cex = 2)}
        Show_img(img = as.array(fake.Monet_img)[,,,i])
        if (i == num_show_img) {text(0.5, 0.9, 'fake monet', col = 'blue', cex = 2)}
        Show_img(img = as.array(mirror.Photo_img)[,,,i])
        if (i == num_show_img) {text(0.5, 0.9, 'mirror photo', col = 'red', cex = 2)}
        Show_img(img = as.array(restored.Photo_img)[,,,i])
        if (i == num_show_img) {text(0.5, 0.9, 'restored photo', col = 'red', cex = 2)}
      }
      
      # Show speed
      
      speed_per_batch <- as.numeric(Sys.time() - t0, units = 'secs') / (current_batch + 1)
      
      # Show loss
      
      current_loss <- batch_logger %>% sapply(., mean) %>% formatC(., 4, format = 'f')
      
      message('Epoch [', j, '] Batch [', current_batch, '] loss list (', formatC(speed_per_batch, 2, format = 'f'), ' sec/batch):')
      message(paste(paste(names(current_loss), current_loss, sep = ': '), collapse = '\n'))

    }
    
    current_batch <- current_batch + 1
    
  }
  
  # Record logger
  
  mean_loss <- batch_logger %>% lapply(., mean)
  
  for (i in 1:length(logger)) {
    logger[[i]][j] <- mean_loss[[i]]
  }
  
  # Save image
  
  pdf(paste0('result/CycleGAN_', model_name, '/epoch_', j, '.pdf'), height = num_show_img * 6, width = 12)
  
  par(mfrow = c(num_show_img * 2, 4), mar = c(0.1, 0.1, 0.1, 0.1))
  
  for (i in 1:num_show_img) {
    Show_img(img = as.array(my_values[['monet']])[,,,i])
    if (i == 1) {text(0.5, 0.1, 'real monet', col = 'blue', cex = 2)}
    Show_img(img = as.array(fake.Photo_img)[,,,i])
    if (i == 1) {text(0.5, 0.1, 'fake photo', col = 'red', cex = 2)}
    Show_img(img = as.array(mirror.Monet_img)[,,,i])
    if (i == 1) {text(0.5, 0.1, 'mirror monet', col = 'blue', cex = 2)}
    Show_img(img = as.array(restored.Monet_img)[,,,i])
    if (i == 1) {text(0.5, 0.1, 'restored monet', col = 'blue', cex = 2)}
  }
  
  for (i in 1:num_show_img) {
    Show_img(img = as.array(my_values[['photo']])[,,,i])
    if (i == num_show_img) {text(0.5, 0.9, 'real photo', col = 'red', cex = 2)}
    Show_img(img = as.array(fake.Monet_img)[,,,i])
    if (i == num_show_img) {text(0.5, 0.9, 'fake monet', col = 'blue', cex = 2)}
    Show_img(img = as.array(mirror.Photo_img)[,,,i])
    if (i == num_show_img) {text(0.5, 0.9, 'mirror photo', col = 'red', cex = 2)}
    Show_img(img = as.array(restored.Photo_img)[,,,i])
    if (i == num_show_img) {text(0.5, 0.9, 'restored photo', col = 'red', cex = 2)}
  }
  
  dev.off()
  
  # Save logger
  
  save(logger, file = paste0('result/CycleGAN_', model_name, '_logger.RData'))
  
  # Save models
  
  M2P_gen_model <- list()
  M2P_gen_model$symbol <- M2P_gen
  M2P_gen_model$arg.params <- M2P_gen_executor$ref.arg.arrays[-1]
  M2P_gen_model$aux.params <- M2P_gen_executor$ref.aux.arrays
  class(M2P_gen_model) <- "MXFeedForwardModel"
  
  mx.model.save(model = M2P_gen_model, prefix = paste0('model/CycleGAN_', model_name, '/M2P_gen_', model_name), iteration = j)
  
  P2M_gen_model <- list()
  P2M_gen_model$symbol <- P2M_gen
  P2M_gen_model$arg.params <- P2M_gen_executor$ref.arg.arrays[-1]
  P2M_gen_model$aux.params <- P2M_gen_executor$ref.aux.arrays
  class(P2M_gen_model) <- "MXFeedForwardModel"
  
  mx.model.save(model = P2M_gen_model, prefix = paste0('model/CycleGAN_', model_name, '/P2M_gen_', model_name), iteration = j)
  
  Monet_dis_model <- list()
  Monet_dis_model$symbol <- Monet_dis
  Monet_dis_model$arg.params <- Monet_dis_executor$ref.arg.arrays[-1]
  Monet_dis_model$aux.params <- Monet_dis_executor$ref.aux.arrays
  class(Monet_dis_model) <- "MXFeedForwardModel"
  
  mx.model.save(model = Monet_dis_model, prefix = paste0('model/CycleGAN_', model_name, '/Monet_dis_', model_name), iteration = j)
  
  Photo_dis_model <- list()
  Photo_dis_model$symbol <- Photo_dis
  Photo_dis_model$arg.params <- Photo_dis_executor$ref.arg.arrays[-1]
  Photo_dis_model$aux.params <- Photo_dis_executor$ref.aux.arrays
  class(Photo_dis_model) <- "MXFeedForwardModel"
  
  mx.model.save(model = Photo_dis_model, prefix = paste0('model/CycleGAN_', model_name, '/Photo_dis_', model_name), iteration = j)
  
}

