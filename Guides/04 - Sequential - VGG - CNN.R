library(keras)

# generate dummy data
x_train <- array(runif(100 * 100 * 100 * 3), dim = c(100, 100, 100, 3))

y_train <- runif(100, min = 0, max = 9) %>% 
  round() %>%
  matrix(nrow = 100, ncol = 1) %>% 
  to_categorical(num_classes = 10)

x_test <- array(runif(20 * 100 * 100 * 3), dim = c(20, 100, 100, 3))

y_test <- runif(20, min = 0, max = 9) %>% 
  round() %>%
  matrix(nrow = 20, ncol = 1) %>% 
  to_categorical(num_classes = 10)

# create model
model <- keras_model_sequential()

# define and compile model
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', 
                input_shape = c(100,100,3)) %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_dense(units = 10, activation = 'softmax') %>% 
  compile(
    loss = 'categorical_crossentropy', 
    optimizer = optimizer_sgd(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = TRUE)
  )

# train
model %>% fit(x_train, y_train, batch_size = 32, epochs = 10)

# evaluate
score <- model %>% evaluate(x_test, y_test, batch_size = 32)