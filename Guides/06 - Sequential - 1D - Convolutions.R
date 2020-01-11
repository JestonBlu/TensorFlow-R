model <- keras_model_sequential()
model %>% 
  layer_conv_1d(filters = 64, kernel_size = 3, activation = 'relu',
                input_shape = c(seq_length, 100)) %>% 
  layer_conv_1d(filters = 64, kernel_size = 3, activation = 'relu') %>% 
  layer_max_pooling_1d(pool_size = 3) %>% 
  layer_conv_1d(filters = 128, kernel_size = 3, activation = 'relu') %>% 
  layer_conv_1d(filters = 128, kernel_size = 3, activation = 'relu') %>% 
  layer_global_average_pooling_1d() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid') %>% 
  compile(
    loss = 'binary_crossentropy',
    optimizer = 'rmsprop',
    metrics = c('accuracy')
  )

model %>% fit(x_train, y_train, batch_size = 16, epochs = 10)
score <- model %>% evaluate(x_test, y_test, batch_size = 16)