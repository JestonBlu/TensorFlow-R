model <- keras_model_sequential() 
model %>% 
  layer_embedding(input_dim = max_features, output_dim - 256) %>% 
  layer_lstm(units = 128) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid') %>% 
  compile(
    loss = 'binary_crossentropy',
    optimizer = 'rmsprop',
    metrics = c('accuracy')
  )

model %>% fit(x_train, y_train, batch_size = 16, epochs = 10)
score <- model %>% evaluate(x_test, y_test, batch_size = 16)