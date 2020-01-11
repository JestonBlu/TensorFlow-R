library(keras)

# constants
data_dim <- 16
timesteps <- 8
num_classes <- 10
batch_size <- 32

# define and compile model
# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
model <- keras_model_sequential()
model %>% 
  layer_lstm(units = 32, return_sequences = TRUE, stateful = TRUE,
             batch_input_shape = c(batch_size, timesteps, data_dim)) %>% 
  layer_lstm(units = 32, return_sequences = TRUE, stateful = TRUE) %>% 
  layer_lstm(units = 32, stateful = TRUE) %>% 
  layer_dense(units = 10, activation = 'softmax') %>% 
  compile(
    loss = 'categorical_crossentropy',
    optimizer = 'rmsprop',
    metrics = c('accuracy')
  )

# generate dummy training data
x_train <- array(runif( (batch_size * 10) * timesteps * data_dim), 
                 dim = c(batch_size * 10, timesteps, data_dim))
y_train <- matrix(runif( (batch_size * 10) * num_classes), 
                  nrow = batch_size * 10, ncol = num_classes)

# generate dummy validation data
x_val <- array(runif( (batch_size * 3) * timesteps * data_dim), 
               dim = c(batch_size * 3, timesteps, data_dim))
y_val <- matrix(runif( (batch_size * 3) * num_classes), 
                nrow = batch_size * 3, ncol = num_classes)

# train
model %>% fit( 
  x_train, 
  y_train, 
  batch_size = batch_size, 
  epochs = 5, 
  shuffle = FALSE,
  validation_data = list(x_val, y_val)
)