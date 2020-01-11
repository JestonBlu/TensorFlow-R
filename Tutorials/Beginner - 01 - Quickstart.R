library(keras)

# Import data
mnist <- dataset_mnist()

# Convert to values between 0 and 1
mnist$train$x <- mnist$train$x/255
mnist$test$x <- mnist$test$x/255

# Defining a model
model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(28, 28)) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dropout(0.2) %>% 
  layer_dense(10, activation = "softmax")

summary(model)

# Compiling a model
model %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )

# Fit a model
model %>% 
  fit(
    x = mnist$train$x, y = mnist$train$y,
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )

# Make predictions
predictions <- predict(model, mnist$test$x)
head(predictions, 2)

# Access model performance
model %>% 
  evaluate(mnist$test$x, mnist$test$y, verbose = 0)