# TensorFlow example using the iris dataset

library(keras)
library(tfdatasets)
library(tidyverse)
library(rsample)
library(caret)

rm(list = ls())

# Read dataset
data(iris)
head(iris)

iris$Species.num = 0
iris$Species.num[iris$Species == 'setosa'] = 0
iris$Species.num[iris$Species == 'versicolor'] = 1
iris$Species.num[iris$Species == 'virginica'] = 2
iris$Species.num = as.integer(iris$Species.num)

# Split the data frame
split = initial_split(iris, prop = .75)
train = training(split)
test  = testing(split)

# Creat validation set
split = initial_split(train, prop = .75)
train = training(split)
valid = testing(split)

# Convert to tfdataset
df_to_dataset <- function(df, shuffle = TRUE) {
  ds <- df %>% 
    tensor_slices_dataset()
  
  if (shuffle)
    ds <- ds %>% dataset_shuffle(buffer_size = nrow(df))
  
  ds %>% 
    dataset_batch(batch_size = 32)
}


train <- df_to_dataset(train)
valid <- df_to_dataset(valid, shuffle = FALSE)
test  <- df_to_dataset(test, shuffle = FALSE)








train = train %>% 
  tensor_slices_dataset() %>%
  dataset_batch(batch_size = batch_size)

valid = valid %>% 
  tensor_slices_dataset() %>%
  dataset_batch(batch_size = batch_size)

test = test %>% 
  tensor_slices_dataset() %>%
  dataset_batch(batch_size = batch_size)

train %>% 
  reticulate::as_iterator() %>% 
  reticulate::iter_next() %>% 
  str()

# One hot coding
train <- train %>% 
  dataset_map(function(record) {
    record$Species.num <- tf$one_hot(record$Species.num, 3L)
    record
  })

# One hot coding
valid <- valid %>% 
  dataset_map(function(record) {
    record$Species.num <- tf$one_hot(record$Species.num, 3L)
    record
  })

# One hot coding
test <- test %>% 
  dataset_map(function(record) {
    record$Species.num <- tf$one_hot(record$Species.num, 3L)
    record
  })


# Create spec
spec = feature_spec(train, Species.num ~ . - Species) %>%
  step_numeric_column(all_numeric(), normalizer_fn = scaler_standard())

spec_prep = fit(spec)

# Model
model <- keras_model_sequential() %>% 
  layer_dense_features(dense_features(spec_prep)) %>%  
  layer_dense(units = 3, activation = "sigmoid")


model %>% compile(
  loss = loss_binary_crossentropy, 
  optimizer = "adam", 
  metrics = "binary_accuracy"
)

history <- model %>% 
  fit(
    dataset_use_spec(train, spec = spec_prep),
    epochs = 30, 
    validation_data = dataset_use_spec(valid, spec_prep),
    verbose = 2
  )

pred <- predict(model, test)
Metrics::auc(test$Species.num, pred)
