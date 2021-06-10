# Algorithm_02

1. Model
## Model 3 : 7 Layers with 4 Convolution layer

```python
model_3 = keras.models.Sequential([
  keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28, 28,1)),  # layer 1
  keras.layers.MaxPool2D((2,2)),                                                  # layer 2
  keras.layers.Conv2D(64, (3,3), activation = 'relu'),                            # layer 3
  keras.layers.Conv2D(64, (3,3), activation = 'relu'),                            # layer 4
  keras.layers.MaxPool2D((2,2)),                                                  # layer 5
  keras.layers.Conv2D(128, (3,3), activation = 'relu'),                           # layer 6
  keras.layers.Flatten(),
  keras.layers.Dense(10, activation = 'softmax')])                                # layer 7
```

2. Training with Training loss
## Training for 5 epochs
model.fit(train_images, train_labels,  epochs = 5)
