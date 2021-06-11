# Algorithm_02

**1. Model**   
### Model 3 : 7 Layers with 4 Convolution layer  

```python
model_1 = keras.models.Sequential([
  keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28, 28,1)),  # layer 1 
  keras.layers.MaxPool2D((2,2)),                                                  # layer 2 
  keras.layers.Flatten(),
  keras.layers.Dense(10, activation = 'softmax')])                                # layer 3
  
model_2 = keras.models.Sequential([
  keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(28,28,1)),     # layer 1 
  keras.layers.MaxPool2D((2,2)),                                                  # layer 2
  keras.layers.Conv2D(64, (3,3), activation = 'relu'),                            # layer 3 
  keras.layers.MaxPool2D((2,2)),                                                  # layer 4
  keras.layers.Flatten(),
  keras.layers.Dense(10, activation = 'softmax')])                                # layer 5
  
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
<br/>

**2. Training with Training loss**  
### Training for 5 epochs  
```python
model.fit(train_images, train_labels,  epochs = 5)
```   
Model 1   
Epoch 1/5   
1875/1875 [==============================] - 17s 9ms/step - loss: 0.6630 - accuracy: 0.9399  
Epoch 2/5  
1875/1875 [==============================] - 18s 9ms/step - loss: 0.0834 - accuracy: 0.9751  
Epoch 3/5  
1875/1875 [==============================] - 17s 9ms/step - loss: 0.0703 - accuracy: 0.9792  
Epoch 4/5  
1875/1875 [==============================] - 17s 9ms/step - loss: 0.0620 - accuracy: 0.9813  
Epoch 5/5  
1875/1875 [==============================] - 17s 9ms/step - loss: 0.0546 - accuracy: 0.9838  
<tensorflow.python.keras.callbacks.History at 0x24f9a2d61c0>

Model 2  
Epoch 1/5  
1875/1875 [==============================] - 31s 16ms/step - loss: 0.4360 - accuracy: 0.9405  
Epoch 2/5  
1875/1875 [==============================] - 29s 16ms/step - loss: 0.0705 - accuracy: 0.9787  
Epoch 3/5  
1875/1875 [==============================] - 29s 15ms/step - loss: 0.0532 - accuracy: 0.98370s - los  
Epoch 4/5  
1875/1875 [==============================] - 28s 15ms/step - loss: 0.0482 - accuracy: 0.9849  
Epoch 5/5  
1875/1875 [==============================] - 28s 15ms/step - loss: 0.0452 - accuracy: 0.9867   
<tensorflow.python.keras.callbacks.History at 0x1a0f0908fa0>

Model 3   
Epoch 1/5   
1875/1875 [==============================] - 49s 26ms/step - loss: 0.2235 - accuracy: 0.9535   
Epoch 2/5   
1875/1875 [==============================] - 47s 25ms/step - loss: 0.0556 - accuracy: 0.9835   
Epoch 3/5   
1875/1875 [==============================] - 47s 25ms/step - loss: 0.0456 - accuracy: 0.9862   
Epoch 4/5   
1875/1875 [==============================] - 47s 25ms/step - loss: 0.0374 - accuracy: 0.9885   
Epoch 5/5   
1875/1875 [==============================] - 47s 25ms/step - loss: 0.0320 - accuracy: 0.9903   
<tensorflow.python.keras.callbacks.History at 0x2480d6b3970>

<br/>  

**3. Test Accuracy**   
### Perform Test with Test data   
```python
test_loss, accuracy = model.evaluate(test_images, test_labels, verbose = 2)
print('\nTest loss : ', test_loss)
print('Test accuracy :', accuracy)
```
Model 1  
313/313 - 1s - loss: 0.1179 - accuracy: 0.9730

Test loss :  0.11791642010211945  
Test accuracy : 0.9730  

Model 2  
313/313 - 2s - loss: 0.0664 - accuracy: 0.9804  

Test loss :  0.06635146588087082  
Test accuracy : 0.9804

Model 3   
313/313 - 3s - loss: 0.0463 - accuracy: 0.9873

Test loss :  0.04629102349281311   
Test accuracy : 0.9873  

<br/>  

**4. Images and corresponding probability that predicted Right**   
Model 1
![mod1 correct](https://user-images.githubusercontent.com/29995281/121654464-b5fdfc00-cad8-11eb-8c19-52c1bdadaf64.PNG)

Model 2
![mod2 correct](https://user-images.githubusercontent.com/29995281/121654470-b72f2900-cad8-11eb-97ca-2a6678804474.PNG)

Model 3  
![correct number](https://user-images.githubusercontent.com/29995281/121518946-079c7d00-ca2c-11eb-8e31-1314c78cfebe.PNG)  

<br/>  

**5. Images and corresponding probability that predicted Wrong**   

Model 1  
![model1 wrong](https://user-images.githubusercontent.com/29995281/121654473-b8605600-cad8-11eb-8a1b-a0da6cb7353d.PNG)

Model 2  
![mod2 wrong](https://user-images.githubusercontent.com/29995281/121654471-b7c7bf80-cad8-11eb-9663-5873938b31fc.PNG)

Model 3  
![wrong number](https://user-images.githubusercontent.com/29995281/121518954-09fed700-ca2c-11eb-9314-2ac6c23557e9.PNG)









