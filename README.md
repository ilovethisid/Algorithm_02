# Algorithm_02

**1. Model**   
### Model 3 : 7 Layers with 4 Convolution layer  

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
</br>

**2. Training with Training loss**  
### Training for 5 epochs  
```python
model.fit(train_images, train_labels,  epochs = 5)
```   
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


**3. Test Accuracy**   
### Perform Test with Test data   
```python
test_loss, accuracy = model.evaluate(test_images, test_labels, verbose = 2)
print('\nTest loss : ', test_loss)
print('Test accuracy :', accuracy)
```
313/313 - 3s - loss: 0.0463 - accuracy: 0.9873

Test loss :  0.04629102349281311   
Test accuracy : 0.9873  

**4. Images and corresponding probability that predicted Right**   
![correct number](https://user-images.githubusercontent.com/29995281/121518946-079c7d00-ca2c-11eb-8e31-1314c78cfebe.PNG)  


**5. Images and corresponding probability that predicted Wrong**   
![wrong number](https://user-images.githubusercontent.com/29995281/121518954-09fed700-ca2c-11eb-9314-2ac6c23557e9.PNG)









