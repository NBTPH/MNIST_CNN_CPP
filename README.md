## How to use
Training dataset file is not included.

Compile training program
```
g++ -o model_train model_train.cpp cnn_lib.cpp
```
Run to build, train and save model to a txt file
```
./model_train
```


Compile running program
```
g++ -o model_run model_run.cpp cnn_lib.cpp
```
Run to build model from txt file and evaluate model
```
./model_run
```