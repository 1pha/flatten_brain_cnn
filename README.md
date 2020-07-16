# flatten_brain_cnn

## Basic Architecture
+ 07.08(still being used on 07.16)<br>
    **Architecture**
    + Conv2D(6,  10, kernel=5, stride=2) > ReLu > MaxPool(kernel=2, stride=2)
    + Conv2D(10, 20, kernel=5, stride=2) > ReLu > MaxPool(kernel=2, stride=2)
    + Conv2D(20, 40, kernel=5, stride=2) > ReLu > MaxPool(kernel=2, stride=2)
    + Dropout
    + FC 480 > 100
    + FC 100 > 10(for multi, 2 for binary)<br>

    **Parameters**
    + epochs: 200(1000 is too much)
    + Cross Entropy Loss with weight
    + Optimizer : Adam, lr=0.0001
<br>

## LOG
### 07.08
+ Use torch.transform instead(not reshape) -> Works now
+ Since the data starts with **6 Features**, shallow architecture might work better
<br>

### 07.15
+ Initialized repo with basic architecture
+ Data format is fine and model does forward the data well, but severe overfitting
+ Suggested methods to be add to prevent overfitting <br>
    **Methods**
    | Methods | How | Train | Val | Date |
    |---|:---:|:---:|:---:|---:|
    | Dropout | p=0.5 | No | No | 07.15 |
    | Dropout | p=0.2 | Yes | No | 07.15 |
    | Batch Normalization | After every Convnet | Rapid | No | 07.15 |
    | K-Folds | 90(10 folds here) / 10(holdout),<br> with Dropout+BatchNorm | Yes | No | 07.15 |
    | Data Augmentation | planning |  |  |  |
    
    **HyperParams**
    + Different learning rate : current lr=0.001, higher lr doesn't work
    + Different Non-Linear Functions
    + Different Kernel Size
    
+ Change hyperparameters before implementing Data Augmentation
    
### DROPOUT
+ with default(0.5) dropout on every Conv Layer, training doesn't work
+ setting them down to 0.2, training set trains, but still overfitting
+ lower than 0.2 has no...means... I guess...

### BATCHNORM 2D
+ put Batchnorm every after
+ Batchnorm trains very very well(training acc goes very high, about 30 Epochs, acc goes 90%), but still overfitting

### K-Folds
+ 90 / 10 Holdout
+ Do 10 folds on 90
