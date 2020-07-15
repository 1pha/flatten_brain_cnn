# flatten_brain_cnn

## Basic Architecture
+ 07.08
    + Conv2D(6,  10, kernel=5, stride=2) > ReLu > MaxPool(kernel=2, stride=2)
    + Conv2D(10, 20, kernel=5, stride=2) > ReLu > MaxPool(kernel=2, stride=2)
    + Conv2D(20, 40, kernel=5, stride=2) > ReLu > MaxPool(kernel=2, stride=2)
    + Dropout
    + FC 480 > 100
    + FC 100 > 10(for multi, 2 for binary)
<br>

## LOG
### 07.08
+ Use torch.transform instead(not reshape) -> Works now
+ Since the data starts with **6 Features**, shallow architecture might work better
<br>

### 07.15
+ Initialized repo with basic architecture
+ Data format is fine and model does forward the data well, but severe overfitting
+ Suggested methods to be add to prevent overfitting
    + Batch Normaliation
    + K-folds
    + Dropout
    + Data Augmentation
    + Different learning rate : current lr=0.001, higher lr doesn't work
    + Different Non-Linear Functions
    
    
