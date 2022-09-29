# MLB_ML_Pitching_Prediction

 What if the Astros could've been more surreptitious in 2017? And a bit more innovative? With technology companies everywhere implementing machine learning, it is a matter of time before sports teams harness this power for their own teams' performance. Using the scikit-learn and keras libraries, we explore Decision Tree, Random Forest, Neural Network and SVM models to see which will be the most effective in predicting Justin Verlander's next pitch, in real time.

---

## Technologies

This application is written in Python. The main Python libraries used are machine learning libraries such as Scikit-Learn and TensorFlow. Refer to the following documentation:

* [Pandas](https://github.com/pandas-dev/pandas)
* [Numpy](https://github.com/numpy/numpy)
* [Scikit-Learn](https://scikit-learn.org/stable/)
* [TensorFlow](https://www.tensorflow.org/api_docs)

---

## Libraries

These models were extensively built using the sklearn and keras library. Refer to the following import lines. 

For the keras Sequential Neural Network import the following: 

```python
# Import Modules
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import Adadelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import np_utils
%matplotlib inline
```
For the Decision Tree and Random Forest models import the following: 

```python
# Import Modules
import pandas as pd
from pathlib import Path
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pprint import pprint

# Needed for decision tree visualization
import pydotplus
from IPython.display import Image
```

For the SVM model import the following: 

```python 
from sklearn.svm import SVC
from sklearn import svm, datasets
```

---

## Installation Guide

Before running the above libraries, you will need to install them first. 

To install Pandas, go to your terminal and run the following command:

`pip install pandas`

To install Numpy, go to your terminal and run the following command:

`pip install numpy`

To install TensorFlow, go to your terminal and run the following command:

`pip install tensorflow`

To install Scikit-Learn go to your terminal and run the following command: 

`pip install -U scikit-learn`

---

## Database 

All the data used was sourced and downloaded from [Baseball Savant](https://baseballsavant.mlb.com). Thank you to Baseball Savant for providing reliable, accurate player, team and game statistics after each game. 

With that being said, the Justin Verlander dataset used is found in the resources folder: 

* verlander_update.csv

---

## Usage

The objective of these models is to see how machine learning can help teams prepare for opposing pitchers and how pitchers can help themselves become less predictable. The B2B business objective is that this technology can be sold to MLB teams. 

From a B2C perspective, micro, real-time betting could be a market worth exploring. With betting on outcome of games and general actions and occurrences throughout the game already mainstream, the ability to bet in real time on small occurrences in a game might be an unexplored market. 

---

## Results

There are 7 models in the Jake folder of this project. Two are random forest models, one is a combination of the two random forest models, and four are SVM models. The combination of random forest models achieved the highest accuracy during model validation at 53%. The combination model also had relatively balanced f1 scores for SL, CU, and FF pitch types. However, during the live testing of the Verlander game on 9/23/22 the Poly SVM model had the best results. This could be do to variance with one game being a small sample size. More testing of the models is required to make any definitive conclusions. 

(Noah's report and summary)

---

## Presentation


---

## Contributors

[Brittanie Polasek](https://www.linkedin.com/in/brittanie-polasek/), [Rishi Prasadha](https://www.linkedin.com/in/rishi-prasadha-912212133/), Noah Saleh, Jake Wheeler

---

## License

MIT
