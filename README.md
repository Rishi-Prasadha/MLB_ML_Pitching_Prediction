# MLB_ML_Pitching_Prediction

 What if we told you the 2017 Astros had a way to get their pitching data legitimately? Batters are allowed to be coached with signals mid-game, but teams aren't allowed to use a camera to see the catchers signals to inform this mid-game coaching (*cough cough*). So in theory, machine learning models could be used to legitimately help the astros predict pitches. With technology companies everywhere implementing machine learning, it is only a matter of time before sports teams harness this power for their own teams' performance. Using the scikit-learn and keras libraries, we explore Decision Tree, Random Forest, Neural Network and SVM models to see which will be the most effective in predicting Justin Verlander's next pitch, in real time. 

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
from lime import lime_tabular
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import hvplot.pandas

# Needed for decision tree visualization
import pydotplus
from IPython.display import Image
```

For the SVM model import the following: 

```python 
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn import svm, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
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

With that being said, the Justin Verlander dataset used is found in the resources folder, including the September 29th game data: 

* verlander_update.csv
* field_test_data.csv

---

## Usage

The objective of these models is to see how machine learning can help teams prepare for opposing pitchers and how pitchers can help themselves become less predictable. The B2B business objective is that this technology can be sold to MLB teams. 

From a B2C perspective, micro, real-time betting could be a market worth exploring. With betting on outcome of games and general actions and occurrences throughout the game already mainstream, the ability to bet in real time on small occurrences in a game might be an unexplored market. 

In order to use the models in Jake's folder navagate to Part 2 of the notebook. This is where you can change the link to your own Verlander CSV file for any game(s). Usage instructions for editing the models is also including in the notebook is you wish to do so.

<img width="1292" alt="Screen Shot 2022-09-29 at 5 17 27 PM" src="https://user-images.githubusercontent.com/106558893/193152732-2d00959b-349c-4f15-b16b-6a8998e8474f.png">
---

## Results

There are 7 models in the Jake folder of this project. Two are random forest models, one is a combination of the two random forest models, and four are SVM models. The combination of random forest models achieved the highest accuracy during model validation at 53%. The combination model also had relatively balanced f1 scores for SL, CU, and FF pitch types. However, during the live testing of the Verlander game on 9/23/22 the Poly SVM model had the best results out of all of Jake's models. This could be do to variance with one game being a small sample size. More testing of the models is required to make any definitive conclusions. 

In Noah's folder, you'll find 5 of his models. Each next model gradually improved over the previous as Noah tweaked his code. Noah's final model ended up being the best with an accuracy of 59%. However, during the live testing of the Verlander game on 9/23/22, it only achieved a 54% accuracy. His model must have had an "off" day. Noah's model is thought to be the best because he borrowed feature engineering from the other group members' models and added some of his own, giving his model the most features engineered. Some of the engineered features his model had were "count", "bases", "pitch_count", and "score_diff". "Count" combined the strikes and balls columns. Noah listed out all the possible counts (so like 3-2 3-1, 3-0, 2-2, etc...) and labelled them with a rank of how good they were for the batter. So 1st place was 3-0 and last place was 0-2. Noah did similiar engineering to combine the "on_1b", "on_2b", and "on_3b" columns into the "bases" column. The "pitch count" column is a count of how many pitches the pitcher had thrown that day. So it's basically a measure of how soar the pitcher's arm was. Lastly, the "score_diff" column combined the 2 columns that gave the scores of the 2 teams that were playing by simply taking the difference between their scores.

For the neural network model, the 1st model with 100 epochs and the adam optimizer performed the best with the training and validation data as shown below: 

<img width="775" alt="Screen Shot 2022-09-29 at 20 42 00" src="https://user-images.githubusercontent.com/107497500/193174536-df6d0c03-65a8-4b27-bc4b-50d711a845f5.png">

Using the Lime library, the neural network's feature importance could be extracted for an instance in the validation data. Below is the feature importance for what most likely correlates with Justin Verlander throwing a curveball. As we can see a 3-0, 2-1 and 3-1 count has the highest correlation with a curveball being thrown:

<img width="565" alt="Screen Shot 2022-09-29 at 18 30 26" src="https://user-images.githubusercontent.com/107497500/193174503-180b587b-5bb9-46bf-9370-1dea508e0007.png">

After the September 29, 2022 game, the models were tested on game data. Noah's model performed the best at 54% accuracy. Surprisingly, of the neural network models, the non-adaptive SGD model performed the best at 48.5% accuracy, while the adam model performed at 42.6% accuracy. But after further analysis, the SGD model only predicted fastballs and has that accuracy value because Justin Verlander throws ~50% fastballs in an outing. 


<img width="893" alt="Screen Shot 2022-09-29 at 20 42 19" src="https://user-images.githubusercontent.com/107497500/193174525-4ae8a384-5d52-47d1-adee-a8fc697b9534.png">



---

## Presentation

[Here is a link to our presentation!](https://docs.google.com/presentation/d/1tEwB8D-ivI-Zd0uTVI24HkBLhJh_i9LuMLjrsyOQh1k/edit?usp=sharing)

---

## Contributors

[Brittanie Polasek](https://www.linkedin.com/in/brittanie-polasek/), [Rishi Prasadha](https://www.linkedin.com/in/rishi-prasadha-912212133/), Noah Saleh, Jake Wheeler

---

## License

MIT
