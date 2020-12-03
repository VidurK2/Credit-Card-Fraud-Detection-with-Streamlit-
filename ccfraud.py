import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# Setting a seed or random state will generate the same set of random numbers each time. For publication
# purposes, practitioners use random seed so that they achieve exactly the same results every time they run 
# the algorithm.


# Here frac tells us that we only need 20% of the data as our sample
cc_data = pd.read_csv('creditcard.csv')
cc_data = cc_data.sample(frac = 0.2, random_state=42)
print('==================================')
print('Data shape :', cc_data.shape)

cc_fraud = cc_data[cc_data['Class']==1]
cc_valid = cc_data[cc_data['Class']==0]
print('FRAUD CASES :', len(cc_fraud))
print('VALID CASES :', len(cc_valid))
print('==================================')

cc_rows = cc_data.drop(['Class'], axis=1)
cc_class = cc_data['Class']

row_Data = cc_rows.values
class_Data = cc_class.values


# Here class_data is the target variable, which is the class values
# class_data defines if the transaction is fraud or not
# The test size is the split percentage, in this case it is 80/20 
# Random State is used to split the data so that our module is randomly trained
# Everytime we use 42, we get the same output as we did in the first split
X_train, X_test, Y_train, Y_test = train_test_split(row_Data, class_Data, test_size=0.2, random_state=42)


# For this project, I'll use the random forest method ( LINK AT THE TOP )
# It is one of the best methods for prediction modules
forest = RandomForestClassifier()
forest.fit(X_train,Y_train)
predict = forest.predict(X_test)

#============================================================================================================

# Incorrect are the cases where the prediction does not match with the actual dataset
fraud_cases = len(cc_fraud)
incorrect = (predict!=Y_test)
inc_cases = incorrect.sum()

# Accuracy is the fraction of prediction that the model gets right
# Not a very effective metric for imbalanced datasets
ACCURACY = accuracy_score(Y_test, predict)

# Precision score tells you that out of all the ALLEGED fraud transactions, which ones are actually fraudulant
# This is considered a really good metric
PRECISION = precision_score(Y_test, predict)


# Recall tells you that out of all the fraudulant transcations, how many can our model identify
RECALL = recall_score(Y_test, predict)


# F1 combines recall and precision as an average metric of the two
# It combines the features of both metrics 
F1 = f1_score(Y_test, predict)


# Matthews corrcoef takes into account all the 4 values of a confusion matrix unlike any other metric
# The 4 values being TP, TN, FP, FN
# A high value which would be close to 1 means that both classes are predicted well
MATTHEWS = matthews_corrcoef(Y_test, predict)


#============================================================================================================

# Confusion matrix is a table with 4 different combinations
# 1.) True Positive : Prediction is positive and it's true
# 2.) True Negative : Prediction is negative and it's true
# 3.) False Positive : Prediction is positive and it's false
# 4.) False Negative : Prediction is negative and it's false 

labels = ['Valid', 'Fraudulant']
conf_matrix = confusion_matrix(Y_test, predict)

# annot or Annotation shows values over the cells
# fmt reduces the number displayed when it is too big, it is there to increase readability
sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, fmt="d")

plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()


rep = classification_report(Y_test, predict)

while True:
    
    print('='*60)
    print("Welcome to my Credit Card Fraud Detection Program!")
    print('='*60)
    print()
    print('Please selfect a metric throigh which you want to analyse the prediction')
    print('1. ACCURACY')
    print('2. PRECISION')
    print('3. RECALL')
    print('4. F1 SCORE')
    print('5. MATTHEWS CORRELATION COEFFECIENT')
    ch=int(input("Enter an option : "))
    
    if ch==1:
        print("ACCURACY :", ACCURACY)
    
    elif ch==2:
        print("PRECISION :", PRECISION)
    
    elif ch==3:
        print("RECALL :", RECALL)
        
    elif ch==4:
        print("F1 SCORE :", F1)
    
    elif ch==5:
        print('Matthews Correlation Coeffecient :', MATTHEWS)
        
    ch2 = int(input("Do you want to continue and print the final report? : (Y/N) == (0/1) : "))
    
    if ch2==0:
        print(rep)
    
    ch3 = int(input("Do you want to continue? (Y/N)==(0/1) : "))
    
    if ch3==1:
        break

print("Thank you for using the prgram!")