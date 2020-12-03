import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



def bsns():
    st.markdown("<hr>", unsafe_allow_html=True)
    st.title('What is Credit Card Fraud?')
    st.write('''It is when a person uses your stolen credit card information for unauthorized purchases. 
             Credit Card Fraud is one of the biggest problems faced by the government. This can take place
             in numerous ways. Some examples can be : Stolen Card, Fake Phone Calls, Hacked Bank Details, etc.''')
    
    
    st.markdown('''<h3><b>Some Challenges Faced in Fraud Detection :-</b></h3>
                <ul>
                <li>Enormous Amount of Data</l1>
                <li>Highly Unbalanced Data</li>
                <li>Data is mostly private</li>
                <li>Misclassified Data</li>
                <li>Scammers use adaptive techniques against the model</li>
                </ul>
                <h3><b>Some Methods to Face these Challenges:-</b></h3>
                <ul>
                <li>Model should be as fast as possible in terms of fraud classification</li>
                <li>The dimensionality of data can be reduced to protect privacy of user</li>
                <li>Trustworthy sources to be taken to train the model</li>
                </ul>''', unsafe_allow_html=True)
    return



def dataun():
    st.markdown("<hr>", unsafe_allow_html=True)
    st.title('Understanding the data')
    st.markdown('''I had picked up 2 datasets in total. One is without any dimensionality reduction and the other has
                PCA ( Principle Component Analysis ) applied to it.''')
                
    st.markdown('''I am going to look at the raw dataset and understand its numerous features along with other important 
                information it contains.''')
            
    st.markdown('''<h3><b>Some Information</b></h3>
                <ul>
                <li>Columns 1 - 22 are self-explanatory columns where transaction details are given</li>
                <li>The 'is_fraud' Column classifies whether the transaction is fraud or not, where 1 is fraudulant and 0 is not</li>
                <li><b>Valid Cases : 1,28,895</b></li>
                <li><b>Fraud Cases : 773</b></li>
                <li>The data is highly unbalanced</li>
                <li>The fraud cases only account for 0.596% of all transactions</li>
                </ul>''', unsafe_allow_html=True)
    


def prep():
    st.markdown("<hr>", unsafe_allow_html=True)
    st.title('Preparing & Understanding the New Data')
    
    st.markdown('''The original dataset provides with all features and transactional information which violates the privacy
                of all the users. To prevent this from happening, PCA Dimensionality can be applied.''')
    
    st.markdown('''<h3><b>Principle Component Analysis : </b></h3>It is a method that is used to calculate the projection of
                the original data. It can be visualised as a line of best fit in a 
                scatter plot''', unsafe_allow_html=True)
                
    st.markdown('''<h3><b>Steps to PCA</b></h3>
                <ol>
                <li>The first step is to standardise and scale the data using sklearn</li>
                <li>The normalised data should then have a mean of 0 and deviation of 1</li>
                <li>Next, the number of columns / dimensions are reduced and are known as 
                Principle Components</li>
                <li>Now, the 'Explained Variance Ratio' can be found out</li>
                <li>It provides how much information each component holds after reducing the dimensionality</li>
                <li>During this process, some information can also be lost</li>
                <li>We can also plot the numerous records after reducing the dimensionality in a scatterplot form</li>
                <li>PCA will essentially act as a line of best fit in the scatterplot</li>
                </ol>''', unsafe_allow_html=True)

                
def profile():
    st.title("Pandas Profiling Report")
    prof = ProfileReport(sample2, minimal=True)
    st_profile_report(prof)


def acc(predict):
    ACCURACY = accuracy_score(Y_test, predict)
    st.write("Accuracy :", ACCURACY, '======', ACCURACY*100, '%')


def ran():
    
    forest = RandomForestClassifier()
    forest.fit(X_train,Y_train)
    predict = forest.predict(X_test)
    
    acc(predict)
        
    conf_matrix = confusion_matrix(Y_test, predict)
    
    labels = ['Valid', 'Fraudulant']
    sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, fmt="d")

    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    x = plt.show()
    st.pyplot(x)
    


def gnb():
    
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    predict = gnb.predict(X_test)
    
    acc(predict)
    
    conf_matrix = confusion_matrix(Y_test, predict)
    
    labels = ['Valid', 'Fraudulant']
    sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, fmt="d")

    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    x = plt.show()
    st.pyplot(x)



def log():
    
    logistic = LogisticRegression()
    logistic.fit(X_train, Y_train)
    predict = logistic.predict(X_test)
    
    acc(predict)
    
    conf_matrix = confusion_matrix(Y_test, predict)
    
    labels = ['Valid', 'Fraudulant']
    sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, fmt="d")

    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    x = plt.show()
    st.pyplot(x)
    

def acc_ran():
    forest = RandomForestClassifier()
    forest.fit(X_train,Y_train)
    predict = forest.predict(X_test)
    ACCURACY1 = accuracy_score(Y_test, predict)
    return ACCURACY1
    

def acc_gnb():
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    predict = gnb.predict(X_test)
    ACCURACY2 = accuracy_score(Y_test, predict)
    return ACCURACY2


def acc_log():
    logistic = LogisticRegression()
    logistic.fit(X_train, Y_train)
    predict = logistic.predict(X_test)
    ACCURACY3 = accuracy_score(Y_test, predict)
    return ACCURACY3

#==========================================================================================================================
                
cc_data = pd.read_csv('fraudTrain.csv')
cc_data = cc_data.sample(frac = 0.2, random_state=42)

cc_fraud = cc_data[cc_data['is_fraud']==1]
cc_valid = cc_data[cc_data['is_fraud']==0]

sample = cc_data.sample(frac=0.1, random_state=48)


pca_data = pd.read_csv('creditcard.csv')
pca_data = pca_data.sample(frac = 0.1, random_state=42)

pca_fraud = pca_data[pca_data['Class']==1]
pca_valid = pca_data[pca_data['Class']==0]

sample2 = pca_data.sample(frac = 0.05, random_state=42) 

cc_rows = pca_data.drop(['Class'], axis=1)
cc_class = pca_data['Class']

row_Data = cc_rows.values
class_Data = cc_class.values

X_train, X_test, Y_train, Y_test = train_test_split(row_Data, class_Data, test_size=0.2, random_state=42)
   

st.title("Vidur's Credit Card Fraud Detection Program")
st.markdown('''<h5>Aim : To analyse and predict fraudulant transactions with the help
            of given datasets and python modules</h5>''', unsafe_allow_html=True)


#==========================================================================================================================

st.sidebar.title('Business Understanding')
if st.sidebar.button('Understand the Concept'):
    bsns()

st.sidebar.title('Data Understanding')
if st.sidebar.button('Understand the Dataset'):
    dataun()
    
if st.sidebar.checkbox('Show the dataset (NO PCA)'):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write(sample)

st.sidebar.title('Data Preperation')
if st.sidebar.button('Understand Data Preperation'):
    prep()

if st.sidebar.checkbox('Show the dataset (PCA)'):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write(sample2)
    
if st.sidebar.checkbox('Pandas Profiling Report'):
    st.markdown("<hr>", unsafe_allow_html=True)
    profile()
    
st.sidebar.markdown('<hr></hr>', unsafe_allow_html=True)
    

if st.sidebar.checkbox('Show Fraud & Valid Transactions'):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write('FRAUD CASES :', len(pca_fraud))
    st.write('VALID CASES :', len(pca_valid))
    
mod_op = ['Random Forest', 'Logistic Regression', 'Naive Bayes']
sel = st.sidebar.selectbox('Select Model', mod_op)

st.set_option('deprecation.showPyplotGlobalUse', False)

if st.sidebar.button('Run Model'):
    if sel=='Random Forest':
        st.markdown("<hr>", unsafe_allow_html=True)
        st.title("Random Forest")
        ran()
        
        st.markdown('''<br>
                    <h3>About the Classifier</h3>
                    <p>
                    Random Forest consists of a large number of decision trees that act as an ensemble. Each individual tree
                    in the forest predicts a class, and the class with the most votes becomes the model's prediction. The reason
                    it works so well is because a large number of uncorrelated models operate as a committee to decide on a 
                    prediction. Ultimately, the trees protect each other from individual errors.
                    </p>
                    <h3><b>Strengths</b></h3>
                    <ul>
                    <li>It can be used for both regression and classification tasks</li>
                    <li>It is pretty straightforward and produces good predictions</li>
                    <li>Random Forest deals with overfitting pretty well with sufficient number of trees</li>
                    </ul>
                    <br>
                    <h3><b>Weaknesses</b></h3>
                    <ul>
                    <li>Doesn't do a good job in continuous problems</li>
                    <li>Can make a program slow due to the large number of trees</li>
                    </ul>
                    ''', unsafe_allow_html=True)
        

    if sel=='Naive Bayes':
        st.markdown("<hr>", unsafe_allow_html=True)
        st.title("Naive Bayes")
        gnb()
        
        st.markdown('''<br>
                    <h3>About the Classifier</h3>
                    <p>
                    A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task and
                    works on the principle of conditional probability. The crux of the classifier is based on the Bayes theorem.
                    It assumes that the presence of one feature is completely unrelated to the presence of others.
                    </p>
                    <h3><b>Strengths</b></h3>
                    <ul>
                    <li>Simple and easy to implement</li>
                    <li>Not sensitive to irrelevant features</li>
                    <li>Doesn't require as much training data</li>
                    </ul>
                    <b>
                    <h3><b>Weaknesses</b></h3>
                    <ul>
                    <li>Relies on the assumption of equally important and independent features resulting in biased
                    probabilities</li>
                    <li>Doesn't work well with continuous data</li>
                    </ul>
                    ''', unsafe_allow_html=True)
  

      
    if sel=='Logistic Regression':
        st.markdown("<hr>", unsafe_allow_html=True)
        st.title("Logistic Regression")
        log()
        st.markdown('''<br>
                    <h3>About the Classifier</h3>
                    <p>
                    Logistic regression is a classification algorithm. It is used to predict a binary outcome based on a set
                    of independent variables. It is a type of regression analysis ( Regression analysis is a type of predictive 
                    modeling technique which is used to find the relationship between a dependent variable and independent 
                    variable(s) ).
                    </p>
                    <h3><b>Strengths</b></h3>
                    <ul>
                    <li>Much easier to implement than other methods</li>
                    <li>Works well for linearly seperable datasets</li>
                    <li>Provides useful insights (coeffecient size, direction of relationship, etc.)</li>
                    </ul>
                    <h3><b>Weaknesses</b></h3>
                    <ul>
                    <li>Fails to predict a continuous outcome</li>
                    <li>May not be accurate if the sample size is small</li>
                    </ul>
                    ''', unsafe_allow_html=True)
                    

st.sidebar.write('<hr>', unsafe_allow_html=True)
    
if st.sidebar.button('Compare Models'):
    
    st.write('<hr>', unsafe_allow_html=True)
    
    st.title('Bar Graph')
    fig = plt.figure(figsize = (10, 5))        
    mods = ['Random Forest', 'Naive Bayes', 'Logistic Regression']
    accs = [acc_ran(), acc_gnb(), acc_log()]
    acc2 = [round(acc_ran(), 5), round(acc_gnb(), 5), round(acc_log(), 5)]
    plt.bar(mods, accs)
    
    for i in range(len(mods)):
        plt.text(x=mods[i], y=accs[i], s = acc2[i], size=10, ha='center')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    st.pyplot(plt.show())
    
    
    st.title('Heatmap')
    sns.heatmap(sample2.corr())
    st.pyplot(plt.show())
    
    
    # Image.MAX_IMAGE_PIXELS = 933120000
    # st.title('Pair Plot')
    # samp3 = pca_data.sample(100)
    # samp3.drop(samp3[11:28], axis=1)
    # sns.pairplot(samp3, size=1)
    # st.pyplot(plt.show())