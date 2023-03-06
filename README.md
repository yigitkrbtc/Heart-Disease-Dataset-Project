# Heart-Disease-Dataset-Project

LIST OF CONTENTS
1. About Dataset
2. Converting and Normalizing
3. KNN Algorithm
4. MLP Algorithm
5. NB Algortihm
6. Comparing The Results

1-)ABOUT DATASET
Context
Our data set name is Heart Disease.This data set dates from 1988 and consists of four databases: 
Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted 
attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to 
the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.
Content
 Attribute Information:
1. age
2. sex
3. chest pain type (4 values) as cp
4. resting blood pressure as trestbps
5. serum cholestoral in mg/dl as chol
6. fasting blood sugar > 120 mg/dl as fbs 
7. resting electrocardiographic results (values 0,1,2) as restecg 
8. maximum heart rate achieved as thalach
9. exercise induced angina as exang
10. oldpeak = ST depression induced by exercise relative to rest
11. the slope of the peak exercise ST segment as slop
12. number of major vessels (0-3) colored by flourosopy as ca
13. thal: 0 = normal; 1 = fixed defect; 2 = reversable defect

DATA SET 

![image](https://user-images.githubusercontent.com/112809652/223134313-fd275c65-49c6-4c05-b850-5d2cc818729e.png)

Import Part

First, import the necessary libraries and load the heart disease dataset into a Pandas 
DataFrame
![image](https://user-images.githubusercontent.com/112809652/223134523-e1b060dc-ba69-489b-9103-cf64c549d418.png)

Here, description and information about our dataset
![image](https://user-images.githubusercontent.com/112809652/223134927-25bcb69d-2f4d-468d-9fd5-b782e7acfa41.png)

![image](https://user-images.githubusercontent.com/112809652/223134981-0966c312-1531-40ec-a4d0-c1d941911698.png)

2-)CONVERTING AND NORMALIZING

Next, convert the categorical values to numerical values by using dummies. After that split 
the data into trainig and testing sets (test_size=0.2, random_state=0)
 
 ![image](https://user-images.githubusercontent.com/112809652/223135256-91940e74-ae01-43c3-88de-66d635fab8a8.png)

Once the categorical values are converted to numerical values, use the StandardScaler to 
normalize all the attributes:
This will normalize all the attributes in the dataset and make them have a mean of zero and 
a standard deviation of one

![image](https://user-images.githubusercontent.com/112809652/223135440-630a1774-6d9a-4b79-9234-6b60fbc04b00.png)

3-)KNN ALGORITHM

---First, apply KNN with k=3 and calculate the accuracy, precision, recall and F1 score

![image](https://user-images.githubusercontent.com/112809652/223135715-25e3797c-541f-4339-9636-4c5db01973a7.png)


---Apply KNN with k=7 and calculate the accuracy, precision, recall and F1 score

![image](https://user-images.githubusercontent.com/112809652/223135886-d0372226-d629-48ca-9a5b-8dbcd21447d5.png)


---Apply KNN with k=11 and calculate the accuracy, precision, recall and F1 score

![image](https://user-images.githubusercontent.com/112809652/223136135-acebb195-8ff9-4b7d-a5a0-d0d959cfc71f.png)

This codes will apply the k-NN algorithm three times, each time with a different value of the 
k parameter. The fit method will fit the model to the training data, and the score method 
will evaluate the model on the testing data and return the accuracy.

4-)MLP ALGORITHM

---Apply MLP with 1 hidden layer and 32 neurons
![image](https://user-images.githubusercontent.com/112809652/223136641-f631545f-7087-4be4-9591-8f39107a0175.png)

---Apply MLP with 2 hidden layer and 32 neurons
![image](https://user-images.githubusercontent.com/112809652/223136788-2822d5f7-98a4-4c84-9864-096d1e72f3c6.png)

---Apply MLP with 3 hidden layer and 32 neurons
![image](https://user-images.githubusercontent.com/112809652/223136910-abc69daf-3149-4191-8338-983e2f251a19.png)

This codes will apply the MLP classifier three times, each time with a different number of 
hidden layers. The fit method will fit the model to the training data, and the score method 
will evaluate the model on the testing data and return the accuracy.

5-)NB(Naive Bayes) ALGORITHM
Apply the NB algorithm using the default parameters
![image](https://user-images.githubusercontent.com/112809652/223137177-ac70d11e-79fd-4d1d-b658-5c4c39aa21f7.png)

6-)COMPARING THE RESULTS
To compare the performance results of the KNN, MLP, and NB algorithms on the heart 
disease dataset, we create a table or plot to visualize the results. Here is an example of how 
compare the accuracy, precision, recall, and F1 scores of the algorithms:
![image](https://user-images.githubusercontent.com/112809652/223137324-410d236b-3d76-4ddb-9427-77430afbec16.png)


Based on the results , it seems that the MLP algorithm with 2 or 3 hidden layers performs 
the best, as it has the highest accuracy, precision, recall, and F1 score. KNN with k=3 also 
performs well, but KNN with k=7 and k=11 have lower performance. The Naive Bayes 
algorithm has the lowest performance among the three algorithms




