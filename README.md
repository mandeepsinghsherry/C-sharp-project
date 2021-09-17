# C-sharp-project (Image Classification With Spatial Pooler)
# Introduction : Testing Approach

Scope of the Testing: SP Model is been trained using four different types of apple images as explained in following steps (1 to 4). Then 4 different images like Banana, Ananas, Apple and Chilli are given as an input to SP model for classification as explained in step 5. Expected result is for Apple, model should give more than 90 % as Prediction Percentage related to trained data i.e. Classifying it as an Apple and on the other hand , for Banana, Chilli and Ananas model should give less than 90 % Prediction Percentage related to trained data i.e. classifying it as not an Apple. The result of this test is depicted 


Step 1: Project Structure
In this project we have different libraries, classes with different specifications.
In the UnitTestProject we have main unit test method in ImageClassificationSpatialPoolerTest.cs  which gives the information about Implementation of this project. 
Helping classes like Binarizer.cs and HelperImageClassificationSpatialPooler.cs have methods defined which are used in UnitTestProject class.


Step 2:  Defining input and output parameters of Images.

Step 3:  Setting input space dimensions and Column space dimensions. 

Step 4: Setting the parameters of Potential connection.

Step 5:  TrainingImagePath is the image path contains images file name.

Step 6:  Training Data

Step 7:  GetHammingDistance 

Step 8: Deciding Model Trained or not

Step 9: Input Images for Classification Prediction

Step 10: Converting Prediction Images into Binary

Step 11: Computing Prediction Percentage

Step 12: Output of Classification Prediction
