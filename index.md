# Practical Machine Learning Course Project
Bhaskar Kamble  
May 9, 2016  



## Introduction

This report is part of the project submission for the course Practical Machine Learning by Jeff Leek, PhD, Professor at Johns Hopkins University, Bloomberg School of Public Health, taught as part of the Johns Hopkins Data Science Specialization offered by Coursera.


Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks.

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

More information is available from the website http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

In this report we use the data from the accelerometers to build predictive models using relevant R packages. We shall describe the transformations and preprocessing carried out on the data, the choices made during model selection and cross validation, and evaluating the out of sample error calculation.

## Getting and Cleaning the Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

First of all, we download the csv files from the above sources into a particular directory. Next we load the data in R by the following commands, where the "NA", "#DIV/0" and "" are specified to be NA values:


```r
train_data <- read.csv("pml-training.csv",na.strings=c("NA","#DIV/0!",""))
test_data <- read.csv("pml-testing.csv",na.strings=c("NA","#DIV/0!",""))
```

Run the str command to get a feel for the training data:


```r
str(train_data, list.len = 15)
```

```
## 'data.frame':	19622 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##   [list output truncated]
```

There are 19622 observations of 160 variables. The variable to be predicted is the "classe" variable, which takes five values: A, B, C, D and E. The outcomes are well distributed with no skewness, as can be checked with the `table` or the `prop.table` commands. 160 is a huge number of variables, so let us check if any variables in `train_data` have nearly zero variance and if so let us remove them. It is important to remove these variables from *both* `train_data` and `test_data`.


```r
suppressMessages(library(caret))
nzv <- nearZeroVar(train_data,saveMetrics=TRUE)
train_v2 <- train_data[,nzv$nzv==FALSE]
test_v2 <- test_data[,nzv$nzv==FALSE]
dim(train_v2)
```

```
## [1] 19622   124
```

We're down to 124 variables, which is still a large number. A quick analysis will show that certain variables in the training data have NAs for almost all observations (> 19000), while the other variables have valid measurements for all observations. Let us remove those variables in the training data which have more than 19000 NA values, and let us remove these columns from the test data as well:


```r
threshold <- 19000
na_check <- is.na(train_v2)  
train_v3 <- train_v2[colSums(na_check) < threshold]
test_v3 <- test_v2[colSums(na_check) < threshold]
dim(train_v3)
```

```
## [1] 19622    59
```

We can further remove the first two columns from both the data sets , since they are simply the index numbers and the names of the participants:


```r
train_v3 <- train_v3[,-c(1,2)]
test_v3 <- test_v3[,-c(1,2)]
```

so finally we shall work with the following variables


```r
names(train_v3)
```

```
##  [1] "raw_timestamp_part_1" "raw_timestamp_part_2" "cvtd_timestamp"      
##  [4] "num_window"           "roll_belt"            "pitch_belt"          
##  [7] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
## [10] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [13] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [16] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [19] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [22] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [25] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [28] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [31] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [34] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [37] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [40] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [43] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [46] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [49] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [52] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [55] "magnet_forearm_y"     "magnet_forearm_z"     "classe"
```

Now we split the training data (`train_v3`) into two sets: one for training, which we call `training` and the other for cross-validating, which we call `testing`.


```r
set.seed(343)
inTrain <- createDataPartition(y=train_v3$classe,p=0.7,list=FALSE)
training <- train_v3[inTrain,]
testing <- train_v3[-inTrain,]
```

Let us now try to fit different models on the `training` set.

## Model Training with Gradient Boosting

Since this is a classification problem with a large number of variables, boosting and random forests are the natural choices for training a model. Of course trees can also be used but they are not expected to be as accurate as the other two methods. Let us first train a model with the Gradient Boosting method.

We will use the interface provided by caret to fit the gradient boost model. The `trainControl` function can be used to specify which type of cross validation should be done. We choose the k-fold cross validation which can be chosen by setting `method="cv"` in the `trainControl` function (http://topepo.github.io/caret/training.html#control). The `number` argument in `trainControl` specifies that a cross-validation with `number` folds will be carried out; in this case we have set `number = 5`. Equivalently, one could have used `trainControl(method = repeatedcv, number = 5, repeats = 1)` to carry out a 5-fold cross-validation `repeats` number of times. Having thus specified the cross validation technique to be used, we shall train the model using the `train` command. This is achieved with the following code:


```r
set.seed(4553)
fitcontrol <- trainControl(method="cv", number=5)
modFitgbm <- train(classe~. , method="gbm", data=training, trControl=fitcontrol, verbose=FALSE)
```

```
## Loading required package: gbm
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```

```
## Loading required package: plyr
```

Training the model took me about 5 minutes on my ASUS notebook with an Intel-Core i5-5200U, 2.7 GHz, and 8 GB processor, with the Ubuntu 14.04 OS. Let us see its performance on the cross validation set:


```r
pred <- predict(modFitgbm,newdata=testing)
confusionMatrix(pred,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    2    0    0    0
##          B    0 1135    0    0    0
##          C    0    1 1020    9    0
##          D    0    1    6  954    1
##          E    0    0    0    1 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9964          
##                  95% CI : (0.9946, 0.9978)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9955          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9965   0.9942   0.9896   0.9991
## Specificity            0.9995   1.0000   0.9979   0.9984   0.9998
## Pos Pred Value         0.9988   1.0000   0.9903   0.9917   0.9991
## Neg Pred Value         1.0000   0.9992   0.9988   0.9980   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1929   0.1733   0.1621   0.1837
## Detection Prevalence   0.2848   0.1929   0.1750   0.1635   0.1839
## Balanced Accuracy      0.9998   0.9982   0.9960   0.9940   0.9994
```

The model has an accuracy of 0.996 when tested on the cross validation set. Hence the out-of-sample error rate is 1-0.996, or 0.4%. This is pretty good. Next, let us fit an random forest model to see if we can improve the result.

## Model Training with Random Forest

Training a model with random forest takes considerably longer compared to gradient boost. Hence let run a 2-fold cross validation during the training process. This time it took about 10 minutes to train the model. Let us also see its performance on the cross-validation set.


```r
fitcontrol <- trainControl(method="cv",number=2)
set.seed(4553)
modFitrf <- train(classe~.,data=training,method="rf",trControl=fitcontrol,prox=TRUE,verbose=TRUE,allowParallel=TRUE)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
pred <- predict(modFitrf,newdata=testing)
confusionMatrix(pred,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1139    0    0    0
##          C    0    0 1026    7    0
##          D    0    0    0  957    1
##          E    0    0    0    0 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9986          
##                  95% CI : (0.9973, 0.9994)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9983          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   0.9927   0.9991
## Specificity            1.0000   1.0000   0.9986   0.9998   1.0000
## Pos Pred Value         1.0000   1.0000   0.9932   0.9990   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   0.9986   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1935   0.1743   0.1626   0.1837
## Detection Prevalence   0.2845   0.1935   0.1755   0.1628   0.1837
## Balanced Accuracy      1.0000   1.0000   0.9993   0.9963   0.9995
```

In this case the accuracy is 0.9986, which is a slight improvement compared to the gbm model. Thus the out-of-sample error rate is 1-0.9986, or 0.14%.

## Results on the test set

Let us now go back to the test data `test_v3`, on which we have to predict the `classe` variable by applying our model. We shall apply the random forest model on this data set to predict the outcomes on this data set.


```r
pred_test <- predict(modFitrf,newdata=test_v3)
pred_test
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

There is a perfect match between all the 20 predicted outcomes and the correct outcomes upon submitting these results in the Course Project Prediction Quiz.

## Summary and Conclusions

We were provided with train and test data which contained measurements of various parameters carried out on six different vounteers while they were exercising. The goal was to use these measurements to train a machine learning model to predict the outcomes on the test data- There were five possible values for the outcome which corresponded to how correctly the exercise was being done.

After getting the data, we removed certain unimportant variables, for example those with nearly zero variance and variables with mostly NA values, from both the training and testing sets. Then we split the training data further into a training set and a cross validation set and fitted a model with the gradient boosting method and a random forest method. In both cases we chose the k-fold cross-validation method. After developing the models we gauged their accuracy on the cross validation set. Both showed extremely high accuracy, almost 99.9%, with the random forest performing slightly better than the gradient boost method.

We applied the random forest model on the test data and were able to predict all the 20 outcomes correctly.
