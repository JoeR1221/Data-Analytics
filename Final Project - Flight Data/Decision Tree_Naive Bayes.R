# Jeffrey Ronay
# Final Project - Decision Tree and Naive Bayes
#install.packages("rpart")
#library(rpart)

#install.packages("rpart.plot")
#library(rpart.plot)

install.packages("rattle")
library(rattle)

dir <- "C:/Users/Home/Documents/Syracuse/IST 707/FinalProject"
setwd(dir)


# Step 1 - Collect the data - Top_50_Airports.csv
# Identify the factors linked to a delayed flight

# Step 2 - Explore and prepare the data
top50 <- read.csv("Top_50_Airports.csv",header = TRUE, stringsAsFactors = TRUE)
# top50$arr_del15 <- as.factor(top50$arr_del15)

top50$X <- NULL


str(top50)

#top50 <- top50[1:100,]
str(top50)



# let's examine the table output for a couple of flight features
# that seem likely to predict a late flight. 
# The carrier and airport are recorded as categorical variables (factor)
nrow(top50)

# remove airport name column
top50$airport_name <- NULL
top50$City <- NULL
top50$City_State <- NULL
str(top50)



summary(top50$arr_del15)

# create a categorical variable representing
# ontime <= the medium of delay minutes, <= 49 minutes

top50$ontime <- as.factor(ifelse(top50$arr_del15 <= 49, "YES", "NO" ))

top50$arr_del15 <- as.factor(top50$arr_del15)

str(top50$ontime)
summary(top50$ontime)

# Create a training and testing set
RNGversion("3.5.2")
set.seed(123)

# create a vector of 900 random integers
# using a 70/30 split -- 70% train, 30% test
# we have 21,196 rows, so 70% ==> 14,837
#train_sample <- sample(21196,14837)
train_sample <- sample(21126,14788)


str(train_sample)

# use the vector to select rows from the flight data
# to split the data into 90% training and 10% testing data

flight_train <- top50[train_sample, ]
flight_test <- top50[-train_sample, ]

# verifiy the random split row counts
nrow(flight_train)
nrow(flight_test)

# step 3 - training the model on data
#  using the C5.0 algorithm
install.packages("C50")
library(C50)
flight_train["arr_del15"]

str(flight_train)

vars <- c("year", "month", "carrier", "airport", "arr_flights", "carrier_ct",
         "weather_ct", "nas_ct", "security_ct", "late_aircraft_ct", "arr_cancelled",
         "arr_diverted", "arr_delay", "carrier_delay", "weather_delay", "nas_delay",
         "security_delay", "late_aircraft_delay")



install.packages("gmodels")
library(gmodels)
# columnsToProcess <- subset(flight_train, select = -c(arr_del15))


flight_model <- C5.0(flight_train[, vars], y = flight_train$ontime)
flight_model
summary(flight_model)

plot(flight_model, subtree = 3)
str(flight_test)

flight_pred <- predict(flight_model, flight_test) 

# compare the vector of predicted class values to the actual class values

CrossTable(flight_test$ontime, flight_pred,
           prop.chisq = FALSE, prop.c =  FALSE, prop.r = FALSE,
           dnn = c('actual delay > 15', 'predicted delay >15'))

flight_model_boost10 <- C5.0(flight_train[, vars], y = flight_train$ontime, trials = 10)
flight_model_boost10
summary(flight_model_boost10)

flight_pred_boost10 <- predict(flight_model_boost10, flight_test) 

CrossTable(flight_test$ontime, flight_pred_boost10,
           prop.chisq = FALSE, prop.c =  FALSE, prop.r = FALSE,
           dnn = c('actual delay > 15', 'predicted delay >15'))

# construct a cost matrix
matrix_dimensions <- list(c("NO", "YES"), c("NO", "YES"))
names(matrix_dimensions) <- c("predicted", "actual")
matrix_dimensions

# assign error cost
error_cost <- matrix(c(0,1000,1000,0), nrow = 2, dimnames = matrix_dimensions)
error_cost

flight_model_boost10_cost <- C5.0(flight_train[, vars], 
                                                 y = flight_train$ontime, trials = 10, costs = error_cost )
flight_model_boost10_cost
summary(flight_model_boost10_cost)

flight_pred_boost10_cost <- predict(flight_model_boost10_cost, flight_test) 

CrossTable(flight_test$ontime, flight_pred_boost10_cost,
           prop.chisq = FALSE, prop.c =  FALSE, prop.r = FALSE,
           dnn = c('actual delay > 15', 'predicted delay >15'))

plot(flight_model_boost10_cost, trial = 10, subtree = 3)

install.packages("partykit")
library(partykit)
myTree2 <- C50::as.party.C5.0(flight_model_boost10_cost)
plot(myTree2[1])
plot(myTree2[3])

install.packages("e1071")
library(e1071)




# begin Naive Bayes
NB <- naiveBayes(ontime ~., data = flight_train, kernel="polynomial", cost=100, scale=FALSE)
print(NB)

# test the model
NB_predict <- predict(NB, flight_test, type = "class")
print(NB_predict)

ontimeLabels <- flight_test$ontime

CrossTable(NB_predict, ontimeLabels,
           prop.chisq = FALSE, prop.c =  FALSE, prop.r = FALSE,
           dnn = c('predicted delay > 15', 'actual delay >15'))


# begin Naive Bayes
NB2 <- naiveBayes(ontime ~., data = flight_train, kernel="polynomial", cost=100, scale=FALSE, laplace = 1)
print(NB2)

# test the model
NB_predict2 <- predict(NB2, flight_test, type = "class")
print(NB_predict2)

ontimeLabels <- flight_test$ontime

CrossTable(NB_predict2, ontimeLabels,
           prop.chisq = FALSE, prop.c =  FALSE, prop.r = FALSE,
           dnn = c('predicted delay > 15', 'actual delay >15'))


