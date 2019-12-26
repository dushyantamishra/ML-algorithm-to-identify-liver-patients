# Libraries
library(tidyverse)
library(caret)

# Now we download the data
columnNames <- c('age', # Age of the patient 
                 'sex', # Sex of the patient 
                 'tb', # Total Bilirubin
                 'db', # Direct Bilirubin 
                 'alkphos', # Alkaline Phosphotase
                 'sgpt', # Alamine Aminotransferase
                 'sgot', # Aspartate Aminotransferase
                 'tp', # Total Protiens
                 'alb', # Albumin
                 'ag', # Ratio	Albumin and Globulin Ratio 
                 'outcome') # Selector field used to split the data into two sets

fullData <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv",
                       sep=',',
                       header=FALSE,
                       col.names=columnNames)

# Format in a more comprehensible manner
fullData <- subset(fullData, complete.cases(fullData))
fullData <- fullData %>% 
  mutate(outcome = as.character(outcome)) %>% 
  mutate(outcome = replace(outcome, outcome == '1', 'Care')) %>%
  mutate(outcome = replace(outcome, outcome == '2', 'Control')) %>%
  mutate(outcome = as.factor(outcome))

head(fullData)

# Splitting the data
set.seed(1)
trainIndex <- createDataPartition(fullData$outcome, p=.7, list=FALSE)
train <- fullData[trainIndex,]
test <- fullData[-trainIndex,]
rm(trainIndex)

## Column by column exploration
#Age
train %>% 
  ggplot(aes(x = age)) + 
  geom_histogram(binwidth = 20) + 
  theme_minimal() +
  facet_grid(~ outcome)

# Sex 
train %>% 
  ggplot(aes(x = sex)) + 
  geom_bar() + 
  theme_minimal() +
  facet_grid(~ outcome)

# Bilirubin
train %>% 
  ggplot(aes(x = tb)) + 
  geom_histogram(binwidth = 5) + 
  theme_minimal() +
  facet_grid(~ outcome)

# Direct bilirubin
train %>% 
  ggplot(aes(x = db)) + 
  geom_histogram(binwidth = 1) + 
  theme_minimal() +
  facet_grid(~ outcome)

# Enzymes
# Alkphos
train %>% 
  ggplot(aes(x = alkphos)) + 
  geom_histogram(binwidth = 200) + 
  theme_minimal() +
  facet_grid(~ outcome)

# sgpt
train %>% 
  ggplot(aes(x = sgpt)) + 
  geom_histogram(binwidth = 200) + 
  theme_minimal() +
  facet_grid(~ outcome)

# sgot
train %>% 
  ggplot(aes(x = sgot)) + 
  geom_histogram(binwidth = 200) + 
  theme_minimal() +
  facet_grid(~ outcome)

# sgot vs bilirubin
train %>% 
  ggplot(aes(x = alkphos, y = tb, shape = outcome, color = outcome)) + 
  geom_point() +
  scale_y_log10() + 
  scale_x_log10() +
  geom_vline(xintercept = 700) +
  geom_hline(yintercept = 7.5) +
  theme_minimal() 

# Proteins
# Albumin
train %>% 
  ggplot(aes(x = alb)) + 
  geom_histogram(binwidth = 0.5) + 
  theme_minimal() +
  facet_grid(~ outcome)

# Total protein
train %>% 
  ggplot(aes(x = tp)) + 
  geom_histogram(binwidth = 0.5) + 
  theme_minimal() +
  facet_grid(~ outcome)

# Albumin and globulin
subset(train, !is.na(ag)) %>% 
  ggplot(aes(x = ag)) + 
  geom_histogram(binwidth = 0.1) + 
  theme_minimal() +
  facet_grid(~ outcome)

## Data preparation
# Correlated predictors
cor(subset(train, select = -c(sex, outcome)))

# Bilirubin
train %>% 
  ggplot(aes(x = tb, y = db)) + 
  geom_point() + 
  theme_minimal() +
  facet_grid(~ outcome)

# Dropping direct bilirubin from list of predictors
train <- train %>% subset(select = -c(db))
test <- test %>% subset(select = -c(db))

# Using same approach for sgpt and sgot columns
train %>% 
  ggplot(aes(x = sgot, y = sgpt, color = outcome, shape = outcome)) + 
  geom_point() + 
  theme_minimal()

train <- train %>% subset(select = -c(sgpt))
test <- test %>% subset(select = -c(sgpt))

# And for proteins
train %>% 
  ggplot(aes(x = tp, y = alb, color = outcome, shape = outcome)) + 
  geom_point() + 
  theme_minimal()

train <- train %>% subset(select = -c(alb))
test <- test %>% subset(select = -c(alb))

# Creating data frame to store results
results <- data.frame(Model = character(), 
                      Accuracy = double(), 
                      Sensitivity = double(), 
                      Specificity = double(), 
                      stringsAsFactors = FALSE)

## Modelling approaches
# Naive Bayes
nb_model = train(outcome ~ ., data = train, method = "nb")
predictions = predict(nb_model, newdata = test)
confusionMatrix <- confusionMatrix(predictions, test$outcome)
results[nrow(results) + 1, ] <- c(as.character('Naive Bayes (nb)'), 
                                  confusionMatrix$overall['Accuracy'],  
                                  confusionMatrix$byClass['Sensitivity'], 
                                  confusionMatrix$byClass['Specificity'])
rm(nb_model, predictions)
confusionMatrix

# Linear classifier
lc_model = train(outcome ~ ., data = train, method = "glmboost")
predictions = predict(lc_model, newdata = test)
confusionMatrix <- confusionMatrix(predictions, test$outcome)
results[nrow(results) + 1, ] <- c(as.character('Linear Classifier (glmboost)'), 
                                  confusionMatrix$overall['Accuracy'],  
                                  confusionMatrix$byClass['Sensitivity'], 
                                  confusionMatrix$byClass['Specificity'])
rm(lc_model, predictions)
confusionMatrix

# repeating confusion matrix, this time including the prevalence
nb_model = train(outcome ~ ., data = train, method = "nb")
predictions = predict(nb_model, newdata = test)
confusionMatrix <- confusionMatrix(predictions, test$outcome, prevalence = 0.06)
rm(nb_model, predictions)
confusionMatrix

lc_model = train(outcome ~ ., data = train, method = "glmboost")
predictions = predict(lc_model, newdata = test)
confusionMatrix <- confusionMatrix(predictions, test$outcome, prevalence = 0.06)
rm(lc_model, predictions)
confusionMatrix

# Logistic regression
lr_model = train(outcome ~ ., data = train, method = "bayesglm")
predictions = predict(lr_model, newdata = test)
confusionMatrix <- confusionMatrix(predictions, test$outcome, prevalence = 0.06)
results[nrow(results) + 1, ] <- c(as.character('Logistic Regression (bayesglm)'), 
                                  confusionMatrix$overall['Accuracy'],  
                                  confusionMatrix$byClass['Sensitivity'], 
                                  confusionMatrix$byClass['Specificity'])
rm(lr_model, predictions)
confusionMatrix

# K-Nearest Neighbours
knn_model = train(outcome ~ ., data = train, method = "knn", preProcess=c('knnImpute'))
knn_model

predictions = predict(knn_model, newdata = test)
confusionMatrix <- confusionMatrix(predictions, test$outcome, prevalence = 0.06)
results[nrow(results) + 1, ] <- c(as.character('K-nearest neighbours (knn)'), 
                                  confusionMatrix$overall['Accuracy'],  
                                  confusionMatrix$byClass['Sensitivity'], 
                                  confusionMatrix$byClass['Specificity'])
rm(knn_model, predictions)
confusionMatrix

# Random forrest
rf_model = train(outcome ~ ., data = train, method = "rf")
predictions = predict(rf_model, newdata = test)
confusionMatrix <- confusionMatrix(predictions, test$outcome, prevalence = 0.06)
results[nrow(results) + 1, ] <- c(as.character('Random Forest (rf)'), 
                                  confusionMatrix$overall['Accuracy'],  
                                  confusionMatrix$byClass['Sensitivity'], 
                                  confusionMatrix$byClass['Specificity'])
rm(rf_model, predictions)
confusionMatrix

rm(confusionMatrix)

## Results
results %>% arrange(Accuracy)

