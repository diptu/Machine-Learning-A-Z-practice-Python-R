#import dataset
dataset = read.csv('50_Startups.csv')

#Encoding categorical data
dataset$State = factor(dataset$State,
                         levels = c('New York','California','Florida'),
                         labels = c(1,2,3))
#Split dataset
library(caTools)
set.seed(123)
split = sample.split(Y = dataset$Profit, SplitRatio = 0.8)
train_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Fit Multiple Linear Regression
lr = lm(formula = Profit ~ . ,
        data = train_set)

#Predicting test set result
y_pred = predict(lr, test_set)

# Fit Multiple Linear Regression excluing columns P values > 0.05,
# dosen't have any statastical significent

# Buliding optimal model using Backward Elimination
lr = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
        data = dataset)
summary(lr)

lr = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend ,
        data = dataset)
summary(lr)

lr = lm(formula = Profit ~ R.D.Spend  + Marketing.Spend ,
        data = dataset)
summary(lr)

# Marketing spend is near the statistical significat lavel , so keep that for now

#Fit Multiple Linear Regression to Final model
lr = lm(formula = Profit ~ R.D.Spend  + Marketing.Spend ,
        data = train_set)

#Predicting test set result
y_pred = predict(lr, test_set)