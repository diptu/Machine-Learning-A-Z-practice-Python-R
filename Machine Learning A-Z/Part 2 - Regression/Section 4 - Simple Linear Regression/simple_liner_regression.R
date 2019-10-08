#import dataset
dataset = read.csv('Salary_Data.csv')

#Split dataset
library(caTools)
set.seed(123)
split = sample.split(Y = dataset$Salary, SplitRatio = 0.8)

train_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Fitting LinerRegression Model
lr = lm(formula = Salary ~ YearsExperience,
        data = train_set)

#Prediction on test set
y_pred = predict(lr,test_set)

#Visualization
library(ggplot2)
#Traingin set
ggplot() +
  geom_point(aes(x = train_set$YearsExperience, y = train_set$Salary),
             colour = 'red') +
  geom_line(aes(x = train_set$YearsExperience, y = predict(lr,train_set)),
            colour = 'blue') +
  ggtitle("Salary vs Experience(Training set)") +
  xlab("Years of Experience") +
  ylab("Salary in USD")

#Test set
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = train_set$YearsExperience, y = predict(lr,train_set)),
            colour = 'blue') +
  ggtitle("Salary vs Experience(Testing set)") +
  xlab("Years of Experience") +
  ylab("Salary in USD")