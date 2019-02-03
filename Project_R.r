rm(list=ls())
rm(churn_data_final)
setwd("C:/Users/Rajashekar/Videos/Project/Churn")
getwd()

x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')
install.packages("corrgram")
lapply(x, require, character.only = TRUE)

#reading data
churn_data=read.csv("Train_data.csv",sep = ",")
colnames(churn_data)
churn_data[0:6,]
str(churn_data)
summary(churn_data)

head(churn_data[, 1:6]) %>% kable(caption = "churn reduction  (Columns: 1-6)",
                            booktabs = TRUE, longtable = TRUE)
head(churn_data[, 7:15]) %>% kable(caption = "churn reduction (Columns: 7-15)",
                             booktabs = TRUE, longtable = TRUE)
head(churn_data[, 16:21]) %>% kable(caption = "churn reduction (Columns: 16-21)",
                                   booktabs = TRUE, longtable = TRUE)
var <- colnames(churn_data)[-ncol(churn_data)]
num <- 1:length(var)
df = data.frame(S.No. = num, Predictor = var)
kable(df, caption = "Predictor Variables", booktabs = TRUE,
      longtable = TRUE)
#Missing value analysis
#training data
churn_train=read.csv("Train_data.csv",sep=",")
missing_val_train= data.frame(apply(churn_train,2,function(x){sum(is.na(x))}))
missing_val_train
#test data
churn_test=read.csv("Test_data.csv",sep=",")
missing_val_test= data.frame(apply(churn_test,2,function(x){sum(is.na(x))}))
missing_val_test

#boxplot

# churn score boxplots for train
library(ggplot2)
num_index=sapply(churn_train,is.numeric)
num_data=churn_train[,num_index]
cnames=colnames(num_data)
for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = 'Churn'), data = churn_train)+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="Churn")+
           ggtitle(paste("Box plot of Churn vs",cnames[i])))
}
gridExtra::grid.arrange(gn1,gn5,gn2,gn3,gn4,gn6,ncol=6)
gridExtra::grid.arrange(gn7,gn8,gn9,gn10,gn11,ncol=5)
gridExtra::grid.arrange(gn12,gn13,gn14,gn15,gn16,ncol=5)

# churn score boxplots for test
library(ggplot2)
num_index=sapply(churn_test,is.numeric)
num_data=churn_test[,num_index]
cnames=colnames(num_data)
for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = 'Churn'), data = churn_test)+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="Churn")+
           ggtitle(paste("Box plot of Churn vs",cnames[i])))
}
gridExtra::grid.arrange(gn1,gn5,gn2,gn3,gn4,gn6,ncol=6)
gridExtra::grid.arrange(gn7,gn8,gn9,gn10,gn11,ncol=5)
gridExtra::grid.arrange(gn12,gn13,gn14,gn15,gn16,ncol=5)

#correlation analysis
library(corrgram)
numeric_index= sapply(churn_train,is.numeric)
numeric_data=churn_train[,numeric_index]
corrgram(churn_data[,numeric_index],order=F,upper.panel=panel.pie,text.panel=panel.txt,
         main="correlation plot")

#chi-square test
factor_index=sapply(churn_train,is.factor)
factor_data=churn_train[,factor_index]
colnames(factor_data)

for(i in 1:4)
{
  print(names(factor_data[i]))
  print(chisq.test(table(factor_data$Churn,factor_data[,i])))
  
}

#dimensionality reduction
churn_data_train=subset(churn_data,select=-c(phone.number,total.day.charge,total.eve.charge
                                             ,total.night.charge,total.intl.charge))
colnames(churn_data)

#Decision Tree for classification
churn_train=read.csv("Train_data_c.csv",sep=",")
churn_test=read.csv("Test_data_c.csv",sep=",")

churn_train_final=subset(churn_train,select=-c(phone.number,total.day.charge,total.eve.charge,total.night.charge,total.intl.charge))
churn_test_final=subset(churn_test,select=-c(phone.number,total.day.charge,total.eve.charge,total.night.charge,total.intl.charge))

##Decision tree for classification
#Develop Model on training data
C50_model = C5.0(Churn ~., churn_train_final, trials = 35, rules = TRUE)

#Summary of DT model
summary(C50_model)

#write rules into disk
write(capture.output(summary(C50_model)), "c50Rules.txt")


#Lets predict for test cases
C50_Predictions = predict(C50_model, churn_test_final[,-16], type = "class")

DT_output=cbind(C50_Predictions,churn_test_final$Churn)

##Evaluate the performance of classification model

ConfMatrix_C50 = table(factor(C50_Predictions),factor(churn_test_final$Churn))
confusionMatrix(ConfMatrix_C50)

#False Negative rate
FNR = FN/FN+TP 
#accuracy =95.8
#False Negative rate =27.67

# removed random forest as state attribute has 51 levels and RF can't handle morethan 32 levels

#Logistic Regression
#logit Model
logit_model = glm(Churn ~ ., data = churn_train_final, family = "binomial")

#summary of the model
summary(logit_model)

#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = churn_test_final, type = "response")

#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)

logit_output=cbind(logit_Predictions,churn_test_final$Churn)

##Evaluate the performance of classification model
ConfMatrix_lt = table(churn_test_final$Churn, logit_Predictions)
ConfMatrix_lt

#accuracy=87.04
#False negative rate=75.44

#Naive baise 
#Develop model
NB_model = naiveBayes(responded ~ ., data = train)


#KNN Classification
#Data reading for KNN
churn_train_knn=read.csv("Train_data.csv",sep=",")
churn_test_knn=read.csv("Test_data.csv",sep=",")
churn_train_final_knn=subset(churn_train_knn,select=-c(phone.number,total.day.charge,total.eve.charge,total.night.charge,total.intl.charge))
churn_test_final_knn=subset(churn_test_knn,select=-c(phone.number,total.day.charge,total.eve.charge,total.night.charge,total.intl.charge))

#assigning levels to factor predicators

for(i in 1:ncol(churn_train_final_knn)){
  
  if(class(churn_train_final_knn[,i]) == 'factor'){
    
    churn_train_final_knn[,i] = factor(churn_train_final_knn[,i], labels=(1:length(levels(factor(churn_train_final_knn[,i])))))
    
  }
}

for(i in 1:ncol(churn_test_final_knn)){
  
  if(class(churn_test_final_knn[,i]) == 'factor'){
    
    churn_test_final_knn[,i] = factor(churn_test_final_knn[,i], labels=(1:length(levels(factor(churn_test_final_knn[,i])))))
    
  }
}
#converting factor predicators to numeric to apply KNN classification
cnames=sapply(churn_train_final_knn,is.factor)
factor_data=churn_train_final_knn[,cnames]
factor_index=colnames(factor_data)
factor_index[1]
for(i in 1:length(factor_index)-1){
  churn_train_final_knn[,factor_index[i]]=as.numeric(churn_train_final_knn[,factor_index[i]])
}
for(i in 1:length(factor_index)-1){
  churn_test_final_knn[,factor_index[i]]=as.numeric(churn_test_final_knn[,factor_index[i]])
}

#applying Knn classification model

KNN_Predications=knn(churn_train_final_knn[,1:15],churn_test_final_knn[,1:15],churn_train_final_knn$Churn,k=1)

str(churn_train_final)

knn_output=cbind(churn_test_final_knn$Churn,KNN_Predications)

#confusion matrix to calculate error metrics
Conf_matrix = table(observed = churn_test_final_knn[,16], predicted = KNN_Predications)
confusionMatrix(Conf_matrix)
KNN_Predications

#Accuray=82.3
#False negative rate=63.39

#data read for bayes

churn_train_b=read.csv("Train_data.csv",sep=",")
churn_test_b=read.csv("Test_data.csv",sep=",")
churn_train_final_b=subset(churn_train_b,select=-c(phone.number,total.day.charge,total.eve.charge,total.night.charge,total.intl.charge))
churn_test_final_b=subset(churn_test_b,select=-c(phone.number,total.day.charge,total.eve.charge,total.night.charge,total.intl.charge))

#naive Bayes
library(e1071)

#Develop model
NB_model = naiveBayes(Churn ~ ., data = churn_train_final_b)

#predict on test cases #raw
NB_Predictions = predict(NB_model,churn_test_final_b[,0:15], type = 'class')

#Look at confusion matrix
Conf_matrix = table(predicted = NB_Predictions,observed = churn_test_final_b[,16])
cbind(churn_test_final_b[,16],NB_Predictions)
confusionMatrix(Conf_matrix)
#Accuracy 88.12
#False Negative Rate=71.43


   

