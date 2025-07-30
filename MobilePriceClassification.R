library(data.table)    
library(ggplot2) 
library(dplyr)
library(gridExtra)
library(glmnet) 
library(ggcorrplot)
library(caTools)
library(rpart)
library(caret)
library(rpart.plot)
library(e1071)
library(caTools)
library(class)
library(nnet)
library(randomForest)

tr_data <- read.csv("~/Downloads/train.csv")


##correlation map 
corr <- round(cor(tr_data), 8)
print(ggcorrplot(
  corr,
  method = "square",       # or "circle" for circular markers
  type = "full",           # shows full matrix (use "lower" or "upper" for half)
  colors = c("coral", "bisque", "salmon"),  # Brown -> White -> Orange
  outline.color = "darkorange4",  # border color of tiles
  ggtheme = theme_bw(),    # clean theme
  lab = TRUE,              # display correlation coefficients
  lab_size = 3,            # size of correlation text
  tl.cex = 10              # size of variable labels
))
## Feature Selection 

x = model.matrix(price_range~., tr_data)       # matrix of predictors
y = tr_data$price_range                       # vector y values
set.seed(123)                                # replicate  results
lasso_model <- cv.glmnet(x, y, alpha=1)      # alpha=1 is lasso
best_lambda_lasso <- lasso_model$lambda.1se
best_lambda_lasso# largest lambda in 1 SE
lasso_coef <- lasso_model$glmnet.fit$beta[,lasso_model$glmnet.fit$lambda == best_lambda_lasso]
coef_l = data.table(lasso = lasso_coef)      # build table
coef_l[, feature := names(lasso_coef)]       # add feature names
to_plot_r = melt(coef_l                      # label table
                 , id.vars='feature'
                 , variable.name = 'model'
                 , value.name = 'coefficient')
ggplot(data=to_plot_r,                       # plot coefficients
       aes(x=feature, y=coefficient, fill=model)) +
  coord_flip() +         
  geom_bar(stat='identity', fill='salmon', color='brown',) +
  facet_wrap(~ model) + guides(fill=FALSE) 

##STEPWISE SELECTION 



## MAKING GRAPHS 
tr_data$blue <- as.factor(tr_data$blue)
tr_data$dual_sim <- as.factor(tr_data$dual_sim)
tr_data$four_g <- as.factor(tr_data$four_g)
tr_data$three_g<- as.factor(tr_data$three_g)
tr_data$touch_screen <- as.factor(tr_data$touch_screen)
tr_data$wifi <- as.factor(tr_data$wifi)
tr_data$n_cores <- as.factor(tr_data$n_cores)
tr_data$price_range <- as.factor(tr_data$price_range)


p1 <-  ggplot(tr_data, aes(x=blue, fill=blue)) + theme_bw() + geom_bar() + ylim(0, 1050) + labs(title = "Bluetooth") + scale_x_discrete(labels = c('Not Supported','Supported')) + 
  scale_fill_manual(values = c("salmon", "orange"))
p2 <- ggplot(tr_data, aes(x=dual_sim, fill=dual_sim)) + theme_bw() + geom_bar() + ylim(0, 1050) + labs(title = "Dual Sim") + scale_x_discrete(labels = c('Not Supported','Supported'))+
  scale_fill_manual(values = c("salmon", "orange"))
p3 <- ggplot(tr_data, aes(x=four_g, fill=four_g)) + theme_bw() + geom_bar() + ylim(0, 1050) + labs(title = "4G") + scale_x_discrete(labels = c('Not Supported','Supported'))+
  scale_fill_manual(values = c("salmon", "orange"))
p4 <- ggplot(tr_data, aes(x=price_range, fill=price_range)) + theme_bw() + geom_bar() + ylim(0, 600) + labs(title = "Price") + scale_x_discrete(labels = c('0','1','2','3'))+
  scale_fill_manual(values = c("salmon", "orange", "coral1","coral4"))
p5 <- ggplot(tr_data, aes(x=three_g, fill=three_g)) + theme_bw() + geom_bar() + ylim(0, 1600) + labs(title = "3G") + scale_x_discrete(labels = c('Not Supported','Supported'))+
  scale_fill_manual(values = c("salmon", "orange"))
p6 <- ggplot(tr_data, aes(x=touch_screen, fill=touch_screen)) + theme_bw() + geom_bar() + ylim(0,1050) + labs(title ="Touch Screen") + scale_x_discrete(labels = c('Not Supported','Supported'))+
  scale_fill_manual(values = c("salmon", "orange"))
p7 <- ggplot(tr_data, aes(x=wifi, fill=wifi)) + theme_bw() + geom_bar() + ylim(0, 1050) + labs(title = "Wifi") + scale_x_discrete(labels = c('Not Supported','Supported'))+
  scale_fill_manual(values = c("salmon", "orange"))
p8 <- ggplot(tr_data, aes(x=n_cores, fill=n_cores))  + geom_bar() + ylim(0, 500) + labs(title = "Number of Processor Cores ") + 
  scale_fill_manual(values = c("salmon", "orange","chocolate", "darksalmon","coral1", "coral2","coral3", "coral"))

print(grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, nrow = 3))

print(prop.table(table(tr_data$blue))) # cell percentages
print(prop.table(table(tr_data$dual_sim))) # cell percentages
print(prop.table(table(tr_data$four_g))) # cell percentages

##Decision Tree
#Importing Libraries And reading The file ----


set.seed(123)
split<-sample.split(tr_data,SplitRatio=0.8)
train_data<-subset(tr_data,split==TRUE)
ts_data<-subset(tr_data,split==FALSE)

#set.seed(12345)
# Training with classification tree
model <- rpart(price_range ~ram+battery_power+px_height+px_width, data=train_data, method="class")
#print(model, digits = 3)
rpart.plot(model)
printcp(model)
plotcp(model)
pred <- predict(model, ts_data, type = "class")
#pred
# Accuracy and other metrics
t1<-table(ts_data$price_range, pred)
confusionMatrix(t1, mode="everything", positive="1")


##KNN 
df<- train_data[,c(1,5,11,12,13,14,21)]


#Applying knn for k = 1----
classifier_knn <- knn(train = tr_data, test = ts_data, cl = tr_data$price_range, k = 1)
#classifier_knn
#summary(classifier_knn)
cm <- table(ts_data$price_range, classifier_knn)
confusionMatrix(cm, mode="everything", positive="1")
#Applying knn for k = 3----
classifier_knn <- knn(train = tr_data, test = ts_data, cl = tr_data$price_range, k = 3)
#classifier_knn
#summary(classifier_knn)
cm <- table(ts_data$price_range, classifier_knn)
confusionMatrix(cm, mode="everything", positive="1")
#Applying knn for k = 5----
classifier_knn <- knn(train = tr_data, test = ts_data, cl = tr_data$price_range, k = 5)
#classifier_knn
#summary(classifier_knn)
cm <- table(ts_data$price_range, classifier_knn)
confusionMatrix(cm, mode="everything", positive="1")
#Applying knn for k = 7----
classifier_knn <- knn(train = tr_data, test = ts_data, cl = tr_data$price_range, k = 7)
#classifier_knn
#summary(classifier_knn)
cm <- table(ts_data$price_range, classifier_knn)
confusionMatrix(cm, mode="everything", positive="1")
#Applying knn for k = 9----
classifier_knn <- knn(train = tr_data, test = ts_data, cl = tr_data$price_range, k = 9)
#classifier_knn
#summary(classifier_knn)
cm <- table(ts_data$price_range, classifier_knn)
confusionMatrix(cm, mode="everything", positive="1")
#Applying knn for k = 11----
classifier_knn <- knn(train = tr_data, test = ts_data, cl = tr_data$price_range, k = 11)
#classifier_knn
#summary(classifier_knn)
cm <- table(ts_data$price_range, classifier_knn)
confusionMatrix(cm, mode="everything", positive="1")
#Applying knn for k = 13----
classifier_knn <- knn(train = tr_data, test = ts_data, cl = tr_data$price_range, k = 13)
#classifier_knn
#summary(classifier_knn)
cm <- table(ts_data$price_range, classifier_knn)
confusionMatrix(cm, mode="everything", positive="1")
#Applying knn for k = 17----
classifier_knn <- knn(train = tr_data, test = ts_data, cl = tr_data$price_range, k = 17)
#classifier_knn
#summary(classifier_knn)
cm <- table(ts_data$price_range, classifier_knn)
confusionMatrix(cm, mode="everything", positive="1")
#Applying knn for k = 15----
classifier_knn <- knn(train = tr_data, test = ts_data, cl = tr_data$price_range, k = 15)
#classifier_knn
#summary(classifier_knn)
cm <- table(ts_data$price_range, classifier_knn)
confusionMatrix(cm, mode="everything", positive="1")




##Naive Bayes 
model <- naiveBayes(price_range ~ram+battery_power+px_height+px_width, data = tr_data)


print(summary(model))


predicted <- predict(model, ts_data)
head(predicted)


accu<-mean(predicted == ts_data$price_range)  #accuracy of the multinomial logistic regression
t1<-table(ts_data$price_range, predicted)
confusionMatrix(t1, mode="everything", positive="1")



## Multinomial Logistic regression 

modelML <- nnet::multinom(price_range ~ram+battery_power+px_height+px_width, tr_data)

summary(modelML)

predicted <- modelML %>% predict(ts_data)
head(predicted)
ts_data <- cbind(ts_data, predicted)


accu<-mean(predicted == ts_data$price_range)  #accuracy of the multinomial logistic regression
t1<-table(ts_data$price_range, predicted)
confusionMatrix(t1, mode = "everything", positive="1")


#Building Random Forest classifier
rf1 <- randomForest(price_range ~ ram + battery_power + px_height + px_width,
                    data = train_data, importance = TRUE)


# Make predictions (will be factors)
pred1 <- predict(rf1, ts_data)
cm1 = table(ts_data$price_range, pred1)
confusionMatrix(cm1, mode = "everything", positive = "1")




