#**************************************************************************************
#Multivariate Analysis
#Spring Term 2017
#Final Project
#Lecturer: Tomas Aluja
#
# Date: 01.04.2017
# Jakob Gerstenlauer
# jakob.gerstenlauer@gjakob.de
###############################################################################################

#remove old objects for safety resons
rm(list=ls(all=TRUE))

#utility function
glue<-function(...){paste(...,sep="")}

#define path of standard directories
source("workingDir.R")

setwd(trainDir)
#read the inputs (features/predictors)
d<-read.table("X_train.txt")
dim(d)
#[1] 7352  561

#Are there any missing values in the inputs?
table(complete.cases(d))
#TRUE 
#7352 

#How many input variables are there?
(maxColumns<-dim(d)[2])
# 561

#read the outputs
dy<-read.table("y_train.txt")
dim(dy)
#[1] 7352    1

#Are there any missing values in the outputs?
table(complete.cases(dy))
#TRUE 
#7352

#Let's have a look at the distribution of labels:
table(dy$V1)
# 1    2    3    4    5    6 
# 1226 1073  986 1286 1374 1407

#Maybe we have to weight the predictions for label 3 more strongly,
#because there are less observations. 
#If we don't do so, the final model might be less accurate in correctly 
#predicting label 3.
includeColum<-vector()

#Test for all input variables if they are significantly related to the labels.
for(numVar in 1:maxColumns){
  d2 <- data.frame(dy$V1, d[,numVar])
  names(d2)<-c("y","x")
  m1.aov<-aov(as.numeric(x)~as.factor(y), data=d2)
  m0.aov<-aov(as.numeric(x)~1, data=d2)
  p.value<-anova(m1.aov,m0.aov)[["Pr(>F)"]][2]
  if(p.value<0.05){
    includeColum<-c(includeColum,TRUE)
  }else{
    includeColum<-c(includeColum,FALSE)
  }
}

#remove columns without effect
table(includeColum)
#includeColum
# FALSE  TRUE 
# 6   555

d<-d[,includeColum]

(input.means<-apply(d,2,mean))
#the inputs are not centered!
(input.sd<-apply(d,2,sd))
hist(input.sd)
#The standard deviation of inputs is bounded between 0.0 and 0.8.
#As we know from the data description:
#"Features are normalized and bounded within [-1,1]".
#Nevertheless, we should scale the inputs because we do not know if the
#variance of measurement inputs has any meaning.
#Probably measurement with higher variance are even less reliable.

Xs <- as.matrix(scale(d, center = TRUE, scale = TRUE))

#We have to convert Y into a matrix with 6 vectors, 
#each vector being a dummy variable for one of the possible labels.
Y<-model.matrix( ~ as.factor(dy$V1) - 1)
dim(Y)
#[1] 7352    6

subjects <- as.factor(readLines("subject_train.txt"))
table(subjects)
# 1  11  14  15  16  17  19  21  22  23  25  26  27  28  29   3  30   5   6   7   8 
# 347 316 323 328 366 368 360 408 321 372 409 392 376 382 344 341 383 302 325 308 281

#library("DMwR")
#outlier_scores<- lofactor(Xs, k=5)
#setwd(plotDir)
#jpeg("Outliers.jpg")
#plot(density(outlier_scores))
#dev.off()
#higher than 1.7 seem to be the real bad guys!
# pick top x outliers
#outliers <- outlier_scores[outlier_scores > 1.7]
#hist(outliers)
#Which are the outliers?
#print(outliers)
#[1] 1.745032 2.057507 1.721322 1.805697
#We excluded the top 4 outliers.
# hist(outlier_scores)
# which(outlier_scores > 1.7)
# [1]   71 1905 3935 5067

indeces.to.exclude<-c(71, 1905, 3935, 5067)

#Here we exclude these 4 outliers from both Xs and Y
Xs <- Xs[-indeces.to.exclude,]
Y <- Y[-indeces.to.exclude,]
subjects <- subjects[-indeces.to.exclude]

#Conclusion: All variables show an effect on the output!
library(FactoMineR)
pca1.fm <- PCA(as.data.frame(Xs),
               #a boolean, if TRUE (value set by default) then data are scaled to unit variance
               scale.unit = TRUE, 
               #number of dimensions kept in the results (by default 5)
               ncp = 100, 
               graph = FALSE
)

plot(pca1.fm)
summary(pca1.fm)
str(pca1.fm)

barplot(pca1.fm$eig$`cumulative percentage of variance`)
pca1.fm$eig$`cumulative percentage of variance`[1:59]
#Based on the 90% rule, I would need the first 59 principal components!
nc<-59

#perform a varimax rotation
pca1.varimax<-varimax(pca1.fm$var$cor[,1:nc])

#extract the loadings of the variables on the nc significant principal components: 
loadings<-pca1.varimax$loadings
dim(loadings)
#[1] 555  59

#Feature extraction: 
#For each significant component do:
cut.off<-0.8
#Lower threshold if no variables were selected for default cut-off
cut.off.2<-0.7

for(i in 1:nc){
  loading<-loadings[,i]
  #1. Select all variables above a certain threshold for the absolute value of the loading.
  index<-abs(loading)>cut.off
  if(length(index)==0){
    index<-abs(loading)>cut.off.2
  }
  if(length(index)==0){
    next;
  }
  m<- as.matrix(Xs[,index])
  #2. If the loading is negative, multiply the value of the variable with -1.
  b<-ifelse(loading[index]<0, -1, 1)
  #3. Calculate the mean value of al selected variables and store it as a new feature.
  c<-apply(m %*% b, 1, mean)
  if(exists("d.pc")){
    d.pc<-cbind(d.pc,c)
  }else{
    d.pc<-c
  }
  rm(c)
}

dim(d.pc)
#[1] 7348   59


#*******************************************************************
#Instead of using that many principal components,
#it's better to use partial least squares regression.
#The advantage of PLSR is that the selected components 
#are also correlated with the response labels.
#*******************************************************************

library(pls)
#Warning: This call can take a long, long time!
#m1.pls2 <- plsr(Y ~ Xs, validation = "LOO")
#fit model without validation:
m1.pls2 <- plsr(Y ~ Xs, validation = "none")

#TODO Try out parallelised cross-validation:
# ## Parallelised cross-validation, using persistent cluster:
# library(parallel)
# ## This creates the cluster:
# pls.options(parallel = makeCluster(4, type = "PSOCK"))
# ## The cluster can be used several times:
# yarn.pls <- plsr(density ~ NIR, 6, data = yarn, validation = "CV")

summary(m1.pls2)

# Plot of R2 for each class 
plot(R2(m1.pls2), legendpos = "bottomright")

#calculate the mean R2 over all classes for an increasing number of components:
r2.mean<-apply(R2(m1.pls2)$val[1,,],2,mean)

#plot the mean R2 for the first 50 components:
plot(1:50,r2.mean[1:50],pch="+")

# Calculate the coefficient of determination 
# based on generalized cross-validation:
n <- nrow(Xs)
#p <- ncol(Xs)
p<-50
q <- ncol(Y)

R2cv<-rep(-1,p)
for (i in 1:p) {
  lmY <- lm(Y~m1.pls2$scores[,1:i])
  PRESS  <- apply((lmY$residuals/(1-ls.diag(lmY)$hat))^2,2,sum)
  R2cv[i]   <- mean(1 - PRESS/(sd(Y)^2*(n-1)));
}

setwd(plotDir)
jpeg("NrOfPLSIIComponents.jpeg")
#plot generalized CV estimate first
plot(1:p,R2cv,type="l",
     xlab="number of components",
     ylab="coefficient of determination")
     points(1:p,r2.mean[1:p],type="l",col="red")
dev.off()

#Conclusion: Based on generalized CV 30 components are sufficient.
nc<-30

###################################################################################################################
# Cluster analysis
###################################################################################################################

#the loadings of the inputs on the components
Psi<-m1.pls2$loadings[,1:nc]

#loadings of the individuals
scores<-m1.pls2$scores[,1:nc]

#... is transformed into a distance matrix.
dist.matrix<-dist(scores, method = "euclidean")

#Then I perform hierarchical clustering based on this distance matrix
#and the corrected Ward algorithm:
clusters <- hclust(dist.matrix, method = "ward.D2")

setwd(plotDir)
jpeg("hierarchical_clustering_WARD.jpg")
plot(clusters)
dev.off()

jpeg("hierarchical_clustering_WARD_inertia_explained.jpg")
barplot(clusters$height[1:10])
dev.off()

#Let's use 5 splits / 6 clusters, which also corresponds to our 6 classes!
cl <- cutree(clusters, 6)
table(cl)
# cl
# 1    2    3    4    5    6 
# 2621 1445  845 1100  173 1168 

setwd(plotDir)
jpeg("hierarchical_clustering_ward_6classes_PC1_PC2.jpeg")
plot(Psi[,1],Psi[,2],type="n",main="Clustering of observations into 4 classes")
points(Psi[,1],Psi[,2],col=cl,pch=dy$V1, cex = 0.6)
abline(h=0,v=0,col="gray")
legend("topleft",c("c1","c2","c3","c4","c5","c6"),pch=20,col=c(1:6))
dev.off()

setwd(plotDir)
jpeg("hierarchical_clustering_ward_6classes_PC1_PC3.jpeg")
plot(Psi[,1],Psi[,3],type="n",main="Clustering of observations into 4 classes")
points(Psi[,1],Psi[,3],col=cl,pch=dy$V1, cex = 0.6)
abline(h=0,v=0,col="gray")
legend("topleft",c("c1","c2","c3","c4","c5","c6"),pch=20,col=c(1:6))
dev.off()

#Is there any correspondence between the clusters and the labels?
ct<-data.frame(labels=dy$V1,clusters=cl)
table(ct)
# clusters
# labels    1    2    3    4    5    6
# 1    0    0  818    3   23  382
# 2    0    0   27  316    2  728
# 3    0    0    0  781  148   57
# 4 1215   70    0    0    0    1
# 5 1369    5    0    0    0    0
# 6   37 1370    0    0    0    0

# cluster 1 -> label 4+5
# cluster 2 -> label 6
# cluster 3 -> label 1
# cluster 4 -> label (2)+3
# cluster 5 -> label 3
# cluster 6 -> label (1)+2

#Conclusion: 
#There is a lot of overlap between labels 1+2+3 and 4+5!

#Now let's run a consolidated cluster analysis using the centroids of this partitioning:
#Consolidation of the partition:

#I use the centroids of the 6 clusters found with hierarchical clustering (WARD)
#as starting point for k-means:

#Calculate the centroids of the 6 clusters
#in the nc significant dimensions (principal components)
cdg <- aggregate(as.data.frame(scores),list(cl),mean)[,2:(nc+1)]

k6 <- kmeans(scores, centers=cdg)

Bss <- sum(rowSums(k6$centers^2)*k6$size) # = k5$betweenss
Wss <- sum(k6$withinss) # = k5$tot.withinss
(Ib <- 100*Bss/(Bss+Wss))
#[1] 75.88375

#Is there any correspondence between the clusters and the labels?
ct2<-data.frame(labels=dy$V1,clusters=k6$cluster)
table(ct2)
# clusters
# labels    1    2    3    4    5    6
#      1    0    0  812  151   62  201
#      2    0    0   47  176    3  847
#      3    0    0   40  634  159  153
#      4 1189   89    0    0    0    8
#      5 1372    0    0    0    0    2
#      6   21 1372    0    0    0   14


#****************************************************************************************
# Next step: 
# a) Use a regression tree to predict the label using the 25 significant components as input!
# b) Use a random forrest to predict the label using the 25 significant components as input!
# c) Use relevance vector machine or support vector machine using the 25 significant components as input!
#****************************************************************************************
Psi<-m1.pls2$Yscores[,1:nc]
training.data <- data.frame(Psi)
training.data$subjects<-subjects
dim(training.data)
#[1] 7348   31

# choosing the training and test data
setwd(trainDir)
Y_train <- as.factor(dy$V1)

#append the class labels, exclude outliers
training.data$y <- Y_train[-indeces.to.exclude]
setwd(testDir)
Y_test <- read.table("y_test.txt")
Y_test$V1<-as.factor(Y_test$V1)
X_test <- read.table("X_test.txt")
#The test inputs have to be transformed in the same way as the training inputs!

#First we have to exclude 6 columns which we excluded for the training set
X_test <- X_test[,includeColum]
N <- nrow(X_test)
p <- ncol(X_test)


############# Subtract the mean values of the columns of the training data ############
correction.matrix<-as.matrix(rep(1,N)) %*% t(as.matrix(input.means))
dim(correction.matrix)
dim(as.matrix(X_test))
#check if the mean of the columns is identical to column means 
table(input.means == apply(correction.matrix,2,mean))
#check if the median of the columns is identical to column means 
table(input.means == apply(correction.matrix,2,median))
#check if the min of the columns is identical to column means 
table(input.means == apply(correction.matrix,2,min))

#subtract the means of the inputs in the training set
X_test<-X_test - correction.matrix

############# Divide by the standard deviation of the columns of the training data ############
correction.matrix<-as.matrix(rep(1,N) %*% t(input.sd))
#check if the mean of the columns is identical to column sd 
table(input.sd == apply(correction.matrix,2,mean))
#divide by the standard deviation of the inputs in the training set
X_test<-X_test / correction.matrix

############# regression tree ############
library(rpart)
set.seed(567)
#Use Gini index as impurity criterion:
m1.rp <- rpart(Adjusted ~ ., method="class", data=d.train, control=rpart.control(cp=0.001, xval=10))
printcp(m1.rp)




############# random forest  ############
set.seed(9019)
#install.packages("randomForest")
library(randomForest)




#Possible hyperparameter:
# ntree: Number of trees to grow. This should not be set to too small a number, to ensure that every input row gets predicted at least a few times.
# mtry:	 Number of variables randomly sampled as candidates at each split. Note that the default values are different for classification (sqrt(p) where p is number of variables in x) and regression (p/3)
# classwt:  Priors of the classes. Need not add up to one. Ignored for regression.
# strata: Maybe define the subject (the person) as strata 
#A (factor) variable that is used for stratified sampling.
# sampsize	
# Size(s) of sample to draw. For classification, if sampsize is a vector of the length
# the number of strata, then sampling is stratified by strata, 
# and the elements of sampsize indicate the numbers to be drawn from the strata.
# nodesize: use 10!	
# Minimum size of terminal nodes. Setting this number larger causes smaller trees to be grown (and thus take less time). Note that the default values are different for classification (1) and regression (5).
# maxnodes	

m1.rf <- randomForest(y~., 
                      ntree=1000, 
                      mtry=10,
                      classwt= rep(1/6,6), 
                      importance=TRUE, 
                      data=training.data,
                      xtest=X_test, 
                      ytest=Y_test, 
                      nodesize=10, 
                      maxnodes=10)

#Plot the error rate for an increasing number of trees:
setwd(plotDir)
jpeg("ErrorRate_NrOfTrees.jpeg")
plot(m1.rf)
dev.off()
#The error rate does not decrease above 200 trees.

pred_rf_data <- predict(rf_data,newdata = X_test) 
table(pred_rf_data,Y_test)
plot(rf_data)


#Step 1: define the LH scheme 
require("lhs")

#number of samples from the LHC 
SampleSize<-16;
NumVariables<-2;   

#Now define the ranges for all four parameters of the LHC:
#V1: mtry
low_V1= 5;
high_V1= nc;

#V2: number of trees
low_V2  = 200;
high_V2 = 600;
  
#V3: nodesize
low_V3  = 1;
high_V3 = 8;

#set-up the Latin Hypercube sampling scheme
LHS<-improvedLHS(n=SampleSize, k=NumVariables, dup=1)

for (simulation in seq(1,dim(LHS)[1]))
{
  for (arguments in seq(1,NumVariables))
  {   
    #Here we use the quantile function for the uniform distribution to "translate" from the standard uniform distribution to the respective trait range
    eval(parse(text=paste(
      'A',arguments,'<-round(qunif(LHS[simulation,',arguments,'], min=low_V',arguments,', max=high_V',arguments,'),digits=3)'
      ,sep="")));
    
    m1.rf <- randomForest(y~., ntree=A2, mtry=A1, nodesize = A3, classwt= rep(1/6,6), strata=subject, importance=TRUE, data=training.data)
    #TODO: Evaluate model quality based on prediction error! Remember best model!
  }
}

#################finish of random forest######################
