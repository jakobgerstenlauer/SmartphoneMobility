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

str(d)
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

#We have to convert Y into a matrix with 6 vectors, 
#each vector being a dummy variable for one of the possible labels.
Y<-model.matrix( ~ as.factor(dy$V1) - 1)
dim(Y)
#[1] 7352    6

#How many input variables are there?
maxColumns<-dim(Xs)[2]
includeColum<-vector()

#Test for all input variables if they are significantly related to the labels.
for(numVar in 1:maxColumns){
  d2 <- data.frame(dy$V1, Xs[,numVar])
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
Xs<-Xs[,includeColum]
table(includeColum)
#TRUE 
#561

#Read additional data set with IPs belonging to Sybills
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
Xs <- Xs[,-indeces.to.exclude]
Y <- Y[,-indeces.to.exclude]
subjects <- subjects[,-indeces.to.exclude]

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
pca1.fm$eig$`cumulative percentage of variance`[1:63]
#Based on the 90% rule, I would need the first 63 principal components!

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

# Plot of R2 for each digit 
plot(R2(m1.pls2), legendpos = "bottomright")

#calculate the mean R2 over all digits for an increasing number of components:
r2.mean<-apply(R2(m1.pls2)$val[1,,],2,mean)

#plot the mean R2 for the first 50 components:
plot(1:50,r2.mean[1:50],pch="+")

# Calculate the coefficient of determination 
# based on generalized cross-validation:
n <- nrow(Xs)
#p <- ncol(Xs)
p<-40
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

#Conclusion: Based on generalized CV 25 components are sufficient.
nc<-25

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
barplot(clusters$height)
dev.off()

#It is pretty obvious that 3 splits / 4 clusters make sense!
cl <- cutree(clusters, 4)

setwd(plotDir)
jpeg("hierarchical_clustering_ward_4classes_PC1_PC2.jpeg")
plot(Psi[,1],Psi[,2],type="n",main="Clustering of observations into 4 classes")
points(Psi[,1],Psi[,2],col=cl,pch=dy$V1, cex = 0.6)
abline(h=0,v=0,col="gray")
legend("topleft",c("c1","c2","c3","c4"),pch=20,col=c(1:4))
dev.off()

setwd(plotDir)
jpeg("hierarchical_clustering_ward_4classes_PC1_PC3.jpeg")
plot(Psi[,1],Psi[,3],type="n",main="Clustering of observations into 4 classes")
points(Psi[,1],Psi[,3],col=cl,pch=dy$V1, cex = 0.6)
abline(h=0,v=0,col="gray")
legend("topleft",c("c1","c2","c3","c4"),pch=20,col=c(1:4))
dev.off()

#Is there any correspondence between the clusters and the labels?
ct<-data.frame(labels=dy$V1,clusters=cl)
table(ct)
# clusters
# labels    1    2    3    4
#      1    0    0  953  273
#      2    0    0  933  140
#      3    0    0  287  699
#      4 1264   21    1    0
#      5 1373    1    0    0
#      6   37 1369    1    0

# cluster 1 -> label 3+4
# cluster 2 -> label 6
# cluster 3 -> label 1+2+ (3)
# cluster 4 -> label (1+2)+3

#Conclusion: 
#The clusters map pretty well to the labels!

#Now let's run a consolidated cluster analysis using the centroids of this partitioning:
#Consolidation of the partition:

#I use the centroids of the 4 clusters found with hierarchical clustering (WARD)
#as starting point for k-means:

#Calculate the centroids of the 4 clusters
#in the nc significant dimensions (principal components)
cdg <- aggregate(as.data.frame(scores),list(cl),mean)[,2:(nc+1)]

k4 <- kmeans(scores, centers=cdg)

Bss <- sum(rowSums(k4$centers^2)*k4$size) # = k5$betweenss
Wss <- sum(k4$withinss) # = k5$tot.withinss
(Ib5 <- 100*Bss/(Bss+Wss))
#[1] 72.88688

#Is there any correspondence between the clusters and the labels?
ct2<-data.frame(labels=dy$V1,clusters=k4$cluster)
table(ct2)
# clusters
# labels    1    2    3    4
# 1    0    0  905  321
# 2    0    0  971  102
# 3    0    0  356  630
# 4 1192   91    3    0
# 5 1373    0    1    0
# 6   21 1375   11    0


#********************************************
#Now let's use five clusters (5 labels!)
#********************************************
cl <- cutree(clusters, 5)

setwd(plotDir)
jpeg("hierarchical_clustering_ward_5classes_PC1_PC2.jpeg")
plot(Psi[,1],Psi[,2],type="n",main="Clustering of observations into 4 classes")
points(Psi[,1],Psi[,2],col=cl,pch=dy$V1, cex = 0.6)
abline(h=0,v=0,col="gray")
legend("topleft",c("c1","c2","c3","c4","c5"),pch=20,col=c(1:4))
dev.off()

setwd(plotDir)
jpeg("hierarchical_clustering_ward_5classes_PC1_PC3.jpeg")
plot(Psi[,1],Psi[,3],type="n",main="Clustering of observations into 4 classes")
points(Psi[,1],Psi[,3],col=cl,pch=dy$V1, cex = 0.6)
abline(h=0,v=0,col="gray")
legend("topleft",c("c1","c2","c3","c4","c5"),pch=20,col=c(1:4))
dev.off()

#Is there any correspondence between the clusters and the labels?
ct<-data.frame(labels=dy$V1,clusters=cl)
table(ct)
# clusters
# labels    1    2    3    4    5
# 1    0    0  953  250   23
# 2    0    0  933  140    0
# 3    0    0  287  574  125
# 4 1264   21    1    0    0
# 5 1373    1    0    0    0
# 6   37 1369    1    0    0

# cluster 1 -> label 4+5
# cluster 2 -> label 6
# cluster 3 -> label 1+2+ (3)
# cluster 4 -> label (1+2)+3
# cluster 5 -> label 3

#Now let's run a consolidated cluster analysis using the centroids of this partitioning:
#Consolidation of the partition:

#I use the centroids of the 5 clusters found with hierarchical clustering (WARD)
#as starting point for k-means:

#Calculate the centroids of the 5 clusters
#in the nc significant dimensions (principal components)
cdg <- aggregate(as.data.frame(scores),list(cl),mean)[,2:(nc+1)]

k5 <- kmeans(scores, centers=cdg)

Bss <- sum(rowSums(k5$centers^2)*k5$size) # = k5$betweenss
Wss <- sum(k5$withinss) # = k5$tot.withinss
(Ib5 <- 100*Bss/(Bss+Wss))
#[1] 75.45319

ct2<-data.frame(labels=dy$V1,clusters=k5$cluster)
table(ct2)
# clusters
# labels    1    2    3    4    5
# 1    0    0  692  483   51
# 2    0    0  861  208    4
# 3    0    0  185  629  172
# 4 1192   89    5    0    0
# 5 1373    0    1    0    0
# 6   21 1373   13    0    0

# cluster 1 -> label 4+5
# cluster 2 -> label 6
# cluster 3 -> label 1+2+ (3)
# cluster 4 -> label 1+(2)+3
# cluster 5 -> label 3

#****************************************************************************************
# Next step: 
# a) Use random forrest to predict the label using the 25 significant components as input!
# b) Use relevance vector machine or support vector machine using the 25 significant components as input!
#****************************************************************************************
############# random forest    Yunxuan Dou May.11############
Psi<-m1.pls2$loadings[,1:nc]
set.seed(9019)
#install.packages("randomForest")
library(randomForest)

rf_data <- data.frame(Psi)
# suppose we have 7352 individuals
# choosing the trainning and test data
X_train <- rf_data
Y_train <- read.table("Y_train.txt")
rf_train <- data.frame(rf_data,Y_train)
X_test <-  read.table("X_test.txt")
Y_test <- read.table("Y_test.txt")

#Possible hyperparameter:
# ntree: 	
# Number of trees to grow. This should not be set to too small a number, to ensure that every input row gets predicted at least a few times.
#
# mtry:	
# Number of variables randomly sampled as candidates at each split. Note that the default values are different for classification (sqrt(p) where p is number of variables in x) and regression (p/3)
# replace 
#
# classwt:	
# Priors of the classes. Need not add up to one. Ignored for regression.
# cutoff	(Classification only) A vector of length equal to number of classes. The ‘winning’ class for an observation is the one with the maximum ratio of proportion of votes to cutoff. Default is 1/k where k is the number of classes (i.e., majority vote wins).

# strata: Maybe define the subject (the person) as strata 
#A (factor) variable that is used for stratified sampling.

# sampsize	
# Size(s) of sample to draw. For classification, if sampsize is a vector of the length
# the number of strata, then sampling is stratified by strata, 
# and the elements of sampsize indicate the numbers to be drawn from the strata.
#
#
# nodesize: use 10!	
# Minimum size of terminal nodes. Setting this number larger causes smaller trees to be grown (and thus take less time). Note that the default values are different for classification (1) and regression (5).
# maxnodes	
# Maximum number of terminal nodes trees in the forest can have. If not given, trees are grown to the maximum possible (subject to limits by nodesize). If set larger than maximum possible, a warning is issued.
# importance	
# Should importance of predictors be assessed?
# localImp	
# Should casewise importance measure be computed? (Setting this to TRUE will override importance.)
# nPerm	
# Number of times the OOB data are permuted per tree for assessing variable importance. Number larger than 1 gives slightly more stable estimate, but not very effective. Currently only implemented for regression.
# proximity	
# Should proximity measure among the rows be calculated?
# oob.prox	
# Should proximity be calculated only on “out-of-bag” data?
# norm.votes	
# If TRUE (default), the final result of votes are expressed as fractions. If FALSE, raw vote counts are returned (useful for combining results from different runs). Ignored for regression.
# do.trace	
# If set to TRUE, give a more verbose output as randomForest is run. If set to some integer, then running output is printed for every do.trace trees.
# keep.forest	
# If set to FALSE, the forest will not be retained in the output object. If xtest is given, defaults to FALSE.
# corr.bias	
# perform bias correction for regression? Note: Experimental. Use at your own risk.
# keep.inbag	
# Should an n by ntree matrix be returned that keeps track of which samples are “in-bag” in which trees (but not how many times, if sampling with replacement)
# ...	
# optional parameters to be passed to the low level function randomForest.default.

randomForest(x, y=NULL,  xtest=NULL, ytest=NULL, ntree=500,
             mtry=if (!is.null(y) && !is.factor(y))
               max(floor(ncol(x)/3), 1) else floor(sqrt(ncol(x))),
             replace=TRUE, classwt=NULL, cutoff, strata,
             sampsize = if (replace) nrow(x) else ceiling(.632*nrow(x)),
             nodesize = if (!is.null(y) && !is.factor(y)) 5 else 1,
             maxnodes = NULL,
             importance=FALSE, localImp=FALSE, nPerm=1,
             proximity, oob.prox=proximity,
             norm.votes=TRUE, do.trace=FALSE,
             keep.forest=!is.null(y) && is.null(xtest), corr.bias=FALSE,
             keep.inbag=FALSE, ...)

rf_data <- randomForest(Y_train~.,data=rf_train, mtry=10, importance=TRUE)
pred_rf_data <- predict(rf_data,newdata = X_test) 
table(pred_rf_data,Y_test)
plot(rf_data)
#################finish of random forest######################
