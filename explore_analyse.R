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
y<-dy$V1

#Let's have a look at the distribution of labels:
table(y)
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

X <- as.matrix(d)

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

#Here we exclude these 4 outliers from both Xs, y, and the list of subjects
X <- X[-indeces.to.exclude,]
y <- y[-indeces.to.exclude]
subjects <- subjects[-indeces.to.exclude]

#############################################################################
#PCA analysis allowing different weights per individual:

#number of observations:
n<-dim(X)[1]

# a. Define the matrix N of weights of individuals 
# The weights differ between observations from different individuals.
num.subjects<-length(unique(subjects))
weights<-as.data.frame(table(subjects))
weights$weight<- (1/weights$Freq) * (1/num.subjects)
weights$subjects<-as.numeric(weights$subjects)
ds.subjects<-data.frame("subjects"=as.numeric(subjects))
ds.weights<-merge(ds.subjects, weights, by="subjects")
str(ds.weights)

sum(ds.weights$weight)  
#1
table(tapply(ds.weights$weight, ds.weights$subjects, sum))
#0.0476190476190476 
#21 
#Conclusion: All subjects have the same sum of weights!

#Assign these weights to the diagonal of the weight matrix:
N<-diag(ds.weights$weight)
dim(N)
# 7348 7348

#check sum of diagonal, must be 1
sum(diag(N))==1
#TRUE

# b. Compute the weighted centroid G of all individuals.
centroid<-t(X) %*% N %*% rep(1,n)

# c. Compute the centered and standardized X matrix.
#center and standardize the matrix because the measure were taken in different units:
#center
Means<-matrix(rep(centroid,n),byrow=TRUE,nrow=n)
X<-scale(X,center=TRUE,scale=FALSE)

# Compute the covariance matrix of X 
M  <- t(X) %*% N %*% X
hist(table(100*diag(M)/sum(diag(M))))
#Some variables contribute more than others,
#because they have a higher variance!
#That is not what we want here!

# scale the matrix
Xs <- X %*% diag(sqrt(diag(M))**(-1))
# Compute the Correlation matrix of X 
C  <- t(Xs) %*% N %*% Xs
table(round(diag(C),10))
#1
#555
#The diagonal now contains only ones because the data matrix was standardized!

#calculate the contribution of each variable
table(100*diag(C)/sum(diag(C)))
#Now all variables contribute equally!

C.eigen<-eigen(C)

#the eigenvalues (lambda) of the correlation matrix
C.eigen$values[1:10]
# [1] 285.040738  37.018159  15.380096  13.950463  10.472981   9.591672
# [7]   7.671311   6.748827   5.596248   5.365080

#the eigenvectors of the covariance matrix (u)
dim(C.eigen$vectors)
#555 555

# e. Do the screeplot of the eigenvalues and define the number of significant
# dimensions. How much is the retained information?
nc<-50
plot(1:nc,C.eigen$values[1:nc], type="l",pch="|",
     xlab=paste("order of eigenvalue"),
     ylab="eigenvalue")

#Kaiser rule: Take all eigenvalues > the mean.
length(C.eigen$values[C.eigen$values>mean(C.eigen$values)])
#60

total.inertia<-sum(C.eigen$values)
eigenvalues.cumulative<-cumsum(C.eigen$values)/total.inertia
length(eigenvalues.cumulative[eigenvalues.cumulative<0.9])
#59

#Based on both the 90% rule and the Kaiser rule, 
#we should work with the first 60 principal components.
lambda.max<-60

#####################################################################
# f. Compute the projections of individuals on the significant components:
psi <- Xs %*% C.eigen$vectors[,1:lambda.max]
dim(psi)
#[1] 7348 60

#calculate eigenvalues again:
diag(t(psi) %*% N %*% psi)

#####################################################################
# Compute the projection of variables on the significant dimensions:

#number of inputs p:
p<-dim(Xs)[2]
#phi <- U %*% diag(sqrt(lambda))
sqrt.lambda<-matrix(rep(t(sqrt(C.eigen$values[1:lambda.max])),p),byrow=TRUE,nrow=p)
phi<- C.eigen$vectors[,1:lambda.max] * sqrt.lambda
dim(phi)
#[1] 555 60

#check if multiplication was correct:
sqrt(C.eigen$values[2]) * C.eigen$vectors[,2] == phi[,2]

# h. Plot a sample of 500 individuals in the first factorial plane of Rp. 
# Color the individuals according to the class label.

sampleI <-sample(1:n,500)

setwd(plotDir)
jpeg("Individuals_first_factorial_plane.jpg")
plot(psi[sampleI,1], psi[sampleI,2],
     pch="+",col=y,
     main = 
       expression(paste("Projection of 500 Individuals in R"^"p")),
     xlab="Component 1",
     ylab="Component 2"
)
#add labels:
#text(psi, labels=,col=)
dev.off()

# i. Plot the first 100 variables (as arrows) in the first factorial plane of Rn.
library("plotrix")
setwd(plotDir)
jpeg("Projection_of_First_100_Variables_First_Factorial_Plane.jpg")
plot(c(-1,1), c(-1,1), type="n", 
     asp = 1, 
     xlim = c(-1, 1),
     ylim = c(-1, 1),
     main = 
       expression(paste("Projection of 100 Variables in R"^"n")),
     xlab="Component 1",
     ylab="Component 2"
)
draw.circle(0, 0, 1, nv = 1000, border = NULL, col = NA, lty = 1, lwd = 1)
arrows(x0=rep(0,length(phi[1:100,1])), y0=rep(0,length(phi[1:100,2])), x1 = phi[1:100,1], y1 = phi[,2])
#add labels:
#text(phi, labels=,col=)
dev.off()

#perform a varimax rotation
pc.rot = varimax(phi)

#extract the loadings of the variables on the nc significant principal components: 
phi.rot = pc.rot$loadings
dim(phi.rot)
#[1] 555  60

#Feature extraction: 
#For each significant component do:
cut.off<-0.8
#Lower threshold if no variables were selected for default cut-off
cut.off.2<-0.7

for(i in 1:lambda.max){

  loading<-phi.rot[,i]
  #1. Select all variables above a certain threshold for the absolute value of the loading.
  index<-abs(loading)>cut.off
  if(length(index[index])==0){
    index<-abs(loading)>cut.off.2
  }
  if(length(index[index])==0){
    next;
  }
  m<- as.matrix(Xs[,index])
  #2. If the loading is negative, multiply the value of the variable with -1.
  b<-ifelse(loading[index]<0, -1, 1)
  #3. Calculate the mean value of al selected variables and store it as a new feature.
  c<- (m %*% b) / length(index[index])
  if(exists("d.pc")){
    d.pc<-cbind(d.pc,c)
  }else{
    d.pc<-c
  }
  rm(c)
}

dim(d.pc)
#[1] 7348   46

###################################################################################################################
# Cluster analysis
###################################################################################################################
#****************************************************************************
# Perform a hierarchical clustering (in the reduced PCA space)
# with the significant factors, decide the number of final classes to obtain
# and perform a consolidation operation of the clustering.
#****************************************************************************

#The reduced subspace given by the 59 new features
#TODO Ask Tomas if this is correct!
Psi<-as.matrix(d.pc)
#... is transformed into a distance matrix.
dist.matrix<-dist(Psi, method = "euclidean")

#Then I perform hierarchical clustering based on this distance matrix
#and the default complete linkage method:
clusters <- hclust(dist.matrix)
str(clusters)

setwd(plotDir)
jpeg("hierarchical_clustering_complete_linkage.jpg")
plot(clusters)
dev.off()

#use "mean linkage" method instead:
clusters <- hclust(dist.matrix, method = 'average')

setwd(plotDir)
jpeg("hierarchical_clustering_average_linkage.jpg")
plot(clusters)
dev.off()

#use "WARD" method instead:
clusters <- hclust(dist.matrix, method = "ward.D2")

setwd(plotDir)
jpeg("hierarchical_clustering_WARD.jpg")
plot(clusters)
dev.off()

jpeg("hierarchical_clustering_WARD_inertia_explained.jpg")
barplot(clusters$height[1:10])
dev.off()

cl <- cutree(clusters, 4)

jpeg("hierarchical_clustering_ward_4_classes_PC1_PC2.jpeg")
plot(Psi[,1],Psi[,2],type="p",pch="+",col=y, main="Clustering of observations in 4 classes")
abline(h=0,v=0,col="gray")
legend("topleft",c("c1","c2","c3","c4"),pch=20,col=c(1:4))
dev.off()

jpeg("hierarchical_clustering_ward_4_classes_PC1_PC3.jpeg")
plot(Psi[,1],Psi[,3],type="p",pch="+",col=y, main="Clustering of observations in 4 classes")
abline(h=0,v=0,col="gray")
legend("topleft",c("c1","c2","c3","c4"),pch=20,col=c(1:4))
dev.off()

# LETS SEE THE QUALITY OF THE HIERARCHICAL PARTITION

#Calculate the centroids of the 4 clusters in the dimensions
#defined by the 46 newly created features:
cdg <- aggregate(as.data.frame(Psi),list(cl),mean)[,-1]

#between clusters sum of squares
Bss <- sum(rowSums(cdg^2)*as.numeric(table(cl)))
#total sum of squares
Tss <- sum(rowSums(Psi^2))
Tss/n
#[1] 43.75862

(Ib4 <- 100 * Bss/Tss)
#[1] 13.00816

#Consolidation of the partition:
#I use the centroids of the 4 clusters found with hierarchical clustering (WARD)
#as starting point for k-means:

k4 <- kmeans(Psi,centers=cdg)
Bss <- sum(rowSums(k4$centers^2)*k4$size) 
Wss <- sum(k4$withinss) 
(Ib4 <- 100*Bss/(Bss+Wss))
#[1] 15.55882

#The consolidated result is not much better, 
#than the end result of the hierarchical clustering!

#****************************************************************************
#Step 5: Interpret and name the obtained clusters
# and represent them in the first factorial display.
#****************************************************************************

jpeg("consolidated_clustering_k_means_5classes_PC1_PC2.jpeg")
plot(Psi[,1],Psi[,2],type="n",
     main="Consolidated K-means Clustering of Countries in 4 classes",
     xlab="Principal Component 1",ylab="Principal Component 2")
text(Psi[,1],Psi[,2],col=k4$cluster,labels=y, cex = 0.6)
abline(h=0,v=0,col="gray")
legend("topleft",c("c1","c2","c3","c4"),pch=20,col=c(1:4))
dev.off()

jpeg("consolidated_clustering_k_means_4classes_PC1_PC3.jpeg")
plot(Psi[,1],Psi[,3],type="n",
     main="Consolidated K-means Clustering of Countries in 4 classes",
     xlab="Principal Component 1",ylab="Principal Component 3")
text(Psi[,1],Psi[,3],col=k4$cluster,labels=y, cex = 0.6)
abline(h=0,v=0,col="gray")
legend("topleft",c("c1","c2","c3","c4"),pch=20,col=c(1:4))
dev.off()

#Link clusters to the classes:
table(k4$cluster, y)
# y
#     1   2   3   4   5   6
# 1   2  38   1 382 298 610
# 2  23   3  13 827 863 776
# 3 651 134 610   0   0   2
# 4 550 898 361  77 211  18

#Conclusion:
#Cluster 1 & 2 -> classes 4,5,6
#Cluster 3 -> classes 1,3,(2)
#Cluster 4 -> classes 1,2,3,(5)
#It is not possible to clearly separate the classes!

#What if we restrict the new features to those significantly related to the classes?
includeColum2<-vector()
maxColumns<-dim(d.pc)[2]
  
#Test for all input variables if they are significantly related to the labels.
for(numVar in 1:maxColumns){
  d2 <- data.frame(y, d.pc[,numVar])
  names(d2)<-c("y","x")
  m1.aov<-aov(as.numeric(x)~as.factor(y), data=d2)
  m0.aov<-aov(as.numeric(x)~1, data=d2)
  p.value<-anova(m1.aov,m0.aov)[["Pr(>F)"]][2]
  if(p.value<0.0001){
    includeColum2<-c(includeColum2,TRUE)
  }else{
    includeColum2<-c(includeColum2,FALSE)
  }
}

#remove columns without effect
table(includeColum2)
# includeColum2
# FALSE  TRUE 
# 1    44 

#Only one column (feature) would be excluded.
#This can not have any effect on the result of our cluster analysis.

#****************************************************************************************
# Next step: 
# a) Use a regression tree to predict the label using the 25 significant components as input!
# b) Use a random forrest to predict the label using the 25 significant components as input!
# c) Use relevance vector machine or support vector machine using the 25 significant components as input!
#****************************************************************************************
training.data <- data.frame(Psi)
training.data$subjects<-subjects
dim(training.data)
#[1] 7348   47

# choosing the training and test data
setwd(trainDir)

#append the class labels, exclude outliers
training.data$y <- as.factor(y)
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
#Do not include subject as predictor!
index.subjects<-which(names(training.data)=="subjects")
m1.rp <- rpart(y ~ ., method="class", data=training.data[,-index.subjects], control=rpart.control(cp=0.001, xval=10))
printcp(m1.rp)

setwd(plotDir)
jpeg("CrossValidatedPredictionErrorRegressionTree.jpeg")
plotcp(m1.rp)
dev.off()

m2.rp<-prune(m1.rp, cp = 0.005)
plotcp(m2.rp)
printcp(m2.rp)
# CP nsplit rel error  xerror      xstd
# 1  0.2305621      0   1.00000 1.00673 0.0056122
# 2  0.2063278      1   0.76944 0.76944 0.0069943
# 3  0.1236957      2   0.56311 0.56311 0.0071843
# 4  0.0888590      3   0.43941 0.44009 0.0069070
# 5  0.0811175      4   0.35056 0.36267 0.0065677
# 6  0.0100976      5   0.26944 0.27331 0.0059858
# 7  0.0085830      6   0.25934 0.26641 0.0059309
# 8  0.0072366      7   0.25076 0.25917 0.0058716
# 9  0.0067317      9   0.23628 0.24958 0.0057901
# 10 0.0055537     10   0.22955 0.23309 0.0056421
# 11 0.0053012     13   0.21171 0.22669 0.0055819
# 12 0.0051330     15   0.20111 0.21996 0.0055166
# 13 0.0050000     17   0.19084 0.21087 0.0054256

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
                      ntree=500, 
                      mtry=10,
                      classwt= rep(1/6,6), 
                      importance=TRUE, 
                      data=training.data,
                      xtest=X_test, 
                      ytest=Y_test, 
                      nodesize=10, 
                      maxnodes=15)

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
