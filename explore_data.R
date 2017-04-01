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

#read the outputs
dy<-read.table("y_train.txt")
dim(dy)
#[1] 7352    1

#Are there any missing values in the inputs?
table(complete.cases(d))
#TRUE 
#7352 

#Are there any missing values in the outputs?
table(complete.cases(dy))
#TRUE 
#7352

#There are no missing values, but maybe there are outliers?
#Outlier detection using the Mahalanobis distance:
# library(chemometrics)
# outlier.results<-Moutlier(d, quantile = 0.975, plot = TRUE)
# Error in solve.default(cov, ...) : 
#   Lapack routine dgesv: system is exactly singular: U[219,219] = 0
# In addition: Warning message:
#   In covMcd(X) : The covariance matrix of the data is singular.
# str(outlier.results)

d<-as.matrix(d)
#the covariance matrix without any centering or standardization
cov<-t(d)%*%d
cov.eigen <- eigen(cov)
str(cov.eigen)
barplot(cov.eigen$values[1:10])
barplot(log(cov.eigen$values[1:10]))
#It seems that the first two components are sufficient)

library(FactoMineR)
pca1.fm <- PCA(d,
               #a boolean, if TRUE (value set by default) then data are scaled to unit variance
               scale.unit = TRUE, 
               #number of dimensions kept in the results (by default 5)
               ncp = 10, 
               graph = FALSE
 )
plot(pca1.fm)
summary(pca1.fm)
str(pca1.fm)

barplot(pca1.fm$eig$`cumulative percentage of variance`)
#Based on the 90% rule, I would need the first 63 principal components!

d<-cbind(d,dy)
