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
(input.means<-apply(d,2,sum))
#the inputs are not centered!
(input.sd<-apply(d,2,sd))
hist(input.sd)
#The standard deviation of inputs is bounded between 0.0 and 0.8.
#As we know from the data description:
#"Features are normalized and bounded within [-1,1]".
#Therefore it is not useful to standardize the sensor data, but we should center it!

Xs <- as.matrix(scale(d, center = TRUE, scale = FALSE))

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

#We have to convert Y into a matrix with 6 vectors, 
#each vector being a dummy variable for one of the possible labels.
Y<-model.matrix( ~ as.factor(dy$V1) - 1)
dim(Y)
#[1] 7352    6

#There are no missing values, but maybe there are outliers?
#Outlier detection using the Mahalanobis distance:
# library(chemometrics)
# outlier.results<-Moutlier(d, quantile = 0.975, plot = TRUE)
# Error in solve.default(cov, ...) : 
#   Lapack routine dgesv: system is exactly singular: U[219,219] = 0
# In addition: Warning message:
#   In covMcd(X) : The covariance matrix of the data is singular.
# str(outlier.results)

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
m1.pls2 <- plsr(Y ~ Xs, validation = "LOO")
summary(m1.pls2)

# Plot of R2 for each digit 
plot(R2(m1.pls2), legendpos = "bottomright")

#Inspect the object:
dim(R2(m1.pls2)$val[1,,])
# 10 257
#rows: response
#columns: cumulative number of components

#calculate the mean R2 over all digits for an increasing number of components:
r2.mean<-apply(R2(m1.pls2)$val[1,,],2,mean)

# Calculate the coefficient of determination based on generalized cross-validation:
n <- nrow(Xs)
p <- ncol(Xs)
q <- ncol(Y)

R2cv<-rep(-1,p)
for (i in 1:p) {lmY <- lm(Y~m1.pls2$scores[,1:i])
PRESS  <- apply((lmY$residuals/(1-ls.diag(lmY)$hat))^2,2,sum)
R2cv[i]   <- mean(1 - PRESS/(sd(Y)^2*(n-1)));
}

#plot generalized CV estimate first
plot(1:p,R2cv,type="l")
#add LOO-CV estimate
points(1:p,r2.mean[1:p],type="l",col="red")
#It seems like the LOOCV gives a more realistic estimate of the generalization error,
#because the PRESS estimate is really quite flat.
#I now work only with the LOOCV estimate as criterium!

#zoom in:
setwd(plotDir)
jpeg("NrOfPLSIIComponents.jpeg")
plot(1:50,r2.mean[1:50],type="l",xlab="number of components", ylab="coefficient of determination")
abline(v=13)
text("13 components",x=22,y=0.1)
dev.off()
#Conclusion: Based on LOOCV 13 components are optimal! 





###################################################################################################################
# Cluster analysis
###################################################################################################################

#TODO Define the significant components here
Psi<-

#... is transformed into a distance matrix.
dist.matrix<-dist(Psi, method = "euclidean")

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
