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

###################################################################################################################
# Cluster analysis
###################################################################################################################
#****************************************************************************
# Perform a hierarchical clustering (in the reduced PCA space)
# with the significant factors, decide the number of final classes to obtain
# and perform a consolidation operation of the clustering.
#****************************************************************************

#The reduced subspace given by the 59 first principal components
Psi<-pca1.fm$ind$coord[,1:nc]
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
barplot(clusters$height)
dev.off()

cl <- cutree(clusters, 5)

jpeg("hierarchical_clustering_ward_5classes_PC1_PC2.jpeg")
plot(Psi[,1],Psi[,2],type="n",main="Clustering of countries in 5 classes")
text(Psi[,1],Psi[,2],col=cl,labels=names.cases, cex = 0.6)
abline(h=0,v=0,col="gray")
legend("topleft",c("c1","c2","c3","c4","c5"),pch=20,col=c(1:5))
dev.off()

jpeg("hierarchical_clustering_ward_5classes_PC1_PC3.jpeg")
plot(Psi[,1],Psi[,3],type="n",main="Clustering of countries in 5 classes")
text(Psi[,1],Psi[,3],col=cl,labels=names.cases, cex = 0.6)
abline(h=0,v=0,col="gray")
legend("topleft",c("c1","c2","c3","c4","c5"),pch=20,col=c(1:5))
dev.off()

# LETS SEE THE QUALITY OF THE HIERARCHICAL PARTITION

#Calculate the centroids of the 5 clusters in the 6 significant dimensions (principal components)
cdg <- aggregate(as.data.frame(Psi),list(cl),mean)[,2:(nd+1)]

#between clusters sum of squares
Bss <- sum(rowSums(cdg^2)*as.numeric(table(cl)))
#total sum of squares
Tss <- sum(rowSums(Psi^2))
Tss/n
#[1] 7.594025

sum(pca1.fm$eig$eigenvalue[1:nd])
#Ratio of inertia explained by clusters divided by total inertia:
(Ib4 <- 100 * Bss/Tss)
#[1] 43.89669

#Consolidation of the partition:
#I use the centroids of the 5 clusters found with hierarchical clustering (WARD)
#as starting point for k-means:

k5 <- kmeans(Psi,centers=cdg)
Bss <- sum(rowSums(k5$centers^2)*k5$size) # = k5$betweenss
Wss <- sum(k5$withinss) # = k5$tot.withinss
(Ib5 <- 100*Bss/(Bss+Wss))
#59.83553

#The consolidated result is considerably better than the end result of the hierarchical clustering!
#Hierarchical clustering: clusters explained 44% of total variance.
#Consolidated clustering: clusters explain 60% of total variance.

#****************************************************************************
#Step 5: Interpret and name the obtained clusters
# and represent them in the first factorial display.
#****************************************************************************

jpeg("consolidated_clustering_k_means_5classes_PC1_PC2.jpeg")
plot(Psi[,1],Psi[,2],type="n",
     main="Consolidated K-means Clustering of Countries in 5 classes",
     xlab="Principal Component 1",ylab="Principal Component 2")
text(Psi[,1],Psi[,2],col=k5$cluster,labels=names.cases, cex = 0.6)
abline(h=0,v=0,col="gray")
legend("topleft",c("c1","c2","c3","c4","c5"),pch=20,col=c(1:5))
dev.off()

jpeg("consolidated_clustering_k_means_5classes_PC1_PC3.jpeg")
plot(Psi[,1],Psi[,3],type="n",
     main="Consolidated K-means Clustering of Countries in 5 classes",
     xlab="Principal Component 1",ylab="Principal Component 3")
text(Psi[,1],Psi[,3],col=k5$cluster,labels=names.cases, cex = 0.6)
abline(h=0,v=0,col="gray")
legend("topleft",c("c1","c2","c3","c4","c5"),pch=20,col=c(1:5))
dev.off()

#Link clusters to the original variables:

#I have to ignore Cuba because I did not include it in the PCA!
d_without_cuba<-d[-index.cuba,]
d_without_cuba$clusters<-as.factor(k5$cluster)

#Index of the cluster variable: 11
test.catdes<-catdes(d_without_cuba, num.var=11)

#description of each category of the num.var by each category of all the categorical variables
test.catdes$category
# $`1`
# Cla/Mod   Mod/Cla   Global      p.value    v.test
# demo=3 68.421053 59.090909 41.30435 2.360839e-02  2.263443
# demo=1  6.666667  4.545455 32.60870 8.942578e-05 -3.917625
# 
# $`2`
# Cla/Mod Mod/Cla   Global      p.value    v.test
# demo=1      40     100 32.60870 0.0005343329  3.462927
# demo=3       0       0 41.30435 0.0316019771 -2.149409
# 
# $`3`
# Cla/Mod Mod/Cla   Global    p.value   v.test
# pais=Sud-Vietnam     100      50 2.173913 0.04347826 2.019086
# pais=Bolivie         100      50 2.173913 0.04347826 2.019086
# 
# $`4`
# Cla/Mod   Mod/Cla   Global     p.value    v.test
# demo=1 46.666667 58.333333 32.60870 0.039737357  2.056469
# demo=3  5.263158  8.333333 41.30435 0.007259914 -2.684681
# 
# $`5`
# NULL


#The description of each category of the num.var variable by the quantitative variables.
test.catdes$quanti

# $`1`
# v.test Mean in category Overall mean sd in category Overall sd      p.value
# farm     4.496867         97.49545     92.82609       2.577080   6.668988 6.896198e-06
# Gini     4.210871         80.59545     71.19783       8.177045  14.333701 2.543884e-05
# Laboagr  3.311864         53.68182     42.43478      15.070701  21.811098 9.267655e-04
# Instab   3.156425         14.54091     12.38261       1.140040   4.391657 1.597162e-03
# Gnpr    -3.624185        287.68182    563.56522     160.689250 488.908040 2.898740e-04
# 
# $`2`
# v.test Mean in category Overall mean sd in category Overall sd      p.value
# Gnpr     3.681397       1256.33333    563.56522     517.720538  488.90804 0.0002319599
# Rent     2.770175         42.76667     23.12391      16.002048   18.42244 0.0056026202
# Laboagr -3.744422         11.00000     42.43478       3.464102   21.81110 0.0001808094
# 
# $`3`
# v.test Mean in category Overall mean sd in category Overall sd      p.value
# Death 5.838112            831.5     73.58696          168.5  185.67006 5.279572e-09
# ecks  2.126970             51.5     21.60870            1.5   20.09919 3.342260e-02
# 
# $`4`
# v.test Mean in category Overall mean sd in category Overall sd      p.value
# Gnpr     2.948224        925.25000    563.56522     372.784148 488.908040 3.196054e-03
# ecks    -2.136539         10.83333     21.60870      13.433995  20.099187 3.263551e-02
# Laboagr -2.728845         27.50000     42.43478      14.459714  21.811098 6.355666e-03
# Rent    -2.941810          9.52500     23.12391       8.308643  18.422437 3.263006e-03
# Gini    -4.246376         55.92500     71.19783       7.807809  14.333701 2.172557e-05
# farm    -4.846043         84.71667     92.82609       4.080611   6.668988 1.259482e-06
# 
# $`5`
# v.test Mean in category Overall mean sd in category Overall sd      p.value
# Instab -5.483531             0.75     12.38261       1.299038   4.391657 4.169198e-08



#Let's see the average values for all clusters:

with(d_without_cuba, tapply(Gini, clusters, mean))
with(d_without_cuba, tapply(Gnpr, clusters, mean))
with(d_without_cuba, tapply(Laboagr, clusters, mean))
with(d_without_cuba, tapply(Death, clusters, mean))
with(d_without_cuba, tapply(ecks, clusters, mean))


# > with(d_without_cuba, tapply(Gini, clusters, mean))
# 1        2        3        4        5 
# 80.59545 71.81667 80.45000 55.92500 59.77500 
# > with(d_without_cuba, tapply(Gnpr, clusters, mean))
# 1         2         3         4         5 
# 287.6818 1256.3333   99.5000  925.2500  188.7500 
# > with(d_without_cuba, tapply(Laboagr, clusters, mean))
# 1        2        3        4        5 
# 53.68182 11.00000 68.50000 27.50000 59.50000 
# > with(d_without_cuba, tapply(Death, clusters, mean))
# 1           2           3           4           5 
# 77.2272727   0.1666667 831.5000000   0.5833333   3.7500000 
# > with(d_without_cuba, tapply(ecks, clusters, mean))
# 1         2         3         4         5 
# 27.272727  7.333333 51.500000 10.833333 29.250000

#Which clusters differ significantly from each other?


#install.packages("lsmeans")
#install.packages("multcompView")
library(lsmeans)
library("multcompView")

#The function cld() uses the Piepho (2004) algorithm 
#(as implemented in the multcompView package) 
#to generate a compact letter display of all pairwise comparisons of least-squares means.
#The function obtains (possibly adjusted) P values
#for all pairwise comparisons of means, using the contrast function with method = "pairwise".
#When a P value exceeds alpha, then the two means have at least one letter in common.

m1.lm<- lm(Gini ~ clusters, d_without_cuba)
cld(lsmeans(m1.lm, specs = "clusters"))
# clusters   lsmean       SE df lower.CL upper.CL .group
# 4        55.92500 2.856121 41 50.15695 61.69305  1    
# 5        59.77500 4.946946 41 49.78444 69.76556  12   
# 2        71.81667 4.039164 41 63.65941 79.97392   23  
# 3        80.45000 6.996038 41 66.32121 94.57879   23  
# 1        80.59545 2.109385 41 76.33547 84.85544    3  

m1.lm<- lm(Gnpr ~ clusters, d_without_cuba)
cld(lsmeans(m1.lm, specs = "clusters"))
# clusters    lsmean        SE df  lower.CL  upper.CL .group
# 3          99.5000 217.49566 41 -339.7414  538.7414  1    
# 5         188.7500 153.79265 41 -121.8406  499.3406  1    
# 1         287.6818  65.57741 41  155.2456  420.1181  1    
# 4         925.2500  88.79223 41  745.9305 1104.5695   2   
# 2        1256.3333 125.57118 41 1002.7372 1509.9295   2

m1.lm<- lm(Laboagr ~ clusters, d_without_cuba)
cld(lsmeans(m1.lm, specs = "clusters"))
# clusters   lsmean       SE df   lower.CL upper.CL .group
# 2        11.00000 5.692411 41 -0.4960581 22.49606  1    
# 4        27.50000 4.025143 41 19.3710594 35.62894  1    
# 1        53.68182 2.972765 41 47.6781977 59.68544   2   
# 5        59.50000 6.971752 41 45.4202618 73.57974   2   
# 3        68.50000 9.859546 41 48.5882433 88.41176   2

m1.lm<- lm(Death ~ clusters, d_without_cuba)
cld(lsmeans(m1.lm, specs = "clusters"))
# clusters      lsmean       SE df  lower.CL  upper.CL .group
# 2          0.1666667 36.11256 41 -72.76414  73.09747  1    
# 4          0.5833333 25.53544 41 -50.98653  52.15320  1    
# 5          3.7500000 44.22868 41 -85.57163  93.07163  1    
# 1         77.2272727 18.85917 41  39.14040 115.31414  1    
# 3        831.5000000 62.54880 41 705.18014 957.81986   2

m1.lm<- lm(ecks ~ clusters, d_without_cuba)
cld(lsmeans(m1.lm, specs = "clusters"))
# clusters    lsmean        SE df   lower.CL upper.CL .group
# 2         7.333333  7.336459 41 -7.4829465 22.14961  1    
# 4        10.833333  5.187660 41  0.3566414 21.31003  1    
# 1        27.272727  3.831341 41 19.5351775 35.01028  12   
# 5        29.250000  8.985291 41 11.1038373 47.39616  12   
# 3        51.500000 12.707120 41 25.8374506 77.16255   2   













#the loadings of the inputs on the components
Psi<-as.matrix(d.pc)

#TODO Where to I get the scores from? 
#TODO How do I project the individuals onto these new components?
#loadings of the individuals
#scores<-m1.pls2$scores[,1:nc]

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
