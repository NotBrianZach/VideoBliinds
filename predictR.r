#! /usr/bin/Rscript --vanilla
 
# install.packages("kernalb")

rm(list=ls(all=TRUE))
library("kernlab")

features_test = as.matrix(read.table("features.txt"))
predicted_dmos=predict(svm_model,features)
write.table(predicted_dmos,file="predicted_dmos.txt",row.names=FALSE,col.names=FALSE)
