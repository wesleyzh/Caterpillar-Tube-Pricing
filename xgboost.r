# This is implementation of XGboost model in R


# library required

library(data.table)
library(xgboost)
library(Matrix)
library(methods)

# you must know why I am using set.seed()
set.seed(546)


# Importing data into R

train  <- read.csv("../input/train_set.csv",header = T)
test  <- read.csv("../input/test_set.csv",header = T)
bom  <- read.csv("../input/bill_of_materials.csv",header = T)
specs  <- read.csv("../input/specs.csv",header = T)
tube  <- read.csv("../input/tube.csv",header = T)

# Merging the data

train$id  <- -(1:nrow(train))
test$cost  <- 0

data  <- rbind(train,test)

data  <- merge(data,tube,by="tube_assembly_id",all = T)
data  <- merge(data,bom,by="tube_assembly_id",all = T)
data  <- merge(data,specs,by="tube_assembly_id",all = T)

# extracting year and month for quote_date

data$quote_date  <- strptime(data$quote_date,format = "%Y-%m-%d", tz="GMT")
data$year <- year(as.IDate(data$quote_date))
data$month <- month(as.IDate(data$quote_date))
data$week <- week(as.IDate(data$quote_date))

# dropping variables
data$quote_date  <- NULL
data$tube_assembly_id  <- NULL


# converting NA in to '0' and '" "' for mode Matrix Generation

for(i in 1:ncol(data)){
  if(is.numeric(data[,i])){
    data[is.na(data[,i]),i] = 0
  }else{
    data[,i] = as.character(data[,i])
    data[is.na(data[,i]),i] = " "
    data[,i] = as.factor(data[,i])
  }
}


# converting data.frame to sparse matrix for modelling

train  <- data[which(data$id < 0), ]
test  <- data[which(data$id > 0), ]

ids  <- test$id
cost  <- train$cost

#dropping some more variables

train$id  <- NULL 
test$id  <- NULL
#train$cost  <- 0
test$cost  <- NULL


# this is a very crude way of generating sparse matrix and might take a bit time
# if anybody has a better way feel free to comment;)

tr.mf  <- model.frame(as.formula(paste("cost ~",paste(names(train),collapse = "+"))),train)
tr.m  <- model.matrix(attr(tr.mf,"terms"),data = train)
tr  <- Matrix(tr.m)
t(tr)


te.mf  <- model.frame(as.formula(paste("~",paste(names(test),collapse = "+"))),test)
te.m  <- model.matrix(attr(te.mf,"terms"),data = test)
te  <- Matrix(te.m)
t(te)



# generating xgboost model 

# tr.x  <- xgb.DMatrix(tr,lable=log(names(train)+1))
cost.log  <- log(cost+1) # treating cost as log transfromation is working good on this data set

tr.x  <- xgb.DMatrix(tr,label = cost.log)
te.x  <- xgb.DMatrix(te)


# parameter selection
par  <-  list(booster = "gblinear",
              objective = "reg:linear",
              min_child_weight = 6,
              gamma = 2,
              subsample = 0.85,
              colsample_bytree = 0.75,
              max_depth = 10,
              verbose = 1,
              scale_pos_weight = 1)


#selecting number of Rounds
n_rounds= 200


#modeling

x.mod.t  <- xgb.train(params = par, data = tr.x , nrounds = n_rounds)
pred  <- predict(x.mod.t,te.x)
head(pred)

for(i in 1:50){
  x.mod.t  <- xgb.train(par,tr.x,n_rounds)
  pred  <- cbind(pred,predict(x.mod.t,te.x))
}

pred.sub  <- exp(rowMeans(pred))-1


# generating data frame for submission


sub.file = data.frame(id = ids, cost = pred.sub)
sub.file = aggregate(data.frame(cost = sub.file$cost), by = list(id = sub.file$id), mean)

write.csv(sub.file, "submit.csv", row.names = FALSE, quote = FALSE)
