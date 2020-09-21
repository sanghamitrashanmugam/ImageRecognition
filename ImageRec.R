


if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("EBImage")


library(EBImage)
library(keras)
library(tensorflow)

install_tensorflow()


pics<-c('b1.jpg','b2.jpg','b3.jpg','b4.jpg','b5.jpg','b6.jpg','b7.jpg','b8.jpg','b9.jpg','b10.jpg','b11.jpg','b12.jpg','b13.jpg','b14.jpg','b15.jpg',
        'c1.jpg','c2.jpg','c3.jpg','c4.jpg','c5.jpg','c6.jpg','c7.jpg','c8.jpg','c9.jpg','c10.jpg','c11.jpg','c12.jpg','c13.jpg','c14.jpg','c15.jpg')


mypics<-list()

for ( i in 1:30) {mypics[[i]]<- readImage(pics[i])}

#Exploring the images

print(mypics[[1]])

display(mypics[[8]])

summary(mypics[[1]])

hist(mypics[[12]])

str(mypics)


# to ensure same dimensions we resize

for (i in 1:30) {mypics[[i]] <- resize(mypics[[i]], 28,28)}


# restructure the data

for (i in 1:30){mypics[[i]]<-array_reshape(mypics[[i]],c(28,28,3))}

str(mypics)


# split the data into train test and sample sets


# train_x contains the predictors 
train_x<-NULL



#add the bike images to train_x

for (i in 1:9){train_x<-rbind(train_x,mypics[[i]])}

# add the car images to train_x

for (i in 16:24){train_x<-rbind(train_x,mypics[[i]])}

str(train_x)


#test_x contains the predictors for the test set

test_x<-NULL

# add the bike images to the test set

for (i in 10:12){test_x<-rbind(test_x,mypics[[i]])}

# add the car images to the test set

for (i in 25:27){test_x<-rbind(test_x,mypics[[i]])}


# add the bike images to the sample set

sample<-NULL

for (i in 13:15){sample<-rbind(sample,mypics[[i]])}

# add the car images to the sample set

for (i in 28:30){sample<-rbind(sample,mypics[[i]])}



# the response variable for train set is stored as train_y

# 0 for bike, 1 for car 

train_y <- c(0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1)

test_y  <- c(0,0,0,1,1,1)


# creating dummy variables

trainLabels<-to_categorical(train_y)
testLabels <-to_categorical(test_y)


# creating the model

model <- keras_model_sequential()

model %>%
  layer_dense(units = 256,activation = 'relu', input_shape =c(2352)) %>% # first hidden layer 
  layer_dense(units = 128,activation = 'relu') %>%  # second hidden layer
  layer_dense(units = 2,activation = 'softmax') # output layer


summary(model)

model %>% 
  compile(loss = 'binary_crossentropy',
          optimizer = optimizer_rmsprop(),
          metrics= c('accuracy'))


# Fit Model

model_1<-model%>%
  fit(train_x,
      trainLabels,
      epochs = 50,
      batch_size = 32,
      validation_split = 0.2)


#model Evaluation and prediction - train data

model %>% evaluate(train_x,trainLabels)

pred <- model%>% predict_classes(train_x)

table(Predicted = pred, Actual = train_y)

prob <- model %>% predict_proba(train_x)


cbind(prob,Predicted=pred,Actual = train_y)



# Evaluation & Prediction - test data

model %>% evaluate(test_x,testLabels)

pred <- model %>% predict_classes(test_x)

table(Predicted = pred, Actual = test_y)



# prediction on the sample

actual_sample<-c(0,0,0,1,1,1)

pred<-model%>%predict_classes(sample)

table(Predicted = pred, Actual = actual_sample)


accuracy(pred,actual_sample)


library(Metrics)
