library(reticulate)
library(tensorflow)
library(keras)
library(unet)
#
library(EBImage)
#
config = tf$compat$v1$ConfigProto(gpu_options = tf$compat$v1$GPUOptions(allow_growth=TRUE),
                                  allow_soft_placement=TRUE,
                                  log_device_placement=FALSE,
                                  device_count = dict('GPU', 1))

sess = tf$compat$v1$Session(config=config)
#
#---------------------------------------------------------------------#
load('TRAIN-DATA.RData')
dat = readRDS('RAW/dat.Rds')
#---------------------------------------------------------------------#
model = unet(input_shape = c(sF, sF, 1),num_classes = 2)
source('dice_metrics.R')
#
model %>% compile(
  #optimizer = optimizer_rmsprop(lr = 1e-5),
  optimizer = optimizer_adam(lr = 1e-4),
  loss = "binary_crossentropy",
  metrics = list(dice, metric_binary_accuracy)
)
#
# model %>% compile(
#   loss = 'categorical_crossentropy',
#   optimizer = 'adam',
#   metrics = c(dice,'accuracy')
# )

#--------------------------------------------------------------------#
#Teach model
#--------------------------------------------------------------------#
#Only take subset of data#
#it = sort(sample(1:nrow(dat$img),2500))
#learn = model %>% fit(x=dat$img[it,,,,drop=F],y=dat$mask[it,,,,drop=F],validation_split=0.25,epochs=100)
#rm(it)
#...Or everything!
learn = model %>% fit(x=dat$img,y=dat$mask,validation_split=0.2,epochs=30)
#--
#lomet = model %>% evaluate(dat$img, dat$mask)
#summary(model)
plot(learn)
#--
model %>% save_model_hdf5("NuCytoFromTL.h5")

#--------------------------------------------------------------------#
#Check model
#-------------------------------------------------- ------------------#

test = predict(model,dat$img[1:256,,,,drop=F])
test = list(N=test[,,,1],C=test[,,,2])
test = lapply(test,function(x){do.call('abind',list(lapply(1:nrow(x),function(i)x[i,,]),along=3))})
test = lapply(test,function(x)tile(x,nx=16,lwd=0))
#
testim = do.call('abind',list(lapply(1:256,function(i)dat$img[i,,,]),along=3))
testim = tile(testim,nx=16,lwd=0)
#
testmask = do.call('abind',list(lapply(1:256,function(i)dat$mask[i,,,]),along=3))
testmask = tile(testmask,nx=16,lwd=0)




