library(tensorflow)
library(keras)
library(tfdatasets)
library(unet)
library(EBImage)
#
tf$config$run_functions_eagerly(TRUE)

#=============================================================================#
#CREATE GENERATORS
#=============================================================================#
load('U2OS.RData')

#--------------------------------------------------------------------------------
im.dir = list.dirs('PATCHS',recursive=F,full.names=T)  
filenames = lapply(im.dir,function(x)naturalsort::naturalsort(list.files(x,full.names = TRUE)));names(filenames)=c('mk','raw') #remove "head" for full data set
#
ratio = 0.1
sind = tail(1:length(filenames[[1]]),floor(length(filenames[[1]])*ratio))
sind = list(tr=setdiff(1:length(filenames[[1]]),sind),val=sind)
#
tr.dataset = tf$data$Dataset$from_tensor_slices(tensors = list(filenames=lapply(filenames,function(x)x[sind$tr])))
val.dataset = tf$data$Dataset$from_tensor_slices(tensors = list(filenames=lapply(filenames,function(x)x[sind$val])))
#
parse_ds = function(tensors){
  #Original images
  ims.raw = tf$io$read_file(tensors$filenames$raw)
  image.raw = tf$image$decode_png(ims.raw,channels = 1)%>%tf$reshape(shape=c(sF,sF,1L))
  #Bind masks
  ims.mk = tf$io$read_file(tensors$filenames$mk)
  image.mk = tf$image$decode_png(ims.mk,channels = 3)%>%tf$slice(begin=list(0L,0L,0L),size=list(sF,sF,2L))
  #
  return(list(image=image.raw,label=image.mk))
}
train_ds = tr.dataset$map(parse_ds)
val_ds = val.dataset$map(parse_ds)

#--------------------------------------------------------------------------------
augment = function(images){
  images = tf$image$random_flip_left_right(images)
  images = tf$image$random_flip_up_down(images)
  images = tf$image$rot90(images)
  return(images)
}
#
preprocess <- function(record, augment=T) {
  record$image <- tf$cast(tf$math$divide(record$image,255L), tf$float32)
  record$label <- tf$cast(tf$math$divide(record$label,255L), tf$float32)
  record = tuple(record$image, record$label)
  if(augment){
  record = tf$concat(record,axis=2L)%>%augment()%>%tf$unstack(axis=2L)
  for(i in 1:length(record)){record[[i]]=tf$reshape(record[[i]],shape=c(sF,sF,1L))}
  record = tuple(record[[1]],tf$concat(record[2:3],axis=2L))
  }
  return(record)
}
#
batch_size = 4
#
train = train_ds %>%
  dataset_map(function(x)preprocess(x,augment=T)) %>%
  #dataset_shuffle(100) %>% #no shuffling with untiled images
  dataset_batch(batch_size)

val = val_ds %>%
  dataset_map(function(x)preprocess(x,augment=F)) %>%
  dataset_batch(batch_size)

#=============================================================================#
#CREATE FCNN AN TRAIN IT
#=============================================================================#

#--------------------------------------------------------------------#
#Define model
#--------------------------------------------------------------------#
              
source('FUNCS/dice_metrics.R')
model = unet(input_shape = c(sF, sF, 1),num_classes = 2,filters = 32, num_layers = 3,dropout=0.5,output_activation='sigmoid')
#
model %>% compile(
  optimizer = optimizer_adam(lr = 1e-4),
  loss = cce_dice_loss,
  metrics = dice_multi
)

#--------------------------------------------------------------------#
#Teach model
#--------------------------------------------------------------------#

learn = model %>% fit(train,validation_data=val,epochs=30,workers=12)
#--
model %>% save_model_hdf5("tl512d-ncm.h5")

#--------------------------------------------------------------------#
#Check model
#-------------------------------------------------- ------------------#

fp = naturalsort::naturalsort(filenames$raw)[1]
im1 = readImage(fp)
pred1 = model%>%predict(abind(abind(im1,along=3),along=0))
#
display(pred[1,,,])





