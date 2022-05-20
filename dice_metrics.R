#Dice functions:monoclass------------------------------------------
dice <- custom_metric("dice", function(y_true, y_pred, smooth=1.0){
  intersection = k_sum(y_true * y_pred, axis=c(1,2,3))
  union = k_sum(y_true, axis=c(1,2,3)) + k_sum(y_pred, axis=c(1,2,3))
  d = k_mean((2 * intersection + smooth)/(union + smooth), axis=0)
  return(d)
})
#
soc <- custom_metric("soc", function(y_true, y_pred, smooth=1.0){
  intersection = k_sum(k_abs(y_true * y_pred), axis=c(1,2,3))
  union = k_sum(y_true, axis=c(1,2,3)) + k_sum(y_pred, axis=c(1,2,3)) -intersection
  d = k_mean((intersection + smooth)/(union + smooth), axis=0)
  return(d)
})
#
dice_loss <- function(y_true, y_pred, smooth = 1.0){
  intersection = k_sum(y_true * y_pred, axis=c(1,2,3))
  union = k_sum(y_true, axis=c(1,2,3)) + k_sum(y_pred, axis=c(1,2,3))
  l=1-(k_mean((2 * intersection + smooth)/(union + smooth), axis=0))
  return(l)
}
#
bce_dice_loss <- function(y_true, y_pred, smooth = 1.0){
  intersection = k_sum(y_true * y_pred, axis=c(1,2,3))
  union = k_sum(y_true, axis=c(1,2,3)) + k_sum(y_pred, axis=c(1,2,3))
  l=keras::loss_binary_crossentropy(y_true, y_pred)+(1-(k_mean((2 * intersection + smooth)/(union + smooth), axis=0)))
  return(l)
}
#
bce_soc_loss <- function(y_true, y_pred, smooth = 1.0){
  intersection = k_sum(k_abs(y_true * y_pred), axis=c(1,2,3))
  union = k_sum(y_true, axis=c(1,2,3)) + k_sum(y_pred, axis=c(1,2,3)) -intersection
  l=keras::loss_binary_crossentropy(y_true, y_pred)+(1-(k_mean((intersection + smooth)/(union + smooth), axis=0)))
  return(l)
}
#Dice functions:multiclass-----------------------------------------
dice_multi<-custom_metric("dice_multi",function(y_true, y_pred,smooth=1.0, n=2L){
  d=0;for(i in 1:n){
    intersection = k_sum(y_true[,,,i,drop=FALSE] * y_pred[,,,i,drop=FALSE], axis=c(1,2,3))
    union = k_sum(y_true[,,,i,drop=FALSE], axis=c(1,2,3)) + k_sum(y_pred[,,,i,drop=FALSE], axis=c(1,2,3))
    d = d+(k_mean((2 * intersection + smooth)/(union + smooth), axis=0))
  }
  return(d/n)
})
#
dice_multi_loss <- function(y_true, y_pred,smooth=1.0, n=2L){
  l=0
  for(i in 1:n){
    intersection = k_sum(y_true[,,,i,drop=FALSE] * y_pred[,,,i,drop=FALSE], axis=c(1,2,3))
    union = k_sum(y_true[,,,i,drop=FALSE], axis=c(1,2,3)) + k_sum(y_pred[,,,i,drop=FALSE], axis=c(1,2,3))
    l=l+(1-(k_mean((2 * intersection + smooth)/(union + smooth), axis=0)))
  }
  return(l/n)
}
#
cce_dice_loss <- function(y_true, y_pred, smooth = 1.0,n=2L){
  l=0
  for(i in 1:n){
    intersection = k_sum(y_true[,,,i,drop=FALSE] * y_pred[,,,i,drop=FALSE], axis=c(1,2,3))
    union = k_sum(y_true[,,,i,drop=FALSE], axis=c(1,2,3)) + k_sum(y_pred[,,,i,drop=FALSE], axis=c(1,2,3))
    l=l+(keras::loss_binary_crossentropy(y_true[,,,i,drop=FALSE], y_pred[,,,i,drop=FALSE])+(1-(k_mean((2 * intersection + smooth)/(union + smooth), axis=0))))
  }
  return(l/n)
}
