# dice <- custom_metric("dice", function(y_true, y_pred, smooth = 1.0) {
#   y_true_f <- k_flatten(y_true)
#   y_pred_f <- k_flatten(y_pred)
#   intersection <- k_sum(y_true_f * y_pred_f)
#   (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
# })
dice <- custom_metric("dice", function(y_true, y_pred, smooth=1.0){
  intersection = k_sum(y_true * y_pred, axis=c(1,2,3))
  union = k_sum(y_true, axis=c(1,2,3)) + k_sum(y_pred, axis=c(1,2,3))
  d = k_mean((2 * intersection + smooth)/(union + smooth), axis=0)
  return(d)
})