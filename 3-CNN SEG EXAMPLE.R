library(EBImage)
library(RBioFormats)
library(MetaxpR)
library(MorphR)
###
library(doParallel)
library(pbapply)
library(magrittr)
###
library(tensorflow)
library(keras)
library(unet)
#=================================================================================================================================================================#
#CONFIGURE GPU
#=================================================================================================================================================================#
config = tf$compat$v1$ConfigProto(gpu_options = tf$compat$v1$GPUOptions(allow_growth=TRUE),
                                  allow_soft_placement=TRUE,
                                  log_device_placement=FALSE,
                                  device_count = dict('GPU', 1))

sess = tf$compat$v1$Session(config=config)

#=================================================================================================================================================================#
#GATHER ALL PLATE INFORMATION
#=================================================================================================================================================================#
SERVER = 'NEO-SERVER'
DB = GetMDCInfo(SERVER)
#-----
PID = c(911);names(PID)=PID 
PLT = DB[sapply(PID,function(p)tail(which(DB$PlateID==p),1)),]
PIF = lapply(PID,function(p)GetPINFO(SERVER,PlateID=p))
alls = lapply(PIF,function(x)lapply(1:nrow(x$SID),function(i)x$SID[i,]))
#-----
INF = lapply(PID,function(p)lapply(GetIMPath(SERVER,PlateID=p,WellID=PIF[[as.character(p)]]$WID[1],SiteID=alls[[as.character(p)]][[1]]),function(x)read.metadata(x)))
cn = lapply(INF,function(x){ci=sapply(x,function(y)y$globalMetadata$'image-name')});print(cn)
cn = apply(do.call('cbind',cn),1,unique);print(cn) #If all cn are the same
names(cn)=cn
Di = min(sapply(INF,function(x)x[[1]]$coreMetadata$sizeX))
#----
PM.loc = '/media/Hcs-screen10-vi/D/Platemap/'
PM = lapply(PID,function(p)subset(read.csv(paste0(PM.loc,SERVER,'/',p,'.csv'),sep=',',stringsAsFactors = F),Drug!=''))
#-----
rgp = lapply(PID,function(p){
  woi = intersect(subset(PM[[as.character(p)]],Drug=='DMSO')$WellHeader,PIF[[as.character(p)]]$WID) 
  if(length(woi)>5){woi = woi[sample(1:length(woi),5,replace=T)]}
  IM = lapply(seq_along(cn),function(i)readImage(sapply(woi,function(w)GetIMPath(SERVER,WellID=w,SiteID=alls[[as.character(p)]][[1]],PlateID=p)[i])))
  rg = lapply(IM,function(x){qu=quantile(x,probs=seq(0,1,5*10**-4));return(qu[c(2,length(qu)-1)])});names(rg)=cn
  return(rg)
})
#----
nCores=16
woi = lapply(PID,function(p){
  id = c(seq(1,length(PIF[[as.character(p)]]$WID),by=16),length(PIF[[as.character(p)]]$WID)+1)
  lapply(1:(length(id)-1),function(i){sort(PIF[[as.character(p)]]$WID)[id[i]:(id[i+1]-1)]})
}) #Forced to split Wells in "Well sets" not to saturate RAM
#----
sF = 128 # Size of image units
save.image('PDATA.RData')
#=================================================================================================================================================================#
#LOAD MODEL
#=================================================================================================================================================================#

source('/media/Hcs-screen10-vi/Q/Allan/DeepLearning/CNN-Segmentation/NUCYTO-128/DL-TRAIN/dice_metrics.R')
model = load_model_hdf5('/media/Hcs-screen10-vi/Q/Allan/DeepLearning/CNN-Segmentation/NUCYTO-128/DL-TRAIN/NuCytoFromTL.h5',c("dice" = dice))
#-------------------#
nt = (Di/sF)**2 #Number of tiles composing each image
dir.create('SEGS');sapply(PID,function(p){dir.create(paste0('SEGS/',p))})

#=================================================================================================================================================================#
#PROCESS
#=================================================================================================================================================================#
for(p in PID){
  ct=1
  for(wset in woi[[as.character(p)]]){
    #IMAGES BATCH SPLIT#
    ####################
    cl = makeCluster(nCores)
    clusterExport(cl,setdiff(ls(),'DB'))
    invisible({clusterEvalQ(cl,{library(MetaxpR);library(EBImage);library(magrittr);library(MorphR);library(S4Vectors)})})
    print(paste0('Splitting images, Plate #',p,' well batch ',ct,'/',length(woi[[as.character(p)]]),'...'))
    pdat = pblapply(wset,function(w){
      wdat = lapply(alls[[as.character(p)]],function(sxy){
        #Read images ------------------------------ ----------------------------
        IP = GetIMPath(SERVER,PlateID=p,WellID=w,SiteID=sxy);names(IP)=cn
        IM = readImage(IP['TL 50'])
        
        #Adjust histograms-----------------------------------------------------
        IN = normalize(IM,inputRange=rgp[[as.character(p)]][['TL 50']])
        
        #untile---------------------------------------------------------------
        TL = untile(IN,nim=rep(Di/sF,2),lwd=0) #Splited image in blocks of size 128
        return(TL)
      })
      wdat = do.call('abind',list(wdat,along=3))
      gc()
      return(wdat)
    },cl=cl)
    stopCluster(cl);rm(cl)
    pdat = do.call('abind',list(pdat,along=3))
    pdat = do.call(abind,list(lapply(1:dim(pdat)[3],function(i)pdat[,,i]),along=0))
    
    #PREDICTION#
    ############
    print(paste0('Predicting masks from images, Plate #',p,' well batch ',ct,'/',length(woi[[as.character(p)]]),'...'))
    pdat = predict(model,do.call('abind',list(pdat,along=4)))#Create OverallMask
    gc()
    
    #RECONSTITUTION#
    ################
    print(paste0('Reconstituting masks, Plate #',p,' well batch ',ct,'/',length(woi[[as.character(p)]]),'...'))
    Ii = c(seq(1,nrow(pdat),by=nt),nrow(pdat)) #Image indexes in matrix
    pdat = lapply(1:(length(Ii)-1),function(id){
      N=tile(do.call('abind',list(lapply(Ii[id]:(Ii[id+1]-1),function(i){pdat[i,,,1]}),along=3)),nx=Di/sF,lwd=0)
      C=tile(do.call('abind',list(lapply(Ii[id]:(Ii[id+1]-1),function(i){pdat[i,,,2]}),along=3)),nx=Di/sF,lwd=0)
      return(list(N=N,C=C))
    });gc()
    #Labels
    labs = unlist(lapply(wset,function(w){
      sapply(alls[[as.character(p)]],function(sxy){
        nm=paste0(w,'_sx',sxy[1],'_sy',sxy[2])
        return(nm)
      })
    }))
    names(pdat)=labs
    
    #ADDITIONNAL PROCESSING
    #######################
    print(paste0('Post-processing masks, Plate #',p,' well batch ',ct,'/',length(woi[[as.character(p)]]),'...'))
    #Process and separate nuclei
    cl=makeCluster(nCores)
    pdat = pblapply(pdat,function(IMi){
      library(EBImage);library(MorphR);library(magrittr)
      NM = IMi$N%>%medianFilter(size=5)%>%sigmoNormalize(scaled=T,z0=0.75,lambda=10**3);NM=NM>otsu(NM)
      NM = propagate(NM,opening(NM,makeBrush(21,'disc')),NM)
      NM = watershed(distmap(NM),tolerance=3)
      #
      CM = pmax(IMi$N,IMi$C)%>%medianFilter(size=5)%>%sigmoNormalize(scaled=T,z0=0.7,lambda=10**3)
      CM = CM>otsu(CM);CM = closing(CM,makeBrush(5,'disc'))
      CM = propagate(CM,opening(CM,makeBrush(41,'disc')),CM)
      CM = propagate(CM,NM,CM)
      return(list(NM=NM,CM=CM))
    },cl=cl)
    stopCluster(cl);rm(cl)
    gc()
    
    #EXPORT#
    ########
    print(paste0('Exporting masks, Plate #',p,' well batch ',ct,'/',length(woi[[as.character(p)]]),'...'))
    pbsapply(names(pdat),function(nm){
      SaveSeg(pdat[[nm]]$N,ExportDir=paste0('SEGS/',p),FileName=paste0('N_',nm))
      SaveSeg(pdat[[nm]]$C,ExportDir=paste0('SEGS/',p),FileName=paste0('C_',nm))
    })
    rm(pdat);gc()
    ct=ct+1
  }
};rm(p,ct);gc()


