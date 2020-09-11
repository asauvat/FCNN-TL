library(EBImage)
library(RBioFormats)
library(MetaxpR)
library(MorphR)
###
library(MetaxpR)
library(MorphR)
library(EBImage)
library(RBioFormats)
#
library(doParallel)
library(pbapply)
#
library(magrittr)

#=================================================================================================================================================================#
#GATHER ALL PLATE INFORMATION
#=================================================================================================================================================================#
SERVER = 'NEO-SERVER'
DB = GetMDCInfo(SERVER)
#-----
PID = c(906);names(PID)=PID 
PLT = DB[sapply(PID,function(p)tail(which(DB$PlateID==p),1)),]
PIF = lapply(PID,function(p)GetPINFO(SERVER,PlateID=p))
alls = lapply(PIF,function(x)lapply(1:nrow(x$SID),function(i)x$SID[i,]))
#-----
INF = lapply(PID,function(p)lapply(GetIMPath(SERVER,PlateID=p,WellID=PIF[[as.character(p)]]$WID[1],SiteID=alls[[as.character(p)]][[1]]),function(x)read.metadata(x)))
cn = lapply(INF,function(x){ci=sapply(x,function(y)y$globalMetadata$'image-name')});print(cn)
cn = apply(do.call('cbind',cn),1,unique);print(cn) #If all cn are the same
names(cn)=cn
Di = min(sapply(INF,function(x)x[[1]]$coreMetadata$sizeX))
#-----
rgp = lapply(PID,function(p){
  woi = c('A01','O01')
  IM = lapply(seq_along(cn),function(i)readImage(sapply(woi,function(w)GetIMPath(SERVER,WellID=w,SiteID=alls[[as.character(p)]][[1]],PlateID=p)[i])))
  rg = lapply(IM,function(x){qu=quantile(x,probs=seq(0,1,5*10**-4));return(qu[c(2,length(qu)-1)])});names(rg)=cn
  return(rg)
})
#----
sF = 128 # Size of image units: maybe using 256 will yield better results !
save.image('TRAIN-DATA.RData')

#=================================================================================================================================================================#
#CREATE TABLE, ONLY NULCEI for now
#=================================================================================================================================================================#

nCores=15
dat = list()
for(p in PID){
  cl = makeCluster(nCores)
  clusterExport(cl,setdiff(ls(),'DB'))
  invisible({clusterEvalQ(cl,{library(MetaxpR);library(EBImage);library(magrittr);library(MorphR);library(S4Vectors)})})
  pdat = pblapply(PIF[[as.character(p)]]$WID,function(w){
    wdat = lapply(alls[[as.character(p)]],function(sxy){
      #Read images ------------------------------ ----------------------------
      IP = GetIMPath(SERVER,PlateID=p,WellID=w,SiteID=sxy);names(IP)=cn
      IM = lapply(IP,function(x)suppressWarnings(readImage(x)))
      
      #Adjust histograms-----------------------------------------------------
      IN = lapply(cn,function(x)normalize(IM[[x]],inputRange=rgp[[as.character(p)]][[x]]))
      
      #Nuclei segmentation---------------------------------------------------
      NG = autoNormalize(IN$DAPI,step=5*10**-4) #autoNormalize required because of AF
      NG = NG-gblur(NG,50);NG[which(NG<0)]=0
      #
      NM = sigmoNormalize(NG,scaled=T,z0=5.5*10**-2,lambda=10**3)
      NM = NM>(otsu(NM)*1)
      NM = fillHull(closing(opening(NM,makeBrush(5,'disc')),makeBrush(3,'disc')))
      #
      NM = propagate(NM,opening(NM,makeBrush(25,'disc')),NM)
      #NM = watershed(distmap(NM),tolerance=3,ext=1)
      #display(paintObjects(NM,toRGB(NG)))
      
      #CYTOPLASM-------------------------------------------------------------
      CG = autoNormalize(LowPass(IN$FITC))
      #
      CM = sigmoNormalize(CG,scaled=T,z0=5.5*10**-2,lambda=10**3) 
      CM = pmax(CM,NM)%>%medianFilter(size=3)
      CM = CM>(otsu(CM)*1)
      CM = closing(CM,makeBrush(3,'disc'))%>%opening(kern=makeBrush(7,'disc'))
      CM = propagate(CM,opening(CM,makeBrush(41,'disc')),CM)
      #
      CYM = CM&!NM
      #display(paintObjects(CYM,toRGB(IN$FITC)))
      
      #Export--------------------------------------------------------------
      TL = untile(IN$`TL 50`,nim=rep(Di/sF,2),lwd=0) #Splited image in blocks of size 256
      MKn = untile(NM,nim=rep(Di/sF,2),lwd=0) #Splitted mask
      MKc = untile(CYM,nim=rep(Di/sF,2),lwd=0) #Splitted mask
      return(abind(TL,MKn,MKc,along=4))
    })
    wdat = do.call('abind',list(wdat,along=3))
    gc()
    return(wdat)
  },cl=cl)
  stopCluster(cl);rm(cl)
  pdat = do.call('abind',list(pdat,along=3))
  dat = c(dat,list(pdat));rm(pdat)
};rm(p);gc()
names(dat)=PID
#=================================================================================================================================================================#
#CREATE TABLE WITH IMAGE INFORMATION
#=================================================================================================================================================================#
dir.create('RAW')
#
dat = do.call('abind',list(dat,along=3))
dat = list(img=dat[,,,1,drop=F],mask=dat[,,,2:3])
dat = lapply(dat,function(x){
  y = lapply(1:dim(x)[3],function(i)x[,,i,])
  y = do.call('abind',list(y,along=0))
  if(length(dim(y))==3)y=do.call('abind',list(y,along=4))
  return(y)
})
#
saveRDS(dat,'RAW/dat.Rds')

