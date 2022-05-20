library(EBImage)
library(RBioFormats)
library(MetaxpR)
library(MorphR)
#
library(doParallel)
library(pbapply)
library(magrittr)

#=================================================================================================================================================================#
#GATHER ALL PLATE INFORMATION
#=================================================================================================================================================================#
SERVER = 'HTS-SERVER' #MDCstore database
DB = GetMDCInfo(SERVER) #Plate records
#-----
PLT = subset(DB,grepl('TM-screening-02',PlateName))
PID = PLT$PlateID;names(PID)=PID #Extract plate IDs with image of interest
#-----
PIF = lapply(PID,function(p)GetPINFO(SERVER,PlateID=p)) #Plate acquisition information
#-----
INF = lapply(PID,function(p)lapply(GetIMPath(SERVER,PlateID=p,WellID=PIF[[as.character(p)]]$WID[1],SiteID=PIF[[as.character(p)]]$SID[1,]),function(x)read.metadata(x)))
cn = lapply(INF,function(x){ci=sapply(x,function(y)y$globalMetadata$'image-name')});print(cn)
cn = apply(do.call('cbind',cn),1,unique);print(cn) #If all cn are the same
names(cn)=cn
Di = min(sapply(INF,function(x)x[[1]]$coreMetadata$sizeX)) #image dimension
#-----
PM.loc = '/media/Hcs-screen10-vi/D/Platemap/' #plate layout location
PM = lapply(PID,function(p)read.csv(paste0(PM.loc,SERVER,'/',p,'.csv'),stringsAsFactors = F)) #plate maps
#-----
rgp = pblapply(PID,function(p){
  woi = subset(PM[[as.character(p)]],NAME=='CM')$WellID;if(length(woi)>4){woi=c(head(woi,2),tail(woi,2))}
  IM = lapply(seq_along(cn),function(i)readImage(sapply(woi,function(w)GetIMPath(SERVER,WellID=w,SiteID=PIF[[as.character(p)]]$SID[1,],PlateID=p)[i])))
  rg = lapply(IM,function(x){qu=quantile(x,probs=seq(0,1,5*10**-4));return(qu[c(2,length(qu)-1)])});names(rg)=cn #image optimal depth range
  return(rg)
}) 
#----
woi = lapply(PID,function(p){tapply(PM[[as.character(p)]]$WellID,PM[[as.character(p)]]$NAME,function(x)head(x,n=1))%>%sort()}) #Only one well/image per condition
#----
sF = 512L # Reduced image dimension (earns memory, decreases precision) 
save.image('U2OS.RData') #Save training set plate info

#=================================================================================================================================================================#
#CREATE PATCHS: RAW, NUC, CYTO
#=================================================================================================================================================================#

dir.create('PATCHS');sapply(c('RAW','MASK'),function(x){dir.create(paste0('PATCHS/',x))})
#
mywd = getwd()
nCores = detectCores()
                             #
for(p in PID){ #Can remove limitation to get more samples
  cl = makeCluster(nCores)
  clusterExport(cl,setdiff(ls(),'DB'))
  invisible({clusterEvalQ(cl,{library(MetaxpR);library(EBImage);library(magrittr);library(MorphR);library(S4Vectors)})})
  pbsapply(woi[[as.character(p)]],function(w){
    apply(PIF[[as.character(p)]]$SID,1,function(sxy){
    #Read images ------------------------------ ----------------------------
    IP = GetIMPath(SERVER,PlateID=p,WellID=w,SiteID=sxy);names(IP)=cn
    IM = lapply(IP,function(x)suppressWarnings(readImage(x)))
    
    #Adjust histograms-----------------------------------------------------
    IN = lapply(cn,function(x)normalize(IM[[x]],inputRange=rgp[[as.character(p)]][[x]]))
    
    #Nuclear mask creation-------------------------------------------------
    NG = whiteTopHat(IN$DAPI,makeBrush(77,'disc',step=F)) #image background removal
    #
    NM = sigmoNormalize(NG,scaled=T,z0=9.25*10**-2,lambda=10**3)>0.95
    NM = closing(NM,kern=makeBrush(3,'disc'))%>%opening(kern=makeBrush(5,'disc'))%>%opening(kern=makeBrush(11,'disc'))
    NM = propagate(NM,opening(NM,makeBrush(27,'disc')),NM) 
    #
    #display(paintObjects(NM,toRGB(IN$DAPI)))
    
    #Cytoplasm mask creation------------------------------------------------
    CG = pmax(autoNormalize(LowPass(IN$FITC)),autoNormalize(LowPass(IN$`Texas Red`))) #if 2 cytoplasmic markers
    CM = sigmoNormalize(CG,scaled=T,z0=6*10**-2,lambda=10**3)>0.95
    CM = CM|dilate(NM>0,makeBrush(5,'disc'))
    #
    CM = closing(CM,makeBrush(5,'disc'))%>%opening(kern=makeBrush(5,'disc'))
    CM = propagate(CM,opening(CM,makeBrush(41,'disc')),CM)
    #
    CM = CM-NM 
    
    #Split--------------------------------------------------------------
    TL = IN$`TL 25`%>%resize(w=sF) #input TL image adjusted and rescaled 
    MK = rgbImage(red=(NM%>%resize(w=sF))>0.9,green=(CM%>%resize(w=sF))>0.9) #mask stack
    
    #Export--------------------------------------------------------------
    writeImage(TL,paste0(mywd,'/PATCHS/RAW/',paste0(p,'_',w,'_',paste(sxy,collapse='-'),'.png')))
    writeImage(MK,paste0(mywd,'/PATCHS/MASK/',paste0(p,'_',w,'_',paste(sxy,collapse='-'),'.png')))
    #
    return(NULL)
    })
    return(NULL)
  },cl=cl)
  stopCluster(cl);rm(cl)
};rm(p);gc()

