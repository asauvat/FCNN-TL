library(tensorflow)
library(keras)
tf$print('Tensorflow initialized')
###
library(EBImage)
library(RBioFormats)
library(MetaxpR)
library(MorphR)
###
library(doParallel)
library(pbapply)
library(magrittr)

#=================================================================================================================================================================#
#GATHER ALL PLATE INFORMATION
#=================================================================================================================================================================#
SERVER = 'NEO-SERVER'
DB = GetMDCInfo(SERVER)
#-----
PID = 21328;names(PID)=PID 
PLT = DB[tail(which(DB$PlateID==PID),1),]
PIF = GetPINFO(SERVER,PlateID=PID)
#-----
INF = lapply(GetIMPath(SERVER,PlateID=PID,WellID=PIF$WID[1],SiteID=PIF$SID[1,]),function(x)read.metadata(x)) #Read metadata
cn = sapply(INF,function(x)x$globalMetadata$'image-name');names(cn)=cn #Channel acquisition names
Di = INF[[1]]$coreMetadata$sizeX #Image dimension
#----
PM.loc = '/media/Hcs-screen10-vi/TEAM_DATA/Platemap/' #Platemap location
PM = subset(read.table(paste0(PM.loc,SERVER,'/',PID,'.csv'),sep=',',stringsAsFactors = F,header=T,encoding='latin1'),Drug!='') #Plate layout
#-----
nCores=32;nF=3
id = c(seq(1,length(PIF$WID),by=nCores*nF),length(PIF$WID)+1)
woi = lapply(1:(length(id)-1),function(i){sort(PIF$WID)[id[i]:(id[i+1]-1)]}) #split well in batches
rm(id)
#----
sF = 512L # Size of image after binning
save.image('PDATA.RData') #keep data stored

#=================================================================================================================================================================#
#LOAD MODEL
#=================================================================================================================================================================#
tr.dir = '/media/Hcs-screen10-vi/T/Allan/DEEPLEARNING/HDF5/' #Trained models location
#
source(paste0(tr.dir,'FUNCS/dice_metrics.R')) 
modsem = unet::unet(input_shape = c(sF,sF,1),num_classes = 2,filters=32,num_layers = 3)%>%load_model_weights_hdf5(paste0(tr.dir,'tl512d-ncm.h5')) #model import
#-------------------#
dir.create('SEGS');sapply(1:(PIF$TID[2]),function(ti){dir.create(paste0('SEGS/T',ti))}) #Folder where masks are going to be exported

#=================================================================================================================================================================#
#PROCESS
#=================================================================================================================================================================#
for(ti in seq_len(PIF$TID[2])){ #Loop over time
  ct=1
  for(wset in woi){ #Loop over well batches
    
    #IMAGES IMPORT & BINNING#
    ########################
    cl = makeCluster(nCores)
    clusterExport(cl,setdiff(ls(),'DB'))
    invisible({clusterEvalQ(cl,{library(MetaxpR);library(EBImage);library(magrittr);library(MorphR);library(S4Vectors)})})
    print(paste0('Importing & preprocessing images, TimePoint#',ti,' well batch ',ct,'/',length(woi),'...'))
    mw = pblapply(wset,function(w){ #Loop over well, multithread
      ms = apply(PIF$SID,1,function(sxy){ #Loop over sites
        IP = GetIMPath(SERVER,PlateID=PID,WellID=w,SiteID=sxy,TimePoint=ti) #Get image path
        IM = readImage(IP)%>%autoNormalize(step=5*10**-4)%>%resize(w=sF) #Read, normalize, and bin image
        return(IM)
      },simplify=F)
      ms = do.call('abind',list(ms,along=0))
      gc()
      return(ms)
    },cl=cl)
    stopCluster(cl);rm(cl)
    names(mw)=wset
    gc()
    
    #GPU-based SEMANTIC SEGMENTATION#
    #################################
    print(paste0('Predicting masks from images, TimePoint#',ti,' well batch ',ct,'/',length(woi),'...'))
    mw = pblapply(names(mw),function(w){pd = predict(modsem,mw[[w]],batch_size = 8);rownames(pd)=paste(w,1:nrow(pd),sep='/');return(pd)})
    names(mw) = wset
    gc()
    
    #PROCESSING RAW PROBABILITY MASKS#
    ##################################
    print(paste0('Post-processing masks, Timepoint#',ti,' well batch ',ct,'/',length(woi),'...'))
    cl=makeCluster(nCores)
    clusterExport(cl,'Di')
    mw = pblapply(mw,function(tsr){
      apply(tsr,1,function(IMi){
        library(EBImage);library(MorphR);library(magrittr)
        #
        NM = IMi[,,1]%>%whiteTopHat(makeBrush(21,'disc',step=F))%>%thresh(w=10,h=10,offset=0.25)
        NM = propagate(NM,opening(NM,makeBrush(7,'disc')),NM)
        NM = watershed(distmap(NM),tolerance=1)
        #
        CM = (pmax(IMi[,,1],IMi[,,2])%>%sigmoNormalize(z0=5*10**-2,lambda=10**3))>0.95
        CM = propagate(CM,opening(CM,makeBrush(11,'disc')),CM)
        CM = propagate(IMi[,,2],NM,CM)-NM
        #
        MK = abind(NM,CM,along=3)%>%resize(w=Di,filter='none',antialias=FALSE)
        return(MK)
      },simplify = FALSE)
    },cl=cl)
    stopCluster(cl);rm(cl)
    gc()
    
    #EXPORT REFINED MASKS#
    #####################
    print(paste0('Exporting Masks, Time#',ti,' well batch ',ct,'/',length(woi),'...'))
    cl=makeCluster(nCores)
    clusterExport(cl,c('SERVER','cn','PID','PIF','ti'))
    invisible({clusterEvalQ(cl,{library(MetaxpR);library(EBImage);library(magrittr);library(MorphR);library(S4Vectors)})})
    pbsapply(mw,function(tsr){
      w = strsplit(names(tsr),'/')[[1]][1]
      sapply(seq_along(tsr),function(i){
        SaveSeg(tsr[[i]][,,1],ExportDir=paste0('SEGS/T',ti),FileName=paste0('N_',w,'_sx',PIF$SID[i,1],'_sy',PIF$SID[i,2]),asRDS=T)
        SaveSeg(tsr[[i]][,,2],ExportDir=paste0('SEGS/T',ti),FileName=paste0('C_',w,'_sx',PIF$SID[i,1],'_sy',PIF$SID[i,2]),asRDS=T)
      })
    },cl=cl)
    stopCluster(cl);rm(cl)
    rm(mw);gc()
    ct=ct+1
  }
};rm(ti,ct);gc()


