margs = list(SERVER=SERVER,PlateID=PID,WellID=sort(PIF$WID)[1],SiteID=c(1,1),TimePoint=1)
#
IM = do.call(GetIMPath,margs)%>%readImage()%>%autoNormalize(step=5*10**-4) #Get image from margs info
MK = ReadSeg(fi=paste0('SEGS/T',margs$TimePoint,'/C_',margs$WellID,'_sx',margs$SiteID[1],'_sy',margs$SiteID[2],'.Rds'),asRDS=T) #Get mask from margs info
#
OL = toRGB(IM)+1/3*(colorLabels(MK)) #Overlay image
display(OL)

