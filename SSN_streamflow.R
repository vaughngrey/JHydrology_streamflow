### SSN_streamflow.R

#' Author: Vaughn Grey: vaughn.grey@unimelb.edu.au
#' Created: 6 August 2024 

#' Code of Spatial Stream Network model for paper: Grey et al 2025 Journal of Hydrology, 
#' Harnessing the strengths of machine learning and geostatistics to improve streamflow prediction in ungauged basins; the best of both worlds

# ' Training, validation and testing of a SSN model, estimating daily flows for the Melbourne region
#' Final model

#' Note: run of University of Melbourne Spartan High Performance Computing system 

start.time <- Sys.time()
print(start.time)

##########################################################################################
### Set up
##########################################################################################

### For Spartan
command_args <- commandArgs(trailingOnly = TRUE)
date.rep <- as.numeric(command_args[1]) # Import trailing argument 

library(dplyr)
library(ggplot2)
library(sf)

setwd("..")                      #SET FOR SPARTAN
lib.lcn <- "R_libs/"             #SET FOR SPARTAN
library(SSN2, lib.loc = lib.lcn) #SET FOR SPARTAN

# Set appropriate WD to access SSN2 object / distance matrix etc
setwd("SSN_Flow_H")                     #SET FOR SPARTAN

# Set appropriate WD and load stream network already in SSN format
load("SSN_Flow_Preds_Spartan.RData")

print("loaded")

##############################################################################################################################################################################
### For loop starts here
##############################################################################################################################################################################

###################
### Model training
###################

mod.out.Store.h <- vector('list')

setwd("SSN_H_Pred_All.ssn/") #setting WD here for distance matrix !!!!SPARTAN!!!!

# Subset DateIndex
no.breaks <- 20
starting.no <- 1
DateIndex.length <- length(h.data.daily.DateIndex) - starting.no
no.dates.per.break <- floor(DateIndex.length/no.breaks)

exp.rep.end <- date.rep * no.dates.per.break + starting.no

if (date.rep == no.breaks) {
  exp.rep.end <- as.numeric(length(h.data.daily.DateIndex))
}

exp.rep.start <- (date.rep-1) * no.dates.per.break + starting.no

X.lines <- seq(exp.rep.start,exp.rep.end,by=1) ## set X to within exp.rep bounds

DateIndex.subset <- h.data.daily.DateIndex[X.lines]

for (exp.rep.2 in 1:length(DateIndex.subset)){

  ##########################################
  ### Select flow date
  ##########################################
  
  # Source new predictive data for Site and DateTime of interest
  h.target.Date <- DateIndex.subset[exp.rep.2]
  h.data.new <- h.data.daily[h.data.daily$Date == h.target.Date,] #cut to date of interest
  h.data.new <- h.data.new[!duplicated(h.data.new$gauge_id),]
  
  ## extract predictors and join
  mwstr_ssn.obs <- mwstr_ssn.obs.master  ## Use master so don't have to reload mwstr_ssn object!
  mwstr_ssn.obs$f_mean <- NULL # remove old data
  mwstr_ssn.obs$q_mm <- NULL # remove old data
  mwstr_ssn.obs <- left_join(mwstr_ssn.obs, h.data.new[,c("rid","q_mm")], by = "rid")

  ## now put back into ssn object - just obs 
  mwstr_ssn$obs <- mwstr_ssn.obs #note data removed in above step
  
  #####################
  ### Create models
  #####################
  print("pre-mod")
  
  ## Use "tryCatch" to skip where eigenvalue errors occurring, can be resolved by fixing correlation structure parameters
  
  skip_to_next <- FALSE
  
  tryCatch(
  
  ### Model 5
  ssn_mod <- ssn_lm(
    formula = q_mm ~ elev + meanP + ti,
    ssn.object = mwstr_ssn,
    tailup_type = "mariah",
    taildown_type = "mariah",
    euclid_type = "spherical",
    nugget_type = "nugget",
    additive = "afvArea",
    estmethod = "reml"
  )
  ,
  error = function(e) { skip_to_next <<- TRUE})
  
  if(skip_to_next) { next }
  
  print("post-mod")
  
  ###################
  ### Prediction at sites
  ##################
  
  mwstr_ssn.preds <- predict(ssn_mod, newdata = "preds",se.fit = TRUE, interval = "none")
  mwstr_ssn.preds <- as.data.frame(mwstr_ssn.preds)
  
  aug_preds <- augment(ssn_mod, newdata = "preds")

  aug_preds <- cbind(aug_preds, mwstr_ssn.preds)
  
  aug_preds <- sf::st_drop_geometry(aug_preds[,c("rid","fit","se.fit")])
  aug_preds$DateTime <- h.target.Date
  
  mod.out.Store.h[[exp.rep.2]] <- aug_preds
  
  setwd("..")
  write.csv(x = aug_preds, file = paste0("SSN_H_Pred_All/","SSN_daily_pred_rep_",date.rep,"_Date_",h.target.Date,".csv"), row.names = F)  ### Save output 
  setwd("SSN_H_Pred_All.ssn/")
  
  print(exp.rep.2)
  
  
} ## Close for-loop here

mod.out.h <- do.call(rbind, mod.out.Store.h)
print("rbound")

setwd("..")
write.csv(x = mod.out.h, file = paste0("SSN_H_Pred_All/SSN_daily_pred_all_rep_",date.rep,".csv"), row.names = F)  ### Save output 

print("saved")

end.time <- Sys.time()
elapsed.time <- end.time - start.time
print(elapsed.time)
