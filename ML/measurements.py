import numpy as np

class measurements:
    def AIC(self,numberOfVariables, predictions, observations):
        residuals = np.array(observations - predictions)
        sumSquaredError = np.sum(residuals**2)
        
        return 2*numberOfVariables - 2*np.log(sumSquaredError)
    
    def BIC(self,numberOfObservations, numberofVariables, predictions, observations):
        residuals = np.array(observations - predictions)
        sumSquaredError = np.sum(residuals**2)
        
        return numberOfObservations*np.log(sumSquaredError/numberOfObservations)\
                    + numberofVariables*np.log(numberOfObservations)
    
    def RMSLE(self,predictions,observations):
        yhat = np.array(predictions)
        y = np.array(observations)
        
        le = np.log(y) - np.log(yhat)
        sle = np.square(le)
        msle = sle.mean()
        rmsle = np.sqrt(msle)
        return rmsle
