import numpy as np
import mltools as ml
import matplotlib.pyplot as plt

# Fix the required "not implemented" functions for the homework ("TODO")

################################################################################
## LOGISTIC REGRESSION BINARY CLASSIFIER #######################################
################################################################################


class logisticClassify2(ml.classifier):
    """A binary (2-class) logistic regression classifier

    Attributes:
        classes : a list of the possible class labels
        theta   : linear parameters of the classifier
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for logisticClassify2 object.

        Parameters: Same as "train" function; calls "train" if available

        Properties:
           classes : list of identifiers for each class
           theta   : linear coefficients of the classifier; numpy array
        """
        self.classes = [0,1]              # (default to 0/1; replace during training)
        self.theta = np.array([])         # placeholder value before training

        if len(args) or len(kwargs):      # if we were given optional arguments,
            self.train(*args,**kwargs)    #  just pass them through to "train"


## METHODS ################################################################

    def plotBoundary(self,X,Y):
        """ Plot the (linear) decision boundary of the classifier, along with data """
        if len(self.theta) != 3: raise ValueError('Data & model must be 2D');
        ax = X.min(0),X.max(0); ax = (ax[0][0],ax[1][0],ax[0][1],ax[1][1]);
        ## TODO: find points on decision boundary defined by theta0 + theta1 X1 + theta2 X2 == 0
        x1b = np.array([ax[0],ax[1]]);  # at X1 = points in x1b
        x2b = -1*(self.theta[0]+self.theta[1]*x1b)/self.theta[2];    
        ## Now plot the data and the resulting boundary:
        A = Y==self.classes[0];                                             
        plt.plot(X[A,0],X[A,1],'r.',X[~A,0],X[~A,1],'b.',x1b,x2b,'k-'); 
        plt.axis(ax); 
        plt.draw();

    def predictSoft(self, X):
        """ Return the probability of each class under logistic regression """
        raise NotImplementedError
        ## You do not need to implement this function.
        ## If you *want* to, it should return an Mx2 numpy array "P", with
        ## P[:,1] = probability of class 1 = sigma( theta*X )
        ## P[:,0] = 1 - P[:,1] = probability of class 0
        return P

    def predict(self, X):
        """ Return the predictied class of each data point in X"""
        # Predicting a class based on the linear response
        ## TODO: compute linear response r[i] = theta0 + theta1 X[i,1] + theta2 X[i,2] + ... for each i
        ## TODO: if z[i] > 0, predict class 1:  Yhat[i] = self.classes[1]
        ##       else predict class 0:  Yhat[i] = self.classes[0]
        Yhat = [];
        R = self.theta[0] + X.dot(self.theta[1:]);
        for r in R:
            if r > 0:
                Yhat.append(self.classes[1]) ;            
            else:
                Yhat.append(self.classes[0]);       
        return np.array(Yhat);
        


    def train(self, X, Y, initStep=1.0, stopTol=1e-4, stopEpochs=5000, plot=None, regularization=False, alpha=2):
        """ Train the logistic regression using stochastic gradient descent """
        M,N = X.shape;                     # initialize the model if necessary:
        self.classes = np.unique(Y);       # Y may have two classes, any values
        XX = np.hstack((np.ones((M,1)),X)) # XX is X, but with an extra column of ones
        YY = ml.toIndex(Y,self.classes);   # YY is Y, but with canonical values 0 or 1
        if len(self.theta)!=N+1: self.theta=np.random.rand(N+1);
        # init loop variables:
        epoch=0; done=False; Jnll=[]; J01=[];
        while not done:
            stepsize, epoch = initStep*2.0/(2.0+epoch), epoch+1; # update stepsize
            # Do an SGD pass through the entire data set:
            for i in np.random.permutation(M):
                ri = XX[i].dot(self.theta);     # TODO: compute linear response r(x)
                if ri > 0:
                    sigi = 1 / (1 + np.exp(-ri))
                else:
                    sigi = np.exp(ri) / (1 + np.exp(ri))
                if YY[i]:
                    gradi = -(1-sigi) * XX[i,:];     # TODO: compute gradient of NLL loss
                else:
                    gradi = sigi * XX[i,:];
                if regularization:
                    gradi += 2*alpha*self.theta;
                self.theta -= stepsize * gradi;  # take a gradient step

            J01.append( self.err(X,Y) )  # evaluate the current error rate

            ## TODO: compute surrogate loss (logistic negative log-likelihood)
            ##  Jsur = sum_i [ (log si) if yi==1 else (log(1-si)) ]
            Sig = [];
            for x in XX:
                r = x.dot(self.theta);
                if r > 0:
                    Sig.append(1 / (1 + np.exp(-r)));
                else:
                    Sig.append(np.exp(r) / (1 + np.exp(r)));
            Sig = np.array(Sig);                   
            
            Jsum = 0;
            counter = 0;
            for s in Sig:
                if (abs(1-s) < 1e-6):
                    Jsum += YY[int(counter)]*np.log(s);
                elif (abs(s) < 1e-6):
                    Jsum += (1-YY[int(counter)])*np.log(1-s);
                else:
                    Jsum += (YY[int(counter)]*np.log(s)+(1-YY[int(counter)])*np.log(1-s));
                counter += 1;
            Jsur = -Jsum / len(YY);
            Jnll.append( Jsur ) # TODO evaluate the current NLL loss

            # plt.figure(1); plt.plot(Jnll,'b-',J01,'r-'); plt.draw();    # plot losses
            # if N==2: plt.figure(2); self.plotBoundary(X,Y); plt.draw(); # & predictor if 2D
            # plt.pause(.01);                    # let OS draw the plot

            ## For debugging: you may want to print current parameters & losses
            # print self.theta, ' => ', Jnll[-1], ' / ', J01[-1]
            # raw_input()   # pause for keystroke

            # TODO check stopping criteria: exit if exceeded # of epochs ( > stopEpochs)
            done = (epoch > stopEpochs) or (epoch>1 and abs(Jnll[-1]-Jnll[-2])<stopTol);   # or if Jnll not changing between epochs ( < stopTol )
        # plot
        self.numberOfIterations = epoch
        if self.plotFlag:
            plt.semilogx(range(epoch), np.abs(Jnll), label='Surrogate Loss')
            plt.semilogx(range(epoch), np.abs(J01), label='Error Rate')
            plt.legend(loc='upper right')
            plt.xlabel('# of iterations')
            plt.ylabel('Losses')
            plt.show()


################################################################################
################################################################################
################################################################################
