import sys

class optimiser:
    def __init__(self):
        pass
    
    def compute(self):
        pass

class gradientDescent(optimiser):
    def __init__(self,gamma,model):
        self.gamma = gamma
        self.counter = 3
        self.history = []
        self.model = model
        
    def compute(self, max_iteration = 100000, tol=10e-8):
        iteration = 0
        counter = 0
        current_beta = self.model.beta
        current_cost = self.model.cost()
        while counter < self.counter:
            beta = self.model.beta - self.gamma*self.model.dcost()
            self.model.setBeta(beta)
            
            
            if (self.model.beta == current_beta).all():
                counter += 1

            current_beta = self.model.beta
            if self.model.cost() > current_cost:
                print(' ')
                print('iteration stopped due to cost() increasing, currently cost() is ',self.model.cost())
                print('tip : If cost() still too large try to lower gamma parameter')
                break
            if abs(self.model.cost() - current_cost) <= tol:
                print(' ')
                print('iteration stopped due to tolerance difference reached')
                break
            current_cost = self.model.cost()
            self.history.append(current_cost)
            
            sys.stdout.write("\r"+"Progress on iteration {:2.1%}".format(iteration / max_iteration))
            if iteration / max_iteration%0.1 == 0.0:
                print('current cost',current_cost)
            sys.stdout.flush()
            iteration+=1
            if iteration == max_iteration:
                print('iteration stopped due to max iteration reached')
                break
        return self.history
    
class stochasticGradientDescent(optimiser):
    def __init__(self,gamma,model,resamplingNumber):
        self.gamma = gamma
        self.counter = 3
        self.history = []
        self.model = model
        self.sampling = resamplingNumber
        
    def compute(self, max_iteration = 100000 , tol=10e-8):
        iteration = 0
        counter = 0
        current_beta = self.model.beta
        current_cost = self.model.cost()
        while counter < self.counter:
            beta = self.model.beta - self.gamma*self.model.dcost(self.sampling)
            self.model.setBeta(beta)
            if (self.model.beta == current_beta).all():
                counter += 1

            current_beta = self.model.beta
            if self.model.cost() > current_cost:
                print(' ')
                print('iteration stopped due to cost() increasing, currently cost() is ',self.model.cost())
                print('tip : If cost() still too large try to lower gamma parameter')
                break
            
            if abs(self.model.cost() - current_cost) <= tol:
                print(' ')
                print('iteration stopped due to tolerance difference reached')
                break
            current_cost = self.model.cost()
            self.history.append(current_cost)
            sys.stdout.write("\r"+"Progress on iteration {:2.1%}".format(iteration / max_iteration))
            if iteration / max_iteration%0.1 == 0.0:
                print('current cost',current_cost)
            sys.stdout.flush()
            iteration+=1
            
            if iteration == max_iteration:
                print('iteration stopped due to max iteration reached')
                break
        return self.history