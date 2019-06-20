from SVDD.DeepSVDD import DeepSVDD, DeepSVDD_Option


class toolkit(object):
    
    def __init__(self):
        self.x = ''

    def setAlgorthem(self,algorthem:str):
        self.alg_name = algorthem
        if algorthem == "deepsvdd" :
            return DeepSVDD_Option()
        
        if algorthem == "deepsvdd" :
            #return needed options
            return
        
        return
    def test(self):
        if self.alg_name == "deepsvdd" :
           self.alg.test()

        if self.alg_name == "xxxx":
           #do xxxx algorthem
           return

        return 

    def train(self, option):
        self.option = option
        if self.alg_name == "deepsvdd" :
            self.alg = DeepSVDD(option)
            self.alg.train()

        if self.alg_name == "xxxx":
           #do xxxx algorthem
           return
        
        return
