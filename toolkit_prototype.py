from SVDD.DeepSVDD import DeepSVDD, DeepSVDD_Option
from REPEN.rankod import REPEN, REPEN_Option
from GAN.GAN import GAN, GAN_Option


class toolkit(object):
    
    def __init__(self):
        self.x = ''

    def setAlgorthem(self,algorthem:str):
        self.alg_name = algorthem
        if algorthem == "deepsvdd" :
            return DeepSVDD_Option()
        
        if algorthem == "repen" :
            return REPEN_Option()
            
        if algorthem == "GAN" :
            return GAN_Option()
        
        return
    def test(self):
        if self.alg_name == "deepsvdd" :
            self.alg.test()
        
        if self.alg_name == "repen" :
            self.alg.test()
            
        if self.alg_name == "GAN" :
            self.alg.test()
        
        return
    
    def train(self, option):
        self.option = option
        if self.alg_name == "deepsvdd" :
            self.alg = DeepSVDD(option)
            self.alg.train()
        
        if self.alg_name == "repen" :
            self.alg = REPEN(option)
            self.alg.train()
            
        if self.alg_name == "GAN" :
            self.alg = GAN(option)
            self.alg.train()
        
        return
        
        
        return
