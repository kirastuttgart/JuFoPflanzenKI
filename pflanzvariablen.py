class PflanzVariablen:

    def __init__(self):
        self.nlsg = 0
        self.wasserstand = 0
        self.lichtrot = 0
        self.lichtweiss = 0
        self.wachstum = 0
        self.zustand = 0

    def copy(self):
        newV = PflanzVariablen()
        newV.nlsg = self.nlsg
        newV.wasserstand = self.wasserstand
        newV.lichtrot = self.lichtrot
        newV.lichtweiss = self.lichtweiss
        newV.wachstum = self.wachstum
        newV.zustand = self.zustand
        return newV

    def printVars(self):
        print(self.nlsg)
        print(self.wasserstand)
        print(self.lichtrot)
        print(self.lichtweiss)
        print(self.wachstum)
        print(self.zustand)
  
    