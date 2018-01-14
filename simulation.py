import pflanzvariablen as pv

class Simulator(object):
  def __init__(self):
    self.time=0
    self.zustand = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0]
    self.wachstum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0]
    self. Pflanzennummer=0
    self.allplants=[]
    
    for i in range(40):
      self.allplants.append([])
      for j in range(40):
        self.allplants[i].append([])

  def fillStupid(self):
    for j in range(40):
      for i in range(40):
        neueVariable = pv.PflanzVariablen()
        neueVariable.nlsg = 0.1
        neueVariable.wasserstand = -.2
        neueVariable.lichtrot = .1
        neueVariable.lichtweiss = 0.2
        self.putNewOutput(j, neueVariable, i)
      
    
  def getPlantHistory(self, Pflanzennummer):
    return self.allplants[Pflanzennummer]

  def getAllData(self):
    return self.allplants
  
  def putNewOutput(self, Pflanzennummer, neueVariable, Zeit):
      wachstum=0
      wachstum=neueVariable.nlsg*0.0125                                                                                   #Nährlösung
      if neueVariable.nlsg >0.6:
          wachstum=0
      if neueVariable.nlsg >0.7:
          wachstum=neueVariable.nlsg*(-1)

      if neueVariable.wasserstand == 0 and wachstum+neueVariable.zustand <= 0.8:                                       #Wasserstand
          wachstum=wachstum+0.0125
      if neueVariable.wasserstand < 0:
          wachstum=wachstum+0.2*neueVariable.wasserstand
      if neueVariable.wasserstand > 0:
          wachstum=wachstum-0.334*neueVariable.wasserstand

      if neueVariable.lichtrot >= -0.7:                                                                                   #LichtRot
          wachstum=wachstum+(neueVariable.lichtrot + 1)*0.007
      else:
          wachstum=wachstum-0.003

      if neueVariable.lichtweiss >= -0.8:                                                                                  #LichtWeiss
          wachstum=wachstum+(neueVariable.lichtweiss+1)*0.02
      else:
          wachstum=wachstum-0.08

      if Zeit>1:
          neueVariable.zustand=self.allplants[Pflanzennummer][Zeit-1].zustand+wachstum
      else:
          neueVariable.zustand=wachstum
      if neueVariable.zustand>1:
          neueVariable.zustand=1
      if neueVariable.zustand<-1:
          neueVariable.zustand=-1

      neueVariable.wachstum=wachstum
      self.allplants[Pflanzennummer][Zeit] = neueVariable


if __name__ == "__main__":
  sim = Simulator()
  for i in range(40):
    neueVariable = pv.PflanzVariablen()
    neueVariable.nlsg = 0.1
    neueVariable.wasserstand = -.2
    neueVariable.lichtrot = .1
    neueVariable.lichtweiss = 0.2
    sim.putNewOutput(1, neueVariable, i)
    
  for i in range(40):
    sim.getPlantHistory(1)[i].printVars()
    


