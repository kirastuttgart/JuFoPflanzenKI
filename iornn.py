#reads the stored data to input
#also processes the data produced by the RNN and grows the plants
#additionally gets feedback from the Robot and adds that as input

import tensorflow as tf
import numpy as np
import configuration as cf

def parsePflanzvariable(list):
  """Parses list with object pflanzvariable
  
  Turns all the values into integers, as they can be processed by the KI
  Additionally normalizes them? This is still all in python
  
  Args: list, in the shape of [plant][timestep]
  
  Returns:
    A list in the shape of [plant][timestep][data]
  """
  
  finallist = []
  for plant in range(len(list)):
    finallist[plant] = []
    for timestep in range(len(list[0])):
      finallist[plant][timestep] = []
      finallist[plant][timestep][0] = list[plant][timestep].nlsg
      finallist[plant][timestep][1] = list[plant][timestep].wasserstand
      finallist[plant][timestep][2] = list[plant][timestep].lichtrot
      finallist[plant][timestep][3] = list[plant][timestep].lichtweiss
      
      finallist[plant][timestep][4] = list[plant][timestep].wachstum
      finallist[plant][timestep][5] = list[plant][timestep].fitness
      
      
      finallist[plant].append([])
  
    finallist.append([])
  
  return finallist
      
def tensorInputProducer(pass_data, name=None):
  """Iterate on the raw data
  
  Produces a tensor based on raw data
  Raw Data should be provided as a list of objects in the form of pflanzvariablen. The structure should be a list [plant][timestep]. 
  
  Args: raw_data: as a list of pflanzvariables
        batch_size: int
        num_steps: int, number of unrolls
        
  Returns:
    A Tensor in the shape of [plant][timestep]
    
  """
  raw_data = []
  
  plants = len(pass_data)
  timesteps = len(pass_data[0])
  
  for plant_num in range(plants):
    raw_data.append([])
    for time_num in range(timesteps):
      raw_data[plant_num].append([])
      for i in range(6):
        raw_data[plant_num] [time_num].append([])
      raw_data[plant_num][time_num][0] = pass_data[plant_num][time_num].nlsg
      raw_data[plant_num][time_num][1] = pass_data[plant_num][time_num].wasserstand
      raw_data[plant_num][time_num][2] = pass_data[plant_num][time_num].lichtrot
      raw_data[plant_num][time_num][3] = pass_data[plant_num][time_num].lichtweiss
      raw_data[plant_num][time_num][4] = pass_data[plant_num][time_num].zustand
      raw_data[plant_num][time_num][5] = pass_data[plant_num][time_num].wachstum

  #batch_num = plants / batch_size

  with tf.name_scope(name):
    data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.float32);
    #This is only possible bc there is a natural limit for timesteps (probs 14 or smth)
    num_steps_total = timesteps
    
    #I don't think we really need to reshape the data
    #It would probably required, if we had more data (eg plants)
      
    epoch_size = num_steps_total #I aint understanding this yet
    
    #---Not really important just protocol--
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    
    """
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")
      num_steps_total = tf.identity(num_steps_total, name="num_steps_total")"""
    #---Not really important ends here --
    
    print("Shape of final input tensor: ")
    print(data.get_shape());
    
    return data, plants, timesteps 
    
    
if __name__ == "__main__":
  tensorInputProducer(data, 4, 2);
    
