class DefaultConfig(object):
  """Small config."""
  "Copied directly from the tensor flow example"
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  #rnn_mode = BLOCK
  
class CustomConfig(object):
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  #num_steps = 14 #bc the program will run 7 days and check the plants twice a day
  #epoch_size = 40 #bc we have 40 plants
  hidden_size = 200 #Also the size of the outputs. As such there will be 50 different states for each pflanzvariable
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  #batch_size = 4
  batch_size = 4 #We have so little 
  principle_magnitude = 1
  #rnn_mode = BLOCK
  run_name = "run_0/"
  log_path = "/Users/Kira/Documents/JuFo/logs/" + run_name