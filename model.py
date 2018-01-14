import tensorflow as tf
import numpy as np
import configuration as cf
import iornn
import utilrnn
import time

WACHSTUM = 5
class PflanzInput(object):
  def __init__(self, config, data, name=None):
    
    self.all_data, self.plants_num, self.num_steps = iornn.tensorInputProducer(data, name)
    self.batch_size = config.batch_size
    self.batch_num = int(np.floor(self.plants_num / self.batch_size))
    self.epoch_size = self.batch_num
    self.batch_num = tf.placeholder(shape=None, name="batch_num", dtype=tf.int32)
    
    self.make_batches()
    
    self.input_data = self.input_is_one_batch(self.batch_num)
    
  def make_batches(self):
    with tf.name_scope("batch_input"):
      self.batched_data = tf.reshape(self.all_data, [self.batch_num, self.batch_size, self.num_steps, 6])
  
  def input_is_one_batch(self, batch_num):
    with tf.name_scope("one_batch"):
      self.input_data = tf.reshape(tf.slice(self.batched_data, [batch_num, 0,0,0], [1,self.batch_size, -1, -1]), [self.batch_size, self.plants_num, 6])
      return self.input_data
    
class PflanzModel(object):
  """The Pflanz Model"""
  
  def __init__(self, loss_time, is_training, config, input_):
    """__init__
    Give input for 1 single plant. In the form of [timestep][vars]"""
    
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.plants_num = input_.plants_num
    self.batch_size = input_.batch_size
    self.num_steps_total = input_.num_steps
    self.num_steps = input_.num_steps
    self._loss_time = loss_time
    
    size = config.hidden_size
    
    inputs = self._input.input_data
    
    #Left dropout out for now
    #can be included, if overfitting occurs
    
    output, state = self._build_rnn_graph(inputs, config, is_training, self._loss_time)
    
    
    self._final_state = state
    
    nlsg, wasser, lrot, lweiss = self._softmax_for_pflanzvariable(output)
    
    outputs = tf.concat([nlsg, wasser, lrot, lweiss], 1, name="concat_output")
    
    nlsg = self._exploration(nlsg, config)
    wasser = self._exploration(wasser, config)
    lrot = self._exploration(lrot, config)
    lweiss = self._exploration(lweiss, config)
        
    nlsg_ind = self._choose_action(nlsg)
    wasser_ind = self._choose_action(wasser)
    lrot_ind = self._choose_action(lrot)
    lweiss_ind = self._choose_action(lweiss)
    
    #nlsg_offset = tf.constant(0, dtype=tf.int32)
    #wasser_offset = tf.constant(49, dtype=tf.int32)
    #lrot_offset = tf.constant(99, dtype=tf.int32)
    #lweiss_offset = tf.constant(149, dtype=tf.int32)
    
    
    #nlsg_out_ind = tf.add(nlsg_ind, tf.fill([config.batch_size, 50], nlsg_offset))
    #wasser_out_ind = tf.add(wasser_ind, tf.fill([config.batch_size, 50], wasser_offset))
    #lrot_out_ind = tf.add(lrot_ind, tf.fill([config.batch_size, 50], lrot_offset))
    #lweiss_out_ind = tf.add(lweiss_ind, tf.fill([config.batch_size, 50], lweiss_offset))
    
    with tf.variable_scope("process_indices"):
      all_ind = tf.concat([nlsg_ind, wasser_ind, lrot_ind, lweiss_ind], 1, name="concat_indices")
      tf.summary.histogram("UnprocessedIndices", all_ind)

      self._final_output = all_ind
      self._final_output = self._index_to_value(all_ind)

      tf.summary.scalar("nlsg", self._final_output[0,0])
      tf.summary.scalar("wasser", tf.reduce_mean(self._final_output[:,1]))
      tf.summary.scalar("lrot", tf.reduce_mean(self._final_output[:,2]))
      tf.summary.scalar("lweiss", tf.reduce_mean(self._final_output[:,3]))
    
    self._summary_op = tf.summary.merge_all()
    self.saver = tf.train.Saver()
    self._learn(self._loss_time, inputs, outputs, all_ind, config)
  
  def _build_rnn_graph(self, inputs, config, is_training, loss_timestep):
    #should probs include Dropout Wrapper later
    
    cell = tf.contrib.rnn.MultiRNNCell([self._get_lstm_cell(config, is_training) for _ in range(config.num_layers)], state_is_tuple=True)
    
    self._initial_state = cell.zero_state(config.batch_size, tf.float32)
    state = self._initial_state
    
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(loss_timestep + 1):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        #outputs.append(cell_output) #This gives outputs for each timestep which is quite ridiculous in our case, bc we can't change the outputs at previous timesteps
        outputs = cell_output
        #That's why for us output = the last cell_output
        
      outputs = cell_output
      output = tf.reshape(outputs, [self.batch_size, 4, -1], name="reshape_output")
      
    return output, state
  
  def _get_lstm_cell(self, config, is_training):
    return tf.contrib.rnn.LSTMBlockCell(config.hidden_size, forget_bias=0.0)
  
  def _softmax_for_pflanzvariable(self, output):
    with tf.variable_scope("softmax"):
      with tf.variable_scope("slice_outputs"):
        nlsg = tf.slice(output, [0, 0, 0], [-1, 1, -1])
        wasserstand = tf.slice(output, [0, 1, 0], [-1, 1, -1])
        lichtrot = tf.slice(output, [0, 2, 0], [-1, 1, -1])
        lichtweiss = tf.slice(output, [0, 3, 0], [-1, 1, -1])
    
      sm_nlsg = tf.reshape(tf.nn.softmax(nlsg, dim=-1), [self.batch_size, -1])

      sm_wasserstand = tf.reshape(tf.nn.softmax(wasserstand, dim=-1), [self.batch_size, -1])

      sm_lichtrot = tf.reshape(tf.nn.softmax(lichtrot, dim=-1), [self.batch_size, -1])

      sm_lichtweiss = tf.reshape(tf.nn.softmax(lichtweiss, dim=-1), [self.batch_size, -1])
  
    return sm_nlsg, sm_wasserstand, sm_lichtrot, sm_lichtweiss
  
  def _exploration(self, output, config):
    with tf.variable_scope("add_random_explore"):
      explore = tf.multiply(tf.pow(output, config.principle_magnitude), tf.random_uniform([config.batch_size, 50], 0, 1))
    return explore
  
  def _choose_action(self, output):
    """
    Choose action with the greatest probability
    
    Do this for each plant in a batch (in most cases 1)
    Need to add a random component later for exploration
    
    Args:
        output: Output, but normalized with softmax
    """
    with tf.variable_scope("choose_action_with_argmax"):
      best_indices = tf.reshape(tf.argmax(output, output_type=tf.int32, axis=1), [self.batch_size, 1])
    return best_indices
  
  def _index_to_value(self, indices):
    with tf.variable_scope("indices_to_values"):
      values = tf.subtract(tf.divide(indices, 25), 1)
    return values
  
  def _learn(self, loss_timestep, inputs, outputs, action_indices, config):
    with tf.variable_scope("Learn"):
      reward = tf.placeholder(shape=[None], dtype=tf.float32)
      with tf.variable_scope("calculate_reward"):
        reward = tf.reduce_mean(inputs[:, loss_timestep, WACHSTUM]) #Loss should be calculated considering the responsible weight as well QLearning? I don't get QLearning or Policy Learning

      with tf.variable_scope("calculate_loss"):
        self.responsible_outputs = tf.gather(outputs, action_indices, axis=1)

        loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * reward)
        self._cost = tf.reduce_sum(loss) #Just making sure this is one-dimensional, though it should automatically be. Cost should be loss directly since there is only one variable on which we measure it

      if self._is_training:
    
        self._lr = tf.Variable(1.0, trainable=False, name="lr")
        tvars = tf.trainable_variables()

        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), config.max_grad_norm, name="gradients") #Gradients find minimun for cost

        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.train.get_or_create_global_step())
    
    return
  
  def save(self, session, config, time_point):
    self.saver.save(session, config.log_path+"modvar"+str(time_point))
  
  def restore(self, session, config, time_point):
    self.saver.restore(session, config.log_path+"modvar"+str(time_point))
    
  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name
  
  @property
  def final_output(self):
    return self._final_output
  
  @property
  def summary_op(self):
    return self._summary_op
    
def run_batch(session, model, eval_op=None, loss_time=0, verbose=False, config=None, current_batch=0):
  """Runs the model on the given data."""
  session.run([model.initial_state])
  writer = tf.summary.FileWriter(config.log_path + "time_" + str(loss_time), graph=tf.get_default_graph())
  start_time = time.time()
  costs = 0.0
  iters = 0

  actions, summary = session.run([model.final_output, model.summary_op])
  """
  cost = vals["cost"]
  state = vals["final_state"]
  actions = vals["chosen_actions"]

  costs += cost
  iters += model.input.num_steps
  """
  writer.add_summary(summary, current_batch)


  return actions