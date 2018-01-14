import tensorflow as tf
import simulation
import model as mod
import configuration as cf

def main(_):
  config = cf.CustomConfig()
  sim = simulation.Simulator()
  sim.fillStupid()
  raw_data = sim.getAllData()
  
  metagraphs = []
  current_time = 6
  
  
  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    
    with tf.name_scope("Input"):
      train_input = mod.PflanzInput(config=config, data=raw_data, name="input")
      train_input.make_batches()
      train_input.input_is_one_batch(0)
        
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
      m = mod.PflanzModel(current_time, is_training=True, config=config, input_=train_input)
      
  
    with tf.Session() as session:
      session.run(tf.global_variables_initializer())
      state = session.run(m.initial_state)
      session.run([m.final_state])
      
      m.save(session, config, 6)
    
  
  actions = get_params(current_time, config, sim)
  for i in actions:
    for j in i:
      for k in j:
        print(k)
      print("oneplant")
    print("onebatch")
      
def get_params(max_time, config, sim):
  raw_data = sim.getAllData()
  
  with tf.Graph().as_default():
    with tf.variable_scope("Input", reuse=None):
      train_input = mod.PflanzInput(config=config, data=raw_data, name="input")
      train_input.make_batches()
      train_input.input_is_one_batch(0)
    
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    with tf.variable_scope("Model", reuse=tf.AUTO_REUSE):
      model = mod.PflanzModel(max_time, is_training=False, config=config, input_=train_input)
        
    with tf.Session() as session:
      session.run(tf.global_variables_initializer())
      model.restore(session,config, max_time)
      output = []
      for batch in range(train_input.batch_num):
        session.run(train_input.input_data, feed_dict={train_input.batch_num:batch})
        output.append(mod.run_batch(session, model, loss_time=max_time, config=config, current_batch=batch))

  return output
    
if __name__ == "__main__":
  tf.app.run()
    
    
      