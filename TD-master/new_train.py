import TD.train as tr
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main(_):
  tr.main(_)

if __name__ == "__main__":
  tr.init_flags(env="local_lewis",eval_every=2000,train_epochs=12000000)
  #tr.tf.flags.DEFINE_string("env", "local_lewis", "Which environment to use.")
  tr.tf.app.run()
