import TD.scripts.prune.eval as eval
import TD.ga.ga as ga
import tensorflow as tf
import os
import TD.train as mytrain
import numpy as np
import copy
import shutil
import time

from TD.training import flags
from TD.models.registry import get_model
from TD.hparams.registry import get_hparams
from TD.data.registry import get_input_fns
from TD.scripts.prune.prune import get_prune_fn

os.environ["CUDA_VISIBLE_DEVICES"]="0"

best_path = "bestpath"
ga_path="gapath"
check_file = "checkpoint.csv"
check_file_model="cp_model"
need_do_layers = {"resnet_mask": ["resnet/unit_1_0/sub1/conv1/DW", "resnet/unit_1_0/sub2/conv2/DW",
                                  "resnet/unit_1_1/sub1/conv1/DW", "resnet/unit_1_1/sub2/conv2/DW",
                                  "resnet/unit_1_2/sub1/conv1/DW", "resnet/unit_1_2/sub2/conv2/DW",
                                  "resnet/unit_1_3/sub1/conv1/DW", "resnet/unit_1_3/sub2/conv2/DW",
                                  "resnet/unit_1_4/sub1/conv1/DW", "resnet/unit_1_4/sub2/conv2/DW",
                                  "resnet/unit_2_0/sub1/conv1/DW", "resnet/unit_2_0/sub2/conv2/DW",
                                  "resnet/unit_2_1/sub1/conv1/DW", "resnet/unit_2_1/sub2/conv2/DW",
                                  "resnet/unit_2_2/sub1/conv1/DW", "resnet/unit_2_2/sub2/conv2/DW",
                                  "resnet/unit_2_3/sub1/conv1/DW", "resnet/unit_2_3/sub2/conv2/DW",
                                  "resnet/unit_2_4/sub1/conv1/DW", "resnet/unit_2_4/sub2/conv2/DW",
                                  "resnet/unit_3_0/sub1/conv1/DW", "resnet/unit_3_0/sub2/conv2/DW",
                                  "resnet/unit_3_1/sub1/conv1/DW", "resnet/unit_3_1/sub2/conv2/DW",
                                  "resnet/unit_3_2/sub1/conv1/DW", "resnet/unit_3_2/sub2/conv2/DW",
                                  "resnet/unit_3_3/sub1/conv1/DW", "resnet/unit_3_3/sub2/conv2/DW",
                                  "resnet/unit_3_4/sub1/conv1/DW", "resnet/unit_3_4/sub2/conv2/DW"],
                  "cifar100_resnet_mask": ["resnet/unit_1_0/sub1/conv1/DW", "resnet/unit_1_0/sub2/conv2/DW",
                                  "resnet/unit_1_1/sub1/conv1/DW", "resnet/unit_1_1/sub2/conv2/DW",
                                  "resnet/unit_1_2/sub1/conv1/DW", "resnet/unit_1_2/sub2/conv2/DW",
                                  "resnet/unit_1_3/sub1/conv1/DW", "resnet/unit_1_3/sub2/conv2/DW",
                                  "resnet/unit_1_4/sub1/conv1/DW", "resnet/unit_1_4/sub2/conv2/DW",
                                  "resnet/unit_2_0/sub1/conv1/DW", "resnet/unit_2_0/sub2/conv2/DW",
                                  "resnet/unit_2_1/sub1/conv1/DW", "resnet/unit_2_1/sub2/conv2/DW",
                                  "resnet/unit_2_2/sub1/conv1/DW", "resnet/unit_2_2/sub2/conv2/DW",
                                  "resnet/unit_2_3/sub1/conv1/DW", "resnet/unit_2_3/sub2/conv2/DW",
                                  "resnet/unit_2_4/sub1/conv1/DW", "resnet/unit_2_4/sub2/conv2/DW",
                                  "resnet/unit_3_0/sub1/conv1/DW", "resnet/unit_3_0/sub2/conv2/DW",
                                  "resnet/unit_3_1/sub1/conv1/DW", "resnet/unit_3_1/sub2/conv2/DW",
                                  "resnet/unit_3_2/sub1/conv1/DW", "resnet/unit_3_2/sub2/conv2/DW",
                                  "resnet/unit_3_3/sub1/conv1/DW", "resnet/unit_3_3/sub2/conv2/DW",
                                  "resnet/unit_3_4/sub1/conv1/DW", "resnet/unit_3_4/sub2/conv2/DW"],
                  "cifar_lenet_mask": ["lenet/conv1/DW", "lenet/conv2/DW"],
                  "mnist_lenet": ["lenet/conv1/DW", "lenet/conv2/DW"],
                  "cifar100_lenet_mask": ["lenet/conv1/DW", "lenet/conv2/DW"],
                  "cifar100_vgg16_mask": ["vgg/conv1_1/DW", "vgg/conv1_2/DW", "vgg/conv2_1/DW",
                                  "vgg/conv2_2/DW", "vgg/conv3_1/DW", "vgg/conv3_2/DW",
                                  "vgg/conv3_3/DW", "vgg/conv4_1/DW", "vgg/conv4_2/DW",
                                  "vgg/conv4_3/DW", "vgg/conv5_1/DW", "vgg/conv5_2/DW",
                                  "vgg/conv5_3/DW"],
                  "mobilenetv2_default": ["vgg/conv1_1/DW", "vgg/conv1_2/DW", "vgg/conv2_1/DW",
                                  "vgg/conv2_2/DW", "vgg/conv3_1/DW", "vgg/conv3_2/DW",
                                  "vgg/conv3_3/DW", "vgg/conv4_1/DW", "vgg/conv4_2/DW",
                                  "vgg/conv4_3/DW", "vgg/conv5_1/DW", "vgg/conv5_2/DW",
                                  "vgg/conv5_3/DW"],
                  "cifar100_vgg19_mask": ["vgg/conv1_1/DW", "vgg/conv1_2/DW", "vgg/conv2_1/DW",
                                  "vgg/conv2_2/DW", "vgg/conv3_1/DW", "vgg/conv3_2/DW",
                                  "vgg/conv3_3/DW", "vgg/conv3_4/DW", "vgg/conv4_1/DW", "vgg/conv4_2/DW",
                                  "vgg/conv4_3/DW", "vgg/conv4_4/DW", "vgg/conv5_1/DW", "vgg/conv5_2/DW",
                                  "vgg/conv5_3/DW","vgg/conv5_4/DW"]}

def saver_restore_part(ckpt_dir,eval_input_fn,model_fn):

  #sess.run(tf.global_variables_initializer())
  #sess.run(tf.local_variables_initializer())
  tf.reset_default_graph()
  features, labels = eval_input_fn()
  gs = tf.train.get_or_create_global_step()
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  model_p = model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)
  all_var_list = [var for var in tf.trainable_variables()]
  saver = tf.train.Saver(var_list=all_var_list)  
  #saver = tf.train.Saver()
  saver.restore(sess, ckpt_dir)
  return sess,model_p
def saver_restore(ckpt_dir,eval_input_fn,model_fn):
  tf.reset_default_graph()
  features, labels = eval_input_fn()
  gs = tf.train.get_or_create_global_step()
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  model_p = model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)

  saver = tf.train.Saver()
  saver.restore(sess, ckpt_dir)
  return sess,model_p
def saver_save(sess, output_dir):
  saver = tf.train.Saver()
  saver.save(sess, output_dir,write_meta_graph=False)

def print_tf_variables(sess):
    variable_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variable_names)
    for k, v in zip(variable_names, values):
      print("Variable: ", k)
      print("Shape: ", v.shape)
      print(v)
def weights_to_masks(orig_weights):
  masks = {}
  for key, value in orig_weights.items():
    mask = value != 0
    mask = mask.astype(int)
    masks[key] = copy.deepcopy(mask)
  return masks

def takeFirst(elem):
  return elem[0]

def write_to_check_file(model_dir,content):
  if not os.path.exists(model_dir):
    os.mkdir(model_dir)
  cfile = os.path.join(model_dir, "checkpoint")
  fp=open(cfile,"w")
  fp.write('model_checkpoint_path: "'+os.path.basename(content)+'"')
  fp.close()

def is_prunable_weight(weight):
  necessary_tokens = ["DW:0","DW_mask:0"]
  is_prunable = any(t in weight.name for t in necessary_tokens)
  return is_prunable

def get_current_weights(sess):
  weights = {}
  variables = {}
  graph = tf.get_default_graph()
  for tensor_name in tf.contrib.graph_editor.get_tensors(graph):
      if is_prunable_weight(tensor_name):
          name = tensor_name.name.strip(":0")
          variables[name] = tensor_name
  for variable_name in tf.global_variables():
      if is_prunable_weight(variable_name):
          name = variable_name.name.strip(":0")
          variables[name] = variable_name

  for weight_name, w in variables.items():
    weights[weight_name] = sess.run(w)

  return weights


def get_weights_dire(output_dir, eval_input_fn,model_fn):
  print("Loading model from...", output_dir)
  sess,model_p=saver_restore_part(output_dir,eval_input_fn,model_fn)
  #print_tf_variables(sess)
  orig_weights = get_current_weights(sess)
  return orig_weights,sess,model_p

def check_in_needdolayer(weight_name,need_do_layer):
  for name in need_do_layer:
    if name in weight_name:
      return True
  return False


def compute_model_prune_dire(output_dir, need_do_layer,eval_input_fn,model_fn):
  orig_weights,sess,model_p = get_weights_dire(output_dir=output_dir,eval_input_fn=eval_input_fn,model_fn=model_fn)
  weights = dict(orig_weights)
  total_masks=ga.weights_to_masks(orig_weights)
  total_size=0
  total_prune=0
  for e, weight_name in enumerate(orig_weights):
    if check_in_needdolayer(weight_name,need_do_layer):
      size = np.size(total_masks[weight_name])
      nonzero = np.count_nonzero(total_masks[weight_name])
      #nonzero = np.nonzero(total_masks[weight_name])
      #nonzero = len(nonzero[0])
      total_size+=size
      total_prune+=size-nonzero
  total_per = 0.
  if total_size>0:
    total_per = float(total_prune)/float(total_size)
  return total_per,total_prune,total_size,orig_weights,sess,model_p

def save_bests(sess, saver, output_path, id=0):
  output_dir = os.path.join(output_path, "best-" + str(id))
  saver.save(sess, output_dir)
  return output_dir

def set_mask(weight_dict, mask):
  pruned_w = mask * weight_dict
  return pruned_w

def new_prune_weights(prune_fn,
                      weight,
                      needdo_weight_name,
                      mask):
  weights_pruned = {}

  pre_prune_nonzero = 0
  pre_prune_total = 0


  pre_prune_nonzero += np.count_nonzero(weight)
  pre_prune_total += weight.size

  weights_pruned[needdo_weight_name] = prune_fn(weight,  mask)
  return weights_pruned, {
    "pre_prune_nonzero": pre_prune_nonzero,
    "pre_prune_total": pre_prune_total
  }

def new_prune_weights1(prune_fn,
                  weights,
                  needdo_weight_name,
                  mask):#mask是单层
  weights_pruned = {}

  pre_prune_nonzero = 0
  pre_prune_total = 0

  for weight_name in weights:
    if "variational" in weight_name:
      print("WARN variational: not pruning {}".format(weight_name))
      continue
    if weight_name == needdo_weight_name:
      pre_prune_nonzero += np.count_nonzero(weights[weight_name])
      pre_prune_total += weights[weight_name].size

      weights_pruned[weight_name] = prune_fn(weights, weight_name,mask)
      break

  return weights_pruned, {
      "pre_prune_nonzero": pre_prune_nonzero,
      "pre_prune_total": pre_prune_total
  }



def weight_to_saver1(FLAGS, hparams,file_name_frefix,cnt,weight_name,bests,meta_name,old_ckpt,eval_input_fn,model_fn):
  models=[]
  for i,p in enumerate(bests):
    filename=file_name_frefix+str(cnt)+"_"+str(i)
    ckpt_dir = old_ckpt
    #saver = tf.train.Saver(max_to_keep=1)
    print("weight_to_saver Loading model from...", ckpt_dir)
    orig_weights,sess,model_p = get_weights_dire(ckpt_dir, eval_input_fn, model_fn)
    mask = weights_to_masks(orig_weights)
    hparams.prune_mask = mask
    sess,model_p=saver_restore(ckpt_dir,eval_input_fn,model_fn)
    #saver = tf.train.import_meta_graph(meta_name,clear_devices=True)
    #sess.run(tf.global_variables_initializer())
    #sess.run(tf.initialize_all_variables())
    #saver.restore(sess, ckpt_dir)
    prune_fn = get_prune_fn(FLAGS.prune)()
    w_copy = dict(orig_weights)
    post_weights_pruned, weight_counts = new_prune_weights1(
      prune_fn,
      w_copy,
      weight_name,
      p[1])
    for v in tf.trainable_variables():
      if is_prunable_weight(v) and v.name.strip(":0")==weight_name:
        assign_op = v.assign(
          np.reshape(post_weights_pruned[v.name.strip(":0")], v.shape))
        sess.run(assign_op)
    #print_tf_variables(sess)
    #print(filename)
    #tf.train.get_or_create_global_step()
    saver_save(sess, filename)
    mytrain.copy_meta(meta_name, filename)
    #saver.save(sess, filename)#,write_meta_graph=False)
    models.append([p[0],filename,p[1]])
  hparams.prune_mask = None
  return models,sess,model_p

def mask_weight_to_saver(FLAGS, hparams,file_name,need_do_layer,meta_name,eval_input_fn,model_fn,masks):
    models=[]

    print("weight_to_saver Loading model from...", file_name)
    orig_weights,sess,model_p = get_weights_dire(file_name, eval_input_fn, model_fn)
    hparams.prune_mask = masks
    sess,model_p=saver_restore(file_name,eval_input_fn,model_fn)

    prune_fn = get_prune_fn(FLAGS.prune)()
    w_copy = dict(orig_weights)
    for weight_name in need_do_layer:
        mask = masks[weight_name]
        mask = mask.reshape([-1, mask.shape[-1]])
        post_weights_pruned, weight_counts = new_prune_weights1(
          prune_fn,
          w_copy,
          weight_name,
          mask)
        for v in tf.trainable_variables():
          if is_prunable_weight(v) and v.name.strip(":0")==weight_name:
            assign_op = v.assign(
              np.reshape(post_weights_pruned[v.name.strip(":0")], v.shape))
            sess.run(assign_op)
            break
    saver_save(sess, file_name)
    mytrain.copy_meta(meta_name, file_name)
    return sess,model_p

def init_flags():
  tf.flags.DEFINE_integer("oi_select_num",3,"Number of selected models for every outermost iteration")
  tf.flags.DEFINE_integer("direction", 1, "True means that ga proecess is from bottom to top, False means that ga proecess is from top to bottom.")
  tf.flags.DEFINE_integer("debug",0,"Whether to debug.")
  tf.flags.DEFINE_float("prune_rate",0.7,"Rate of prune for every layer.")
  tf.flags.DEFINE_float("variation_rate",0.2,"Rate of variation for the ga process.")
  tf.flags.DEFINE_float("cross_rate", 0.8, "Rate of cross for the ga process.")

  tf.flags.DEFINE_string("model", None, "Which model to use.")
  tf.flags.DEFINE_string("data", None, "Which data to use.")
  tf.flags.DEFINE_string("env", "local", "Which environment to use.")
  tf.flags.DEFINE_string("hparams", "mnist_lenet", "Which hparams to use.")
  tf.flags.DEFINE_string("hparam_override", "",
                         "Run-specific hparam settings to use.")
  tf.flags.DEFINE_string("output_dir", None, "The output directory.")
  tf.flags.DEFINE_string("data_dir", None, "The data directory.")
  tf.flags.DEFINE_string(
      "post_weights_dir", "",
      "folder of the weights, if not set defaults to output_dir")
  tf.flags.DEFINE_string("prune_percent", "0.5",
                         "percent of weights to prune, comma separated")
  tf.flags.DEFINE_string("prune", "prune_mask", "one_shot or fisher")#tf.flags.DEFINE_string("prune", "weight", "one_shot or fisher")#
  tf.flags.DEFINE_boolean("variational", False, "use evaluate")
  tf.flags.DEFINE_string("eval_file", "eval_prune_results",
                         "file to put results")
  tf.flags.DEFINE_boolean("fresh", False, "Remove output_dir before running.")
  
  tf.flags.DEFINE_integer("eval_steps",10,"Steps of evaluation for the ga process.")
  
  tf.flags.DEFINE_integer("Outermost_iterations", 200,"Number of outside iteration.")
  tf.flags.DEFINE_integer("train_epochs_easy", 10000,"Max number of easy training's epochs.")
  tf.flags.DEFINE_integer("best_go_max_easy", 20,"Number of early-stop times for easy training.")
  tf.flags.DEFINE_integer("train_epochs",50000,"Number of train epochs for every training process.")
  tf.flags.DEFINE_integer("ga_iterations",20,"Number of ga iterations.")
  tf.flags.DEFINE_integer("population_num",100,"Number of population for every generation")
  tf.flags.DEFINE_integer("train_steps", 10000,"Number of training steps to perform.")
  tf.flags.DEFINE_integer("eval_every", 2000,"Number of steps between evaluations.")
  tf.flags.DEFINE_integer("gen_best_num",2,"Number of selected models for every ga process.")
  tf.flags.DEFINE_integer("layer_select_num",2,"Number of selected models for every layer")
  tf.flags.DEFINE_integer("model_select_num",1,"Number of best models for every outermost iteration.")
  tf.flags.DEFINE_integer("best_go_max",50,"Max times of being lower than best")
  


def find_weight_in_needdolayer(weightname,need_do_layer,direction):
  l = len(need_do_layer)
  try:
    pos = need_do_layer.index(weightname)
  except ValueError:
    if direction:
      pos = 0
    else:
      pos = l - 1

  if direction:
    r = range(pos,l)
  else:
    r = range(pos,-1,-1)
  if weightname == "":
    return pos,need_do_layer[pos],r
  return need_do_layer.index(weightname),weightname,r

def get_next_weightname(doing_weightname,need_do_layer,direction):
  l = len(need_do_layer)
  try:
    pos = need_do_layer.index(doing_weightname)
  except ValueError:
    if direction:
      pos = 0
    else:
      pos = l - 1
  if direction:
    pos = pos + 1
    if(pos>=l):
      pos=0
  else:
    pos = pos - 1
    if(pos<0):
      pos = l - 1
  return need_do_layer[pos]

def test_eval_model(sess,model_p,old_ckpt):
  fits=[]
  stamps=[]

  def saver_restore_part(sess, ckpt_dir):
    all_var_list = [var for var in tf.trainable_variables()]
    saver = tf.train.Saver(var_list=all_var_list)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver.restore(sess, ckpt_dir)

  ckpt_dir = old_ckpt
  print("Loading model from...", ckpt_dir)
  saver_restore_part(sess, ckpt_dir)
  for i in range(100):
    print(sess.run(model_p.eval_metric_ops['acc']))#loss))[0]))
  print(sess.run(model_p.eval_metric_ops['acc'][0]))
  sess.run(tf.local_variables_initializer())
  print(sess.run(model_p.eval_metric_ops['acc']))
  print(sess.run(model_p.eval_metric_ops['acc']))
  return fits,stamps    

def direction_run():
  init_flags()
  FLAGS = tf.app.flags.FLAGS
  if FLAGS.debug:
    FLAGS.Outermost_iterations = 20
    FLAGS.train_epochs_easy=2000
    FLAGS.best_go_max_easy=200
    FLAGS.train_epochs = 20000
    FLAGS.ga_iterations = 20
    FLAGS.population_num = 100
    FLAGS.train_steps = 1000
    FLAGS.eval_every = 2000
    FLAGS.gen_best_num = 3
    FLAGS.layer_select_num = 2
    FLAGS.model_select_num=3
    FLAGS.best_go_max = 200


  eval_file = open(FLAGS.eval_file, "w")

  hparams_list = FLAGS.hparams.split(",")
  total_evals = {}
  hparam_name=hparams_list[0]

  hparams = get_hparams(hparam_name)
  hparams = hparams.parse(FLAGS.hparam_override)
  hparams = flags.update_hparams(FLAGS, hparams,hparam_name)

  FLAGS.eval_steps = int(1000/hparams.batch_size)
  hparams.eval_steps = int(10000/hparams.batch_size)

  need_do_layer = need_do_layers[hparam_name]
  train_input_fn, eval_input_fn, test_input_fn = get_input_fns(hparams, generate=False)
  features, labels = eval_input_fn()

  model_fn = get_model(hparams)
  sess = tf.Session()
  gs=tf.train.get_or_create_global_step()
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  model_p = model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)

  models=[]
  model_cnt=0
  checkpoint_arr=[]
  same_layer=False
  if(os.path.exists(check_file)):
    f = open(check_file, 'r')
    for l in f.readlines():
      l = l[:-1]
      ls = l.split(",")
      weight_name = ""
      weight_id=-1
      if(len(ls)>1):
        weight_id=int(ls[2])
        weight_name=ls[3]
      acc = float(ls[0])
      whole_copy_file_name = ls[1] + "_copy"
      src_file_name = os.path.basename(ls[1])
      copy_file_name = src_file_name + "_copy"
      mytrain.filecopy(src=ls[1], des=hparams.output_dir, src_file_name=src_file_name, des_file_name=copy_file_name)
      models.append([acc,whole_copy_file_name,weight_id,weight_name])
      cnt = len(checkpoint_arr)
      if(weight_id>-1):
        if(same_layer):
            checkpoint_arr[cnt-1].append(models[model_cnt])
        else:
            checkpoint_arr.append([])
            checkpoint_arr[cnt].append(models[model_cnt])
        same_layer=True    
      else:
        checkpoint_arr.append([])
        checkpoint_arr[cnt].append(models[model_cnt])
        same_layer=False
      model_cnt+=1  
    f.close()
  else:
    output_dir = tf.train.latest_checkpoint(hparams.output_dir)
    if output_dir is None:
      return
    models = [[1.0, output_dir,-1,""]]
    checkpoint_arr.append([])
    checkpoint_arr[0].append(models[0])

  #test_eval_model(sess, model_p, models[0][1])
  meta_name=models[0][1]+".meta"


  for m in models:
    
    write_to_check_file(content=m[1],model_dir=hparams.output_dir)
    output_dir111 = tf.train.latest_checkpoint(hparams.output_dir)
    per, prune, size, orig_weights,sess,model_p = compute_model_prune_dire(
      output_dir=output_dir111, need_do_layer=need_do_layer,eval_input_fn=eval_input_fn,model_fn=model_fn)
    print("@@@@@@@@@@@@@@@prune  ")
    print(prune)
    #mytrain.copy_meta(meta_name,m[1])
    acc1 = mytrain.test_new_eval_test(FLAGS, hparams, model_fn, eval_input_fn, test_input_fn,sess=sess)
    print(acc1)

  ga_file_name_frefix = os.path.join(hparams.output_dir,"ga-")

  def train_best(bests, train_epochs, best_go_max):
    v_acc_s = []
    for c, j in enumerate(bests):
      tf.reset_default_graph()
      features, labels = eval_input_fn()
      gs = tf.train.get_or_create_global_step()
      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      model_p = model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)
      saver = tf.train.Saver()
      saver.restore(sess, j[1])

      orig_weights,sess,model_p = get_weights_dire(output_dir=j[1], eval_input_fn=eval_input_fn,model_fn=model_fn)
      masks = weights_to_masks(orig_weights)
      hparams.prune_mask = masks

      sess, model_p = mask_weight_to_saver(FLAGS, hparams, j[1], need_do_layer, meta_name, eval_input_fn, model_fn,
                                           masks)
      write_to_check_file(content=j[1], model_dir=hparams.output_dir)
      acc1 = mytrain.test_new_eval_test(FLAGS, hparams, model_fn, eval_input_fn, test_input_fn, sess=sess)
      #mytrain.copy_meta(meta_name,j[1])
      v_acc = mytrain.test1(FLAGS, hparams, train_epochs, model_fn, train_input_fn, eval_input_fn,
                            restore_file=j[1], save_file=j[1], gens=[first_l, c, j[0]], best_go_max=best_go_max)
      #newmodels = weight_to_saver_in_train_best(FLAGS=FLAGS, filename=j[1], weight_names=weight_names, sess=sess,
      #                            bests=mask, meta_name=meta_name, old_ckpt=j[1])
      '''
      per, prune, size, orig_weights,sess,model_p = compute_model_prune_dire(output_dir=j[1], need_do_layer=need_do_layer,eval_input_fn=eval_input_fn,model_fn=model_fn)
      print("@@@@@@@@@@@@@@@####  ")
      # print(orig_weights)
      print(prune)
      '''
      sess, model_p = mask_weight_to_saver(FLAGS, hparams,j[1],need_do_layer,meta_name,eval_input_fn,model_fn,masks)

      write_to_check_file(content=j[1], model_dir=hparams.output_dir)

      acc1 = mytrain.test_new_eval_test(FLAGS, hparams, model_fn, eval_input_fn, test_input_fn, sess=sess)
      print("########   acc1:")
      print(acc1)
      v_acc_s.append([acc1['eval']['acc'], j[1], acc1['test']['acc']])
      print("v_acc and t_acc: " + str(acc1['eval']['acc']) + " ," + str(acc1['test']['acc']))
      '''
      per, prune, size, orig_weights,sess,model_p = compute_model_prune_dire(output_dir=j[1], need_do_layer=need_do_layer,eval_input_fn=eval_input_fn,model_fn=model_fn)
      print("@@@@@@@@@@@@@@@####  ")
      print(prune)      
      '''
    return v_acc_s,model_p

  for first_l in range(FLAGS.Outermost_iterations):
    size = len(checkpoint_arr)
    best_models=[]
    for i in range(size):
      selected_layer_best=[]
      doing_weightname=checkpoint_arr[i][0][3]
      weight_id, doing_weightname, r = find_weight_in_needdolayer(doing_weightname, need_do_layer, FLAGS.direction)
      for ch in checkpoint_arr[i]:
        selected_layer_best.append([ch[0], ch[1], doing_weightname])
      cnt = 0
      for e in r:
        S=[]
        doing_weightname = need_do_layer[e]
        prefix_k=ga_file_name_frefix + str(cnt) +"_"
        for k,lbest in enumerate(selected_layer_best):
          output_dir = lbest[1]
          old_ckpt = output_dir

          orig_weights,sess,model_p = get_weights_dire(output_dir=output_dir, eval_input_fn=eval_input_fn,model_fn=model_fn)
          weights = dict(orig_weights)
          write_to_check_file(model_dir=hparams.output_dir, content=output_dir)
          #mytrain.copy_meta(meta_name,output_dir)
          myga=ga.Ga_layer(FLAGS=FLAGS, hparam_name=hparam_name,hparams=hparams,
                           layer_name=doing_weightname,orig_weights=orig_weights,
                           sess=sess,model_fn=model_fn,eval_input_fn=eval_input_fn,
                           out_gen=[first_l,i,doing_weightname],ep=FLAGS.ga_iterations,
                           population_num=FLAGS.population_num,best_num=FLAGS.gen_best_num,
                           prune_rate=FLAGS.prune_rate,v_rate=FLAGS.variation_rate,
                           c_rate=FLAGS.cross_rate,eval_steps=FLAGS.eval_steps,meta_name=meta_name,
                           model_p=model_p)
          layer_best,sess=myga.evo(weight_name=doing_weightname, old_ckpt=old_ckpt,eval_input_fn=eval_input_fn,model_fn=model_fn)
          total_evals[hparam_name] = layer_best
          newmodels,sess,model_p = weight_to_saver1(FLAGS, hparams ,prefix_k, k, doing_weightname, layer_best,  meta_name,old_ckpt,eval_input_fn,model_fn)
          per, prune, size, orig_weights,sess,model_p = compute_model_prune_dire(output_dir=newmodels[0][1], need_do_layer=need_do_layer,eval_input_fn=eval_input_fn,model_fn=model_fn)
          print("@@@@@@@@@@@@@@@####  ")
          # print(orig_weights)
          print(prune)
          S = S + newmodels

        v_acc_s,model_p = train_best(S,train_epochs=FLAGS.train_epochs_easy,best_go_max=FLAGS.best_go_max_easy)
        v_acc_s.sort(key=takeFirst, reverse=True)
        v_acc_len = len(v_acc_s)
        if(v_acc_len>FLAGS.layer_select_num):
            v_acc_len = FLAGS.layer_select_num
        selected_layer_best = v_acc_s[0:v_acc_len]
        models_acc = []
        next_weigthname = get_next_weightname(doing_weightname,need_do_layer,FLAGS.direction)
        checkpoint_arr[i]=[]
        for v_acc_cnt in range(v_acc_len):
          per,prune,size,orig_weights,sess,model_p = compute_model_prune_dire(output_dir=selected_layer_best[v_acc_cnt][1], need_do_layer=need_do_layer,eval_input_fn=eval_input_fn,model_fn=model_fn)
          models_acc.append(selected_layer_best[v_acc_cnt][0])
          models_acc.append(selected_layer_best[v_acc_cnt][2])
          models_acc.append(per)
          models_acc.append(prune)
          models_acc.append(size)
          models_acc.append(selected_layer_best[v_acc_cnt][1])
          checkpoint_arr[i].append([selected_layer_best[v_acc_cnt][0],selected_layer_best[v_acc_cnt][1],e+1,next_weigthname])
          
        str_checkpoint = ""
        for str_c in checkpoint_arr:
          for str_s in str_c:
            str_checkpoint = str_checkpoint + str(str_s[0])+ ","+str_s[1] + ","+str(str_s[2]) + ","+str_s[3] + "\n"
        f = open(check_file, 'w')
        f.writelines(str_checkpoint)
        f.close()
        
        ga.writer_to_csv("layer_best_models.csv", [[first_l,i,doing_weightname] + models_acc])
        cnt = (cnt + 1) % 2
      start_pos = len(best_models)
      for e,lbest in enumerate(selected_layer_best):
        pos=start_pos+e
        check_file_name = check_file_model + str(pos)
        src_file_name = os.path.basename(selected_layer_best[e][1])
        src = selected_layer_best[e][1]
        new_file_name=mytrain.filecopy(src=src, des=hparams.output_dir, src_file_name=src_file_name, des_file_name=check_file_name)
        selected_layer_best[e][1]=new_file_name

      best_models = best_models + selected_layer_best

    best_models.sort(key=takeFirst, reverse=True)
    best_models_len = len(best_models)
    if (best_models_len > FLAGS.model_select_num):
      best_models_len = FLAGS.model_select_num
    models = best_models[0:best_models_len]
    models,model_p = train_best(models, train_epochs=FLAGS.train_epochs, best_go_max=FLAGS.best_go_max)
    models_acc = []
    checkpoint_arr=[]
    for v_acc_cnt in range(best_models_len):
      per, prune, size, orig_weights,sess,model_p = compute_model_prune_dire(output_dir=models[v_acc_cnt][1], need_do_layer=need_do_layer,eval_input_fn=eval_input_fn,model_fn=model_fn)
      models_acc.append(models[v_acc_cnt][0])
      models_acc.append(models[v_acc_cnt][2])
      models_acc.append(per)
      models_acc.append(prune)
      models_acc.append(size)
      c_file_name = models[v_acc_cnt][1]
      models_acc.append(c_file_name)
      src_file_name = os.path.basename(c_file_name)
      copy_file_name = src_file_name + "_copy"
      new_str=mytrain.filecopy(src=c_file_name, des=hparams.output_dir, src_file_name=src_file_name, des_file_name=copy_file_name)
      models[v_acc_cnt][1]=new_str
      whole_check_file_name = c_file_name#os.path.join(hparams.output_dir, c_file_name)
      checkpoint_arr.append([])
      checkpoint_arr[v_acc_cnt].append([models[v_acc_cnt][0],whole_check_file_name,-1,""])

    ga.writer_to_csv("best_models.csv", [[first_l, i] + models_acc])
    str_checkpoint = ""
    for str_c in checkpoint_arr:
      for str_s in str_c:
        str_checkpoint = str_checkpoint + str(str_s[0])+ ","+str_s[1] + ","+str(str_s[2]) + ","+str_s[3] + "\n"
    f = open(check_file, 'w')
    f.writelines(str_checkpoint)
    f.close()
    hparams.prune_mask=None

  tf.reset_default_graph()
  eval_file.close()




if __name__ == "__main__":
  #eval.main()
  #print("aa")
  direction_run()#,)#,new_run()#debug=True,env="local_lewis",env="local_lewis"

