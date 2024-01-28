import TD.scripts.prune.eval as eval
import TD.ga.ga as ga
import tensorflow as tf
import os
import TD.train as mytrain
import numpy as np
import copy

from TD.training import flags
from TD.models.registry import get_model
from TD.hparams.registry import get_hparams
from TD.data.registry import get_input_fns
from TD.scripts.prune.prune import get_prune_fn

best_path = "bestpath"
ga_path="gapath"
check_file = "checkpoint.csv"
check_file_model="cp_model"
need_do_layers = {"resnet_mask":["sub","init_conv"],"cifar_lenet_mask":["lenet/conv1/DW","lenet/conv2/DW"],"cifar100_vgg16_no_dropout":["vgg/conv1_1/DW","vgg/conv1_2/DW","vgg/conv2_1/DW","vgg/conv2_2/DW","vgg/conv3_1/DW","vgg/conv3_2/DW","vgg/conv3_3/DW","vgg/conv4_1/DW","vgg/conv4_2/DW","vgg/conv4_3/DW","vgg/conv5_1/DW","vgg/conv5_2/DW","vgg/conv5_3/DW"]}
#need_do_layers=["conv1_1","conv1_2","conv2_1","conv2_2","conv3_1","conv3_2","conv3_3","conv4_1","conv4_2","conv4_3","conv5_1","conv5_2","conv5_3"]


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

def get_weights(output_dir, sess, saver):
  print("Loading model from...", output_dir)
  saver.restore(sess, output_dir)
  orig_weights = get_current_weights(sess)
  return orig_weights, saver

def check_in_needdolayer(weight_name,need_do_layer):
  for name in need_do_layer:
    if name in weight_name:
      return True
  return False

def compute_model_prune(output_dir, sess, saver,need_do_layer):
  orig_weights,saver = get_weights(output_dir=output_dir,sess=sess,saver=saver)
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
  return total_per,total_prune,total_size,orig_weights

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


def weight_to_saver(FLAGS, filename, weight_names, sess, bests, saver, old_ckpt):
  models = []
  ckpt_dir = old_ckpt
  saver = tf.train.Saver(max_to_keep=1)
  saver.restore(sess, ckpt_dir)
  orig_weights = get_current_weights(sess)
  for weight_name in weight_names:
    p = bests[weight_name]

    print("weight_to_saver Loading model from...", ckpt_dir)

    w_copy = orig_weights[weight_name]

    post_weights_pruned, weight_counts = new_prune_weights(
      set_mask,
      w_copy,
      weight_name,
      p)
    for v in tf.trainable_variables():
      if is_prunable_weight(v) and v.name.strip(":0") == weight_name:
        assign_op = v.assign(
          np.reshape(post_weights_pruned[v.name.strip(":0")], v.shape))
        sess.run(assign_op)

  saver.save(sess, filename)


  return filename


def test_new_run(debug=False,env="local"):
  if debug:
    n=8
    train_epochs=1
    ep=2
    population_num=10
    train_steps=1
    eval_every=1
    gen_best_num=2
  else:
    n = 200
    train_epochs = 6000
    ep = 20
    population_num = 50
    train_steps = 10000
    eval_every = 5000
    gen_best_num=2

  eval.new_init_flags(train_steps=train_steps, eval_every=eval_every, env=env)
  FLAGS = tf.app.flags.FLAGS
  eval_file = open(FLAGS.eval_file, "w")

  hparams_list = FLAGS.hparams.split(",")
  total_evals = {}
  hparam_name=hparams_list[0]

  hparams = get_hparams(hparam_name)
  hparams = hparams.parse(FLAGS.hparam_override)
  hparams = flags.update_hparams(FLAGS, hparams,hparam_name)

  need_do_layer = need_do_layers[hparam_name]

  model_fn = get_model(hparams)
  train_input_fn, eval_input_fn, test_input_fn = get_input_fns(hparams, generate=False)
  sess = tf.Session()
  gs=tf.train.get_or_create_global_step()
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  features, labels = eval_input_fn()
  model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)
  models=[]
  #if (os.path.exists(check_file)):

  if(os.path.exists(check_file)):
    f = open(check_file, 'r')
    for l in f.readlines():
      l = l[:-1]
      whole_copy_file_name = l + "_copy"
      src_file_name = os.path.basename(l)
      copy_file_name = src_file_name + "_copy"
      mytrain.filecopy(src=l, des=hparams.output_dir, src_file_name=src_file_name, des_file_name=copy_file_name)
      models.append([1.0,whole_copy_file_name])
    f.close()
  else:
    output_dir = tf.train.latest_checkpoint(hparams.output_dir)
    if output_dir is None:
      return
    models = [[1.0, output_dir]]


  forward_saver = tf.train.Saver(max_to_keep=100)

  #orig_weights, saver = get_weights(output_dir=output_dir111, sess=sess, saver=forward_saver)
  #mask = weights_to_masks(orig_weights)
  #hparams.prune_mask = mask



  for m in models:
    write_to_check_file(content=m[1],model_dir=hparams.output_dir)
    acc1 = mytrain.test_new_eval_test(FLAGS, hparams, model_fn, eval_input_fn, test_input_fn,sess=sess)
    print(acc1)
    output_dir111 = tf.train.latest_checkpoint(hparams.output_dir)
    per, prune, size,orig_weights = compute_model_prune(
        output_dir=output_dir111, sess=sess, saver=forward_saver,
        need_do_layer=need_do_layer)
    print("@@@@@@@@@@@@@@@prune  ")
    print(prune)
  ga_file_name_frefix = os.path.join(hparams.output_dir,"ga-")
  for first_l in range(n):
    bests = []
    cnt = 0
    size = len(models)
    #forward_saver = tf.train.Saver(max_to_keep=100)
    for i in range(size):
      output_dir = models[i][1]
      orig_weights,saver = get_weights(output_dir=output_dir,sess=sess,saver=forward_saver)
      weights = dict(orig_weights)
      write_to_check_file(model_dir=hparams.output_dir, content=output_dir)
      old_ckpt=output_dir
      for e, weight_name in enumerate(weights):
        if check_in_needdolayer(weight_name,need_do_layer):
          myga=ga.Ga_layer(FLAGS=FLAGS, hparam_name=hparam_name,hparams=hparams,layer_name=weight_name,orig_weights=orig_weights,sess=sess,model_fn=model_fn,eval_input_fn=eval_input_fn,out_gen=[first_l,i,weight_name],ep=ep,population_num=population_num,best_num=gen_best_num)
          layer_best=myga.evo(weight_name=weight_name, saver=forward_saver,old_ckpt=old_ckpt)
          #print(hparam_name, ":", layer_best)

          total_evals[hparam_name] = layer_best
          #model_fn = get_model(hparams)
          #model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)
          newmodels = eval.weight_to_saver(FLAGS=FLAGS, hparam_name=hparam_name, file_name_frefix=ga_file_name_frefix, cnt=cnt, weight_name=weight_name, sess=sess,
                                           bests=layer_best, orig_weights=orig_weights, saver=forward_saver,old_ckpt=old_ckpt)
          cnt = cnt + len(newmodels)
          bests = bests + newmodels
    v_acc_s = []
    for c, j in enumerate(bests):
      '''
      per,prune,size,orig_weights = compute_model_prune(output_dir=j[1], sess=sess, saver=forward_saver,need_do_layer=need_do_layer)
      print("@@@@@@@@@@@@@@@  ")
      print(orig_weights)
      print(prune)
      '''
      orig_weights, saver = get_weights(output_dir=j[1], sess=sess, saver=forward_saver)
      mask = weights_to_masks(orig_weights)
      #mask = j[2]
      hparams.prune_mask = mask
      weight_names=[]
      for e, weight_name in enumerate(weights):
        if check_in_needdolayer(weight_name,need_do_layer):
          weight_names.append(weight_name)
      write_to_check_file(content=j[1],model_dir=hparams.output_dir)
      #save_file=
      v_acc,t_acc = mytrain.test1(FLAGS, hparams, output_dir, train_epochs, model_fn, train_input_fn, eval_input_fn,
                              test_input_fn, sess, restore_file=j[1],save_file=j[1], gens=[first_l, c, j[0]])
      #v_acc_s.append([v_acc, j[1],t_acc])

      newmodels = weight_to_saver(FLAGS=FLAGS, filename=j[1],weight_names=weight_names, sess=sess,
                                       bests=mask, saver=forward_saver,old_ckpt=j[1])

      per,prune,size,orig_weights = compute_model_prune(output_dir=j[1], sess=sess, saver=forward_saver,need_do_layer=need_do_layer)
      print("@@@@@@@@@@@@@@@####  ")
      #print(orig_weights)
      print(prune)

      write_to_check_file(content=j[1], model_dir=hparams.output_dir)
      acc1 = mytrain.test_new_eval_test(FLAGS, hparams, model_fn, eval_input_fn, test_input_fn, sess=sess)
      print("########   acc1:")
      print(acc1)
      v_acc_s.append([acc1['eval']['acc'], j[1], acc1['test']['acc']])
      print("v_acc and t_acc: "+ str(v_acc) + " ," + str(t_acc))
    v_acc_s.sort(key=takeFirst, reverse=True)
    v_acc_len = len(v_acc_s)
    if(v_acc_len>3):
        v_acc_len = 3
    models = v_acc_s[0:v_acc_len]
    models_acc = []
    checkpoints=[]
    for i in range(v_acc_len):
      per,prune,size,orig_weights = compute_model_prune(output_dir=models[i][1], sess=sess, saver=forward_saver,need_do_layer=need_do_layer)
      models_acc.append(models[i][0])
      models_acc.append(models[i][2])
      models_acc.append(per)
      models_acc.append(prune)
      models_acc.append(size)
      check_file_name = check_file_model + str(i)
      src_file_name = os.path.basename(models[i][1])
      src = models[i][1]
      mytrain.filecopy(src=src, des=hparams.output_dir, src_file_name=src_file_name, des_file_name=check_file_name)
      whole_check_file_name = os.path.join(hparams.output_dir, check_file_name)
      checkpoints.append(whole_check_file_name + "\n")

    ga.writer_to_csv("best_models.csv", [[first_l] + models_acc])
    f = open(check_file,'w')
    f.writelines(checkpoints)
    f.close()
    hparams.prune_mask=None

  tf.reset_default_graph()


  eval_file.close()

if __name__ == "__main__":
  #eval.main()
  #print("aa")
  test_new_run(debug=False,env="local_lewis")#,)#,new_run()#debug=True,env="local_lewis",env="local_lewis"

