import os
import sys
import argparse

import tensorflow as tf
import opennmt as onmt

from opennmt import constants
from opennmt.utils import decay
from opennmt.utils import losses
from opennmt.utils import misc
from opennmt.utils import optim

from tqdm import tqdm


# Define the "base" Transformer model.
source_inputter = onmt.inputters.WordEmbedder("source_vocabulary", embedding_size=300)
target_inputter = onmt.inputters.WordEmbedder("target_vocabulary", embedding_size=300)

encoder = onmt.encoders.BidirectionalRNNEncoder(
  num_layers=2, 
  num_units=300)
decoder = onmt.decoders.AttentionalRNNDecoder(
  num_layers=2,
  num_units=300)
'''
encoder = onmt.encoders.SelfAttentionEncoder(
    num_layers=6,
    num_units=512,
    num_heads=8,
    ffn_inner_dim=2048,
    dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.1)
decoder = onmt.decoders.SelfAttentionDecoder(
    num_layers=6,
    num_units=512,
    num_heads=8,
    ffn_inner_dim=2048,
    dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.1)
'''

def build_model(source, 
                  target, 
                  mode,
                  reuse=False):
  # Encode the source.
  with tf.variable_scope("encoder", reuse=reuse):
    source_embedding = source_inputter.make_inputs(source, training=True)
    memory, _, _ = encoder.encode(source_embedding, source["length"], mode=mode)

  # Decode the target.
  with tf.variable_scope("decoder", reuse=reuse):
    target_embedding = target_inputter.make_inputs(target, training=True)
    logits, _, _ = decoder.decode(
        target_embedding,
        target["length"],
        vocab_size=target_inputter.vocabulary_size,
        mode=mode,
        memory=memory,
        memory_sequence_length=source["length"])
    #logits = tf.Print(logits, [tf.argmax(logits, -1)[0]], summarize=maximum_length)
    #logits = tf.Print(logits, [tr_target['ids_out'][0]], summarize=maximum_length)
  # Compute the loss.
  loss, normalizer, _ = losses.cross_entropy_sequence_loss(
      logits,
      target["ids_out"],
      target["length"],
      label_smoothing=0.1,
      average_in_time=True,
      mode=mode)
  loss /= normalizer  

  return loss

def train(model_dir,
          tr_inputter,
          tr_src,
          tr_tgt,
          maximum_length=1000,
          shuffle_buffer_size=50000,
          gradients_accum=8,
          train_steps=10000,#15000,
          save_every=1000,
          report_every=100):
  """Runs the training loop.
  Args:
    model_dir: Directory where checkpoints are saved.
    example_inputter: The inputter instance that produces the training examples.
    source_file: The source training file.
    target_file: The target training file.
    maximum_length: Filter sequences longer than this.
    shuffle_buffer_size: How many examples to load for shuffling.
    gradients_accum: Accumulate gradients of this many iterations.
    train_steps: Train for this many iterations.
    save_every: Save a checkpoint every this many iterations.
    report_every: Report training progress every this many iterations.
  """
  mode = tf.estimator.ModeKeys.TRAIN

  # Create the dataset.
  tr_dataset = tr_inputter.make_training_dataset(
      tr_src,
      tr_tgt,
      batch_size=128,
      batch_type="tokens",
      shuffle_buffer_size=shuffle_buffer_size,
      #bucket_width=1,  # Bucketize sequences by the same length for efficiency.
      maximum_features_length=maximum_length,
      maximum_labels_length=maximum_length)
  tr_iterator = tr_dataset.make_initializable_iterator()
  tr_source, tr_target = tr_iterator.get_next()

  tr_loss = build_model(tr_source, tr_target, mode) 

  with tf.name_scope('losses'):
    tf.summary.scalar('tr_loss', tr_loss)
  summ_op = tf.summary.merge_all()

  # Define the learning rate schedule.
  step = tf.train.create_global_step()

  # Define the optimization op.
  optimizer = tf.train.AdamOptimizer(0.001)

  optim_variables = tf.trainable_variables()
  gradients = tf.gradients(tr_loss, optim_variables)
  gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
  train_op = optimizer.apply_gradients(zip(gradients, optim_variables), global_step=step)

  # Runs the training loop.
  saver = tf.train.Saver()
  checkpoint_path = None
  if os.path.exists(model_dir):
    checkpoint_path = tf.train.latest_checkpoint(model_dir)
  with tf.Session() as sess:
    bar = tqdm(total=train_steps, desc='Train')
    writer = tf.summary.FileWriter(model_dir, sess.graph)

    if checkpoint_path is not None:
      print("Restoring parameters from %s" % checkpoint_path)
      saver.restore(sess, checkpoint_path)
    else:
      sess.run(tf.global_variables_initializer())

    sess.run(tf.variables_initializer(optim_variables))
    sess.run(tf.tables_initializer())
    sess.run(tr_iterator.initializer)
    last_step = -1
    while True:
      bar.update(1)

      step_, loss_, summ ,_ = sess.run([step, tr_loss, summ_op, train_op])
      if step_ != last_step:
        if step_ % report_every == 0:
          writer.add_summary(summ, step_)
        if step_ % save_every == 0:
          print("Saving checkpoint for step %d" % step_)
          saver.save(sess, "%s/model" % model_dir, global_step=step_)
        if step_ == train_steps:
          break
      last_step = step_
  bar.close()

def train_and_eval(model_dir,
          tr_inputter,
          val_inputter,
          tr_src,
          tr_tgt,
          val_src,
          val_tgt,
          maximum_length=1000,
          shuffle_buffer_size=1000000,
          gradients_accum=8,
          train_steps=15000,
          save_every=1000,
          report_every=50):
  """Runs the training loop.
  Args:
    model_dir: Directory where checkpoints are saved.
    example_inputter: The inputter instance that produces the training examples.
    source_file: The source training file.
    target_file: The target training file.
    maximum_length: Filter sequences longer than this.
    shuffle_buffer_size: How many examples to load for shuffling.
    gradients_accum: Accumulate gradients of this many iterations.
    train_steps: Train for this many iterations.
    save_every: Save a checkpoint every this many iterations.
    report_every: Report training progress every this many iterations.
  """
  mode = tf.estimator.ModeKeys.TRAIN

  # Create the dataset.
  tr_dataset = tr_inputter.make_training_dataset(
      tr_src,
      tr_tgt,
      batch_size=32,
      batch_type="tokens",
      shuffle_buffer_size=shuffle_buffer_size,
      #bucket_width=1,  # Bucketize sequences by the same length for efficiency.
      maximum_features_length=maximum_length,
      maximum_labels_length=maximum_length)
  tr_iterator = tr_dataset.make_initializable_iterator()
  tr_source, tr_target = tr_iterator.get_next()

  val_dataset = val_inputter.make_training_dataset(
      val_src,
      val_tgt,
      batch_size=32,
      batch_type="tokens",
      shuffle_buffer_size=shuffle_buffer_size,
      maximum_features_length=maximum_length,
      maximum_labels_length=maximum_length)
  val_iterator = val_dataset.make_initializable_iterator()
  val_source, val_target = val_iterator.get_next()

  tr_loss = build_model(tr_source, tr_target, mode) 
  val_loss = build_model(val_source, val_target, mode, reuse=True)

  with tf.name_scope('losses'):
    tf.summary.scalar('tr_loss', tr_loss)
    tf.summary.scalar('val_loss', val_loss)
  summ_op = tf.summary.merge_all()

  # Define the learning rate schedule.
  step = tf.train.create_global_step()

  # Define the optimization op.
  optimizer = tf.train.AdamOptimizer(0.001)
  #learning_rate = tf.cond(step > 4000,
  #  lambda: tf.train.exponential_decay(0.7, step, 200, 0.7, staircase=True),
  #  lambda: 0.7)
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  optim_variables = tf.trainable_variables()
  gradients = tf.gradients(tr_loss, optim_variables)
  gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
  train_op = optimizer.apply_gradients(zip(gradients, optim_variables), global_step=step)
  '''
  learning_rate = decay.noam_decay_v2(2.0, step, model_dim=300, warmup_steps=4000)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  gradients = optimizer.compute_gradients(loss)
  train_op, optim_variables = optim.delayed_update(
      optimizer,
      gradients,
      step,
      accum_count=gradients_accum)
  '''

  # Runs the training loop.
  saver = tf.train.Saver()
  checkpoint_path = None
  if os.path.exists(model_dir):
    checkpoint_path = tf.train.latest_checkpoint(model_dir)
  with tf.Session() as sess:
    bar = tqdm(total=train_steps, desc='Train')
    writer = tf.summary.FileWriter(model_dir, sess.graph)

    if checkpoint_path is not None:
      print("Restoring parameters from %s" % checkpoint_path)
      saver.restore(sess, checkpoint_path)
    else:
      sess.run(tf.global_variables_initializer())

    sess.run(tf.variables_initializer(optim_variables))
    sess.run(tf.tables_initializer())
    sess.run(tr_iterator.initializer)
    sess.run(val_iterator.initializer)
    last_step = -1
    while True:
      bar.update(1)

      step_, loss_, summ ,_ = sess.run([step, tr_loss, summ_op, train_op])
      writer.add_summary(summ, step_)
      if step_ != last_step:
        #if step_ % report_every == 0:
        #  print("Step = %d ; Loss = %f" % (step_, loss_))
        if step_ % save_every == 0:
          print("Saving checkpoint for step %d" % step_)
          saver.save(sess, "%s/model" % model_dir, global_step=step_)
        if step_ == train_steps:
          break
      last_step = step_
  bar.close()

def translate(model_dir,
              example_inputter,
              source_file,
              batch_size=32,
              beam_size=10):
  """Runs translation.
  Args:
    model_dir: The directory to load the checkpoint from.
    example_inputter: The inputter instance that produces the training examples.
    source_file: The source file.
    batch_size: The batch size to use.
    beam_size: The beam size to use. Set to 1 for greedy search.
  """
  mode = tf.estimator.ModeKeys.PREDICT

  # Create the inference dataset.
  dataset = example_inputter.make_inference_dataset(source_file, batch_size)
  iterator = dataset.make_initializable_iterator()
  source = iterator.get_next()

  # Encode the source.
  with tf.variable_scope("encoder"):
    source_embedding = source_inputter.make_inputs(source)
    memory, _, _ = encoder.encode(source_embedding, source["length"], mode=mode)

  # Generate the target.
  with tf.variable_scope("decoder"):
    target_inputter.build()
    batch_size = tf.shape(memory)[0]
    start_tokens = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    end_token = constants.END_OF_SENTENCE_ID
    target_ids, _, target_length, _ = decoder.dynamic_decode_and_search(
        target_inputter.embedding,
        start_tokens,
        end_token,
        vocab_size=target_inputter.vocabulary_size,
        beam_width=beam_size,
        memory=memory,
        memory_sequence_length=source["length"],
        length_penalty=0.8)
    target_vocab_rev = target_inputter.vocabulary_lookup_reverse()
    target_tokens = target_vocab_rev.lookup(tf.cast(target_ids, tf.int64))

  # Iterates on the dataset.
  saver = tf.train.Saver()
  checkpoint_path = tf.train.latest_checkpoint(model_dir)
  with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    sess.run(tf.tables_initializer())
    sess.run(iterator.initializer)

    writer = open("test.txt", 'w')
    while True:
      try:
        batch_tokens, batch_length = sess.run([target_tokens, target_length])
        for tokens, length in zip(batch_tokens, batch_length):
          #misc.print_bytes(b" ".join(tokens[0][:length[0] - 1]))
          line = b" ".join(tokens[0][:length[0] - 1])
          writer.write(line.decode('utf-8'))
          writer.write("\n")
      except tf.errors.OutOfRangeError:
        break

def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("run", choices=["train", "train_and_eval", "translate"],
                      help="Run type.")
  parser.add_argument("--tr_src", default="data/train/source.txt", 
                      help="Path to the source file.")
  parser.add_argument("--tr_tgt", default="data/train/target.txt",
                      help="Path to the target file.")
  parser.add_argument("--val_src", default="data/val/source.txt", 
                      help="Path to the source file.")
  parser.add_argument("--val_tgt", default="data/val/target.txt",
                      help="Path to the target file.")
  parser.add_argument("--src_vocab", default="data/vocab/source.txt",
                      help="Path to the source vocabulary.")
  parser.add_argument("--tgt_vocab", default="data/vocab/target.txt",
                      help="Path to the target vocabulary.")
  parser.add_argument("--model_dir", default="checkpoint",
                      help="Directory where checkpoint are written.")
  args = parser.parse_args()

  tr_inputter = onmt.inputters.ExampleInputter(source_inputter, target_inputter)
  tr_inputter.initialize({
      "source_vocabulary": args.src_vocab,
      "target_vocabulary": args.tgt_vocab
  })

  val_inputter = onmt.inputters.ExampleInputter(source_inputter, target_inputter)
  val_inputter.initialize({
      "source_vocabulary": args.src_vocab,
      "target_vocabulary": args.tgt_vocab
  })

  if args.run == "train":
    train(args.model_dir, tr_inputter,args.tr_src, args.tr_tgt)
  elif args.run == "train_and_eval":
    train(args.model_dir, tr_inputter, val_inputter, args.tr_src, args.tr_tgt, args.val_src, args.val_tgt)
  elif args.run == "translate":
    translate(args.model_dir, val_inputter, args.val_src)


if __name__ == "__main__":
  main()