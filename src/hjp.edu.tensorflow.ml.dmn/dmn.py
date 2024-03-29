import sys
import time

import numpy as np
from copy import deepcopy

import tensorflow as tf
from attn import AttentionGRUCell

from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

import babi

class Config(object):
    batch_size = 100
    embed_size = 80
    hidden_size = 80

    max_epochs = 256
    early_stopping = 20

    dropout = 0.9
    lr = 0.001
    l2 = 0.001

    cap_grads = False
    max_grad_val = 10
    noisy_grads = False

    word2vec_init = False
    embedding_init = np.sqrt(3) 

    strong_supervision = False
    beta = 1

    drop_grus = False

    anneal_threshold = 1000
    anneal_by = 1.5

    num_hops = 3
    num_attention_features = 4

    max_allowed_inputs = 130
    num_train = 9000

    floatX = np.float32

    babi_id = "1"
    babi_test_id = ""

    train_mode = True

def _add_gradient_noise(t, stddev=1e-3, name=None):
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

def _position_encoding(sentence_size, embedding_size):
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

class DMN_PLUS(object):
    def load_data(self, debug=False):
        if self.config.train_mode:
            self.train, self.valid, self.word_embedding, self.max_q_len, self.max_input_len, self.max_sen_len, self.num_supporting_facts, self.vocab_size = babi.load_babi(self.config, split_sentences=True)
        else:
            self.test, self.word_embedding, self.max_q_len, self.max_input_len, self.max_sen_len, self.num_supporting_facts, self.vocab_size = babi.load_babi(self.config, split_sentences=True)
        self.encoding = _position_encoding(self.max_sen_len, self.config.embed_size)

    def add_placeholders(self):
        self.question_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_q_len))
        self.input_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_input_len, self.max_sen_len))

        self.question_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))
        self.input_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))

        self.answer_placeholder = tf.placeholder(tf.int64, shape=(self.config.batch_size,))

        self.rel_label_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.num_supporting_facts))

        self.dropout_placeholder = tf.placeholder(tf.float32)

        self.gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)

        if self.config.drop_grus:
            self.gru_cell = tf.contrib.rnn.DropoutWrapper(self.gru_cell,
                    input_keep_prob=self.dropout_placeholder,
                    output_keep_prob=self.dropout_placeholder)

    def get_predictions(self, output):
        preds = tf.nn.softmax(output)
        pred = tf.argmax(preds, 1)
        return pred
      
    def add_loss_op(self, output):
        gate_loss = 0
        if self.config.strong_supervision:
            for i, att in enumerate(self.attentions):
                labels = tf.gather(tf.transpose(self.rel_label_placeholder), 0)
                gate_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=att, labels=labels))

        loss = self.config.beta * tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.answer_placeholder)) + gate_loss

        for v in tf.trainable_variables():
            if not 'bias' in v.name.lower():
                loss += self.config.l2 * tf.nn.l2_loss(v)

        tf.summary.scalar('loss', loss)
        return loss
        
    def add_training_op(self, loss):
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        gvs = opt.compute_gradients(loss)

        if self.config.cap_grads:
            gvs = [(tf.clip_by_norm(grad, self.config.max_grad_val), var) for grad, var in gvs]
        if self.config.noisy_grads:
            gvs = [(_add_gradient_noise(grad), var) for grad, var in gvs]

        train_op = opt.apply_gradients(gvs)
        return train_op  

    def get_question_representation(self, embeddings):
        questions = tf.nn.embedding_lookup(embeddings, self.question_placeholder)

        _, q_vec = tf.nn.dynamic_rnn(self.gru_cell,
                questions,
                dtype=np.float32,
                sequence_length=self.question_len_placeholder
        )
        return q_vec

    def get_input_representation(self, embeddings):
        inputs = tf.nn.embedding_lookup(embeddings, self.input_placeholder)
        inputs = tf.reduce_sum(inputs * self.encoding, 2)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.gru_cell,
                self.gru_cell,
                inputs,
                dtype=np.float32,
                sequence_length=self.input_len_placeholder
        )

        fact_vecs = tf.reduce_sum(tf.stack(outputs), axis=0)
        fact_vecs = tf.nn.dropout(fact_vecs, self.dropout_placeholder)
        return fact_vecs

    def get_attention(self, q_vec, prev_memory, fact_vec, reuse):
        with tf.variable_scope("attention", reuse=True):

            features = [fact_vec * q_vec,
                        fact_vec * prev_memory,
                        tf.abs(fact_vec - q_vec),
                        tf.abs(fact_vec - prev_memory)]

            feature_vec = tf.concat(features, 1)

            attention = tf.layers.dense(feature_vec,
                    self.config.embed_size,
                    activation=tf.nn.tanh,
                    reuse=reuse)

            attention = tf.layers.dense(attention,
                    1,
                    activation=None,
                    reuse=reuse)            
        return attention

    def generate_episode(self, memory, q_vec, fact_vecs, hop_index):
        attentions = [tf.squeeze(
            self.get_attention(q_vec, memory, fv, bool(hop_index) or bool(i)), axis=1)
            for i, fv in enumerate(tf.unstack(fact_vecs, axis=1))]

        attentions = tf.transpose(tf.stack(attentions))
        self.attentions.append(attentions)
        attentions = tf.nn.softmax(attentions)
        attentions = tf.expand_dims(attentions, axis=-1)

        reuse = True if hop_index > 0 else False

        gru_inputs = tf.concat([fact_vecs, attentions], 2)

        with tf.variable_scope('attention_gru', reuse=reuse):
            _, episode = tf.nn.dynamic_rnn(AttentionGRUCell(self.config.hidden_size),
                    gru_inputs,
                    dtype=np.float32,
                    sequence_length=self.input_len_placeholder
            )

        return episode

    def add_answer_module(self, rnn_output, q_vec):
        rnn_output = tf.nn.dropout(rnn_output, self.dropout_placeholder)

        output = tf.layers.dense(tf.concat([rnn_output, q_vec], 1),
                self.vocab_size,
                activation=None)
        return output

    def inference(self):
        embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding")

        with tf.variable_scope("question", initializer=tf.contrib.layers.xavier_initializer()):
            print '==> get question representation'
            q_vec = self.get_question_representation(embeddings)         

        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            print '==> get input representation'
            fact_vecs = self.get_input_representation(embeddings)

        self.attentions = []

        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
            print '==> build episodic memory'

            prev_memory = q_vec

            for i in range(self.config.num_hops):
                print '==> generating episode', i
                episode = self.generate_episode(prev_memory, q_vec, fact_vecs, i)

                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, episode, q_vec], 1),
                            self.config.hidden_size,
                            activation=tf.nn.relu)
            output = prev_memory

        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
            output = self.add_answer_module(output, q_vec)
        return output

    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False):
        config = self.config
        dp = config.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
        total_steps = len(data[0]) / config.batch_size
        total_loss = []
        accuracy = 0

        p = np.random.permutation(len(data[0]))
        qp, ip, ql, il, im, a, r = data
        qp, ip, ql, il, im, a, r = qp[p], ip[p], ql[p], il[p], im[p], a[p], r[p] 

        for step in range(total_steps):
            index = range(step * config.batch_size, (step + 1) * config.batch_size)
            feed = {self.question_placeholder: qp[index],
                  self.input_placeholder: ip[index],
                  self.question_len_placeholder: ql[index],
                  self.input_len_placeholder: il[index],
                  self.answer_placeholder: a[index],
                  self.rel_label_placeholder: r[index],
                  self.dropout_placeholder: dp}
            loss, pred, summary, _ = session.run(
              [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)

            if train_writer is not None:
                train_writer.add_summary(summary, num_epoch * total_steps + step)

            answers = a[step * config.batch_size:(step + 1) * config.batch_size]
            accuracy += np.sum(pred == answers) / float(len(answers))

            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                  step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')
        return np.mean(total_loss), accuracy / float(total_steps)

    def __init__(self, config):
        self.config = config
        self.variables_to_save = {}
        self.load_data(debug=False)
        self.add_placeholders()
        self.output = self.inference()
        self.pred = self.get_predictions(self.output)
        self.calculate_loss = self.add_loss_op(self.output)
        self.train_step = self.add_training_op(self.calculate_loss)
        self.merged = tf.summary.merge_all()
