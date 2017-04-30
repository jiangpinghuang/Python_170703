import tensorflow as tf

from util.nn import weight, bias
from util.attn import AttnGRU

class EpisodeModule:
    def __init__(self, num_hidden, question, facts, is_training, bn):
        self.question = question
        self.facts = tf.unpack(tf.transpose(facts, [1, 2, 0]))

        self.question_transposed = tf.transpose(question)
        self.facts_transposed = [tf.transpose(f) for f in self.facts]

        self.w1 = weight('w1', [num_hidden, 4 * num_hidden])
        self.b1 = bias('b1', [num_hidden, 1])
        self.w2 = weight('w2', [1, num_hidden])
        self.b2 = bias('b2', [1, 1])
        self.gru = AttnGRU(num_hidden, is_training, bn)

    @property
    def init_state(self):
        return tf.zeros_like(self.facts_transposed[0])

    def new(self, memory):
        state = self.init_state
        memory = tf.transpose(memory)

        with tf.variable_scope('AttnGate') as scope:
            for f, f_t in zip(self.facts, self.facts_transposed):
                g = self.attention(f, memory)
                state = self.gru(f_t, state, g)
                scope.reuse_variables()

        return state

    def attention(self, f, m):
        with tf.variable_scope('attention'):
            q = self.question_transposed
            vec = tf.concat(0, [f * q, f * m, tf.abs(f - q), tf.abs(f - m)])
            l1 = tf.matmul(self.w1, vec) + self.b1
            l1 = tf.nn.tanh(l1)
            l2 = tf.matmul(self.w2, l1) + self.b2
            l2 = tf.nn.softmax(l2)
            return tf.transpose(l2)

        return att
