from . import autodiff
from . import session
import numpy as np

class GradientDescentOptimizer(object):
    def __init__(self, learning_rate=1., use_locking=False):
        self.learning_rate = learning_rate
        self.eval_node_list = []
    def compute_gradients(self, var):
        return autodiff.gradients(var, self.eval_node_list)
    def apply_gradients(self, gradients):
        update = [autodiff.assign_op(self.eval_node_list[i], self.eval_node_list[i] - self.learning_rate * gradients[i]) for i in range(len(self.eval_node_list))]
        return autodiff.init_op(update)
    def minimize(self, var):
        for node in session._all_variable_inits:
            if node.target not in self.eval_node_list:
                self.eval_node_list.append(node.target)
        gradients = self.compute_gradients(var)
        return self.apply_gradients(gradients)

class AdamOptimizer(object):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-04):
        self.t = None
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []
        self.v = []

    def minimize(self, obj):
        eval_node_list = []
        for node in session._all_variable_inits:
            if node.target not in eval_node_list:
                eval_node_list.append(node.target)
        gradients = autodiff.gradients(obj, eval_node_list)
        updates = []
        self.t = session.Variable([0])
        for i in range(len(eval_node_list)):
            self.m.append(session.Variable([0]))
            self.v.append(session.Variable([0]))
        update_t = autodiff.assign_op(self.t, self.t + 1)
        rate = autodiff.sqrt_op(1 - self.beta2 ** update_t) / (1 - self.beta1 ** update_t)
        lr_t = self.learning_rate * rate
        for var, g, m, v in zip(eval_node_list, gradients, self.m, self.v):
            update_m = autodiff.assign_op(m, self.beta1 * m + (1 - self.beta1) * g)
            update_v = autodiff.assign_op(v, self.beta2 * v + (1 - self.beta2) * g * g)
            update_var = autodiff.assign_op(var, var - lr_t * update_m / (autodiff.sqrt_op(update_v) + self.epsilon))
            updates.append(update_var)
        return autodiff.init_op(updates)
