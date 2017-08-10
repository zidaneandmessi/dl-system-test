""" library to take autodiff and execute a computation graph """
from __future__ import absolute_import
import ctypes
import numpy as np
# import scipy.signal
from . import c

float32 = ctypes.c_float

class Node(object):
    """Node in a computation graph."""
    def __init__(self):
        """Constructor, new node is indirectly created by Op object call method.

            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object,
                e.g. add_op if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant.
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""
        self._all_variable_inits = []
        self.target = None
        self.axes_indices = []

    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in new node's const_attr
            # 'other' argument is a constant
            new_node = add_op(self, const_op(other))
        return new_node

    def __sub__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = sub_op(self, other)
        else:
            # Add by a constant stores the constant in new node's const_attr
            # 'other' argument is a constant
            new_node = sub_op(self, const_op(other))
        return new_node

    def __rsub__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = sub_op(other, self)
        else:
            # Add by a constant stores the constant in new node's const_attr
            # 'other' argument is a constant
            new_node = sub_op(const_op(other), self)
        return new_node


    def __mul__(self, other):
        """Multiplying two nodes return a new node."""
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            # Mul by a constant stores the constant in new node's const_attr
            # 'other' argument is a constant
            new_node = mul_op(self, const_op(other))
        return new_node

    def __div__(self, other):
        if isinstance(other, Node):
            new_node = div_op(self, other)
        else:
            new_node = div_op(self, const_op(other))
        return new_node

    def __rdiv__(self, other):
        if isinstance(other, Node):
            new_node = div_op(other, self)
        else:
            new_node = div_op(const_op(other), self)
        return new_node

    def __neg__(self):
        new_node = neg_op(self)
        return new_node

    def __pow__(self, other):
        if isinstance(other, Node):
            new_node = pow_op(self, other)
        else:
            new_node = pow_op(self, const_op(other))
        return new_node

    def __rpow__(self, other):
        if isinstance(other, Node):
            new_node = pow_op(other, self)
        else:
            new_node = pow_op(const_op(other), self)
        return new_node

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow     to display node name."""
        return self.name

    def eval(self, feed_dict = None):
        feed_dict = feed_dict if feed_dict else {}
        for n, val in feed_dict.items():
            if not isinstance(val, np.ndarray):
                if not isinstance(val, list):
                    val = [val]
                feed_dict[n] = np.array(val)
        exe = Executor([self])
        ans = exe.run(feed_dict)
        if ans[self].shape == (1,):
            return ans[self][0]
        return ans[self]

    run = eval

class Op(object):
    """Op represents operations performed on nodes."""
    def __call__(self):
        """Create a new node and associate the op object with the node.

        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.
        output_val: output value of the node, modified in-place.
        use_numpy: bool flag whether to use numpy for compute
        """
        raise NotImplementedError

    def gradient(self, node, output_grad):
        """Given output gradient, compute partial gradient to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        """Given shapes of input nodes, compute shape of output node.

        Implementation note:
        It's simpler to treat shape of constants as (1,), so that constants can
        be stored as a numpy array too and you would need fewer special case
        handling.

        Parameters
        ----------
        node: node whose shape is being inferred.
        input_vals: shapes of input nodes.

        Returns
        -------
        A tuple representing the shape of output node.
        """
        raise NotImplementedError

class AssignOp(Op):
    def __call__(self, node, c):
        new_node = Op.__call__(self)
        if isinstance(c, Node):
            val_node = c
        else:
            if not isinstance(c, list):
                c = [c]
            val_node = const_op(c)
        new_node.inputs = [val_node]
        new_node.target = node
        new_node.name = "(%s=%s)" % (node.name, val_node.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        node.target.value = input_vals[0]
        output_val = input_vals[0]

    def gradient(self, node, output_grad):
        return None

    def infer_shape(self, node, input_shapes):
        return input_shapes[0] 

def shape_equal(shape_a, shape_b):
    if len(shape_a) == len(shape_b):
        return shape_a == shape_b
    if shape_a == (1,) and shape_b == (1,):
        return True
    if shape_b == (1,) or shape_a == (1,):
        return False
    while shape_a[0] == 1:
        shape_a = tuple(shape_a[i] for i in range(1, len(shape_a)))
    while shape_a[len(shape_a) - 1] == 1:
        shape_a = tuple(shape_a[i] for i in range(len(shape_a) - 1))
    while shape_b[0] == 1:
        shape_b = tuple(shape_b[i] for i in range(1, len(shape_b)))
    while shape_b[len(shape_b) - 1] == 1:
        shape_b = tuple(shape_b[i] for i in range(len(shape_b) - 1))
    return shape_a == shape_b

class AddOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        # output_val[:] allows modify in-place
        output_val[:] = input_vals[0] + input_vals[1]


    def gradient(self, node, output_grad):
        return [reducesumto_op(output_grad, node.inputs[0]), reducesumto_op(output_grad, node.inputs[1])]

    def infer_shape(self, node, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        """TODO: Your code here"""
        if shape_equal(input_shapes[0], input_shapes[1]):
            return input_shapes[0]
        elif input_shapes[1] == (1,):
            return input_shapes[0]
        elif input_shapes[0] == (1,):
            return input_shapes[1]
        else:
            return broadcast_rule(input_shapes[0], input_shapes[1])

class SubOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s-%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        # output_val[:] allows modify in-place
        output_val[:] = input_vals[0] - input_vals[1]

    def gradient(self, node, output_grad):
        return [reducesumto_op(output_grad, node.inputs[0]), -reducesumto_op(output_grad, node.inputs[1])]

    def infer_shape(self, node, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        """TODO: Your code here"""
        if shape_equal(input_shapes[0], input_shapes[1]):
            return input_shapes[0]
        elif input_shapes[1] == (1,):
            return input_shapes[0]
        elif input_shapes[0] == (1,):
            return input_shapes[1]
        else:
            return broadcast_rule(input_shapes[0], input_shapes[1])

class MulOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        output_val[:] = input_vals[0] * input_vals[1]

    def gradient(self, node, output_grad):
        return [reducesumto_op(node.inputs[1] * output_grad, node.inputs[0]), reducesumto_op(node.inputs[0] * output_grad, node.inputs[1])]

    def infer_shape(self, node, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        """TODO: Your code here"""
        if shape_equal(input_shapes[0], input_shapes[1]):
            return input_shapes[0]
        elif input_shapes[1] == (1,):
            return input_shapes[0]
        elif input_shapes[0] == (1,):
            return input_shapes[1]
        else:
            return broadcast_rule(input_shapes[0], input_shapes[1])

class DivOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s/%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        output_val[:] = input_vals[0] / input_vals[1]

    def gradient(self, node, output_grad):
        return [reducesumto_op(output_grad / node.inputs[1], node.inputs[0]), reducesumto_op(-node.inputs[0] * output_grad / node.inputs[1] / node.inputs[1], node.inputs[1])]

    def infer_shape(self, node, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        """TODO: Your code here"""
        if shape_equal(input_shapes[0], input_shapes[1]):
            return input_shapes[0]
        elif input_shapes[1] == (1,):
            return input_shapes[0]
        elif input_shapes[0] == (1,):
            return input_shapes[1]
        else:
            return broadcast_rule(input_shapes[0], input_shapes[1])

class MatMulOp(Op):
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (
            node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        if ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is False)):
            output_val[:] = np.matmul(input_vals[0], input_vals[1])
        elif ((node.matmul_attr_trans_A is True) and
                (node.matmul_attr_trans_B is False)):
            output_val[:] = np.matmul(
                np.transpose(input_vals[0]), input_vals[1])
        elif ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is True)):
            output_val[:] = np.matmul(
                input_vals[0], np.transpose(input_vals[1]))
        elif ((node.matmul_attr_trans_A is True) and
                (node.matmul_attr_trans_B is True)):
            output_val[:] = np.matmul(
                np.transpose(input_vals[0]), np.transpose(input_vals[1]))

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input.
            
        Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
                        if Y=A^T B, then dA=B dY^T, dB=A dY
                        if Y=AB^T, then dA=dY B, dB=dY^T A
                        if Y=A^T B^T, then dA=B^T dY^T, dB=dY^T A^T
        """
        # """TODO: Your code here"""
        transA = node.matmul_attr_trans_A
        transB = node.matmul_attr_trans_B
        if (transA == False) and (transB == False):
            return [matmul_op(output_grad, node.inputs[1], False, True), matmul_op(node.inputs[0], output_grad, True, False)]
        elif (transA == True) and (transB == False):
            return [matmul_op(node.inputs[1], output_grad, False, True), matmul_op(node.inputs[0], output_grad, False, False)]
        elif (transA == False) and (transB == True):
            return [matmul_op(output_grad, node.inputs[1], False, False), matmul_op(output_grad, node.inputs[0], True, False)]
        elif (transA == True) and (transB == True):
            return [matmul_op(node.inputs[1], output_grad, True, True), matmul_op(output_grad, node.inputs[0], True, True)]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        transA = node.matmul_attr_trans_A
        transB = node.matmul_attr_trans_B
        if len(input_shapes[0]) == 2:
            x0 = 0
            y0 = 1
        else:
            x0 = 1
            y0 = 2
        if len(input_shapes[1]) == 2:
            x1 = 0
            y1 = 1
        else:
            x1 = 1
            y1 = 2
        if (transA == False) and (transB == False):
            return (input_shapes[0][x0], input_shapes[1][y1])
        elif (transA == True) and (transB == False):
            return (input_shapes[0][y0], input_shapes[1][y1])
        elif (transA == False) and (transB == True):
            return (input_shapes[0][x0], input_shapes[1][x1])
        elif (transA == True) and (transB == True):
            return (input_shapes[0][y0], input_shapes[1][x1])

class PlaceholderOp(Op):
    def __call__(self, dtype=float32):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        new_node.name = "Placeholder"
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert False, "placeholder %s values provided by feed_dict" % node.name

    def gradient(self, node, output_grad):
        return None

    def infer_shape(self, node, input_shapes):
        assert False, "placeholder %s shape provided by feed_shape" % node.name
    
class VariableOp(Op):
    def __call__(self):
        new_node = Op.__call__(self)
        new_node.value = None
        new_node.name = "Var"
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy = True):
        output_val[:] = node.value

    def gradient(self, node, output_grad):
        return None

    def infer_shape(self, node, input_shapes):
        return node.value.shape

class ConstOp(Op):
    def __call__(self, c):
        if not isinstance(c, np.ndarray):
            if not isinstance(c, list):
                c = [c]
            c = np.array(c)
        new_node = Op.__call__(self)
        new_node.const_attr = c
        new_node.name = "Const"
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        output_val[:] = node.const_attr
    def gradient(self, node, output_grad):
        return [0]
    def infer_shape(self, node, input_shapes):
        return node.const_attr.shape
    
class InitOp(Op):
    def __call__(self, inits):
        new_node = Op.__call__(self)
        new_node.inputs = inits
        new_node.name = "Init"
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        output_val = None
    def gradient(self, node, output_grad):
        return None
    def infer_shape(self, node, input_shapes):
        return None


class ZerosLikeOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.zeros(node_A.shape)."""
        if not isinstance(node_A, Node):
            node_A = const_op(node_A)
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        output_val[:] = np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        """If input_shape is a vector, simpler to return (1,)"""
        """TODO: Your code here"""
        return input_shapes[0]


class OnesLikeOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.ones(node_A.shape)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        output_val[:] = np.ones(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        """If input_shape is a vector, simpler to return (1,)"""
        """TODO: Your code here"""
        return input_shapes[0]


class ReduceSumOp(Op):
    def __call__(self, node_A, reduction_indices = [0], keepdims = False):
        """Creates a node that represents np.sum(node_A, axis=0).
        Only support common-case axis=0 reduction for simplicity of gradient.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "ReduceSum(%s)" % (node_A.name)
        new_node.axes_indices = reduction_indices
        new_node.const_attr = keepdims
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert(isinstance(input_vals[0], np.ndarray))
        axes = tuple(axis for axis in node.axes_indices)
        output_val[:] = np.sum(input_vals[0], axes, keepdims = node.const_attr)

    def gradient(self, node, output_grad):
        return [broadcastto_op(output_grad, node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        """summation reduction axis = 0
        e.g. (3,4,5)->(4,5)
        for vector, simpler to do (3,)->(1,)
        """
        """TODO: Your code here"""
        while len(input_shapes[0]) > 1 and input_shapes[0][0] == 1:
            input_shapes[0] = tuple(input_shapes[0][i] for i in range(1, len(input_shapes[0])))
        if node.const_attr == True:
            if len(input_shapes[0]) == 1:
                return (1,)
            else:
                ans = []
                for i in range(len(input_shapes[0])):   
                    if i not in node.axes_indices:
                        ans.append(input_shapes[0][i])
                    else:
                        ans.append(1)
                return tuple(x for x in ans)
        if len(input_shapes[0]) == 1:
            return (1,)
        else:
            ans = []
            for i in range(len(input_shapes[0])):   
                if i not in node.axes_indices:
                    ans.append(input_shapes[0][i])
            return tuple(x for x in ans)


class ReduceSumToOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "ReduceSumTo(%s, %s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        if input_vals[0].shape == input_vals[1].shape:
            output_val[:] = input_vals[0]
            return
        ans = input_vals[0]
        x = 0
        for i in range(len(input_vals[1].shape)):
            while x < len(ans.shape) and ans.shape[x] != input_vals[1].shape[i]:
                ans = np.sum(ans, axis = i)
            x += 1
        while x < len(ans.shape):
            ans = np.sum(ans, axis = x)
            x += 1
        output_val[:] = ans.reshape(input_vals[1].shape)

    def gradient(self, node, output_grad):
        return [broadcastto_op(output_grad, node.inputs[0]), zeroslike_op(node.inputs[1])]

    def infer_shape(self, node, input_shapes):
        return input_shapes[1]

class SizeOp(Op):
    def __call__(self, node_A, reduction_indices = [0]):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Size(%s)" % (node_A.name)
        new_node.axes_indices = reduction_indices
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        input_shape = input_vals[0].shape
        if len(input_shape) == 1:
            input_shape = (1, input_shape[0])
        size = 1
        for i in range(len(input_shape)):
            if i not in node.axes_indices:
                size *= input_shape[i]
        output_val[:] = [size]

    def gradient(self, node, output_grad):
        return [0]

    def infer_shape(self, node, input_shapes):
        return (1,)

class BroadcastToOp(Op):
    def __call__(self, node_A, node_B):
        """Creates a node that represents np.broadcast_to(node_A, node_B.shape).
        Only support axis=0. e.g. (3,4)->(2,3,4) to make gradient simple.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "BroadcastTo(%s,%s.shape)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert(len(input_vals)==2)
        if len(input_vals[0].shape) > len(input_vals[1].shape):
            input_vals[0] = input_vals[0].reshape(input_vals[1].shape)
        while len(input_vals[0].shape) < len(input_vals[1].shape):
            if (input_vals[0].shape[0] == input_vals[1].shape[0]):
              input_vals[0] = input_vals[0].reshape(input_vals[0].shape + (1,))
            else: 
              input_vals[0] = input_vals[0].reshape((1,) + input_vals[0].shape)
        output_val[:] = np.broadcast_to(input_vals[0], input_vals[1].shape)

    def gradient(self, node, output_grad):
        grad_A = reducesumaxiszero_op(output_grad)
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        return broadcast_rule(input_shapes[0], input_shapes[1])


class SoftmaxOp(Op):
    def __call__(self, node_A):
        exp_node = exp_op(node_A)
        new_node = exp_node / reducesum_op(exp_node, reduction_indices=[1], keepdims=True)
        new_node.name = "Softmax(%s)" % (node_A.name)
        return new_node


class ReluOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Relu(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        output_val[:] = np.maximum(input_vals[0], 0)

    def gradient(self, node, output_grad):
        return [relu_gradient_op(node.inputs[0], output_grad)]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        return input_shapes[0]


class ReluGradientOp(Op):
    def __call__(self, node_A, node_B):
        """node_B is output_grad"""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "ReluGradient(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        # heaviside function, 0.5 at x=0
        output_val[:] = (np.sign(input_vals[0]) + 1) * 0.5 * input_vals[1]

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        return input_shapes[0]

class LogOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Log(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        output_val[:] = np.log(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad / node.inputs[0]]

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]

class ExpOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Exp(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        output_val[:] = np.exp(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * exp_op(node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]

class NegOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "-(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        output_val[:] = -input_vals[0]

    def gradient(self, node, output_grad):
        return [-output_grad]

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]


class EqualOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s==%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        output_val[:] = np.equal(input_vals[0], input_vals[1])

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]

class ArgmaxOp(Op):
    def __call__(self, node, axis):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "Argmax(%s)" % (node.name)
        new_node.axes_indices = axis
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        output_val[:] = np.argmax(input_vals[0], axis=node.axes_indices)

    def infer_shape(self, node, input_shapes):
        shape = input_shapes[0]
        if len(shape) > 1 and shape[0] == 1:
            shape = tuple(shape[i] for i in range(1, len(shape)))
        if node.axes_indices == None:
            return (1,)
        else:
            ans = []
            for i in range(len(shape)):
                if i != node.axes_indices:
                    ans.append(shape[i])
            return tuple(x for x in ans)
        

class CastOp(Op):
    def __call__(self, node, dtype):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.const_attr = dtype
        new_node.name = "Cast(%s)" % (node.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        output_val[:] = input_vals[0].astype(node.const_attr)

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]


class SqrtOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "Sqrt(%s)" % (node.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        output_val[:] = np.sqrt(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad / sqrt_op(node.inputs[0]) / 2.]

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]


class PowOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s^%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        output_val[:] = input_vals[0] ** input_vals[1]

    def gradient(self, node, output_grad):
        return [node.inputs[1] * pow_op(node.inputs[0], node.inputs[1] - 1), log_op(node.inputs[0]) * pow_op(node.inputs[0], node.inputs[1])]

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]


class ReshapeOp(Op):
    def __call__(self, node, shape):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "Reshape"
        new_node.const_attr = shape
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        output_val[:] = input_vals[0].reshape(node.const_attr)

    def gradient(self, node, output_grad):
        return [reshapeto_op(output_grad, node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        size = 1
        shape = [x for x in node.const_attr]
        for i in range(len(input_shapes[0])):
            size *= input_shapes[0][i]
        for i in range(len(node.const_attr)):
            if shape[i] != -1:
                size /= shape[i]
        for i in range(len(node.const_attr)):
            if shape[i] == -1:
                shape[i] = size
        return tuple(x for x in shape)


class ReshapeToOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "ReshapeTo(%s, %s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        output_val[:] = input_vals[0].reshape(input_vals[1].shape)

    def infer_shape(self, node, input_shapes):
        return input_shapes[1]


# def correlate(input, filter, mode):
#     in_height = input.shape[0]
#     in_width = input.shape[1]
#     filter = np.rot90(filter, k = 2, axes = (0, 1))
#     filter_height = filter.shape[0]
#     filter_width = filter.shape[1]
#     if mode == 'same':
#         input_matrix = np.zeros((in_height + filter_height - 1, in_width + filter_width - 1))
#         padding = (filter_height - 1) / 2
#         input_matrix[padding:-padding, padding:-padding] = input
#         input_matrix = input_matrix.reshape((input_matrix.size,))
#         input_matrix = np.hstack((input_matrix, np.zeros((in_width - filter_width,))))
#         filter_matrix = np.hstack((filter_matrix, np.zeros((filter_height, in_width - filter_width))))
#         filter_matrix = filter_matrix.reshape((filter_matrix.size,))


class Conv2dOp(Op):
    def __call__(self, input, filter, strides=[1, 1, 1, 1], padding='SAME'):
        new_node = Op.__call__(self)
        new_node.inputs = [input, filter]
        new_node.name = "Conv2d"
        new_node.const_attr = (strides, padding)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        # input: [batch, in_height, in_width, in_channels]
        # filter: [filter_height, filter_width, in_channels, out_channels]
        # strides: [1, stride, stride, 1]
        # padding: SAME VALID
        # output: [batch, (in_height - filter_height - padding * 2) / stride + 1, (in_width - filter_width - padding * 2) / stride + 1, out_channels]
        
        input = input_vals[0].astype(np.float64)
        filter = input_vals[1].astype(np.float64)
        # strides = node.const_attr[0]
        # batch = input.shape[0]
        # in_height = input.shape[1]
        # in_width = input.shape[2]
        # in_channels = input.shape[3]
        # filter_height = filter.shape[0]
        # filter_width = filter.shape[1]
        # out_channels = filter.shape[3]
        # output_val[:] = np.zeros(output_val.shape)
        # for i in range(batch):
        #     input_matrix = input[i, :, :, :]
        #     for k in range(out_channels):
        #         filter_matrix = filter[:, :, :, k]
        #         for l in range(in_channels):
        #             output_val[i, :, :, k] += scipy.signal.correlate2d(input_matrix[:, :, l], filter_matrix[:, :, l], mode = 'same')
        c.conv2d(input, filter, output_val)

    def gradient(self, node, output_grad):
        return [conv2d_grad1_op(node.inputs[0], node.inputs[1], output_grad, node.const_attr), conv2d_grad2_op(node.inputs[0], node.inputs[1], output_grad, node.const_attr)]
        
    def infer_shape(self, node, input_shapes):
        batch = input_shapes[0][0]
        in_height = input_shapes[0][1]
        in_width = input_shapes[0][2]
        out_channels = input_shapes[1][3]
        return (batch, in_height, in_width, out_channels)


class Conv2dGrad1Op(Op):
    def __call__(self, node_A, node_B, gradient, data):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, gradient]
        new_node.name = "Conv2dGrad1"
        new_node.const_attr = data
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        input = input_vals[2].astype(np.float64)
        filter = np.rot90(input_vals[1], k = 2, axes = (0, 1)).astype(np.float64)
        # strides = node.const_attr[0]
        # batch = input.shape[0]
        # in_height = input_vals[0].shape[1]
        # in_width = input_vals[0].shape[2]
        # in_channels = input_vals[0].shape[3]
        # filter_height = filter.shape[0]
        # filter_width = filter.shape[1]
        # out_channels = filter.shape[3]
        # output_val[:] = np.zeros(output_val.shape)
        # for i in range(batch):
        #     input_matrix = input[i, :, :, :]
        #     for l in range(out_channels):
        #         for k in range(in_channels):
        #             filter_matrix = filter[:, :, k, l]
        #             output_val[i, :, :, k] += scipy.signal.correlate2d(input_matrix[:, :, l], filter_matrix, mode = 'same')
        c.conv2dgrad1(input, filter, output_val)
      
    def infer_shape(self, node, input_shapes):
        return input_shapes[0]


class Conv2dGrad2Op(Op):
    def __call__(self, node_A, node_B, gradient, data):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, gradient]
        new_node.name = "Conv2dGrad2"
        new_node.const_attr = data
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        input = input_vals[0].astype(np.float64)
        filter = np.rot90(input_vals[2], k = 2, axes = (1, 2)).astype(np.float64)
        # strides = node.const_attr[0]
        # batch = input.shape[0]
        # in_height = input.shape[1]
        # in_width = input.shape[2]
        # in_channels = input.shape[3]
        # filter_height = input_vals[1].shape[0]
        # filter_width = input_vals[1].shape[1]
        # out_channels = input_vals[1].shape[3]
        # output_val[:] = np.zeros(output_val.shape)
        # for i in range(batch):
        #     for k in range(in_channels):
        #         input_matrix = np.zeros((in_height - 1 + filter_height, in_width - 1 + filter_width))
        #         padding = (filter_height - 1) / 2
        #         input_matrix[padding:-padding, padding:-padding] = input[i, :, :, k]
        #         for l in range(out_channels):
        #             filter_matrix = filter[i, :, :, l]
        #             output_val[:, :, k, l] += scipy.signal.correlate2d(input_matrix, filter_matrix, mode = 'valid')
        c.conv2dgrad2(input, filter, output_val)
      
    def infer_shape(self, node, input_shapes):
        return input_shapes[1]


def pooling(input, in_height, in_width, ksize, stride):
    pool_height = ksize[1]
    pool_width = ksize[2]
    ans = np.zeros((in_height / stride, in_width / stride))
    x = 0
    xx = 0
    while x + pool_height <= in_height:
        y = 0
        yy = 0
        while y + pool_width <= in_width:
            ans[xx][yy] = np.max(input[x:x + pool_height, y:y + pool_width])
            y += stride
            yy += 1
        x += stride
        xx += 1
    return ans


class MaxPoolOp(Op):
    def __call__(self, input, ksize, strides=[1, 1, 1, 1], padding='SAME'):
        new_node = Op.__call__(self)
        new_node.inputs = [input]
        new_node.name = "MaxPool"
        new_node.const_attr = (ksize, strides, padding)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        # input: [batch, in_height, in_width, in_channels]
        # ksize: [1, pool_height, pool_width, 1]
        # strides: [1, stride,stride, 1]
        # padding: SAME VALID
        # output: [batch, (in_height - pool_height - padding * 2) / stride + 1, (in_width - pool_width - padding * 2) / stride + 1, in_channels]
        input = input_vals[0].astype(np.float64)
        ksize = node.const_attr[0]
        strides = node.const_attr[1]
        batch = input.shape[0]
        in_height = input.shape[1]
        in_width = input.shape[2]
        in_channels = input.shape[3]
        for i in range(batch):
            for k in range(in_channels):
                padding = ((in_height / strides[1] - 1) * strides[1] + ksize[1] - in_height) / 2
                input_matrix = np.zeros((in_height + padding * 2, in_width + padding * 2))
                if padding == 0:
                    input_matrix[:, :] = input[i, :, :, k]
                else:
                    input_matrix[padding:-padding, padding:-padding] = input[i, :, :, k]
                output_val[i, :, :, k] = pooling(input_matrix, in_height, in_width, ksize, strides[1])
        # c.maxpool(input, ksize, strides[1], output_val)

    def gradient(self, node, output_grad):
        return [maxpool_grad_op(node.inputs[0], output_grad, node.const_attr)]
        
    def infer_shape(self, node, input_shapes):
        batch = input_shapes[0][0]
        in_height = input_shapes[0][1]
        in_width = input_shapes[0][2]
        in_channels = input_shapes[0][3]
        stride = node.const_attr[1][1]
        return (batch, in_height / stride, in_width / stride, in_channels)


def pooling_grad(input, ksize, stride, gradient):
    in_height = input.shape[0]
    in_width = input.shape[1]
    pool_height = ksize[1]
    pool_width = ksize[2]
    ans = np.zeros((in_height, in_width))
    x = 0
    xx = 0
    while x + pool_height <= in_height:
        y = 0
        yy = 0
        while y + pool_width <= in_width:
            mx = np.max(input[x:x + pool_height, y:y + pool_width])
            ans[x:x + pool_height, y:y + pool_width] += np.equal(input[x:x + pool_height, y:y + pool_width], mx) * gradient[xx][yy]
            y += stride
            yy += 1
        x += stride
        xx += 1
    return ans


class MaxPoolGradientOp(Op):
    def __call__(self, node, gradient, data):
        new_node = Op.__call__(self)
        new_node.inputs = [node, gradient]
        new_node.const_attr = data
        new_node.name = "MaxPoolGrad"
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        input = input_vals[0].astype(np.float64)
        gradient = input_vals[1].astype(np.float64)
        ksize = node.const_attr[0]
        strides = node.const_attr[1]
        # batch = input.shape[0]
        # in_height = input.shape[1]
        # in_width = input.shape[2]
        # in_channels = input.shape[3]
        # for i in range(batch):
        #     for k in range(in_channels):
        #         padding = ((in_height / strides[1] - 1) * strides[1] + ksize[1] - in_height) / 2
        #         input_matrix = np.zeros((in_height + padding * 2, in_width + padding * 2))
        #         if padding == 0:
        #             input_matrix[:, :] = input[i, :, :, k]
        #             output_val[i, :, :, k] = pooling_grad(input_matrix, ksize, strides[1], gradient[i, :, :, k])
        #         else:
        #             input_matrix[padding:-padding, padding:-padding] = input[i, :, :, k]
        #             output_val[i, :, :, k] = pooling_grad(input_matrix, ksize, strides[1], gradient[i, :, :, k])[padding:-padding, padding:-padding]
        c.maxpoolgrad(input, gradient, ksize, strides[1], output_val)

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]


class DropOutOp(Op):
    def __call__(self, node, keep_prob):
        new_node = Op.__call__(self)
        new_node.inputs = [node, keep_prob]
        new_node.name = "DropOut"
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        if input_vals[1] == 1:
            output_val[:] = input_vals[0]
            return
        noise_shape = input_vals[0].shape
        random_tensor = np.random.uniform(size=noise_shape)
        node.const_attr = random_tensor < input_vals[1]
        output_val[:] = input_vals[0] * node.const_attr


    def gradient(self, node, output_grad):
        return [dropout_grad_op(node, output_grad), 0]

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]


class DropoutGradientOp(Op):
    def __call__(self, node, gradient):
        new_node = Op.__call__(self)
        new_node.inputs = [node, gradient]
        new_node.name = "DropoutGradient"
        return new_node
        
    def compute(self, node, input_vals, output_val, use_numpy = True):
        output_val[:] = input_vals[0] * node.inputs[0].const_attr
        
    def infer_shape(self, node, input_shapes):
        return input_shapes[0]


# Create global singletons of operators.
assign_op = AssignOp()
add_op = AddOp()
sub_op = SubOp()
mul_op = MulOp()
div_op = DivOp()
matmul_op = MatMulOp()
placeholder_op = PlaceholderOp()
variable_op = VariableOp()
const_op = ConstOp()
init_op = InitOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
zeros = np.zeros
ones = np.ones
reducesum_op = ReduceSumOp()
reducesumto_op = ReduceSumToOp()
size_op = SizeOp()
broadcastto_op = BroadcastToOp()
softmax_op = SoftmaxOp()
relu_op = ReluOp()
relu_gradient_op = ReluGradientOp()
log_op = LogOp()
neg_op = NegOp()
exp_op = ExpOp()
equal_op = EqualOp()
argmax_op = ArgmaxOp()
cast_op = CastOp()
sqrt_op = SqrtOp()
pow_op = PowOp()
reshape_op = ReshapeOp()
reshapeto_op = ReshapeToOp()
conv2d_op = Conv2dOp()
maxpool_op = MaxPoolOp()
maxpool_grad_op = MaxPoolGradientOp()
conv2d_grad1_op = Conv2dGrad1Op()
conv2d_grad2_op = Conv2dGrad2Op()
dropout_op = DropOutOp()
dropout_grad_op = DropoutGradientOp()


class Executor(object):
    """Executor computes values for given set of nodes in computation graph."""
    def __init__(self, eval_node_list, ctx=None):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        ctx: runtime DLContext, default is None which means np.ndarray on cpu
        topo_order: list of nodes in topological order
        node_to_shape_map: dict from node to shape of the node
        node_to_arr_map: dict from node to ndarray.NDArray allocated for node
        feed_shapes: shapes of feed_dict from last run(...)
        """
        self.eval_node_list = eval_node_list
        self.ctx = ctx
        self.topo_order = find_topo_sort(self.eval_node_list)
        self.node_to_shape_map = None
        self.node_to_arr_map = None
        self.feed_shapes = None

    def infer_shape(self, feed_shapes):
        """Given shapes of feed_dict nodes, infer shape for all nodes in graph.

        Implementation note:
        Iteratively calls node.op.infer_shape to infer shapes.
        Node shapes stored in self.node_to_shape_map.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        """TODO: Your code here"""
        self.node_to_shape_map = dict(feed_shapes)
        for node in self.topo_order:
            if node not in self.node_to_shape_map:
                input_shapes = [self.node_to_shape_map[input_node] for input_node in node.inputs]
                self.node_to_shape_map[node] = node.op.infer_shape(node, input_shapes)

    def memory_plan(self, feed_shapes):
        """Allocates ndarray.NDArray for every node except feed_dict nodes.

        Implementation note:
        Option 1: Alloc a ndarray.NDArray per node that persists across run()
        Option 2: Implement a memory pool to reuse memory for nodes of same
                shapes. More details see Lecture 7.

        For both options, self.node_to_arr_map stores node->NDArray mapping to
        allow mapping to persist across multiple executor.run().

        Hint: use ndarray.empty(shape, ctx=self.ctx) to allocate NDArray.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        """TODO: Your code here"""
        self.node_to_arr_map = {}
        self.infer_shape(feed_shapes)
        for node in self.topo_order:
            if node in self.node_to_shape_map:
                self.node_to_arr_map[node] = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx)
        

    def run(self, feed_dict, convert_to_numpy_ret_vals=False):
        """
        Parameters
        ----------
        feed_dict: a dictionary of node->np.ndarray supplied by user.
        convert_to_numpy_ret_vals: whether to convert ret vals to np.array

        Returns
        -------
        A list of values for nodes in eval_node_list. NDArray or np.ndarray.
        """
        def are_feed_shapes_equal(sa, sb):
            if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
                return False
            unmatched_item = set(sa.items()) ^ set(sb.items())
            return len(unmatched_item) == 0

        # Assume self.ctx is None implies numpy array and numpy ops.
        use_numpy = self.ctx is None
        node_to_val_map = {}
        for node, value in feed_dict.items():
            # all values passed in feed_dict must be np.ndarray
            assert isinstance(value, np.ndarray)
            node_to_val_map[node] = value

        # collect shapes for all placeholders
        feed_shapes = {}
        for node in node_to_val_map:
            feed_shapes[node] = node_to_val_map[node].shape

        # infer shape if feed_shapes changed since last run
        # e.g. call run() on test data after trainng
        if (not are_feed_shapes_equal(feed_shapes, self.feed_shapes)):
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes
            # plan memory if using GPU

        # Traverse graph in topo order and compute values for all nodes.
        for node in self.topo_order:
            if node in node_to_val_map:
                # Skip placeholder nodes. Values already provided by feed_dict.
                continue
            input_vals = [node_to_val_map[n] for n in node.inputs]
            node_val = np.empty(shape=self.node_to_shape_map[node])
            # node_val is modified in-place whether np.ndarray or NDArray
            node.op.compute(node, input_vals, node_val, use_numpy)
            node_to_val_map[node] = node_val

        return node_to_val_map


def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    node_to_output_grad = {}
    # Traverse forward graph in reverse topological order
    reverse_topo_order = reversed(find_topo_sort([output_node]))
    for node in reverse_topo_order:
        output_grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = output_grad
        input_grads_list = node.op.gradient(node, output_grad)
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            # Calculate partial adjoint for input nodes.
            node_to_output_grads_list[node.inputs[i]].append(
                input_grads_list[i])

    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list

##################
# Helper Methods #
##################


def find_topo_sort(node_list):
    """Given a list of nodes, return a topo ordering of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a
    topological sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def sum_node_list(node_list):
    """Custom sum func to avoid creating redundant nodes in Python sum func."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)


def broadcast_rule(shape_a, shape_b):
    """Return output shape of broadcast shape_a, shape_b.
    e.g. broadcast_rule((3,2), (4,3,2))
    returns output_shape = (4,3,2)

    Check out explanations and more examples at
    https://docs.scipy.org/doc/numpy-1.10.0/user/basics.broadcasting.html
    http://eli.thegreenplace.net/2015/broadcasting-arrays-in-numpy/
    """
    assert(isinstance(shape_a, tuple))
    assert(isinstance(shape_b, tuple))
    if len(shape_a) > len(shape_b):
        longer_shape, shorter_shape = shape_a, shape_b
    else:
        longer_shape, shorter_shape = shape_b, shape_a
    if len(longer_shape) > len(shorter_shape):
        x = 0
        for i in range(len(longer_shape)):
            suc = 0
            for j in range(len(shorter_shape)):
                if longer_shape[i] != 1 and longer_shape[i] == shorter_shape[j]:
                    x = i - j
                    suc = 1
                    break
            if suc:
                break
        for i in range(x):
            shorter_shape = (1,) + shorter_shape
    len_diff = len(longer_shape) - len(shorter_shape)
    for i in range(len_diff):
        shorter_shape = shorter_shape + (1,)
    assert len(shorter_shape) == len(longer_shape)
    output_shape = list(longer_shape)
    for i in range(len(output_shape)):
        assert (shorter_shape[i] == longer_shape[i]) \
            or (shorter_shape[i] == 1) \
            or (longer_shape[i] == 1)
        output_shape[i] = max(shorter_shape[i], longer_shape[i])
    return tuple(output_shape)
