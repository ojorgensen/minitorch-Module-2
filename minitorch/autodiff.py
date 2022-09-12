variable_count = 1
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol


# ## Module 1

# Variable is the main class for autodifferentiation logic for scalars
# and tensors.

def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    # Make the first set of arguments
    upper = list(vals[:])
    upper[arg] = upper[arg] + epsilon/2

    # Then the second
    lower = list(vals[:])
    lower[arg] = lower[arg] - epsilon/2

    # Return the central difference. Note the * here unpacks the list into the arguments of the function
    return (f(*upper) - f(*lower)) / epsilon




class Variable:
    """
    Attributes:
        history (:class:`History` or None) : the Function calls that created this variable or None if constant
        derivative (variable type): the derivative with respect to this variable
        grad (variable type) : alias for derivative, used for tensors
        name (string) : a globally unique name of the variable
    """

    def __init__(self, history, name=None):
        global variable_count
        assert history is None or isinstance(history, History), history

        self.history = history
        self._derivative = None

        # This is a bit simplistic, but make things easier.
        variable_count += 1
        self.unique_id = "Variable" + str(variable_count)

        # For debugging can have a name.
        if name is not None:
            self.name = name
        else:
            self.name = self.unique_id
        self.used = 0

    def requires_grad_(self, val):
        """
        Set the requires_grad flag to `val` on variable.

        Ensures that operations on this variable will trigger
        backpropagation.

        Args:
            val (bool): whether to require grad
        """
        self.history = History()

    def backward(self, d_output=None):
        """
        Calls autodiff to fill in the derivatives for the history of this object.

        Args:
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)

    @property
    def derivative(self):
        return self._derivative

    def is_leaf(self):
        "True if this variable created by the user (no `last_fn`)"
        return self.history.last_fn is None

    def accumulate_derivative(self, val):
        """
        Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            val (number): value to be accumulated
        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self._derivative is None:
            self._derivative = self.zeros()
        self._derivative += val

    def zero_derivative_(self):  # pragma: no cover
        """
        Reset the derivative on this variable.
        """
        self._derivative = self.zeros()

    def zero_grad_(self):  # pragma: no cover
        """
        Reset the derivative on this variable.
        """
        self.zero_derivative_()

    def expand(self, x):
        "Placeholder for tensor variables"
        return x

    # Helper functions for children classes.

    def __radd__(self, b):
        return self + b

    def __rmul__(self, b):
        return self * b

    def zeros(self):
        return 0.0


# Some helper functions for handling optional tuples.


def wrap_tuple(x):
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


# Classes for Functions.


class Context:
    """
    Context class is used by `Function` to store information during the forward pass.

    Attributes:
        no_grad (bool) : do not save gradient information
        saved_values (tuple) : tuple of values saved for backward pass
        saved_tensors (tuple) : alias for saved_values
    """

    def __init__(self, no_grad=False):
        self._saved_values = None
        self.no_grad = no_grad

    def save_for_backward(self, *values):
        """
        Store the given `values` if they need to be used during backpropagation.

        Args:
            values (list of values) : values to save for backward
        """
        if self.no_grad:
            return
        self._saved_values = values

    @property
    def saved_values(self):
        assert not self.no_grad, "Doesn't require grad"
        assert self._saved_values is not None, "Did you forget to save values?"
        return unwrap_tuple(self._saved_values)

    @property
    def saved_tensors(self):  # pragma: no cover
        return self.saved_values


class History:
    """
    `History` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes:
        last_fn (:class:`FunctionBase`) : The last Function that was called.
        ctx (:class:`Context`): The context for that Function.
        inputs (list of inputs) : The inputs that were given when `last_fn.forward` was called.

    """

    def __init__(self, last_fn=None, ctx=None, inputs=None):
        self.last_fn = last_fn
        self.ctx = ctx
        self.inputs = inputs

    def backprop_step(self, d_output):
        """
        Run one step of backpropagation by calling chain rule.

        Args:
            d_output : a derivative with respect to this variable

        Returns:
            list of numbers : a derivative with respect to `inputs`
        """
        raise NotImplementedError('Need to include this file from past assignment.')


class FunctionBase:
    """
    A function that can act on :class:`Variable` arguments to
    produce a :class:`Variable` output, while tracking the internal history.

    Call by :func:`FunctionBase.apply`.

    """

    @staticmethod
    def variable(raw, history):
        # Implement by children class.
        raise NotImplementedError()

    @classmethod
    def apply(cls, *vals):
        """
        Apply is called by the user to run the Function.
        Internally it does three things:

        a) Creates a Context for the function call.
        b) Calls forward to run the function.
        c) Attaches the Context to the History of the new variable.

        There is a bit of internal complexity in our implementation
        to handle both scalars and tensors.

        Args:
            vals (list of Variables or constants) : The arguments to forward

        Returns:
            `Variable` : The new variable produced

        """
        # Go through the variables to see if any needs grad.
        raw_vals = []
        need_grad = False
        for v in vals:
            if isinstance(v, Variable):
                if v.history is not None:
                    need_grad = True
                v.used += 1
                raw_vals.append(v.get_data())
            else:
                raw_vals.append(v)

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls.forward(ctx, *raw_vals)
        assert isinstance(c, cls.data_type), "Expected return typ %s got %s" % (
            cls.data_type,
            type(c),
        )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = History(cls, ctx, vals)
        return cls.variable(cls.data(c), back)

    @classmethod
    def chain_rule(cls, ctx, inputs, d_output):
        """
        Implement the derivative chain-rule.

        Args:
            ctx (:class:`Context`) : The context from running forward
            inputs (list of args) : The args that were passed to :func:`FunctionBase.apply` (e.g. :math:`x, y`)
            d_output (number) : The `d_output` value in the chain rule.

        Returns:
            list of (`Variable`, number) : A list of non-constant variables with their derivatives
            (see `is_constant` to remove unneeded variables)

        """
        # Tip: Note when implementing this function that
        # cls.backward may return either a value or a tuple.
        raise NotImplementedError('Need to include this file from past assignment.')


# Algorithms for backpropagation


def is_constant(val):
    return not isinstance(val, Variable) or val.history is None


def find_nodes(variable):
    """
    Helper function to find all descendent modules of a module.
    """

    # TODO: PROBLEM: variables are not a class of modules. Hence, we need to implement the below stuff in terms of scalars instead.
    # Have maybe solved this, but it needs checking
    lst = [variable]    
    # find the direct descendents of the node, add these to the list
    children = variable.parents
    lst += children
    # if there are none, return the list
    if children == []:
        return lst
    # otherwise, find the nodes of all descendent modules, and append these to the list. Recursive.
    else:
        for child in children:
            lst += find_nodes(child)
    return lst

def visit(node_n, L):
    """
    Helper function for topological sort.

    Outlines the visit function found in the depth first search pseudocode here:
    https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
    """
    # If node already added, stop the visit
    # There is an issue here: we might have something which is identical here, but not actually the same
    # This means we are getting rid of duplicate inputs!
    if (node_n, node_n.unique_id) in L:
        return L
    # Otherwise, visit each child node
    children = node_n.parents
    for child in children:
        # update the list to reflect what happens after the search
        L = visit(child, L)
    #prepend the list with your original node
    # Unsure how to format this: should L be composed of names? or unique ids?
    L.insert(0, (node_n, node_n.unique_id))
    return L



def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Works by Depth-first search.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.

    # TODO: Also general testing

    # To implement topological sort, we first need to form a list of all nodes, from the first node.
    nodes = find_nodes(variable)
    # I think this part is ok, so the next part is the issue.
    # print("nodes found by find_nodes "+ str(nodes))
    # Empty list which will contain the sorted nodes
    L = []

    while nodes != []:
        #print(nodes)
        node = nodes[0]
        L = visit(node, L)
        #print(L)
        nodes = nodes[1:]
    # Since we have stored not just nodes but their unique identifiers, we return the first part of each tuple
    return [x[0] for x in L]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # Form an ordered queue
    queue = topological_sort(variable)
    # print("queue is " + str(queue))
    # To help us use unique_id, we are going to create a dictionary of modules with their unique_ids
    queue_ids = [x.unique_id for x in queue]
    queue_with_ids = dict(zip(queue_ids, queue))

    # Create dictionary of scalars and current derivatives
    current_backprop = dict()
    current_backprop[variable.unique_id] = deriv

    for node in queue:
        # store derivative
        deriv = current_backprop[node.unique_id]
        # If the node is a leaf, then change its derivative.
        if node.is_leaf():
            #print("leaf node is: " + str(node))
            #print("leaf node derivative is: " + str(deriv))
            node.accumulate_derivative(deriv)
            # print("node, deriv are" + str(node) + str(deriv))
            #print("node derivative (within the list is " + str(node.derivative))
        # If not, call backprop_step on the node, and add deriv to that scalar's total deriv.
        else:
            next_scalars_derivs = node.backprop_step(deriv)
            for (scalar, deriv2) in next_scalars_derivs:
                #print("deriv2 is" + str(deriv2))
                if scalar.unique_id in current_backprop.keys():
                    current_backprop[scalar.unique_id] += deriv2
                else:
                    current_backprop[scalar.unique_id] = deriv2
    #print("current_backprop is" + str(current_backprop))

    #print("variable.derivative is "+ str(variable.parents[1].derivative))

    #assert variable.parents[0].derivative == 5
    return
