from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


class Module:
    """
    Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes:
        _modules (dict of name x :class:`Module`): Storage of the child modules
        _parameters (dict of name x :class:`Parameter`): Storage of the module's parameters
        training (bool): Whether the module is in training mode or evaluation mode

    """

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        "Return the direct child modules of this module."
        m: Dict[str, Module] = self.__dict__["_modules"]
        return list(m.values())

    def train(self) -> None:
        "Set the mode of this module and all descendent modules to `train`."
        # TODO: Implement for Task 0.4.
        self.training = True
        # Unsure if this is how you access / change child modules?
        for child in self.modules():
            child.training = True

    def eval(self) -> None:
        "Set the mode of this module and all descendent modules to `eval`."
        # TODO: Implement for Task 0.4.
        self.training = False
        # Unsure if this is how you access / change child modules?
        for child in self.modules():
            child.training = False


    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """
        Collect all the parameters of this module and its descendents.


        Returns:
            The name and `Parameter` of each ancestor parameter.
        """
        # TODO: Implement for Task 0.4.
        # Problem: this is only getting the parameters from the current module.
        # We want the parameters from all the descendents of these modules too.
        # I think the solution might look recursive?
        p: Dict[str, Parameter] = self.__dict__["_parameters"]
        ls = list(zip(p.keys(), p.values()))



        m: Dict[str, Module] = self.__dict__["_modules"]

        #print("m is " + str(m))

        #print(" list m is " + str(list(zip(m.keys(), m.values()))))

        zipped_m = list(zip(m.keys(), m.values()))

        for (mod_key, mod_value) in zipped_m:
            ls1 = [(mod_key + "." + mod[0], mod[1]) for mod in mod_value.named_parameters()]
            ls += ls1
        return ls

        # print( self.__dict__["_modules"] )
        # #return self.__dict__["_modules"]

        # # For each descendent module, find the named parameters of these modules, add them to the list
        # # Not only add them, add them with the prefix describing the modules they came from!
        # for mod in self.__dict__["_modules"].:
        #     ls1 = [(str(mod) + "." + x[0], x[1]) for x in mod.named_parameters()]
        #     s += ls1
        # return ls

    def parameters(self) -> Sequence[Parameter]:
        "Enumerate over all the parameters of this module and its descendents."
        # TODO: Implement for Task 0.4.
        ls = [y for (x,y) in self.named_parameters()]
        return ls

    def add_parameter(self, k, v):
        """
        Manually add a parameter. Useful helper for scalar parameters.

        Args:
            k (str): Local name of the parameter.
            v (value): Value for the parameter.

        Returns:
            Parameter: Newly created parameter.
        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key, val):
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key):
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        assert False, "Not Implemented"

    def __repr__(self):
        def _addindent(s_, numSpaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """
    A Parameter is a special container stored in a :class:`Module`.

    It is designed to hold a :class:`Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x=None, name=None):
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x):
        "Update the parameter value."
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)
