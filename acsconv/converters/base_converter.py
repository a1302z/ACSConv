import torch.nn as nn
from ..utils import _triple_same
from warnings import warn


class BaseConverter(object):
    """
    base class for converters
    """

    converter_attributes = []
    target_conv = None
    target_tranpose_conv = None

    def __init__(self, model):
        """Convert the model to its corresponding counterparts and deal with original weights if necessary"""
        pass

    def convert_module(self, module):
        """
        A recursive function.
        Treat the entire model as a tree and convert each leaf module to
            target_conv if it's Conv2d,
            3d counterparts if it's a pooling or normalization module,
            trilinear mode if it's a Upsample module.
        """
        for child_name, child in module.named_children():
            if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)):
                arguments = type(child).__init__.__code__.co_varnames[1:]

                # cleaning the tuple due to new PyTorch namming schema (or added new variables)
                arguments = [
                    a
                    for a in arguments
                    if a
                    not in [
                        "device",
                        "dtype",
                        "factory_kwargs",
                        "kernel_size_",
                        "stride_",
                        "padding_",
                        "dilation_",
                    ]
                ]

                kwargs = {k: getattr(child, k) for k in arguments}
                kwargs = self.convert_conv_kwargs(kwargs)
                new_class = (
                    self.__class__.target_conv
                    if isinstance(child, nn.Conv2d)
                    else self.__class__.target_tranpose_conv
                )
                setattr(module, child_name, new_class(**kwargs))
            elif hasattr(nn, child.__class__.__name__) and (
                "pool" in child.__class__.__name__.lower()
                or "norm" in child.__class__.__name__.lower()
            ):
                if hasattr(nn, child.__class__.__name__.replace("2d", "3d")):
                    TargetClass = getattr(
                        nn, child.__class__.__name__.replace("2d", "3d")
                    )
                    arguments = TargetClass.__init__.__code__.co_varnames[1:]
                    arguments = [
                        a
                        for a in arguments
                        if a not in ["device", "dtype", "factory_kwargs"]
                    ]
                    kwargs = {k: getattr(child, k) for k in arguments}
                    if "adaptive" in child.__class__.__name__.lower():
                        for k in kwargs.keys():
                            if not isinstance(kwargs[k], bool):
                                kwargs[k] = _triple_same(kwargs[k])
                    setattr(module, child_name, TargetClass(**kwargs))
                else:
                    raise Exception(
                        "No corresponding module in 3D for 2d module {}".format(
                            child.__class__.__name__
                        )
                    )
            elif isinstance(child, (nn.Upsample, nn.UpsamplingBilinear2d)):
                arguments = type(child).__init__.__code__.co_varnames[1:]
                kwargs = {k: getattr(child, k) for k in arguments}
                kwargs["mode"] = (
                    "trilinear" if kwargs["mode"] == "bilinear" else kwargs["mode"]
                )
                if "size" in kwargs:
                    shape = kwargs["size"]
                    if shape[0] == shape[1]:
                        kwargs["size"] = _triple_same(shape[0])
                    else:
                        warn(
                            f"Size kwarg set for upsample module with different shapes. "
                            "Will use first shape."
                        )
                        kwargs["size"] = (*shape, shape[-1])
                setattr(module, child_name, nn.Upsample(**kwargs))
            else:
                self.convert_module(child)
        return module

    def convert_conv_kwargs(self, kwargs):
        """
        Called by self.convert_module. Transform the original Conv2d arguments
        to meet the arguments requirements of target_conv.
        """
        raise NotImplementedError

    def __getattr__(self, attr):
        return getattr(self.model, attr)

    def __setattr__(self, name, value):
        if name in self.__class__.converter_attributes:
            return object.__setattr__(self, name, value)
        else:
            return setattr(self.model, name, value)

    def __call__(self, x):
        return self.model(x)

    def __repr__(self):
        return self.__class__.__name__ + "(\n" + self.model.__repr__() + "\n)"
