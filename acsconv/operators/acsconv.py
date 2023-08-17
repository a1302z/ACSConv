from .functional import acs_conv_f

from .base_acsconv import _ACSConv

from torch.nn import functional as F


class ACSConv(_ACSConv):
    """
    Vallina ACS Convolution

    Args:
        acs_kernel_split: optional, equally spit if not specified.

        Other arguments are the same as torch.nn.Conv3d.
    Examples:
        >>> import ACSConv
        >>> x = torch.rand(batch_size, 3, D, H, W)
        >>> conv = ACSConv(3, 10, kernel_size=3, padding=1)
        >>> out = conv(x)

        >>> conv = ACSConv(3, 10, acs_kernel_split=(4, 3, 3))
        >>> out = conv(x)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        acs_kernel_split=None,
        bias=True,
        padding_mode="zeros",
        output_padding=0,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            output_padding,
            groups,
            bias,
            padding_mode,
        )
        if acs_kernel_split is None:
            if self.out_channels % 3 == 0:
                self.acs_kernel_split = (
                    self.out_channels // 3,
                    self.out_channels // 3,
                    self.out_channels // 3,
                )
            if self.out_channels % 3 == 1:
                self.acs_kernel_split = (
                    self.out_channels // 3 + 1,
                    self.out_channels // 3,
                    self.out_channels // 3,
                )
            if self.out_channels % 3 == 2:
                self.acs_kernel_split = (
                    self.out_channels // 3 + 1,
                    self.out_channels // 3 + 1,
                    self.out_channels // 3,
                )
        else:
            self.acs_kernel_split = acs_kernel_split

    def forward(self, x):
        """
        Convolution forward function
        Divide the kernel into three parts on output channels based on acs_kernel_split,
        and conduct convolution on three directions seperately. Bias is added at last.
        """

        return acs_conv_f(
            x,
            self.weight,
            self.bias,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
            self.groups,
            self.out_channels,
            self.acs_kernel_split,
        )

    def extra_repr(self):
        s = super().extra_repr() + ", acs_kernel_split={acs_kernel_split}"
        return s.format(**self.__dict__)


class ACSTransposeConv(_ACSConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        acs_kernel_split=None,
        bias=True,
        padding_mode="zeros",
        output_padding=0,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
        )
        if acs_kernel_split is None:
            if self.out_channels % 3 == 0:
                self.acs_kernel_split = (
                    self.out_channels // 3,
                    self.out_channels // 3,
                    self.out_channels // 3,
                )
            if self.out_channels % 3 == 1:
                self.acs_kernel_split = (
                    self.out_channels // 3 + 1,
                    self.out_channels // 3,
                    self.out_channels // 3,
                )
            if self.out_channels % 3 == 2:
                self.acs_kernel_split = (
                    self.out_channels // 3 + 1,
                    self.out_channels // 3 + 1,
                    self.out_channels // 3,
                )
        else:
            self.acs_kernel_split = acs_kernel_split

    def forward(self, x):
        return acs_conv_f(
            x,
            self.weight,
            self.bias,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
            self.groups,
            self.out_channels,
            self.acs_kernel_split,
            output_padding=self.output_padding,
            transposed=True,
            conv_func=F.conv_transpose3d,
        )
