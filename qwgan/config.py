analytic = True
shots = 1000					# ignored if analytic is true
device_type = 'default.qubit.tf' # default.qubit is default, other options are default.qubit.tf

# interface set to 'torch', may not work if set to other options
interface = 'tf' 			# 'autograd' is default, 'tf' for tensorflow, 'torch' for pytorch

torch_device = 'cuda:0'			# 'cuda:0' for gpu, 'cpu' for cpu. Only change if you decide to use pytorch interface (tensorflow is recommended)
diff_method="backprop"			# default is 'best', other option is 'backprop', "finite-diff", "parameter-shift"