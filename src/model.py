'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
from abc import ABCMeta, abstractmethod
from pathlib import Path

from openvino.inference_engine import IENetwork, IECore
from perf_count_decor import perf_counts
from timeit_decor import timeit
from util import Utilities
#from profiling import profile


class OpenVINOModel(metaclass=ABCMeta):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, precision, device='CPU', extensions=None, threshold=0.60):
        """
        TODO: Use this to set your instance variables.
        """
        models_path = Path(os.environ["MODELS_PATH"])
        self.model_weights = str(models_path.joinpath(f"{model_name}/{precision}/{model_name}.bin"))
        self.model_structure =str(models_path.joinpath(f"{model_name}/{precision}/{model_name}.xml"))
        self.threshold = threshold
        self._ie_core = IECore()
        self._exec_net = None
        self.device = device
        self.extensions = extensions
        self.perf_counts = None
        self.stats = {}
        try:
            try:
                self.net = self._ie_core.read_network(model=self.model_structure, weights=self.model_weights)
            except AttributeError:
                self.net = IENetwork(model=self.model_structure, weights=self.model_weights)
        except Exception:
            raise ValueError("Could not Initialise the network. have you entered the correct model path?")
        self.default_input_name = next(iter(self.net.inputs))
        self.default_input_shape = self.net.inputs[self.default_input_name].shape
        self.default_output_name = next(iter(self.net.outputs))
        self.default_output_shape = self.net.outputs[self.default_output_name].shape

    @timeit
    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        # Add any necessary extension
        if self.extensions:
            for extension_path in self.extensions:
                self._ie_core.add_extension(extension_path=extension_path, device_name=self.device)  # self.extensions
                # Get the supported layers of the network
        self._exec_net = self._ie_core.load_network(network=self.net, device_name=self.device, num_requests=1)

    @property
    def device_name(self):
        return self._ie_core.get_metric(metric_name="FULL_DEVICE_NAME", device_name=self.device) #SUPPORTED_METRICS

    @property
    def network_name(self):
        return self._exec_net.get_metric("NETWORK_NAME") # return the network name

    @perf_counts
    @timeit
    #@profile
    def predict(self, *args,**kwargs):
        """
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        """
        input_dict = self.preprocess_input(*args, **kwargs)
        # # Start asynchronous inference for specified request.
        infer_request_handle = self._exec_net.start_async(request_id=0, inputs=input_dict)
        infer_status = infer_request_handle.wait()
        output = None
        # Request status code: OK or RESULT_NOT_READY
        if infer_status == 0:
            if len(infer_request_handle.outputs) > 1:
                output = infer_request_handle.outputs
            else:
                output = infer_request_handle.outputs[self.default_output_name]
        return self.preprocess_output(output)

    def check_model(self):
        layers_map = self._ie_core.query_network(network=self.net, device_name=self.device)
        # Look for unsupported layers
        unsupported_layers = [layer for layer in self.net.layers.keys() if layer not in layers_map]
        # If there're unsupported layers, notify and exit
        if len(unsupported_layers):
            raise Exception("There were unsupported layers on the network, try checking if path \
                                 on --cpu_extension is correct. The unsupported layers were: {0}\
                                 ".format(unsupported_layers))

    def preprocess_input(self, *args, **kwargs) -> dict:
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        batch_size, channels, height, width = self.default_input_shape
        batched_image = Utilities.batch_image(args[0],width, height, batch_size, channels)
        input_dict = {self.default_input_name: batched_image}
        return input_dict

    @abstractmethod
    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError
