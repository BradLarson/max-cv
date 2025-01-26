from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, Shape, TensorType, TensorValue
from pathlib import Path
from typing import List
from .io import normalize_image, restore_image

class ImagePipeline:
    name: str
    _pipeline_dtype: DType
    _shape: Shape
    _graph: Graph
    _model: Model
    input_image: TensorValue
    input_images: List[TensorValue]
    output_image: TensorValue

    def __init__(
        self,
        name: str,
        shape: Shape,
        pipeline_dtype: DType,
        num_inputs: int = 1
    ):
        self.name = name
        self._shape = shape
        self._pipeline_dtype = pipeline_dtype
        input_types = [
            TensorType(DType.uint8, shape=self._shape) for i in range(num_inputs)
        ]
        self._graph = Graph(
            self.name,
            input_types=input_types
        )

    def __enter__(self):
        self._graph = self._graph.__enter__()
        images = self._graph.inputs
        self.input_images = [
            normalize_image(image, dtype=self._pipeline_dtype) for image in images
        ]
        self.input_image = self.input_images[0]
        return self

    def __exit__(self, *exc):
        if hasattr(self, "output_image"):
            restored_image = restore_image(self.output_image)
            self._graph.output(restored_image)
        self._graph.__exit__(*exc)

    def output(self, image: TensorValue):
        """Mark the output of a processing pipeline."""
        self.output_image = image

    def compile(self, device: Device):
        """Compile the pipeline computational graph for a given device."""
        # TODO: Find way to not hardcode this.
        operations_path = Path("operations.mojopkg")
        session = InferenceSession(
            devices=[device],
            custom_extensions=operations_path,
        )

        self._model = session.load(self._graph)
    
    def __call__(self, *images: Tensor) -> Tensor:
        # TODO: Assert that the image tensor resides on the same device.
        return self._model.execute(*images)[0]
