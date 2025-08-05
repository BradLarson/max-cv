import numpy as np
from max.driver import CPU, Device, Tensor
from max.dtype import DType
from max.graph import Graph
from max.engine import InferenceSession
from pathlib import Path
from max_cv.io import load_image_into_tensor


def generate_test_tensor(
    device: Device, dtype: DType, shape: tuple = (4, 6, 3)
) -> Tensor:
    """Generate a test tensor with realistic image data.

    If shape matches standard test image dimensions, loads real image data.
    Otherwise creates realistic synthetic data based on the shape.
    """
    # Use real image data when possible
    if shape == (600, 450, 3):
        # Load small test image
        img = load_image_into_tensor(
            Path("examples/resources/bucky_birthday_small.jpeg"), CPU()
        )
        if dtype != DType.uint8:
            # Convert to float32 and normalize for operations that expect 0-1 range
            img_np = img.to_numpy().astype(np.float32) / 255.0
            return Tensor.from_numpy(img_np).to(device)
        return img.to(device)
    elif shape == (2000, 1500, 3):
        # Load large test image
        img = load_image_into_tensor(
            Path("examples/resources/bucky_birthday.jpeg"), CPU()
        )
        if dtype != DType.uint8:
            img_np = img.to_numpy().astype(np.float32) / 255.0
            return Tensor.from_numpy(img_np).to(device)
        return img.to(device)
    else:
        # For other shapes, create realistic synthetic data
        # Create a gradient pattern with some noise for more realistic testing
        h, w = shape[:2]
        if len(shape) == 3 and shape[2] == 3:  # RGB
            # Create a gradient pattern
            x_grad = np.linspace(0, 1, w).reshape(1, w)
            y_grad = np.linspace(0, 1, h).reshape(h, 1)

            # Create different patterns for each channel
            r_channel = x_grad * 0.7 + y_grad * 0.3
            g_channel = (1 - x_grad) * 0.6 + y_grad * 0.4
            b_channel = x_grad * y_grad * 0.8 + 0.2

            image_array = np.stack([r_channel, g_channel, b_channel], axis=-1)

            # Add some noise for realism
            noise = np.random.normal(0, 0.05, shape)
            image_array = np.clip(image_array + noise, 0, 1)

        elif len(shape) == 3 and shape[2] == 1:  # Grayscale
            # Create grayscale gradient
            x_grad = np.linspace(0, 1, w).reshape(1, w)
            y_grad = np.linspace(0, 1, h).reshape(h, 1)
            image_array = ((x_grad + y_grad) / 2).reshape(h, w, 1)

            # Add noise
            noise = np.random.normal(0, 0.05, shape)
            image_array = np.clip(image_array + noise, 0, 1)
        else:
            # Fallback to zeros for unknown shapes
            image_array = np.zeros(shape, dtype=np.float32)

        # Convert to the requested dtype
        if dtype == DType.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(dtype.to_numpy())

        return Tensor.from_numpy(image_array).to(device)


def run_graph(graph: Graph, input: Tensor, session: InferenceSession) -> Tensor:
    model = session.load(graph)
    result = model.execute(input)[0]
    return result.to(CPU())


def make_graph(name, **kwargs) -> Graph:
    return Graph(
        name,
        **kwargs,
        custom_extensions=[Path(__file__).parent / ".." / "max_cv" / "operations_mojo"],
    )
