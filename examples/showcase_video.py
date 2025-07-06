import click
import cv2 as cv
from collections.abc import Callable
from pathlib import Path
from platform import system

import sys

path_root = Path(__file__).parent.parent
sys.path.append(str(path_root))

from max_cv import ImagePipeline # noqa: E402
from max_cv import operations as ops # noqa: E402
from max.driver import Accelerator, CPU, Device, Tensor # noqa: E402
from max.dtype import DType # noqa: E402
from max.graph import TensorValue # noqa: E402


@click.group()
def showcase_video():
    pass


@showcase_video.command(name="sobel")
@click.option(
    "--value",
    type=float,
    default=1.0,
    show_default=True,
    help="Edge strength.",
)
@click.option("--camera", is_flag=True)
@click.option(
    "--file",
    type=str,
)
def sobel(value, camera, file):
    print("Performing Sobel edge detection with strength:", value)

    def edge_detection(device: Device, input: TensorValue) -> TensorValue:
        processed_image = ops.rgb_to_luminance(device, input)
        return ops.sobel_edge_detection(device, processed_image, strength=1.0)

    if camera:
        run_pipeline_live_video(operations=edge_detection)
    else:
        run_pipeline_video_file(operations=edge_detection, path=file)


def create_pipeline(operations: Callable, sample_frame: Tensor) -> ImagePipeline:
    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device: Device

    try:
        # Unsupported accelerators will throw an error here
        device = Accelerator()
    except ValueError:
        device = CPU()

    with ImagePipeline(
        "video_showcase",
        sample_frame.shape,
        pipeline_dtype=DType.float32,
        device=device,
        num_inputs=1,
    ) as pipeline:
        processed_image = operations(device, pipeline.input_image)
        pipeline.output(processed_image)

    print("Compiling graph...")
    pipeline.compile()
    print("Running pipeline")

    return pipeline


def run_pipeline_video_file(operations: Callable, path: str):
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        print("Cannot open file")
        return

    ret, frame = cap.read()
    pipeline = create_pipeline(operations, Tensor.from_numpy(frame))

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)

    is_mac = system() == "Darwin"
    fourcc = cv.VideoWriter_fourcc(*("mp4v" if is_mac else "XVID"))
    out = cv.VideoWriter(
        f"output.{'mov' if is_mac else 'avi'}", fourcc, fps, (width, height)
    )

    while ret:
        tensor = Tensor.from_numpy(frame)
        result = pipeline(tensor)
        result = result.to(CPU())
        out.write(result.to_numpy())

        ret, frame = cap.read()

    cap.release()
    out.release()


def run_pipeline_live_video(operations: Callable):
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    ret, frame = cap.read()
    pipeline = create_pipeline(operations, Tensor.from_numpy(frame))

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        frame = cv.flip(frame, 1)

        tensor = Tensor.from_numpy(frame)
        result = pipeline(tensor)
        result = result.to(CPU())

        cv.imshow("frame", result.to_numpy())
        if cv.waitKey(1) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    showcase_video()
