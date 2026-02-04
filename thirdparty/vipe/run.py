import hydra
from omegaconf import DictConfig
from pathlib import Path


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(args: DictConfig) -> None:
    from vipe.streams.base import StreamList

    # Gather all video streams
    stream_list = StreamList.make(args.streams)

    from vipe.pipeline import make_pipeline
    from vipe.utils.logging import configure_logging

    # Process each video stream
    logger = configure_logging()
    for stream_idx in range(len(stream_list)):
        try:
            video_stream = stream_list[stream_idx]
            pose_file = Path(args.pipeline.output.path) / "pose" / f"{video_stream.name()}.npz"
            if pose_file.exists():
                logger.info(f"Pose file for {video_stream.name()} already exists. Skipping processing.")
                continue
            logger.info(
                f"Processing {video_stream.name()} ({stream_idx + 1} / {len(stream_list)})"
            )
            pipeline = make_pipeline(args.pipeline)
            pipeline.run(video_stream)
            logger.info(f"Finished processing {video_stream.name()}")
        except Exception as e:
            logger.error(f"Error processing stream {video_stream.name()}: {e}")


if __name__ == "__main__":
    run()
