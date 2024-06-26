from datetime import datetime
from pathlib import Path

import imageio

class VideoRecorder(object):
    def __init__(self, root_dir="./videos", height=256, width=256, fps=20):
        self.root_dir = Path(root_dir)
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []

    def record(self, env):
        frame = env.render("rgb_array", height=self.height, width=self.width)
        self.frames.append(frame)

    def save(self, name, wandb=None):
        path = self.root_dir.joinpath(f"{name}")
        path.mkdir(parents=True, exist_ok=True)
        video_path = path.joinpath(
            f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}-{name}.mp4"
        )
        imageio.mimsave(
            video_path,
            self.frames,
            fps=20,
        )
        if wandb is not None:
            wandb.log(
                {
                    "video": wandb.Video(
                        str(video_path),
                        caption="Final agent behaviour",
                        fps=20,
                        format="mp4",
                    )
                }
            )
