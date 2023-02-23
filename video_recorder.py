import imageio
from pathlib import Path


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

    def save(self, name):
        alg_name, env_name, data_name, version, id = name.split("-")
        path = self.root_dir.joinpath(f"{env_name}-{data_name}/{alg_name}")
        path.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path.joinpath(f"{name}.mp4"), self.frames, fps=20)
