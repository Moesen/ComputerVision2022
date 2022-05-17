from pathlib import Path
import matplotlib.pyplot as plt


class ImgSaver:
    def __init__(self) -> None:
        self._exam_folder_path = Path("../../Exam/")
        self._exam_img_folder_path = self._exam_folder_path / "exercise_imgs"
        self._exam_txt_folder_path = self._exam_folder_path / "exercise_txts"

    def save_fig(self, fn: str) -> None:
        if ".png" not in fn:
            fn = fn.split(".")[0] + ".png"
        plt.savefig((self._exam_img_folder_path / fn).as_posix())

    def save_txt(self, fn: str, content: str):
        if ".txt" not in fn:
            fn = fn.split(".")[0] + ".txt"
        with open(self._exam_txt_folder_path / fn, "w") as f:
            f.write(content)           
