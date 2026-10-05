from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageOps, ImageTk


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_image(path: str | Path, mode: str = "RGB") -> Image.Image:
    """Завантажити незалежну копію зображення з урахуванням його орієнтації."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert(mode)


def apply_visible(
    base_img: Image.Image, marker_img: Image.Image, opacity: float = 0.7
) -> Image.Image:
    """Накласти логотип у правому нижньому куті, враховуючи його прозорість."""
    if not 0 <= opacity <= 1:
        raise ValueError("Непрозорість має бути від 0 до 1.")
    result = base_img.convert("RGBA")
    marker = marker_img.convert("RGBA")
    margin = min(10, max(0, (min(result.size) - 1) // 4))
    max_width = max(1, min(result.width // 3, result.width - 2 * margin))
    max_height = max(1, min(result.height // 3, result.height - 2 * margin))
    marker.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    alpha = marker.getchannel("A").point(lambda value: round(value * opacity))
    marker.putalpha(alpha)
    position = (result.width - marker.width - margin,
                result.height - marker.height - margin)
    result.alpha_composite(marker, dest=position)
    return result.convert("RGB")


def prepare_hidden_marker(marker_img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Білий фон під прозорими ділянками, градації сірого й розмір контейнера."""
    marker = marker_img.convert("RGBA")
    background = Image.new("RGBA", marker.size, "white")
    background.alpha_composite(marker)
    grayscale = background.convert("L").resize(size, Image.Resampling.LANCZOS)
    return np.array(grayscale, dtype=np.uint8)


def embed_lsb(base_img: Image.Image, marker_img: Image.Image) -> Image.Image:
    """Записати чотирирівневий сірий маркер у два молодші біти R, G, B."""
    base = np.array(base_img.convert("RGB"), dtype=np.uint8)
    marker = prepare_hidden_marker(marker_img, base_img.size)
    # 0..63 -> 0; 64..127 -> 1; 128..191 -> 2; 192..255 -> 3.
    marker_bits = marker >> 6
    encoded = (base & np.uint8(252)) | marker_bits[:, :, None]
    return Image.fromarray(encoded)


def extract_lsb(watermarked_img: Image.Image) -> Image.Image:
    """Прочитати молодші біти та відновити рівні сірого 0, 85, 170, 255."""
    pixels = np.array(watermarked_img.convert("RGB"), dtype=np.uint8)
    bits = pixels & np.uint8(3)
    levels = np.rint(bits.mean(axis=2)).astype(np.uint8)
    return Image.fromarray(levels * np.uint8(85))


class WatermarkApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.base_img = None
        self.marker_img = None
        self.result_img = None
        self.result_name = "result.png"
        self.preview_refs = []
        self.opacity = tk.DoubleVar(value=0.7)
        self.status = tk.StringVar(value="Оберіть основне зображення та маркер.")

        root.title("Лабораторна №5 — Маркування інформації")
        root.geometry("1020x620")
        root.minsize(900, 540)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

        inputs = ttk.Frame(root, padding=12)
        inputs.grid(row=0, column=0, sticky="ew")
        ttk.Button(inputs, text="Відкрити зображення", command=self.open_base).pack(side="left")
        ttk.Button(inputs, text="Відкрити маркер", command=self.open_marker).pack(side="left", padx=8)

        actions = ttk.Frame(root, padding=(12, 0, 12, 12))
        actions.grid(row=1, column=0, sticky="ew")
        ttk.Button(actions, text="Нанести видимий", command=lambda: self.process("visible")).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(actions, text="Нанести прихований", command=lambda: self.process("hidden")).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(actions, text="Зчитати прихований", command=lambda: self.process("read")).grid(row=0, column=2, padx=(0, 8))
        self.save_btn = ttk.Button(actions, text="Зберегти результат", command=self.save_result, state="disabled")
        self.save_btn.grid(row=0, column=3)
        ttk.Label(actions, text="Непрозорість видимого знака:").grid(row=1, column=0, columnspan=2, sticky="w", pady=(12, 0))
        ttk.Scale(actions, from_=0, to=1, variable=self.opacity, length=220).grid(row=1, column=2, sticky="w", pady=(12, 0))

        previews = ttk.Frame(root, padding=12)
        previews.grid(row=2, column=0, sticky="nsew")
        previews.rowconfigure(0, weight=1)
        self.preview_labels = []
        for index, title in enumerate(("Основне зображення", "Маркер", "Результат")):
            previews.columnconfigure(index, weight=1, uniform="preview")
            frame = ttk.LabelFrame(previews, text=title, padding=8)
            frame.grid(row=0, column=index, sticky="nsew", padx=4)
            label = ttk.Label(frame, text="Зображення не вибрано", anchor="center")
            label.pack(fill="both", expand=True)
            self.preview_labels.append(label)

        ttk.Label(root, textvariable=self.status, padding=12, wraplength=880).grid(row=3, column=0, sticky="ew")
        ttk.Label(root, padding=(12, 0, 12, 12), wraplength=880,
                  text="LSB: прихований маркер відновлюється у 4 рівнях сірого. "
                       "Зберігайте у PNG; JPEG, зміна розміру та фільтри можуть пошкодити знак. "
                       "Зчитування саме по собі не доводить наявність маркера.").grid(row=4, column=0, sticky="ew")

    def choose_image(self, title: str) -> str:
        return filedialog.askopenfilename(parent=self.root, title=title,
                                         filetypes=[("Зображення", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp"),
                                                    ("Усі файли", "*.*")])

    def open_base(self) -> None:
        path = self.choose_image("Оберіть основне або марковане зображення")
        if not path:
            return
        try:
            self.base_img = load_image(path)
            self.clear_result()
            self.status.set(f"Відкрито: {Path(path).name} ({self.base_img.width} × {self.base_img.height}).")
            self.update_previews()
        except (OSError, ValueError) as error:
            messagebox.showerror("Помилка відкриття", str(error), parent=self.root)

    def open_marker(self) -> None:
        path = self.choose_image("Оберіть зображення водяного знака")
        if not path:
            return
        try:
            self.marker_img = load_image(path, "RGBA")
            self.clear_result()
            self.status.set(f"Обрано маркер: {Path(path).name}.")
            self.update_previews()
        except (OSError, ValueError) as error:
            messagebox.showerror("Помилка відкриття", str(error), parent=self.root)

    def clear_result(self) -> None:
        self.result_img = None
        self.save_btn.config(state="disabled")

    def update_previews(self) -> None:
        self.preview_refs = []
        for label, image in zip(self.preview_labels, (self.base_img, self.marker_img, self.result_img)):
            if image is None:
                label.config(image="", text="Зображення не вибрано")
                continue
            preview = image.copy()
            preview.thumbnail((270, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(preview)
            self.preview_refs.append(photo)
            label.config(image=photo, text="")

    def process(self, mode: str) -> None:
        if self.base_img is None:
            messagebox.showwarning("Потрібне зображення", "Спочатку відкрийте основне зображення.", parent=self.root)
            return
        if mode != "read" and self.marker_img is None:
            messagebox.showwarning("Потрібен маркер", "Спочатку відкрийте зображення маркера.", parent=self.root)
            return
        if mode == "visible":
            self.result_img = apply_visible(self.base_img, self.marker_img, self.opacity.get())
            self.result_name = "visible_watermark.png"
            self.status.set("Видимий знак нанесено у правому нижньому куті.")
        elif mode == "hidden":
            self.result_img = embed_lsb(self.base_img, self.marker_img)
            self.result_name = "hidden_watermark.png"
            self.status.set("Прихований знак нанесено. Збережіть результат, а потім відкрийте його для зчитування.")
        else:
            self.result_img = extract_lsb(self.base_img)
            self.result_name = "extracted_marker.png"
            self.status.set("Молодші біти зчитано. Якщо знак не наносили або зображення змінювали, результат може бути шумом.")
        self.save_btn.config(state="normal")
        self.update_previews()

    def save_result(self) -> None:
        if self.result_img is None:
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(parent=self.root, title="Зберегти результат у PNG",
                                          initialdir=str(OUTPUT_DIR), initialfile=self.result_name,
                                          defaultextension=".png", filetypes=[("PNG без втрат", "*.png")])
        if not path:
            return
        if Path(path).suffix.lower() != ".png":
            messagebox.showwarning("Формат збереження", "Оберіть назву з розширенням .png.", parent=self.root)
            return
        try:
            self.result_img.save(path, format="PNG")
            self.status.set(f"Збережено: {path}")
        except OSError as error:
            messagebox.showerror("Помилка збереження", str(error), parent=self.root)


def main() -> None:
    root = tk.Tk()
    WatermarkApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
