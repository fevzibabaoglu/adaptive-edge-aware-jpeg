import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class PreviewPanel:
    """Manages the preview section of the application with side-by-side image comparison."""

    def __init__(
        self,
        parent,
        process_function,
        preview_path=None,
        title="Preview",
        canvas_bg="#f0f0f0",
        filetypes=(
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ),
        padding=10,
        initial_load_delay=100
    ):
        """
        Initialize the preview panel with configurable options.

        Args:
            parent: Parent tkinter widget
            process_function: Function that processes images, takes PIL Image and returns Image
            preview_path: Path to initial preview image
            title: Title for the panel frame
            canvas_bg: Background color for the canvas
            filetypes: Tuple of file type filters for file dialog
            padding: Padding for the label frame
            initial_load_delay: Delay in ms before loading initial preview
        """
        # Configuration properties
        self.parent = parent
        self.preview_path = preview_path
        self.process_function = process_function
        self.title = title
        self.canvas_bg = canvas_bg
        self.filetypes = filetypes
        self.padding = padding

        # Image holders
        self.original_image = None
        self.processed_image = None

        # Create UI and setup
        self._setup_ui()

        # Auto-load preview if path is provided
        if self.preview_path:
            self.parent.after(initial_load_delay, self.refresh_images)

    def _setup_ui(self):
        """Create and arrange all UI components."""
        # Main frame
        self.frame = ttk.LabelFrame(self.parent, text=self.title, padding=self.padding)

        # Controls bar at top
        self._setup_controls()

        # Canvas for displaying images
        self.canvas = tk.Canvas(self.frame, bg=self.canvas_bg)
        self.canvas.pack(fill='both', expand=True)

        # Update canvas when window is resized
        self.canvas.bind("<Configure>", lambda e: self.refresh_images() if self.original_image else None)

    def _setup_controls(self):
        """Create the control buttons section."""
        control_bar = ttk.Frame(self.frame)
        control_bar.pack(fill='x', pady=(0, 10))

        # Select image button
        select_btn = ttk.Button(
            control_bar,
            text="Select Preview Image",
            command=self.browse_for_image
        )
        select_btn.pack(side='left')

        # Update preview button
        update_btn = ttk.Button(
            control_bar,
            text="Update Preview",
            command=self.process_and_display
        )
        update_btn.pack(side='right')

    def refresh_images(self):
        """Refresh both the original and processed images."""
        self.load_original_image()
        self.process_and_display()

    def browse_for_image(self):
        """Let user select an image file using a dialog."""
        file_path = filedialog.askopenfilename(filetypes=self.filetypes)

        if file_path:
            self.set_preview_path(file_path)

    def set_preview_path(self, path):
        """Set a new preview image path and reload the preview."""
        self.preview_path = path
        self.refresh_images()

    def set_process_function(self, process_function):
        """Update the image processing function."""
        self.process_function = process_function
        self.process_and_display()

    def load_original_image(self):
        """Load and display the original preview image."""
        if not self._ensure_image_exists():
            return

        try:
            # Open the image file
            original_img = Image.open(self.preview_path)

            # Get display dimensions
            dimensions = self._get_display_dimensions()

            # Create thumbnail for display
            img_display = original_img.copy()
            img_display.thumbnail((
                dimensions["width"],
                dimensions["height"] // 2 - 20
            ))
            self.original_image = ImageTk.PhotoImage(img_display)

            # Clear canvas and draw original image
            self.canvas.delete("all")
            self._render_original_image(dimensions)
            self._render_divider(dimensions)

        except Exception as error:
            self._display_error(f"Failed to load preview: {error}")

    def process_and_display(self):
        """Process the image and display the result."""
        if not self._ensure_image_exists():
            return

        try:
            # Get original image and process it
            original_img = Image.open(self.preview_path)
            processed_img = self.process_function(original_img)

            # Prepare for display
            dimensions = self._get_display_dimensions()

            # Create thumbnail for display
            processed_display = processed_img.copy()
            processed_display.thumbnail((
                dimensions["width"],
                dimensions["height"] // 2 - 20
            ))
            self.processed_image = ImageTk.PhotoImage(processed_display)

            # Update display
            self.canvas.delete("processed")
            self._render_processed_image(dimensions)

        except Exception as error:
            self._display_error(f"Failed to process image: {error}")

    def _ensure_image_exists(self):
        """Verify that the image file exists and is accessible."""
        return self.preview_path and os.path.exists(self.preview_path)

    def _get_display_dimensions(self):
        """Get current canvas dimensions for layout calculations."""
        return {
            "width": self.canvas.winfo_width() or 400,
            "height": self.canvas.winfo_height() or 300
        }

    def _render_original_image(self, dimensions):
        """Render the original image on the top half of the canvas."""
        self.canvas.create_image(
            dimensions["width"] // 2,
            dimensions["height"] // 4,
            image=self.original_image,
            anchor='center',
            tags="original"
        )

    def _render_processed_image(self, dimensions):
        """Render the processed image on the bottom half of the canvas."""
        self.canvas.create_image(
            dimensions["width"] // 2,
            3 * dimensions["height"] // 4,
            image=self.processed_image,
            anchor='center',
            tags="processed"
        )

    def _render_divider(self, dimensions):
        """Render a separator line between original and processed images."""
        self.canvas.create_line(
            0, dimensions["height"] // 2,
            dimensions["width"], dimensions["height"] // 2,
            fill="gray", width=2,
            tags="separator"
        )

    def _display_error(self, message):
        """Show an error message dialog to the user."""
        messagebox.showerror("Error", message)
