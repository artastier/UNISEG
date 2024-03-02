from PyQt5.QtWidgets import QFileDialog, QWidget


class PathSelection(QWidget):
    def __init__(self, selection_message: str):
        super().__init__()

        options = QFileDialog.Options()

        # Show the file dialog for selecting a directory
        self.directory = QFileDialog.getExistingDirectory(self, selection_message, "", options=options)

        if not self.directory:
            self.directory = None
