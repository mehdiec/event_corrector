from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication


def is_shift_pressed():
    app = QApplication.instance()
    return app.keyboardModifiers() & Qt.ShiftModifier


def is_left_click_pressed():
    app = QApplication.instance()
    return app.mouseButtons() & Qt.LeftButton


class SegmenterBindings:
    change_correcting_mode = "m"
    save_state = "s"
    cancel_drawing = "Escape"
    update_outline = "u"
    update_outline_alt = "Shift-u"
    is_deletion_mode_activated = is_shift_pressed
    is_left_click_pressed = is_left_click_pressed
    undo = "z"
    free_hand = "f"