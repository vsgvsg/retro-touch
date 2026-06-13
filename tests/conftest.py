import pytest
import tkinter as tk
import sys

@pytest.fixture(scope="session")
def shared_tk_root():
    root = tk.Tk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass

@pytest.fixture
def tk_root(shared_tk_root):
    # Prepare the root window: deiconify (make sure it's mapped so event_generate works), clean up widgets
    shared_tk_root.deiconify()
    for child in shared_tk_root.winfo_children():
        try:
            child.destroy()
        except Exception:
            pass
    shared_tk_root.title("Tk")
    yield shared_tk_root
    # Cleanup after test
    for child in shared_tk_root.winfo_children():
        try:
            child.destroy()
        except Exception:
            pass
    try:
        shared_tk_root.withdraw()
    except Exception:
        pass
