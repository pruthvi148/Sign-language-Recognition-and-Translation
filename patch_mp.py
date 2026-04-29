import ctypes
import os
import re
import mediapipe.python.solution_base as sb

file_path = sb.__file__

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Make sure we don't patch twice
if '_get_sp' not in content:
    replacement = """def _get_dependencies(dependency_mapping):
  import ctypes
  def _get_sp(l):
    b = ctypes.create_unicode_buffer(500)
    ctypes.windll.kernel32.GetShortPathNameW(l, b, 500)
    return b.value
  global __file__
  __file__ = _get_sp(__file__)
"""
    patched = re.sub(r'def _get_dependencies\(.*?\):', replacement, content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(patched)
    print("MediaPipe patched successfully for unicode paths!")
else:
    print("MediaPipe already patched.")
