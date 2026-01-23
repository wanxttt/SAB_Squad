import sys
import shutil
try:
    import traci
    import ultralytics
    import cv2
except ImportError:
    pass

print("--- VS CODE CONFIGURATION CHECK ---")
print(f"Python executable being used: {sys.executable}")
print(f"Python version: {sys.version.split()[0]}")

errors = False

# 1. Check SUMO path
if not shutil.which('sumo'):
    print("❌ ERROR: SUMO not found in system PATH.")
    errors = True

# 2. Check Libraries
if 'traci' not in sys.modules:
    print("❌ ERROR: 'traci' library not found.")
    errors = True
if 'ultralytics' not in sys.modules or 'cv2' not in sys.modules:
    print("❌ ERROR: Vision libraries (ultralytics/opencv) not found.")
    errors = True

if not errors:
    print("\n✅ SUCCESS: VS Code is using the correct Python 3.11 environment with all libraries.")
else:
    print("\n⚠️ FAILURE: VS Code is using the wrong Python interpreter. See errors above.")