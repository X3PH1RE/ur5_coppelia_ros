import subprocess
import threading

movement_detected = threading.Event()

def monitor_output(script_path, trigger_message):
    proc = subprocess.Popen(
        ["python3", "-u", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in proc.stdout:
        print(f"[{script_path}] {line.strip()}")
        if trigger_message in line:
            movement_detected.set() 

    proc.wait()

drum = "drum_manipulator.py"
plane_fit_1 = "first_sweep.py"
adjust_1 = "adjust_dummy_1.py"
plane_fit_2 = "second_sweep.py"
adjust_2 = "adjust_dummy_2.py"

drum_proc = subprocess.Popen(["python3", "-u", drum])
plane_proc_1 = subprocess.Popen(["python3", "-u", plane_fit_1])

adjust_thread = threading.Thread(
    target=monitor_output, args=(adjust_1, "movement complete.")
)
adjust_thread.start()

movement_detected.wait()

print("Starting second sweep...")
plane_proc_2 = subprocess.Popen(["python3", "-u", plane_fit_2])
adjust_proc_2 = subprocess.Popen(["python3", "-u", adjust_2])

drum_proc.wait()
plane_proc_1.wait()
adjust_thread.join()
plane_proc_2.wait()
adjust_proc_2.wait()
