import tkinter as tk
import pybullet as p

def update_camera(physicsClient):
    if ui_toggle.get():
        # UI mode: override camera using the slider values.
        distance = distance_slider.get()
        yaw = yaw_slider.get()
        pitch = pitch_slider.get()
        target = (target_x_slider.get(), target_y_slider.get(), target_z_slider.get())
        p.resetDebugVisualizerCamera(distance, yaw, pitch, target, physicsClientId=physicsClient)
    else:
        # Default navigation: update the UI sliders to reflect the current camera view.
        cam_data = p.getDebugVisualizerCamera(physicsClientId=physicsClient)
        # According to PyBullet documentation, the returned tuple indices are:
        # index 10: distance, index 8: yaw, index 9: pitch, index 11: target tuple.
        distance_slider.set(cam_data[10])
        yaw_slider.set(cam_data[8])
        pitch_slider.set(cam_data[9])
        target_x_slider.set(cam_data[11][0])
        target_y_slider.set(cam_data[11][1])
        target_z_slider.set(cam_data[11][2])
        update_target_canvas()
        update_orientation_canvas()
    
    # Schedule the next update.
    root.after(100, update_camera, physicsClient)

def on_target_canvas_event(event):
    canvas_width = target_canvas.winfo_width()
    canvas_height = target_canvas.winfo_height()
    # Map canvas coordinates to a range of -10 to 10.
    x = (event.x - canvas_width/2) / (canvas_width/2) * 10
    y = (event.y - canvas_height/2) / (canvas_height/2) * 10
    target_x_slider.set(round(x, 2))
    target_y_slider.set(round(y, 2))
    update_target_canvas()

def update_target_canvas():
    target_canvas.delete("all")
    canvas_width = target_canvas.winfo_width()
    canvas_height = target_canvas.winfo_height()
    # Map slider values to canvas coordinates.
    x = (target_x_slider.get() / 10) * (canvas_width/2) + canvas_width/2
    y = (target_y_slider.get() / 10) * (canvas_height/2) + canvas_height/2
    r = 5
    target_canvas.create_oval(x - r, y - r, x + r, y + r, fill="red")

def on_orientation_canvas_event(event):
    canvas_width = orientation_canvas.winfo_width()
    canvas_height = orientation_canvas.winfo_height()
    # Map canvas coordinates:
    # Horizontal (x) maps to yaw in range [-180, 180]
    # Vertical (y) maps to pitch in range [-90, 90]
    yaw = (event.x - canvas_width/2) / (canvas_width/2) * 180
    pitch = (event.y - canvas_height/2) / (canvas_height/2) * 90
    yaw_slider.set(round(yaw, 2))
    pitch_slider.set(round(pitch, 2))
    update_orientation_canvas()

def update_orientation_canvas():
    orientation_canvas.delete("all")
    canvas_width = orientation_canvas.winfo_width()
    canvas_height = orientation_canvas.winfo_height()
    # Map the current slider values to canvas coordinates.
    x = (yaw_slider.get() / 180) * (canvas_width/2) + canvas_width/2
    y = (pitch_slider.get() / 90) * (canvas_height/2) + canvas_height/2
    r = 5
    orientation_canvas.create_oval(x - r, y - r, x + r, y + r, fill="blue")

def preset_selected(preset):
    presets = {
        "Front":      (5,   0,  -10, (0, 0, 0)),
        "Back":       (5, 180,  -10, (0, 0, 0)),
        "Left":       (5, -90,  -10, (0, 0, 0)),
        "Right":      (5,  90,  -10, (0, 0, 0)),
        "Top":        (5,   0,  -90, (0, 0, 0)),
        "Isometric":  (7,  45,  -30, (0, 0, 0))
    }
    if preset in presets:
        d, y, p_val, t = presets[preset]
        distance_slider.set(d)
        yaw_slider.set(y)
        pitch_slider.set(p_val)
        target_x_slider.set(t[0])
        target_y_slider.set(t[1])
        target_z_slider.set(t[2])
        update_target_canvas()
        update_orientation_canvas()

def on_toggle():
    # When switching to UI view, update the slider values based on the current camera state.
    if ui_toggle.get():
        cam_data = p.getDebugVisualizerCamera(physicsClientId=physicsClient)
        distance_slider.set(cam_data[10])
        yaw_slider.set(cam_data[8])
        pitch_slider.set(cam_data[9])
        target_x_slider.set(cam_data[11][0])
        target_y_slider.set(cam_data[11][1])
        target_z_slider.set(cam_data[11][2])
        update_target_canvas()
        update_orientation_canvas()

def run_tk(physicsClient, start_cam):
    global root, distance_slider, yaw_slider, pitch_slider
    global target_x_slider, target_y_slider, target_z_slider, target_canvas
    global orientation_canvas, ui_toggle
    # Store the passed physicsClient as a global variable for access in callbacks.
    physicsClient = physicsClient
    
    init_distance, init_yaw, init_pitch, init_target = start_cam
    
    root = tk.Tk()
    root.title("PyBullet Camera Control")

    # Frame for camera parameter sliders.
    cam_frame = tk.Frame(root)
    cam_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
    
    tk.Label(cam_frame, text="Distance").grid(row=0, column=0, sticky="w")
    distance_slider = tk.Scale(cam_frame, from_=1, to=100, orient=tk.HORIZONTAL)
    distance_slider.set(init_distance)
    distance_slider.grid(row=0, column=1, sticky="ew")
    
    tk.Label(cam_frame, text="Yaw").grid(row=1, column=0, sticky="w")
    yaw_slider = tk.Scale(cam_frame, from_=-180, to=180, orient=tk.HORIZONTAL,
                          command=lambda val: update_orientation_canvas())
    yaw_slider.set(init_yaw)
    yaw_slider.grid(row=1, column=1, sticky="ew")
    
    tk.Label(cam_frame, text="Pitch").grid(row=2, column=0, sticky="w")
    pitch_slider = tk.Scale(cam_frame, from_=-180, to=180, orient=tk.HORIZONTAL,
                            command=lambda val: update_orientation_canvas())
    pitch_slider.set(init_pitch)
    pitch_slider.grid(row=2, column=1, sticky="ew")
    
    tk.Label(cam_frame, text="Presets").grid(row=3, column=0, sticky="w")
    preset_var = tk.StringVar(value="Select Preset")
    preset_menu = tk.OptionMenu(cam_frame, preset_var, "Front", "Back", "Left", "Right", "Top", "Isometric",
                                 command=preset_selected)
    preset_menu.grid(row=3, column=1, sticky="ew")
    
    # Toggle checkbutton: default navigation (off) vs. UI navigation (on)
    ui_toggle = tk.BooleanVar(value=False)
    toggle_cb = tk.Checkbutton(cam_frame, text="UI Navigation", variable=ui_toggle, command=on_toggle)
    toggle_cb.grid(row=4, column=0, columnspan=2, sticky="w")
    
    # Frame for target coordinate sliders.
    target_frame = tk.Frame(root)
    target_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
    
    tk.Label(target_frame, text="Target X").grid(row=0, column=0, sticky="w")
    target_x_slider = tk.Scale(target_frame, from_=-10, to=10, orient=tk.HORIZONTAL,
                               command=lambda val: update_target_canvas())
    target_x_slider.set(init_target[0])
    target_x_slider.grid(row=0, column=1, sticky="ew")
    
    tk.Label(target_frame, text="Target Y").grid(row=1, column=0, sticky="w")
    target_y_slider = tk.Scale(target_frame, from_=-10, to=10, orient=tk.HORIZONTAL,
                               command=lambda val: update_target_canvas())
    target_y_slider.set(init_target[1])
    target_y_slider.grid(row=1, column=1, sticky="ew")
    
    tk.Label(target_frame, text="Target Z").grid(row=2, column=0, sticky="w")
    target_z_slider = tk.Scale(target_frame, from_=-10, to=10, orient=tk.HORIZONTAL)
    target_z_slider.set(init_target[2])
    target_z_slider.grid(row=2, column=1, sticky="ew")
    
    # Canvas for visual target (X/Y) selection.
    tk.Label(root, text="Drag on the canvas to set target X/Y").pack(pady=(10, 0))
    target_canvas = tk.Canvas(root, width=200, height=200, bg="white")
    target_canvas.pack(padx=10, pady=5)
    target_canvas.bind("<Button-1>", on_target_canvas_event)
    target_canvas.bind("<B1-Motion>", on_target_canvas_event)
    
    # Canvas for visual orientation (yaw/pitch) selection.
    tk.Label(root, text="Drag on the canvas to set orientation (yaw/pitch)").pack(pady=(10, 0))
    orientation_canvas = tk.Canvas(root, width=200, height=200, bg="white")
    orientation_canvas.pack(padx=10, pady=5)
    orientation_canvas.bind("<Button-1>", on_orientation_canvas_event)
    orientation_canvas.bind("<B1-Motion>", on_orientation_canvas_event)
    
    update_target_canvas()
    update_orientation_canvas()
    update_camera(physicsClient)
    
    root.mainloop()

# Example usage:
# run_tk(physicsClient, (5, 0, -10, (0, 0, 0)))