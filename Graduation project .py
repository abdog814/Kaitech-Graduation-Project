import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Sample data
data = {
    "sensor_id": ["S1", "S2", "S1", "S3", "S2", "S1"],
    "timestamp": pd.to_datetime([
        "2025-04-28 10:00", "2025-04-28 10:00", "2025-04-28 11:00",
        "2025-04-28 10:00", "2025-04-28 11:00", "2025-04-28 12:00"]),
    "temperature": [35.2, 36.5, 36.1, 34.0, 37.2, 37.0],
    "stress": [12.1, 14.0, 12.5, 11.8, 14.3, 13.0],
    "displacement": [0.002, 0.003, 0.0021, 0.0025, 0.0031, 0.0022],
    "sensor_health_score": [99.5, 92.5, 97.17, 97.17, 90.67, 94.67]
}
df = pd.DataFrame(data)

threshold = 93.0
df["status"] = (df["sensor_health_score"] >= threshold).astype(int)

X = df[["temperature", "stress", "displacement"]]
y = df["status"]
model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X, y)

def calculate_health_score(temp, stress, disp):
    temp_score = max(0, 100 - abs(temp - 35) * 5)
    stress_score = max(0, 100 - abs(stress - 12) * 5)
    disp_score = max(0, 100 - abs(disp - 0.002) * 5000)
    return round((temp_score + stress_score + disp_score) / 3, 2)

root = tk.Tk()
root.title("Sensor Monitoring Dashboard")
root.geometry("1100x900")

style = ttk.Style()
style.configure("TButton", font=("Arial", 10), padding=6)
style.map("TButton",
          foreground=[('active', 'blue')],
          background=[('active', '#e6e6e6')])

tree = ttk.Treeview(root, columns=list(df.columns), show="headings")
for col in df.columns:
    tree.heading(col, text=col)
    tree.column(col, width=120)

def refresh_tree(filtered_df=None):
    for row in tree.get_children():
        tree.delete(row)
    display_df = filtered_df if filtered_df is not None else df
    for _, row in display_df.iterrows():
        tags = ("good",) if row["status"] == 1 else ("bad",)
        tree.insert("", "end", values=list(row), tags=tags)
    tree.tag_configure("good", background="#d4edda")
    tree.tag_configure("bad", background="#f8d7da")

refresh_tree()
tree.pack(expand=True, fill="both", pady=10)

def plot_stress():
    fig, ax = plt.subplots(figsize=(6, 4))
    for sensor in df["sensor_id"].unique():
        subset = df[df["sensor_id"] == sensor]
        ax.plot(subset["timestamp"], subset["stress"], marker='o', label=sensor)
    ax.set_title("Stress Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Stress")
    ax.legend()
    ax.grid(True)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

def show_status_chart():
    fig, ax = plt.subplots()
    counts = df["status"].value_counts().sort_index()
    labels = ["❌ Bad", "✅ Good"]
    ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=["red", "green"])
    ax.set_title("Good vs Bad Sensor Status")

    top = tk.Toplevel(root)
    top.title("Sensor Status Pie Chart")
    canvas = FigureCanvasTkAgg(fig, master=top)
    canvas.draw()
    canvas.get_tk_widget().pack()

def show_summary():
    summary = df.groupby("sensor_id")[["temperature", "stress", "displacement"]].agg(["mean", "min", "max"])
    highest_stress_sensor = df.groupby("sensor_id")["stress"].mean().idxmax()
    high_temp_readings = df[df["temperature"] > 36.0]

    top = tk.Toplevel(root)
    top.title("Statistical Summary")
    txt = tk.Text(top, width=120, height=30)
    txt.pack()
    txt.insert("end", "📊 Summary Stats per Sensor:\n")
    txt.insert("end", str(summary) + "\n\n")
    txt.insert("end", f"🔥 Sensor with highest average stress: {highest_stress_sensor}\n\n")
    txt.insert("end", "🌡️ Readings with temperature > 36.0°C:\n")
    txt.insert("end", str(high_temp_readings) + "\n")

def delete_selected():
    selected = tree.selection()
    if not selected:
        messagebox.showinfo("Warning", "Please select a row to delete")
        return
    for item in selected:
        values = tree.item(item, "values")
        global df
        df = df[~(
            (df["sensor_id"] == values[0]) &
            (df["timestamp"].astype(str) == values[1]) &
            (df["temperature"] == float(values[2]))
        )]
    refresh_tree()

def add_reading():
    try:
        sid = sensor_id_entry.get()
        timestamp = datetime.strptime(timestamp_entry.get(), "%Y-%m-%d %H:%M")
        temp = float(temp_entry.get())
        stress = float(stress_entry.get())
        disp = float(disp_entry.get())

        health = calculate_health_score(temp, stress, disp)
        prediction = model.predict([[temp, stress, disp]])[0]

        global df
        new_row = pd.DataFrame([[sid, timestamp, temp, stress, disp, health, prediction]], columns=df.columns)
        df = pd.concat([df, new_row], ignore_index=True)
        refresh_tree()

        messagebox.showinfo("Success", f"✔️ Data added successfully and classified as {'Good' if prediction == 1 else 'Bad'}")
    except Exception as e:
        messagebox.showerror("Error", f"Input error: {str(e)}")

def make_prediction():
    try:
        temp = float(temp_entry.get())
        stress = float(stress_entry.get())
        disp = float(disp_entry.get())
        prediction = model.predict([[temp, stress, disp]])[0]
        quality = "✅ Good" if prediction == 1 else "❌ Bad"
        messagebox.showinfo("Prediction", f"Predicted Status: {quality}")
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {str(e)}")

def save_to_excel():
    try:
        df.to_excel("sensor_data_saved.xlsx", index=False)
        messagebox.showinfo("Saved", "✔️ Data saved to sensor_data_saved.xlsx")
    except Exception as e:
        messagebox.showerror("Error", f"Save failed: {str(e)}")

def filter_by_sensor():
    sensor = filter_entry.get()
    if sensor:
        filtered = df[df["sensor_id"] == sensor]
        refresh_tree(filtered)
    else:
        refresh_tree()

# Input form
input_frame = ttk.LabelFrame(root, text="Add New Reading")
input_frame.pack(fill="x", padx=10, pady=5)

labels = ["Sensor ID", "Timestamp (YYYY-MM-DD HH:MM)", "Temperature", "Stress", "Displacement"]
entries = []
for i, label in enumerate(labels):
    ttk.Label(input_frame, text=label).grid(row=0, column=i, padx=5)
    entry = ttk.Entry(input_frame)
    entry.grid(row=1, column=i, padx=5)
    entries.append(entry)

sensor_id_entry, timestamp_entry, temp_entry, stress_entry, disp_entry = entries

btn_frame = ttk.Frame(input_frame)
btn_frame.grid(row=2, column=0, columnspan=len(labels), pady=10)

add_btn = ttk.Button(btn_frame, text="➕ Add", command=add_reading)
predict_btn = ttk.Button(btn_frame, text="🔮 Predict Status", command=make_prediction)

add_btn.pack(side="left", padx=10)
predict_btn.pack(side="left", padx=10)

bottom_btn_frame = ttk.Frame(root)
bottom_btn_frame.pack(pady=5)

for text, cmd in [
    ("📈 Show Stress Plot", plot_stress),
    ("📊 Sensor Summary", show_summary),
    ("🧩 Show Status Pie", show_status_chart),
    ("🗑 Delete Selected", delete_selected),
    ("💾 Save to Excel", save_to_excel)
]:
    ttk.Button(bottom_btn_frame, text=text, command=cmd).pack(side="left", padx=5)

control_frame = ttk.Frame(root)
control_frame.pack(pady=5)

filter_entry = ttk.Entry(control_frame)
filter_entry.grid(row=0, column=0, padx=5)
ttk.Button(control_frame, text="🔍 Filter by Sensor", command=filter_by_sensor).grid(row=0, column=1)

root.mainloop()
