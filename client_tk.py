import json
import threading
import tkinter as tk
from tkinter import ttk
from urllib.parse import urljoin

import requests


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bad Words API Tester")
        self.geometry("680x480")

        # Inputs
        self.base_url_var = tk.StringVar(value="http://127.0.0.1:8000")
        self.api_key_var = tk.StringVar(value="")
        self.advanced_var = tk.BooleanVar(value=False)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="Base URL:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.base_url_var, width=40).grid(row=0, column=1, sticky="we", columnspan=3)

        ttk.Label(top, text="API Key:").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.api_key_var, width=40, show="*").grid(row=1, column=1, sticky="we", columnspan=3)

        top.columnconfigure(3, weight=1)

        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True, **pad)

        ttk.Label(mid, text="Text to check:").grid(row=0, column=0, sticky="w")
        self.text_entry = tk.Text(mid, height=6, wrap="word")
        self.text_entry.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=(0, 6))

        self.advanced_chk = ttk.Checkbutton(mid, text="Advanced response (list bad words)", variable=self.advanced_var)
        self.advanced_chk.grid(row=2, column=0, sticky="w")

        btns = ttk.Frame(mid)
        btns.grid(row=2, column=3, sticky="e")
        ttk.Button(btns, text="Health", command=self.on_health).pack(side="left", padx=4)
        ttk.Button(btns, text="Send", command=self.on_send).pack(side="left")

        ttk.Label(mid, text="Server response:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.output = tk.Text(mid, height=10, wrap="word", state="disabled")
        self.output.grid(row=4, column=0, columnspan=4, sticky="nsew")

        mid.rowconfigure(1, weight=1)
        mid.rowconfigure(4, weight=1)
        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=1)
        mid.columnconfigure(2, weight=1)
        mid.columnconfigure(3, weight=1)

        status = ttk.Frame(self)
        status.pack(fill="x", **pad)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status, textvariable=self.status_var).pack(side="left")

    def set_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    def write_output(self, text: str):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")

    def _headers(self):
        key = self.api_key_var.get().strip()
        h = {"Content-Type": "application/json"}
        if key:
            h["X-API-Key"] = key
        return h

    def _safe_join(self, base: str, path: str) -> str:
        if not base.endswith("/"):
            base = base + "/"
        if path.startswith("/"):
            path = path[1:]
        return urljoin(base, path)

    def on_health(self):
        def work():
            try:
                self.set_status("Calling /health ...")
                url = self._safe_join(self.base_url_var.get().strip(), "/health")
                r = requests.get(url, timeout=15, headers=self._headers())
                r.raise_for_status()
                data = r.json()
                text = json.dumps(data, ensure_ascii=False, indent=2)
                self.after(0, lambda: self.write_output(text))
                self.after(0, lambda: self.set_status("OK"))
            except Exception as e:
                self.after(0, lambda: self.write_output(str(e)))
                self.after(0, lambda: self.set_status("Error"))

        threading.Thread(target=work, daemon=True).start()

    def on_send(self):
        payload = {
            "text": self.text_entry.get("1.0", "end").strip(),
            "advanced": bool(self.advanced_var.get()),
        }

        def work():
            try:
                self.set_status("Calling /check ...")
                url = self._safe_join(self.base_url_var.get().strip(), "/check")
                r = requests.post(url, timeout=30, headers=self._headers(), json=payload)
                r.raise_for_status()
                data = r.json()
                text = json.dumps(data, ensure_ascii=False, indent=2)
                self.after(0, lambda: self.write_output(text))
                self.after(0, lambda: self.set_status("OK"))
            except Exception as e:
                self.after(0, lambda: self.write_output(str(e)))
                self.after(0, lambda: self.set_status("Error"))

        threading.Thread(target=work, daemon=True).start()


if __name__ == "__main__":
    App().mainloop()