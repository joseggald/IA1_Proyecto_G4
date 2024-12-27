import tkinter as tk
from tkinter import ttk
import threading
from model import EnhancedTopicAwareChatbot
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledText
import json
from datetime import datetime
import time

class ModernChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Assistant Pro")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # Variables
        self.status_var = tk.StringVar(value="Initializing...")
        self.chatbot = None
        self.is_bot_typing = False
        self.chat_history = []
        self.typing_dots = 0
        
        # Configurar GUI
        self.setup_gui()
        self.setup_bindings()
        
        # Cargar historial
        self.load_chat_history()
        
        # Iniciar chatbot en un hilo separado
        self.init_thread = threading.Thread(target=self.initialize_chatbot)
        self.init_thread.start()
        
        # Iniciar animación de bienvenida
        self.root.after(500, self.show_welcome_message)

    def setup_gui(self):
        # Frame principal
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Sidebar
        self.sidebar = ttk.Frame(self.main_frame, style="secondary.TFrame")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        
        # Botón nuevo chat
        self.new_chat_btn = ttk.Button(
            self.sidebar,
            text="+ New Chat",
            style="info.Outline.TButton",
            command=self.new_chat,
            width=15
        )
        self.new_chat_btn.pack(pady=10, padx=5)
        
        # Lista de chats previos
        self.history_frame = ttk.Frame(self.sidebar)
        self.history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame principal de chat
        self.chat_main = ttk.Frame(self.main_frame)
        self.chat_main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Header
        header_frame = ttk.Frame(self.chat_main)
        header_frame.pack(fill=tk.X, pady=10)
        
        title_label = ttk.Label(
            header_frame,
            text="AI Assistant Pro",
            font=("Segoe UI", 18, "bold"),
            foreground="#7289DA"
        )
        title_label.pack(pady=5)
        
        # Área de mensajes
        self.messages_area = ScrolledText(
            self.chat_main,
            padding=10,
            height=20,
            wrap=tk.WORD,
            autohide=True,
            font=("Segoe UI", 11)
        )
        self.messages_area.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Frame de entrada
        input_frame = ttk.Frame(self.chat_main)
        input_frame.pack(fill=tk.X, pady=10)
        
        self.input_field = ttk.Entry(
            input_frame,
            font=("Segoe UI", 12)
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.send_button = ttk.Button(
            input_frame,
            text="Send",
            style="info.TButton",
            command=self.send_message,
            width=10
        )
        self.send_button.pack(side=tk.RIGHT)
        
        # Barra de estado
        self.status_label = ttk.Label(
            self.chat_main,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            foreground="gray"
        )
        self.status_label.pack(fill=tk.X, pady=5)
        
        # Configurar estilos de mensajes
        self.setup_message_styles()

    def setup_message_styles(self):
        self.messages_area.tag_configure(
            "user",
            foreground="#FFFFFF",
            background="#2F3136",
            spacing1=15,
            spacing3=15,
            lmargin1=20,
            lmargin2=20,
            rmargin=20,
            font=("Segoe UI", 11)
        )
        
        self.messages_area.tag_configure(
            "bot",
            foreground="#E1E1E6",
            background="#36393F",
            spacing1=15,
            spacing3=15,
            lmargin1=20,
            lmargin2=20,
            rmargin=20,
            font=("Segoe UI", 11)
        )
        
        self.messages_area.tag_configure(
            "confidence",
            foreground="#72767D",
            font=("Segoe UI", 9, "italic")
        )
        
        self.messages_area.tag_configure(
            "timestamp",
            foreground="#72767D",
            font=("Segoe UI", 8)
        )

    def show_welcome_message(self):
        welcome_text = "Welcome to AI Assistant Pro! How can I help you today?"
        self.messages_area.insert(tk.END, "Bot: ", "bot")
        self.animate_text(welcome_text, "bot")

    def animate_text(self, text, tag):
        """Anima el texto carácter por carácter"""
        for char in text:
            self.messages_area.insert(tk.END, char, tag)
            self.messages_area.see(tk.END)
            self.messages_area.update()
            time.sleep(0.02)
        self.messages_area.insert(tk.END, "\n\n", tag)

    def new_chat(self):
        self.messages_area.delete(1.0, tk.END)
        self.show_welcome_message()

    def load_chat_history(self):
        try:
            with open('chat_history.json', 'r', encoding='utf-8') as f:
                self.chat_history = json.load(f)
                self.update_history_buttons()
        except:
            self.chat_history = []

    def save_chat_history(self):
        try:
            with open('chat_history.json', 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving chat history: {e}")

    def update_history_buttons(self):
        for widget in self.history_frame.winfo_children():
            widget.destroy()
        
        for i, chat in enumerate(self.chat_history[-3:]):
            chat_btn = ttk.Button(
                self.history_frame,
                text=f"Chat {len(self.chat_history)-i}",
                style="secondary.Outline.TButton",
                width=15
            )
            chat_btn.pack(pady=2)

    def send_message(self, event=None):
        message = self.input_field.get().strip()
        if not message or not self.chatbot:
            return
        
        # Limpiar campo de entrada
        self.input_field.delete(0, tk.END)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M")
        
        # Mostrar mensaje del usuario
        self.messages_area.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.messages_area.insert(tk.END, "You: ", "user")
        self.animate_text(message, "user")
        
        # Guardar en historial
        self.chat_history.append({
            'sender': 'user',
            'message': message,
            'timestamp': timestamp
        })
        self.save_chat_history()
        self.update_history_buttons()
        
        # Iniciar animación de escritura
        self.is_bot_typing = True
        self.simulate_typing()
        
        # Obtener respuesta del bot
        threading.Thread(target=self.get_bot_response, args=(message,)).start()

    def get_bot_response(self, message):
        try:
            # Simular delay para la animación
            time.sleep(1)
            
            response, confidence = self.chatbot.get_response(message)
            
            # Detener animación de escritura
            self.is_bot_typing = False
            self.update_status("Ready")
            
            # Timestamp
            timestamp = datetime.now().strftime("%H:%M")
            
            # Mostrar respuesta
            self.messages_area.insert(tk.END, f"[{timestamp}] ", "timestamp")
            self.messages_area.insert(tk.END, "Bot: ", "bot")
            self.animate_text(response, "bot")
            
            if confidence > 0:
                self.messages_area.insert(
                    tk.END,
                    f"(Confidence: {confidence:.2f})\n",
                    "confidence"
                )
            
            # Guardar en historial
            self.chat_history.append({
                'sender': 'bot',
                'message': response,
                'timestamp': timestamp,
                'confidence': confidence
            })
            self.save_chat_history()
            self.update_history_buttons()
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            print(f"Error getting response: {str(e)}")

    def simulate_typing(self):
        if self.is_bot_typing:
            dots = "." * ((self.typing_dots % 3) + 1)
            self.update_status(f"Bot is typing{dots}")
            self.typing_dots += 1
            self.root.after(500, self.simulate_typing)

    def initialize_chatbot(self):
        try:
            self.update_status("Loading AI model...")
            self.chatbot = EnhancedTopicAwareChatbot()
            
            if self.chatbot.load_model():
                self.update_status("Model loaded successfully!")
            else:
                self.update_status("Training new model...")
                self.chatbot.train_model()
                self.update_status("Ready to chat!")
            
            self.root.after(0, self.enable_input)
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            print(f"Error initializing chatbot: {str(e)}")

    def enable_input(self):
        self.input_field.configure(state="normal")
        self.send_button.configure(state="normal")

    def update_status(self, message):
        self.root.after(0, lambda: self.status_var.set(message))

    def setup_bindings(self):
        self.input_field.bind("<Return>", self.send_message)
        self.input_field.configure(state="disabled")
        self.send_button.configure(state="disabled")

def main():
    root = ttk.Window(themename="darkly")
    app = ModernChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()