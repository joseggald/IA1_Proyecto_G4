import tkinter as tk
from tkinter import ttk, messagebox
import threading
from model import EnhancedTopicAwareChatbot
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledText
import json
from datetime import datetime
import time
import os
import uuid

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
        self.current_chat_id = None
        self.typing_dots = 0
        
        # Asegurar directorio de historial
        self.history_dir = "chat_history"
        os.makedirs(self.history_dir, exist_ok=True)
        
        # Configurar GUI
        self.setup_gui()
        self.setup_bindings()
        
        # Cargar historial e iniciar chatbot
        self.load_chat_history()
        self.init_thread = threading.Thread(target=self.initialize_chatbot)
        self.init_thread.start()

    def setup_gui(self):
        # Frame principal
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configurar sidebar
        self.setup_sidebar()
        
        # Configurar área principal
        self.setup_main_area()
        
        # Configurar estilos de mensajes
        self.setup_message_styles()

    def setup_sidebar(self):
        # Sidebar principal
        self.sidebar = ttk.Frame(self.main_frame, style="secondary.TFrame", width=250)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        self.sidebar.pack_propagate(False)
        
        # Header del sidebar
        sidebar_header = ttk.Frame(self.sidebar, style="secondary.TFrame")
        sidebar_header.pack(fill=tk.X, padx=5, pady=10)
        
        # Botón nuevo chat
        self.new_chat_btn = ttk.Button(
            sidebar_header,
            text="+ New Chat",
            style="primary.TButton",
            command=self.new_chat,
            width=15
        )
        self.new_chat_btn.pack(pady=(0, 5))
        
        # Separador
        ttk.Separator(sidebar_header, orient="horizontal").pack(fill=tk.X, pady=5)
        
        # Título de historial
        history_label = ttk.Label(
            sidebar_header,
            text="Chat History",
            font=("Segoe UI", 10, "bold"),
            foreground="#7289DA",
            style="secondary.TLabel"
        )
        history_label.pack(pady=5)
        
        # Frame scrollable para historial
        self.setup_history_frame()

    def setup_history_frame(self):
        # Contenedor del canvas
        history_container = ttk.Frame(self.sidebar, style="secondary.TFrame")
        history_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas y scrollbar
        self.history_canvas = tk.Canvas(
            history_container,
            bg="#2F3136",
            highlightthickness=0
        )
        history_scrollbar = ttk.Scrollbar(
            history_container,
            orient="vertical",
            command=self.history_canvas.yview
        )
        
        # Frame para los botones de historial
        self.history_frame = ttk.Frame(
            self.history_canvas,
            style="secondary.TFrame"
        )
        
        # Configurar canvas
        self.history_canvas.configure(yscrollcommand=history_scrollbar.set)
        
        # Empaquetar elementos
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Crear ventana en canvas
        self.canvas_frame = self.history_canvas.create_window(
            (0, 0),
            window=self.history_frame,
            anchor="nw",
            width=self.sidebar.winfo_reqwidth() - 30
        )
        
        # Configurar eventos de scroll
        self.history_frame.bind("<Configure>", self.on_frame_configure)
        self.history_canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Bind mouse wheel
        self.history_canvas.bind_all("<MouseWheel>", self.on_mousewheel)

    def setup_main_area(self):
        # Frame principal de chat
        self.chat_main = ttk.Frame(self.main_frame)
        self.chat_main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Header
        self.setup_header()
        
        # Área de mensajes
        self.setup_messages_area()
        
        # Área de entrada
        self.setup_input_area()
        
        # Barra de estado
        self.setup_status_bar()

    def setup_header(self):
        header_frame = ttk.Frame(self.chat_main)
        header_frame.pack(fill=tk.X, pady=10)
        
        title_label = ttk.Label(
            header_frame,
            text="AI Assistant Pro",
            font=("Segoe UI", 18, "bold"),
            foreground="#7289DA"
        )
        title_label.pack(pady=5)

    def setup_messages_area(self):
        self.messages_area = ScrolledText(
            self.chat_main,
            padding=10,
            height=20,
            wrap=tk.WORD,
            autohide=True,
            font=("Segoe UI", 11)
        )
        self.messages_area.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

    def setup_input_area(self):
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
            style="primary.TButton",
            command=self.send_message,
            width=10
        )
        self.send_button.pack(side=tk.RIGHT)

    def setup_status_bar(self):
        self.status_label = ttk.Label(
            self.chat_main,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            foreground="gray"
        )
        self.status_label.pack(fill=tk.X, pady=5)

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

    def on_frame_configure(self, event=None):
        self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))

    def on_canvas_configure(self, event):
        self.history_canvas.itemconfig(self.canvas_frame, width=event.width)

    def on_mousewheel(self, event):
        self.history_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def new_chat(self):
        # Generar nuevo ID para el chat
        self.current_chat_id = str(uuid.uuid4())
        
        # Limpiar área de mensajes
        self.messages_area.delete(1.0, tk.END)
        
        # Actualizar historial
        self.save_current_chat()
        self.update_history_buttons()
        
        # Habilitar entrada
        self.enable_input()

    def load_chat_history(self):
        try:
            history_files = [f for f in os.listdir(self.history_dir) if f.endswith('.json')]
            self.chat_history = []
            
            for file in history_files:
                try:
                    with open(os.path.join(self.history_dir, file), 'r', encoding='utf-8') as f:
                        chat_data = json.load(f)
                        self.chat_history.append(chat_data)
                except:
                    continue
            
            # Ordenar por fecha
            self.chat_history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            # Actualizar botones
            self.update_history_buttons()
            
        except Exception as e:
            print(f"Error loading chat history: {e}")
            self.chat_history = []

    def save_current_chat(self):
        if not self.current_chat_id:
            return
            
        # Obtener contenido actual
        content = self.messages_area.get(1.0, tk.END).strip()
        if not content:
            return
            
        # Crear objeto de chat
        chat_data = {
            'id': self.current_chat_id,
            'timestamp': datetime.now().isoformat(),
            'messages': self.get_messages_from_content(content)
        }
        
        # Guardar a archivo
        try:
            filename = f"chat_{self.current_chat_id}.json"
            with open(os.path.join(self.history_dir, filename), 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving chat: {e}")

    def get_messages_from_content(self, content):
        messages = []
        lines = content.split('\n')
        current_message = None
        
        for line in lines:
            if line.strip():
                if '[' in line and ']' in line:  # Nueva entrada
                    if current_message:
                        messages.append(current_message)
                    
                    # Parsear nueva entrada
                    parts = line.split(']', 1)
                    if len(parts) == 2:
                        timestamp = parts[0].strip('[')
                        message_part = parts[1].strip()
                        
                        if ': ' in message_part:
                            sender, text = message_part.split(': ', 1)
                            current_message = {
                                'timestamp': timestamp,
                                'sender': 'user' if sender.strip() == 'You' else 'bot',
                                'message': text.strip()
                            }
                elif current_message:  # Continuar mensaje actual
                    if 'Confidence:' in line:
                        try:
                            current_message['confidence'] = float(line.split(':')[1].strip().strip('()'))
                        except:
                            pass
                    else:
                        current_message['message'] += '\n' + line.strip()
        
        if current_message:
            messages.append(current_message)
        
        return messages

    def update_history_buttons(self):
        # Limpiar botones existentes
        for widget in self.history_frame.winfo_children():
            widget.destroy()
        
        # Crear botones para cada chat
        for chat_data in self.chat_history:
            try:
                # Crear frame para el botón
                chat_frame = ttk.Frame(self.history_frame, style="secondary.TFrame")
                chat_frame.pack(fill=tk.X, pady=2)
                
                # Obtener título del chat
                first_message = next((msg['message'] for msg in chat_data['messages'] 
                                   if msg['sender'] == 'user'), "New Chat")
                title = (first_message[:30] + '...') if len(first_message) > 30 else first_message
                
                # Crear botón
                chat_btn = ttk.Button(
                    chat_frame,
                    text=title,
                    style="secondary.TButton",
                    command=lambda c=chat_data: self.load_chat_content(c),
                    width=25
                )
                chat_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
                
                # Botón de eliminar
                delete_btn = ttk.Button(
                    chat_frame,
                    text="×",
                    style="secondary.TButton",
                    command=lambda cid=chat_data['id']: self.delete_chat(cid),
                    width=3
                )
                delete_btn.pack(side=tk.RIGHT)
                
            except Exception as e:
                print(f"Error creating history button: {e}")
                continue

    def load_chat_content(self, chat_data):
        try:
            # Actualizar ID actual
            self.current_chat_id = chat_data['id']
            
            # Limpiar área de mensajes
            self.messages_area.delete(1.0, tk.END)
            
            # Insertar mensajes
            for message in chat_data['messages']:
                timestamp = message.get('timestamp', '')
                sender = message['sender']
                msg_text = message['message']
                confidence = message.get('confidence')
                
                self.messages_area.insert(tk.END, f"[{timestamp}] ", "timestamp")
                self.messages_area.insert(tk.END, f"{'You' if sender == 'user' else 'Bot'}: ", sender)
                self.messages_area.insert(tk.END, msg_text + "\n", sender)
                
                if confidence is not None:
                    self.messages_area.insert(
                        tk.END,
                        f"(Confidence: {confidence:.2f})\n",
                        "confidence"
                    )
                
                self.messages_area.insert(tk.END, "\n")
            
            # Scroll al final
            self.messages_area.see(tk.END)
            
        except Exception as e:
            print(f"Error loading chat content: {e}")
            self.update_status("Error loading chat")

    def delete_chat(self, chat_id):
        try:
            # Confirmar eliminación
            if not messagebox.askyesno("Delete Chat", "Are you sure you want to delete this chat?"):
                return
            
            # Eliminar archivo
            filename = f"chat_{chat_id}.json"
            filepath = os.path.join(self.history_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Actualizar lista en memoria
            self.chat_history = [chat for chat in self.chat_history if chat['id'] != chat_id]
            
            # Si el chat actual fue eliminado, crear uno nuevo
            if self.current_chat_id == chat_id:
                self.new_chat()
            else:
                self.update_history_buttons()
                
        except Exception as e:
            print(f"Error deleting chat: {e}")
            self.update_status("Error deleting chat")

    def send_message(self, event=None):
        message = self.input_field.get().strip()
        if not message or not self.chatbot:
            return
        
        # Asegurar que hay un chat actual
        if not self.current_chat_id:
            self.current_chat_id = str(uuid.uuid4())
        
        # Limpiar campo de entrada
        self.input_field.delete(0, tk.END)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M")
        
        # Mostrar mensaje del usuario
        self.messages_area.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.messages_area.insert(tk.END, "You: ", "user")
        self.messages_area.insert(tk.END, message + "\n", "user")
        self.messages_area.insert(tk.END, "\n")
        self.messages_area.see(tk.END)
        
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
            self.messages_area.insert(tk.END, response + "\n", "bot")
            
            if confidence > 0:
                self.messages_area.insert(
                    tk.END,
                    f"(Confidence: {confidence:.2f})\n",
                    "confidence"
                )
            
            self.messages_area.insert(tk.END, "\n")
            self.messages_area.see(tk.END)
            
            # Guardar chat actualizado
            self.save_current_chat()
            self.load_chat_history()
            
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
                self.update_status("Ready")
            else:
                self.update_status("Training new model...")
                self.chatbot.train_model()
                self.update_status("Ready")
            
            self.root.after(0, self.enable_input)
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            print(f"Error initializing chatbot: {str(e)}")

    def enable_input(self):
        self.input_field.configure(state="normal")
        self.send_button.configure(state="normal")
        self.input_field.focus()

    def update_status(self, message):
        self.root.after(0, lambda: self.status_var.set(message))

    def setup_bindings(self):
        self.input_field.bind("<Return>", self.send_message)
        self.input_field.configure(state="disabled")
        self.send_button.configure(state="disabled")
        
        # Bind para cerrar aplicación
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        # Guardar chat actual antes de cerrar
        if self.current_chat_id:
            self.save_current_chat()
        self.root.destroy()

def main():
    root = ttk.Window(themename="darkly")
    app = ModernChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()