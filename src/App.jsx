import { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Menu, Plus, History, Trash2 } from 'lucide-react';
import { Drawer, List, ListItem, ListItemIcon, ListItemText, Fade, Slide, Tooltip } from '@mui/material';

const LoadingScreen = () => (
  <div className="h-screen flex flex-col items-center justify-center bg-gradient-to-b from-gray-900 via-blue-900 to-purple-900 relative overflow-hidden font-['Space_Mono']">
    <div className="absolute inset-0 bg-[radial-gradient(circle,transparent_20%,black_100%)] opacity-20"></div>
    
    <div className="relative w-48 h-48 mb-12">
      <div className="absolute inset-0 flex items-center justify-center">
        <Bot size={96} className="text-blue-400 z-10 animate-pulse" />
      </div>
      <div className="absolute inset-0">
        <div className="w-full h-full border-4 border-t-blue-400 border-transparent rounded-full animate-spin-slow"></div>
        <div className="absolute top-0 left-0 w-full h-full border-4 border-b-purple-400 border-transparent rounded-full animate-reverse-spin"></div>
      </div>
    </div>

    <h1 className="text-8xl font-bold mb-8 animate-gradient-text text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500">
      ChatBot
    </h1>
    
    <div className="flex space-x-2">
      <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce"></div>
      <div className="w-3 h-3 bg-purple-500 rounded-full animate-bounce delay-100"></div>
      <div className="w-3 h-3 bg-pink-500 rounded-full animate-bounce delay-200"></div>
    </div>

    <div className="absolute bottom-8 text-white/50 text-sm tracking-widest animate-pulse">
      Iniciando sistema...
    </div>
  </div>
);

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [chats, setChats] = useState([
    { id: 1, title: 'Chat Principal', messages: [] },
    { id: 2, title: 'Consulta Técnica', messages: [] }
  ]);
  const [activeChat, setActiveChat] = useState(1);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    setTimeout(() => {
      setIsLoading(false);
      setMessages([{ id: 1, text: "¡Hola! Soy tu asistente virtual. ¿En qué puedo ayudarte?", sender: 'bot' }]);
    }, 2000);
  }, []);

  const sendMessage = (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    const newMessage = { id: messages.length + 1, text: input, sender: 'user' };
    setMessages([...messages, newMessage]);
    setInput('');
    
    setMessages(prev => [...prev, { 
      id: prev.length + 1, 
      text: "Procesando respuesta...", 
      sender: 'bot',
      loading: true 
    }]);

    setTimeout(() => {
      setMessages(prev => [
        ...prev.slice(0, -1),
        { id: prev.length, text: "Esta es una respuesta simulada del bot", sender: 'bot' }
      ]);
    }, 1500);
  };

  const newChat = () => {
    const newChatId = chats.length + 1;
    setChats([...chats, { 
      id: newChatId, 
      title: `Nuevo Chat ${newChatId}`,
      messages: []
    }]);
    setActiveChat(newChatId);
    setMessages([{ id: 1, text: "¡Hola! ¿En qué puedo ayudarte?", sender: 'bot' }]);
    setDrawerOpen(false);
  };

  if (isLoading) {
    return <LoadingScreen />;
  }


  return (
    <div className="flex h-screen bg-gray-50 font-['Space_Mono']">
      <Drawer 
        anchor="left" 
        open={drawerOpen} 
        onClose={() => setDrawerOpen(false)}
        PaperProps={{
          className: "bg-gray-900 text-white"
        }}
      >
        <div className="w-72 p-4">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-blue-400">Chats</h2>
            <Tooltip title="Nuevo Chat">
              <button 
                onClick={newChat}
                className="p-2 rounded-lg bg-blue-600 hover:bg-blue-700 transition-all duration-200"
              >
                <Plus size={20} />
              </button>
            </Tooltip>
          </div>
          <List className="space-y-2">
            {chats.map(chat => (
              <ListItem 
                key={chat.id}
                button 
                className={`rounded-lg transition-all duration-200 ${
                  activeChat === chat.id ? 'bg-blue-600' : 'hover:bg-gray-800'
                }`}
                onClick={() => setActiveChat(chat.id)}
              >
                <ListItemIcon className="text-white min-w-0 mr-3">
                  <History size={20} />
                </ListItemIcon>
                <ListItemText primary={chat.title} />
                <Trash2 size={16} className="opacity-0 group-hover:opacity-100 transition-opacity" />
              </ListItem>
            ))}
          </List>
        </div>
      </Drawer>

      <div className="flex flex-col flex-1 overflow-hidden">
        <div className="bg-gradient-to-r from-blue-600 to-blue-800 p-4 flex items-center shadow-lg">
          <button 
            onClick={() => setDrawerOpen(true)} 
            className="text-white p-2 hover:bg-blue-700 rounded-lg transition-all duration-200"
          >
            <Menu />
          </button>
          <h1 className="text-white text-xl font-bold ml-4">ChatBot Pro</h1>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-hide">
          {messages.map((message, index) => (
            <Slide direction="up" in={true} timeout={300 + index * 100} key={message.id}>
              <div className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                <Fade in={true} timeout={500}>
                  <div 
                    className={`max-w-[80%] p-4 rounded-2xl ${
                      message.sender === 'user'
                        ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-br-none'
                        : 'bg-white text-gray-800 rounded-bl-none shadow-md'
                    } transform hover:scale-[1.02] transition-all duration-200`}
                  >
                    <div className="flex items-center space-x-3">
                      <div className={`p-1.5 rounded-full ${
                        message.sender === 'user' ? 'bg-blue-500' : 'bg-gray-100'
                      }`}>
                        {message.sender === 'bot' ? <Bot size={16} /> : <User size={16} />}
                      </div>
                      <p className="leading-relaxed">{message.text}</p>
                      {message.loading && (
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100"></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200"></div>
                        </div>
                      )}
                    </div>
                  </div>
                </Fade>
              </div>
            </Slide>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={sendMessage} className="p-4 bg-white border-t shadow-lg">
          <div className="flex space-x-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Escribe un mensaje..."
              className="flex-1 p-4 border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200"
            />
            <Tooltip title="Enviar mensaje">
              <button
                type="submit"
                className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-4 rounded-xl hover:shadow-lg transition-all duration-200 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <Send size={20} />
              </button>
            </Tooltip>
          </div>
        </form>
      </div>
    </div>
  );
}

export default App;