import { useState, useRef, useEffect } from 'react';
import { Menu } from 'lucide-react';
import { LoadingScreen } from './components/LoadingScreen/LoadingScreen';
import { ChatDrawer } from './components/ChatDrawer/ChatDrawer';
import { MessageList } from './components/MessageList/MessageList';
import { ChatInput } from './components/ChatInput/ChatInput';
import { useChats } from './hooks/useChats';

function App() {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const messagesEndRef = useRef(null);
  
  const { 
    chats, 
    activeChat, 
    currentChat, 
    setActiveChat, 
    addChat, 
    deleteChat, 
    addMessage 
  } = useChats();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [currentChat?.messages]);

  useEffect(() => {
    setTimeout(() => setIsLoading(false), 2000);
  }, []);

  const handleSendMessage = (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    addMessage(activeChat, {
      text: input,
      sender: 'user'
    });

    // Simulate bot response
    setTimeout(() => {
      addMessage(activeChat, {
        text: "Esta es una respuesta simulada del bot",
        sender: 'bot'
      });
    }, 1000);

    setInput('');
  };

  if (isLoading) return <LoadingScreen />;

  return (
    <div className="flex h-screen bg-gray-100 font-['Space_Mono']">
      <ChatDrawer 
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        chats={chats}
        activeChat={activeChat}
        onChatSelect={setActiveChat}
        onNewChat={addChat}
        onDeleteChat={deleteChat}
      />

      <div className="flex flex-col flex-1 overflow-hidden">
        <header className="bg-gradient-to-r from-blue-600 to-blue-800 p-4 flex items-center shadow-lg">
          <button 
            onClick={() => setDrawerOpen(true)} 
            className="text-white p-2 hover:bg-blue-700 rounded-lg transition-all duration-200"
          >
            <Menu />
          </button>
          <h1 className="text-white text-xl font-bold ml-4">
            {currentChat?.title || 'ChatBot Pro'}
          </h1>
        </header>

        <MessageList 
          messages={currentChat?.messages || []} 
          messagesEndRef={messagesEndRef}
        />

        <ChatInput 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onSubmit={handleSendMessage}
        />
      </div>
    </div>
  );
}

export default App;