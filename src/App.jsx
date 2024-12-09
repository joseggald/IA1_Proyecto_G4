import React from 'react';
import { useChat } from './hooks/useChats';
import { Bot } from 'lucide-react';
import { MessageList } from './components/MessageList/MessageList';
import { ChatInput } from './components/ChatInput/ChatInput';

const LoadingScreen = () => (
  <div className="h-screen flex flex-col items-center justify-center bg-gradient-to-b from-gray-900 via-blue-900 to-purple-900 relative overflow-hidden font-['Space_Mono']">
    <div className="absolute inset-0 bg-[radial-gradient(circle,transparent_20%,black_100%)] opacity-20"></div>
    
    <div className="relative w-32 h-32 md:w-48 md:h-48 mb-8 md:mb-12">
      <div className="absolute inset-0 flex items-center justify-center">
        <Bot size={64} className="text-blue-400 z-10 animate-pulse md:scale-150" />
      </div>
      <div className="absolute inset-0">
        <div className="w-full h-full border-4 border-t-blue-400 border-transparent rounded-full animate-spin-slow"></div>
        <div className="absolute top-0 left-0 w-full h-full border-4 border-b-purple-400 border-transparent rounded-full animate-reverse-spin"></div>
      </div>
    </div>

    <h1 className="text-4xl md:text-8xl font-bold mb-6 md:mb-8 animate-gradient-text text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500">
      ChatBot
    </h1>
    
    <div className="flex space-x-2">
      <div className="w-2 h-2 md:w-3 md:h-3 bg-blue-500 rounded-full animate-bounce"></div>
      <div className="w-2 h-2 md:w-3 md:h-3 bg-purple-500 rounded-full animate-bounce delay-100"></div>
      <div className="w-2 h-2 md:w-3 md:h-3 bg-pink-500 rounded-full animate-bounce delay-200"></div>
    </div>

    <div className="absolute bottom-8 text-white/50 text-xs md:text-sm tracking-widest animate-pulse px-4 text-center">
      Cargando modelo...
    </div>
  </div>
);

function App() {
  const {
    messages,
    inputValue,
    isModelReady,
    isProcessing,
    messagesEndRef,
    handleInputChange,
    handleSubmit,
    clearChat
  } = useChat();

  if (!isModelReady) {
    return <LoadingScreen />;
  }

  return (
    <div className="flex h-screen bg-gray-100 font-['Space_Mono']">
      <div className="flex flex-col flex-1 overflow-hidden">
        <header className="bg-gradient-to-r from-blue-600 to-blue-800 p-3 md:p-4 flex items-center justify-between shadow-lg">
          <div className="flex items-center">
            <h1 className="text-white text-lg md:text-xl font-bold ml-2 md:ml-4">
              ChatBot Pro
            </h1>
          </div>
          <button
            onClick={clearChat}
            className="text-white text-sm md:text-base px-3 py-1.5 md:px-4 md:py-2 hover:bg-blue-700 rounded-lg transition-all duration-200"
          >
            Limpiar
          </button>
        </header>

        <MessageList
          messages={messages}
          messagesEndRef={messagesEndRef}
        />

        <ChatInput
          value={inputValue}
          onChange={handleInputChange}
          onSubmit={handleSubmit}
          disabled={isProcessing}
        />
      </div>
    </div>
  );
}

export default App;