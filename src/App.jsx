import { useChat } from './hooks/useChats';
import { Menu } from 'lucide-react';
import { ChatDrawer } from './components/ChatDrawer/ChatDrawer';
import { MessageList } from './components/MessageList/MessageList';
import { ChatInput } from './components/ChatInput/ChatInput';

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

  return (
    <div className="flex h-screen bg-gray-100 font-['Space_Mono']">
      <div className="flex flex-col flex-1 overflow-hidden">
        <header className="bg-gradient-to-r from-blue-600 to-blue-800 p-4 flex items-center justify-between shadow-lg">
          <div className="flex items-center">
            <h1 className="text-white text-xl font-bold ml-4">
              ChatBot Pro
            </h1>
          </div>
          {!isModelReady && (
            <div className="text-white text-sm bg-yellow-600 px-3 py-1 rounded-full">
              Cargando modelo...
            </div>
          )}
          <button
            onClick={clearChat}
            className="text-white px-4 py-2 hover:bg-blue-700 rounded-lg transition-all duration-200"
          >
            Limpiar Chat
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
          disabled={!isModelReady || isProcessing}
        />
      </div>
    </div>
  );
}

export default App;