// src/components/LoadingScreen/LoadingScreen.jsx
import { Bot } from 'lucide-react';

export const LoadingScreen = () => (
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