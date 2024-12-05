// src/hooks/useChats.js
import { useState, useEffect } from 'react';

export const useChats = () => {
  const [chats, setChats] = useState(() => {
    const savedChats = localStorage.getItem('chats');
    return savedChats ? JSON.parse(savedChats) : [
      { 
        id: 1, 
        title: 'Chat Principal',
        messages: [{ 
          id: 1, 
          text: "¡Hola! Soy tu asistente virtual. ¿En qué puedo ayudarte?", 
          sender: 'bot',
          timestamp: new Date().toISOString()
        }]
      }
    ];
  });

  const [activeChat, setActiveChat] = useState(() => {
    return parseInt(localStorage.getItem('activeChat')) || 1;
  });

  useEffect(() => {
    localStorage.setItem('chats', JSON.stringify(chats));
    localStorage.setItem('activeChat', activeChat.toString());
  }, [chats, activeChat]);

  const currentChat = chats.find(chat => chat.id === activeChat);

  const addChat = () => {
    const newChatId = Math.max(...chats.map(chat => chat.id)) + 1;
    const newChat = {
      id: newChatId,
      title: `Chat ${newChatId}`,
      messages: [{
        id: 1,
        text: "¡Hola! ¿En qué puedo ayudarte?",
        sender: 'bot',
        timestamp: new Date().toISOString()
      }]
    };
    setChats([...chats, newChat]);
    setActiveChat(newChatId);
    return newChatId;
  };

  const deleteChat = (chatId) => {
    if (chats.length <= 1) return false;
    const updatedChats = chats.filter(chat => chat.id !== chatId);
    setChats(updatedChats);
    if (activeChat === chatId) {
      setActiveChat(updatedChats[0].id);
    }
    return true;
  };

  const addMessage = (chatId, message) => {
    setChats(currentChats => 
      currentChats.map(chat => {
        if (chat.id === chatId) {
          return {
            ...chat,
            messages: [...chat.messages, {
              ...message,
              id: chat.messages.length + 1,
              timestamp: new Date().toISOString()
            }]
          };
        }
        return chat;
      })
    );
  };

  return {
    chats,
    activeChat,
    currentChat,
    setActiveChat,
    addChat,
    deleteChat,
    addMessage
  };
};