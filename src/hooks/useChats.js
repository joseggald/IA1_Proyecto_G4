import { useState, useEffect } from 'react';
import ModelService from '../services/modelService';

const BOT_RESPONSES = {
  0: ["¡Hola! ¿En qué puedo ayudarte?", "¡Buen día! ¿Cómo estás?"],
  1: ["¡Hasta luego! Que tengas un excelente día.", "¡Nos vemos pronto!"],
  2: ["Entiendo, gracias por compartir eso.", "Interesante, cuéntame más."],
  3: ["Permíteme ayudarte con eso.", "Veamos cómo podemos resolver esto."],
};

export const useChats = () => {
  const [chats, setChats] = useState(() => {
    const savedChats = localStorage.getItem('chats');
    return savedChats ? JSON.parse(savedChats) : [
      { 
        id: 1, 
        title: 'Chat Principal',
        messages: [{ 
          id: Date.now(), // Usar timestamp como ID
          text: "¡Hola! Soy tu asistente virtual. ¿En qué puedo ayudarte?", 
          sender: 'bot',
          timestamp: new Date().toISOString()
        }],
        lastMessageId: Date.now() // Añadir tracking del último ID
      }
    ];
  });

  const [activeChat, setActiveChat] = useState(() => {
    return parseInt(localStorage.getItem('activeChat')) || 1;
  });

  const [modelReady, setModelReady] = useState(false);

  useEffect(() => {
    const initModel = async () => {
      try {
        await ModelService.init();
        setModelReady(true);
      } catch (error) {
        console.error('Error inicializando el modelo:', error);
      }
    };
    initModel();
  }, []);

  useEffect(() => {
    localStorage.setItem('chats', JSON.stringify(chats));
    localStorage.setItem('activeChat', activeChat.toString());
  }, [chats, activeChat]);

  const currentChat = chats.find(chat => chat.id === activeChat);
  const [modelError, setModelError] = useState(null);

  useEffect(() => {
    const initModel = async () => {
      try {
        await ModelService.init();
        setModelReady(true);
      } catch (error) {
        console.error('Error inicializando el modelo:', error);
        setModelError(error.message);
      }
    };
    initModel();
  }, []);
  const addChat = () => {
    const newChatId = Math.max(...chats.map(chat => chat.id)) + 1;
    const initialMessageId = Date.now();
    const newChat = {
      id: newChatId,
      title: `Chat ${newChatId}`,
      messages: [{
        id: initialMessageId,
        text: "¡Hola! ¿En qué puedo ayudarte?",
        sender: 'bot',
        timestamp: new Date().toISOString()
      }],
      lastMessageId: initialMessageId
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

  const getRandomResponse = (category) => {
    const responses = BOT_RESPONSES[category] || BOT_RESPONSES[0];
    return responses[Math.floor(Math.random() * responses.length)];
  };

  const addMessage = async (chatId, message) => {
    // Añadir mensaje del usuario con ID único basado en timestamp
    const messageId = Date.now();
    
    setChats(currentChats => 
      currentChats.map(chat => {
        if (chat.id === chatId) {
          return {
            ...chat,
            messages: [...chat.messages, {
              ...message,
              id: messageId,
              timestamp: new Date().toISOString()
            }],
            lastMessageId: messageId
          };
        }
        return chat;
      })
    );

    // Si es un mensaje del usuario, procesar con el modelo
    if (message.sender === 'user' && modelReady) {
      try {
        const result = await ModelService.processMessage(message.text);
        
        if (result.confidence > 0.6) {
          const botResponse = getRandomResponse(result.prediction);
          const botMessageId = Date.now(); // Nuevo ID único para respuesta del bot
          
          setChats(currentChats => 
            currentChats.map(chat => {
              if (chat.id === chatId) {
                return {
                  ...chat,
                  messages: [...chat.messages, {
                    id: botMessageId,
                    text: botResponse,
                    sender: 'bot',
                    timestamp: new Date().toISOString(),
                    analysis: result.contextAnalysis
                  }],
                  lastMessageId: botMessageId
                };
              }
              return chat;
            })
          );
        }
      } catch (error) {
        console.error('Error procesando mensaje:', error);
        const errorMessageId = Date.now();
        
        setChats(currentChats => 
          currentChats.map(chat => {
            if (chat.id === chatId) {
              return {
                ...chat,
                messages: [...chat.messages, {
                  id: errorMessageId,
                  text: "Lo siento, hubo un error al procesar tu mensaje.",
                  sender: 'bot',
                  timestamp: new Date().toISOString()
                }],
                lastMessageId: errorMessageId
              };
            }
            return chat;
          })
        );
      }
    }
  };

  return {
    chats,
    activeChat,
    currentChat,
    setActiveChat,
    addChat,
    deleteChat,
    addMessage,
    modelReady
  };
};