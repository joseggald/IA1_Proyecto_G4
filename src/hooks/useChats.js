import { useState, useEffect, useRef, useCallback } from 'react';
import { ChatbotModel } from '../models/';

// Datos de entrenamiento y respuestas (importados directamente)
import trainingData from '../data.json';
import responseData from '../data/response.json';

export const useChat = () => {
  // Estado inicial de los mensajes con persistencia en localStorage
  const [messages, setMessages] = useState(() => {
    const savedMessages = localStorage.getItem('chat-messages');
    return savedMessages ? JSON.parse(savedMessages) : [{
      id: Date.now().toString(),
      text: "¡Hola! Soy tu asistente virtual. ¿En qué puedo ayudarte?",
      sender: 'bot',
      timestamp: Date.now()
    }];
  });

  // Estados para el manejo del chat
  const [inputValue, setInputValue] = useState('');
  const [isModelReady, setIsModelReady] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [chatbot, setChatbot] = useState(null);
  const messagesEndRef = useRef(null);

  // Persistir mensajes en localStorage
  useEffect(() => {
    localStorage.setItem('chat-messages', JSON.stringify(messages));
  }, [messages]);

  // Inicializar el modelo del chatbot
  useEffect(() => {
    const initChatbot = async () => {
      try {
        // Inicializar chatbot con los datos importados
        const bot = new ChatbotModel();
        await bot.initialize(trainingData);
        
        setChatbot(bot);
        setIsModelReady(true);
        console.log('Chatbot inicializado correctamente.');
      } catch (error) {
        console.error('Error detallado inicializando el chatbot:', error);
        addMessage({
          text: "Lo siento, hubo un problema al cargar el sistema. Por favor, intenta recargar la página.",
          sender: 'bot'
        });
      }
    };

    initChatbot();
  }, []);

  // Auto scroll al último mensaje
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Función para obtener respuesta basada en el label
  const getResponse = (label) => {
    try {
      const labelResponses = responseData[label] || [];
      return labelResponses[Math.floor(Math.random() * labelResponses.length)] ||
        'Lo siento, no entendí tu mensaje.';
    } catch (error) {
      console.error('Error obteniendo respuesta:', error);
      return 'Lo siento, ocurrió un error al procesar tu mensaje.';
    }
  };

  // Función para añadir mensajes al chat
  const addMessage = useCallback((newMessage) => {
    const messageWithMetadata = {
      id: Date.now().toString(),
      timestamp: Date.now(),
      ...newMessage
    };
    setMessages(prev => [...prev, messageWithMetadata]);
  }, []);

  // Manejador del cambio en el input
  const handleInputChange = useCallback((e) => {
    setInputValue(e.target.value);
  }, []);

  // Manejador del envío de mensajes
  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();

    const userMessage = inputValue.trim();
    if (!userMessage || isProcessing || !chatbot) return;

    setIsProcessing(true);
    setInputValue('');

    // Añadir mensaje del usuario
    addMessage({
      text: userMessage,
      sender: 'user'
    });

    try {
      // Obtener predicción del chatbot
      const label = await chatbot.predict(userMessage);
      
      // Obtener respuesta basada en el label
      const responseText = getResponse(label);

      // Añadir respuesta del bot
      addMessage({
        text: responseText,
        sender: 'bot',
        label: label
      });
    } catch (error) {
      console.error('Error completo en handleSubmit:', error);
      addMessage({
        text: "Lo siento, ocurrió un error al procesar tu mensaje. Por favor, intenta de nuevo.",
        sender: 'bot',
        error: error.message
      });
    } finally {
      setIsProcessing(false);
    }
  }, [inputValue, isProcessing, chatbot, addMessage]);

  // Función para limpiar el chat
  const clearChat = useCallback(() => {
    setMessages([{
      id: Date.now().toString(),
      text: "¡Hola! Soy tu asistente virtual. ¿En qué puedo ayudarte?",
      sender: 'bot',
      timestamp: Date.now()
    }]);
  }, []);

  return {
    messages,
    inputValue,
    isModelReady,
    isProcessing,
    messagesEndRef,
    handleInputChange,
    handleSubmit,
    clearChat
  };
};