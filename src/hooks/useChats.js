import { useState, useEffect, useRef, useCallback } from 'react';
import ModelService from '../services/modelService';

export const useChat = () => {
  const [messages, setMessages] = useState(() => {
    const savedMessages = localStorage.getItem('chat-messages');
    return savedMessages ? JSON.parse(savedMessages) : [{
      id: Date.now().toString(),
      text: "¡Hola! Soy tu asistente virtual. ¿En qué puedo ayudarte?",
      sender: 'bot',
      timestamp: Date.now()
    }];
  });

  const [inputValue, setInputValue] = useState('');
  const [isModelReady, setIsModelReady] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const messagesEndRef = useRef(null);

  // Persistir mensajes en localStorage
  useEffect(() => {
    localStorage.setItem('chat-messages', JSON.stringify(messages));
  }, [messages]);

  // Inicializar el modelo
  useEffect(() => {
    const initModel = async () => {
      try {
        await ModelService.init();
        const tokenizerOk = await ModelService.verifyTokenizer();
        const responsesOk = await ModelService.verifyResponses();
        
        console.log('Model initialization status:', {
          tokenizerStatus: tokenizerOk,
          responsesStatus: responsesOk,
          modelLoaded: ModelService.model !== null
        });

        if (!tokenizerOk || !responsesOk) {
          throw new Error('Verificación del modelo falló');
        }

        setIsModelReady(true);
      } catch (error) {
        console.error('Error detallado inicializando el modelo:', error);
        addMessage({
          text: "Lo siento, hubo un problema al cargar el sistema. Por favor, intenta recargar la página.",
          sender: 'bot'
        });
      }
    };
    initModel();
  }, []);

  // Auto scroll al último mensaje
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const addMessage = useCallback((newMessage) => {
    const messageWithMetadata = {
      id: Date.now().toString(),
      timestamp: Date.now(),
      ...newMessage
    };
    setMessages(prev => [...prev, messageWithMetadata]);
  }, []);

  const handleInputChange = useCallback((e) => {
    setInputValue(e.target.value);
  }, []);

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    
    const userMessage = inputValue.trim();
    if (!userMessage || isProcessing) return;

    setIsProcessing(true);
    setInputValue('');

    // Añadir mensaje del usuario
    addMessage({
      text: userMessage,
      sender: 'user'
    });

    try {
      console.log('Starting message processing for:', userMessage);
      const response = await ModelService.processMessage(userMessage);
      
      console.log('Final response:', response);
      
      // Añadir respuesta del bot
      addMessage({
        text: response.text || "Lo siento, no pude procesar tu mensaje.",
        sender: 'bot',
        analysis: response.contextAnalysis
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
  }, [inputValue, isProcessing, addMessage]);

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