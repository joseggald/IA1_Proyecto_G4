// src/components/MessageList/MessageList.jsx
import { Slide, Fade } from '@mui/material';
import { Bot, User } from 'lucide-react';

import PropTypes from 'prop-types';

export const MessageList = ({ messages, messagesEndRef }) => {
  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
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
                </div>
                <div className="mt-2 text-xs opacity-50">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </div>
              </div>
            </Fade>
          </div>
        </Slide>
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
};

MessageList.propTypes = {
  messages: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      sender: PropTypes.string.isRequired,
      text: PropTypes.string.isRequired,
      timestamp: PropTypes.number.isRequired,
    })
  ).isRequired,
  messagesEndRef: PropTypes.shape({ current: PropTypes.instanceOf(Element) })
};