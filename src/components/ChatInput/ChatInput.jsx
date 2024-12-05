import { Send } from 'lucide-react';
import { Tooltip } from '@mui/material';
import PropTypes from 'prop-types';

export const ChatInput = ({ value, onChange, onSubmit }) => {
  return (
    <form onSubmit={onSubmit} className="p-4 bg-white border-t shadow-lg">
      <div className="flex space-x-2">
        <input
          type="text"
          value={value}
          onChange={onChange}
          placeholder="Escribe un mensaje..."
          className="flex-1 p-4 border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200"
        />
        <Tooltip title="Enviar mensaje">
          <button
            type="submit"
            disabled={!value.trim()}
            className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-4 rounded-xl hover:shadow-lg transition-all duration-200 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
          >
            <Send size={20} />
          </button>
        </Tooltip>
        </div>
    </form>
  );
};

ChatInput.propTypes = {
  value: PropTypes.string.isRequired,
  onChange: PropTypes.func.isRequired,
  onSubmit: PropTypes.func.isRequired,
};
