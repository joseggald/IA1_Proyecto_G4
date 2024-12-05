// src/components/ChatDrawer/ChatDrawer.jsx
import { List, ListItem, ListItemIcon, ListItemText, Drawer, Tooltip } from '@mui/material';
import { Plus, History, Trash2 } from 'lucide-react';
import PropTypes from 'prop-types';

export const ChatDrawer = ({ 
  open, 
  onClose, 
  chats, 
  activeChat, 
  onChatSelect, 
  onNewChat, 
  onDeleteChat 
}) => {
  return (
    <Drawer 
      anchor="left" 
      open={open} 
      onClose={onClose}
      PaperProps={{
        className: "bg-gray-900 text-white w-80"
      }}
    >
      <div className="p-4">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-blue-400">Mis Chats</h2>
          <Tooltip title="Nuevo Chat">
            <button 
              onClick={onNewChat}
              className="p-2 rounded-lg bg-blue-600 hover:bg-blue-700 transition-all duration-200"
            >
              <Plus size={20} />
            </button>
          </Tooltip>
        </div>
        <List className="space-y-2">
          {chats.map(chat => (
            <ListItem 
              key={chat.id}
              className={`group rounded-lg transition-all duration-200 ${
                activeChat === chat.id ? 'bg-blue-600' : 'hover:bg-gray-800'
              }`}
            >
              <button 
                className="flex items-center w-full p-2"
                onClick={() => {
                  onChatSelect(chat.id);
                  onClose();
                }}
              >
                <ListItemIcon className="text-white min-w-0 mr-3">
                  <History size={20} />
                </ListItemIcon>
                <ListItemText 
                  primary={chat.title} 
                  secondary={`${chat.messages.length} mensajes`}
                  className="text-white"
                />
                {chats.length > 1 && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteChat(chat.id);
                    }}
                    className="p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <Trash2 size={16} className="text-red-400 hover:text-red-300" />
                  </button>
                )}
              </button>
            </ListItem>
          ))}
        </List>
      </div>
    </Drawer>
  );
};

ChatDrawer.propTypes = {
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  chats: PropTypes.arrayOf(PropTypes.shape({
    id: PropTypes.string.isRequired,
    title: PropTypes.string.isRequired,
    messages: PropTypes.arrayOf(PropTypes.object).isRequired,
  })).isRequired,
  activeChat: PropTypes.string,
  onChatSelect: PropTypes.func.isRequired,
  onNewChat: PropTypes.func.isRequired,
  onDeleteChat: PropTypes.func.isRequired,
};
