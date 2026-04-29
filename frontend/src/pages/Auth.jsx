import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { MessageSquare, Mail, Lock, User } from 'lucide-react';

export default function Auth() {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({ name: '', email: '', password: '' });
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    setError('');

    if (!formData.email || !formData.password || (!isLogin && !formData.name)) {
      setError('Please fill in all required fields.');
      return;
    }

    if (isLogin) {
      const storedUsers = JSON.parse(localStorage.getItem('signlive_users') || '[]');
      const user = storedUsers.find(u => u.email === formData.email && u.password === formData.password);
      if (user) {
        localStorage.setItem('signlive_active_user', JSON.stringify(user));
        navigate('/dashboard');
      } else {
        setError('Invalid email or password.');
      }
    } else {
      const storedUsers = JSON.parse(localStorage.getItem('signlive_users') || '[]');
      if (storedUsers.some(u => u.email === formData.email)) {
        setError('An account with this email already exists.');
        return;
      }
      const newUser = { id: Date.now(), name: formData.name, email: formData.email, password: formData.password };
      storedUsers.push(newUser);
      localStorage.setItem('signlive_users', JSON.stringify(storedUsers));
      localStorage.setItem('signlive_active_user', JSON.stringify(newUser));
      navigate('/dashboard');
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-logo">
            <MessageSquare className="logo-icon" size={32} />
            <h2>SignLive</h2>
          </div>
          <p>{isLogin ? 'Welcome back! Please enter your details.' : 'Create an account to get started.'}</p>
        </div>

        {error && <div className="auth-error">{error}</div>}

        <form onSubmit={handleSubmit} className="auth-form">
          {!isLogin && (
            <div className="input-group">
              <User className="input-icon" size={18} />
              <input 
                type="text" 
                placeholder="Full Name" 
                value={formData.name}
                onChange={(e) => setFormData({...formData, name: e.target.value})}
              />
            </div>
          )}
          
          <div className="input-group">
            <Mail className="input-icon" size={18} />
            <input 
              type="email" 
              placeholder="Email Address" 
              value={formData.email}
              onChange={(e) => setFormData({...formData, email: e.target.value})}
            />
          </div>
          
          <div className="input-group">
            <Lock className="input-icon" size={18} />
            <input 
              type="password" 
              placeholder="Password" 
              value={formData.password}
              onChange={(e) => setFormData({...formData, password: e.target.value})}
            />
          </div>

          <button type="submit" className="auth-button">
            {isLogin ? 'Sign In' : 'Sign Up'}
          </button>
        </form>

        <div className="auth-footer">
          <p>
            {isLogin ? "Don't have an account? " : "Already have an account? "}
            <span onClick={() => { setIsLogin(!isLogin); setError(''); }} className="auth-toggle">
              {isLogin ? 'Sign up' : 'Log in'}
            </span>
          </p>
        </div>
      </div>
    </div>
  );
}
