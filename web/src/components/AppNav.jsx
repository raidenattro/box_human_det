import { Link, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import './AppNav.css';

export default function AppNav() {
  const { user, authRequired, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logout();
    navigate('/login', { replace: true });
  };

  return (
    <nav className="app-nav">
      <div className="app-nav-links">
        <NavLink to="/" end>
          摄像头总览
        </NavLink>
      </div>
      {authRequired && user && (
        <div className="app-nav-user">
          <Link to="/settings" className="app-nav-settings" title="系统设置">
            ⚙
          </Link>
          <span className="app-nav-name" title={user.username}>
            {user.display_name || user.username}
          </span>
          <button type="button" className="app-nav-logout" onClick={handleLogout}>
            退出
          </button>
        </div>
      )}
    </nav>
  );
}
