import { APP_VERSION } from '../build-info.js';
import './AppFooter.css';

export default function AppFooter() {
  return (
    <footer className="app-footer">
      <span className="app-footer-copy">
        © 2026-2027 米道（基于 MAPS<sup className="tm-mark">TM</sup> 开发）
      </span>
      <span className="app-footer-version">{APP_VERSION}</span>
    </footer>
  );
}
