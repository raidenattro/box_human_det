import { Navigate, useSearchParams } from 'react-router-dom';

/** 旧路径重定向到检测监控页的标注模式 */
export default function AnnotatePage() {
  const [searchParams] = useSearchParams();
  const camera = searchParams.get('camera')?.trim();
  if (!camera) {
    return <Navigate to="/" replace />;
  }
  const qs = new URLSearchParams({ mode: 'annotate', camera });
  return <Navigate to={`/monitor?${qs.toString()}`} replace />;
}
