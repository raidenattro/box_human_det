/** 摄像头流类型与表单字段（与后端 source_type 对齐） */

export const CAMERA_SOURCE_TYPES = [
  {
    value: 'external',
    label: '外部 RTSP',
    hint: '填写完整 RTSP/RTSPS 地址，不经 MediaMTX 托管（仍可播放若指向本机 MTX）。',
  },
  {
    value: 'publisher',
    label: '推流到 MediaMTX',
    hint: 'ffmpeg/设备向本机 MediaMTX 推流；下方为播放地址（可自动生成）。',
  },
  {
    value: 'rtsp_pull',
    label: 'MediaMTX 拉流',
    hint: 'MediaMTX 从上游地址拉流；请区分「上游拉流地址」与「本机播放地址」。',
  },
  {
    value: 'v4l2',
    label: '本地摄像头',
    hint: 'USB 摄像头等设备，由 MediaMTX runOnInit 推流。',
  },
];

export const DEFAULT_RTSP_HOST = '127.0.0.1';
export const DEFAULT_RTSP_PORT = '8554';

export function defaultPlaybackUrl(path, host = DEFAULT_RTSP_HOST, port = DEFAULT_RTSP_PORT) {
  const slug = String(path || '').trim().replace(/^\/+/, '');
  if (!slug) return '';
  return `rtsp://${host}:${port}/${slug}`;
}

export function emptyCameraForm() {
  return {
    path: '',
    name: '',
    source_type: 'external',
    url: '',
    pull_url: '',
    device: '/dev/video0',
    enabled: true,
    settings: {},
  };
}

/** 从 API 摄像头记录填充抽屉表单（勿把 pull_url 填入 url） */
export function cameraToForm(cam) {
  if (!cam) return emptyCameraForm();
  const source_type = cam.source_type || 'external';
  return {
    path: cam.path || cam.id || '',
    name: cam.name || '',
    source_type,
    url: cam.url || '',
    pull_url: cam.pull_url || '',
    device: cam.device || '/dev/video0',
    enabled: cam.enabled !== false,
    settings: { ...(cam.settings || {}) },
  };
}

/** 保存前组装 payload */
export function formToCameraPayload(form) {
  const path = String(form.path || '').trim();
  const name = String(form.name || '').trim();
  const source_type = form.source_type || 'external';
  const payload = {
    path,
    name,
    source_type,
    enabled: form.enabled !== false,
    settings: form.settings || {},
  };

  if (source_type === 'rtsp_pull') {
    payload.pull_url = String(form.pull_url || '').trim();
    payload.url = String(form.url || '').trim() || defaultPlaybackUrl(path);
  } else if (source_type === 'v4l2') {
    payload.device = String(form.device || '/dev/video0').trim();
    payload.url = String(form.url || '').trim() || defaultPlaybackUrl(path);
    payload.pull_url = '';
  } else if (source_type === 'publisher') {
    payload.url = String(form.url || '').trim() || defaultPlaybackUrl(path);
    payload.pull_url = '';
  } else {
    payload.url = String(form.url || '').trim();
    payload.pull_url = '';
  }
  return payload;
}

export function sourceTypeLabel(value) {
  return CAMERA_SOURCE_TYPES.find((t) => t.value === value)?.label || value || '—';
}
