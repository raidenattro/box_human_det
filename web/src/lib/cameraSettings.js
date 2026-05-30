/** 与系统全局配置对齐、可按摄像头覆盖的项 */

export const INFERENCE_MODEL_OPTIONS = [
  { value: 'rtmpose_t', label: 'RTMPose-T + RTMDet-nano（ONNX）', shortLabel: 'RTMPose-T' },
  { value: 'rtmpose_s', label: 'RTMPose-S + RTMDet-nano（ONNX）', shortLabel: 'RTMPose-S' },
  { value: 'rtmpose_m', label: 'RTMPose-M + RTMDet-nano（ONNX）', shortLabel: 'RTMPose-M' },
  { value: 'yolo26n_pose', label: 'YOLO26n-pose（端到端）', shortLabel: 'YOLO26n' },
  { value: 'yolo26s_pose', label: 'YOLO26s-pose（端到端）', shortLabel: 'YOLO26s' },
  { value: 'yolo26m_pose', label: 'YOLO26m-pose（端到端）', shortLabel: 'YOLO26m' },
  { value: 'yolo26l_pose', label: 'YOLO26l-pose（端到端）', shortLabel: 'YOLO26l' },
];

/** @deprecated 使用 INFERENCE_MODEL_OPTIONS */
export const INFERENCE_BACKEND_OPTIONS = INFERENCE_MODEL_OPTIONS;

export const CAMERA_OVERRIDE_FIELDS = [
  {
    key: 'models.backend',
    label: '推理模型',
    type: 'select',
    options: INFERENCE_MODEL_OPTIONS,
    hint: '修改后需重新启动该路智能检测。RTMPose 需 lite / lite-gpu-onnx 镜像；YOLO 需含 ultralytics 的 GPU 镜像。',
  },
  { key: 'inference.frame_rate', label: '推理帧率 (fps)', type: 'number', min: 1, max: 60 },
  { key: 'inference.height', label: '推理高度 (px)', type: 'number', min: 120, max: 2160 },
  { key: 'inference.pose_frame_interval', label: '姿态检测间隔 (帧)', type: 'number', min: 1, max: 120 },
  {
    key: 'debug-info.enabled',
    label: '推理调试日志',
    type: 'boolean',
    hint: '开启后推理容器周期性输出 [DEBUG-INFO]（帧率、资源等）。不影响监控页画面与骨架叠加，生产环境建议关闭。',
  },
];

/** 旧配置 / 族 id → 当前 preset id（与 model_registry._ALIASES 对齐） */
const BACKEND_ALIASES = {
  lite: 'rtmpose_t',
  mp: 'rtmpose_t',
  mediapipe: 'rtmpose_t',
  mmpose: 'rtmpose_t',
  mm: 'rtmpose_t',
  default: 'rtmpose_t',
  rtmpose_onnx: 'rtmpose_t',
  'rtmpose-t': 'rtmpose_t',
  yolo_pose: 'yolo26s_pose',
};

export function normalizeBackendId(value) {
  const v = String(value || '').trim().toLowerCase();
  return BACKEND_ALIASES[v] || v;
}

export function formatSettingDisplayValue(field, value) {
  if (value === undefined || value === null || value === '') return '—';
  if (field.type === 'boolean') return value ? '开' : '关';
  if (field.type === 'select' && field.options) {
    const normalized = normalizeBackendId(value);
    const opt = field.options.find((o) => o.value === normalized);
    return opt?.shortLabel || opt?.label || String(value);
  }
  return String(value);
}

export function backendLabel(value) {
  return formatSettingDisplayValue(
    { type: 'select', options: INFERENCE_MODEL_OPTIONS },
    value,
  );
}

/** 监控页展示：优先用推理容器实际 backend，其次摄像头 effective_settings */
export function resolveCameraModelLabel(camera) {
  if (!camera) return '—';
  const backend =
    camera.inference?.backend || camera.effective_settings?.['models.backend'];
  return backendLabel(backend);
}
