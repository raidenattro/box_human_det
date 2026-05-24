import { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';

async function startWhep(whepUrl, videoEl, pcRef) {
  if (pcRef.current) {
    pcRef.current.close();
    pcRef.current = null;
  }
  const pc = new RTCPeerConnection();
  pcRef.current = pc;
  pc.addTransceiver('video', { direction: 'recvonly' });
  pc.addTransceiver('audio', { direction: 'recvonly' });
  pc.ontrack = (ev) => {
    const [stream] = ev.streams;
    if (stream) videoEl.srcObject = stream;
  };
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  const resp = await fetch(whepUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/sdp' },
    body: offer.sdp,
  });
  if (!resp.ok) {
    let detail = '';
    try {
      const ct = resp.headers.get('content-type') || '';
      if (ct.includes('json')) {
        const j = await resp.json();
        detail = j.hint || j.error || '';
      } else {
        detail = (await resp.text()).slice(0, 120);
      }
    } catch {
      /* ignore */
    }
    throw new Error(
      detail || `WebRTC 信令失败 (HTTP ${resp.status})，请确认 MediaMTX 已开启 WebRTC 端口`,
    );
  }
  const answerSdp = await resp.text();
  if (!answerSdp.trim()) {
    throw new Error('WebRTC 未返回 SDP，请确认该路径在 MediaMTX 上已有画面');
  }
  await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp });
}

function stopWhep(pcRef, videoEl) {
  if (pcRef.current) {
    pcRef.current.close();
    pcRef.current = null;
  }
  if (videoEl) {
    videoEl.srcObject = null;
  }
}

/** 按格式驱动 video/img 预览源 */
export function usePreviewStream({ format, playback, mjpegSrc, videoRef, imgRef, enabled = true }) {
  const hlsRef = useRef(null);
  const pcRef = useRef(null);
  const [streamError, setStreamError] = useState('');

  useEffect(() => {
    if (!enabled) return undefined;
    const video = videoRef?.current;
    const img = imgRef?.current;
    setStreamError('');

    const cleanup = () => {
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }
      stopWhep(pcRef, video);
      if (video) {
        video.removeAttribute('src');
        video.srcObject = null;
      }
      if (img) {
        img.removeAttribute('src');
      }
    };

    if (format === 'mjpeg') {
      cleanup();
      if (img && mjpegSrc) {
        img.src = mjpegSrc;
      }
      return cleanup;
    }

    if (img) {
      img.removeAttribute('src');
    }

    if (!video) return cleanup;

    if (format === 'hls') {
      const hlsUrl = playback?.formats?.hls?.url;
      if (!hlsUrl) {
        setStreamError('HLS 不可用（需 MediaMTX 托管摄像头）');
        return cleanup;
      }
      cleanup();
      if (Hls.isSupported()) {
        const hls = new Hls({ enableWorker: true, lowLatencyMode: true });
        hlsRef.current = hls;
        hls.on(Hls.Events.ERROR, (_, data) => {
          if (data.fatal) setStreamError('HLS 播放失败');
        });
        hls.loadSource(hlsUrl);
        hls.attachMedia(video);
      } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
        video.src = hlsUrl;
      } else {
        setStreamError('当前浏览器不支持 HLS');
      }
      return cleanup;
    }

    if (format === 'webrtc') {
      const whepUrl = playback?.formats?.webrtc?.url;
      if (!whepUrl) {
        setStreamError('WebRTC 不可用（需 MediaMTX 托管摄像头）');
        return cleanup;
      }
      cleanup();
      let cancelled = false;
      (async () => {
        try {
          await startWhep(whepUrl, video, pcRef);
        } catch (err) {
          if (!cancelled) {
            const msg = err?.message || 'WebRTC 连接失败';
            if (err?.name === 'TypeError' && /fetch|network/i.test(msg)) {
              setStreamError('无法连接 WebRTC 信令，请确认 MediaMTX 已启动（8889）');
            } else {
              setStreamError(msg);
            }
          }
        }
      })();
      return () => {
        cancelled = true;
        cleanup();
      };
    }

    return cleanup;
  }, [enabled, format, mjpegSrc, playback, videoRef, imgRef]);

  return { streamError };
}
