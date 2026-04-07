# 最终落地方案（采集训练数据 + 训练动作分类模型）

## 0. 目标与边界
- 目标：在不破坏现有前端报警流程的前提下，新增稳定的旁路特征采集，产出可用于动作分类训练的数据集。
- 现状：当前 `main.py` 仍是原始碰撞逻辑，尚未接入采集器与轨迹稳定化。
- 成功标准：可持续生成高质量 CSV，并能按视频切分训练/验证/测试，训练 LSTM/时序模型区分 `grasp` 与 `touch_pass`。

## 1. 必须修复的 5 个点（按优先级）
1. 在 `main.py` 中接入采集器（当前没有）。
2. 手轨迹关联使用绝对像素坐标，不再用归一化坐标匹配。
3. `video_id` 不能基于固定路径字符串，应基于文件属性（size + mtime）或上传唯一 ID。
4. CSV 按 session 独立文件输出，避免并发污染。
5. 速度/加速度按 `dt` 计算，不能简单帧差。

## 2. 代码改造清单（最小可用版本）

### 2.1 新增导入
- `math`
- `csv`
- `uuid`
- `hashlib`
- `deque`
- `dataclass`

### 2.2 新增类
- `TrackState(abs_x, abs_y, ts_sec)`
- `WristTrackAssigner(max_match_dist=150.0, stale_sec=1.0)`
  - 同一 `hand_side` 内做近邻匹配
  - 匹配距离用像素坐标 `wrist_x, wrist_y`
  - 定时清理 stale track
- `ActionFeatureExtractorV2`
  - 输出字段：
    - `session_id, video_id, ts_sec, frame_idx`
    - `track_id, person_id_raw, person_track_id, hand_side`
    - `norm_x, norm_y, score`
    - `v_x, v_y, a_x, a_y`
    - `is_in_box, dist_to_box_center, box_id`
  - `auto_flush_every` 周期刷盘
  - `close()` 在 finally 中调用

### 2.3 WebSocket 中新增会话信息与采集器
- 生成 `session_id = uuid4()[:8]`
- 生成 `video_id`：
  - 若文件存在：`md5(f"{st_size}_{st_mtime}")[:10]`
  - 否则：`unknown`
- 采集文件名：`localdata/action_dataset_{session_id}.csv`
- 实例化：
  - `feature_extractor = ActionFeatureExtractorV2(csv_filename=csv_name)`
  - `hand_assigner = WristTrackAssigner(...)`

### 2.4 在关键点循环中调用采集器
- 每个 `p_idx` 每帧调用一次 `extract_and_save(...)`
- `ts_sec` 使用 `frame_count / video_fps`
- `person_id_raw` 保留 `p_idx` 仅用于排查，不作为长期稳定 ID

### 2.5 资源释放
- `finally` 里：
  - `feature_extractor.close()`
  - `cap.release()`

### 2.6 诊断日志
- JSON 标注文件不存在时输出明确日志，提示先前端标注。

## 3. 数据采集执行 SOP
1. 启动服务，上传视频，完成标注，启动推理。
2. 每次推理自动生成一个 `localdata/action_dataset_<session_id>.csv`。
3. 采集至少 20 段视频，覆盖：
   - 正常抓取
   - 手掠过
   - 停顿后离开
   - 多人同框与遮挡
4. 采集后合并 CSV（按列对齐）。

## 4. 标注与切窗规则

### 4.1 标签集合（建议）
- `idle`
- `approach`
- `touch_pass`
- `grasp_pick`
- `place_back`

### 4.2 切窗
- 窗口长度：32 帧（可试 48）
- 步长：4 帧
- 样本主键：`video_id + track_id + hand_side + start_frame`

### 4.3 半自动标注工具（必须补齐）
- 不做手工逐行 Excel 标注，使用半自动脚本。
- 脚本输入：`session CSV + 原视频 + 标注框 JSON`。
- 触发候选：当 `is_in_box == 1` 或 `dist_to_box_center < 1.5` 连续出现时，生成事件片段。
- 每个候选片段自动扩展上下文（建议前后各 1 秒），在窗口中循环播放。
- 标注员按键：
  - `1`: `grasp_pick`
  - `2`: `touch_pass`
  - `3`: `place_back`
  - `0`: `idle/other`
- 脚本输出：
  - `segments_labeled.csv`（片段级标签）
  - `windows_labeled.csv`（已切窗、可直接训练）

### 4.4 触发式切窗（替代全局滑窗）
- 禁止全视频无差别滑窗（会产生海量 `idle` 样本）。
- 采用 Trigger-based 采样：仅在“手靠近货架”时生成训练窗。
- 推荐触发条件：`dist_to_box_center < 1.5` 或 `is_in_box == 1`。
- 正负样本策略：
  - 正样本：围绕 `grasp_pick/place_back` 事件中心采样。
  - 难负样本：优先采 `touch_pass`、快速掠过、短暂停留。
  - 纯 `idle` 采样降采样到与正样本接近（例如 1:1 到 3:1）。

### 4.5 置信度清洗规则（训练前必做）
- 对每个 32 帧窗口统计低置信帧数：`score < 0.15`。
- 若低置信帧数 > 10：
  - 默认丢弃该窗口；或
  - 插值修复后仅用于预训练，不用于最终评估。
- 对缺失帧可做线性插值或上一帧保持，但要打 `is_imputed` 标记。

### 4.6 切分数据集
- 严格按 `video_id` 切分，不按行随机切分
- 推荐：70% 训练 / 15% 验证 / 15% 测试
- 对每个类别检查切分后样本量，避免某类在验证集缺失

## 5. 训练基线（LSTM）
- 输入：`[T, D]`（每窗特征序列）
- 模型：2 层 BiLSTM（hidden=128）+ FC
- 优化：AdamW，lr=1e-3，dropout=0.3
- 类别不平衡：class_weight 或 focal loss
- 类别采样：WeightedRandomSampler 或分层采样
- 指标：
  - `grasp_pick` 召回率
  - `touch_pass` 精确率
  - 事件级延迟

## 6. 验证“轻量时序优化导致漏判”的实验
1. 配置 A：`det_interval=1, pose_interval=1`
2. 配置 B：`det_interval=2, pose_interval=1`
3. 配置 C：`det_interval=3, pose_interval=2`（当前）
4. 同一视频集逐项对比：
   - 召回
   - 误报
   - 推理 FPS
5. 若 C 漏判明显高于 A/B，则确认跳帧是主因。

## 7. 风险与后续增强
- 绝对坐标近邻仍可能在极端交叉时串轨迹。
- 后续增强顺序：
  1. 引入 person tracker（ByteTrack/OCSORT）
  2. 在 person 内再做 hand track
  3. 升级 wholebody/hand keypoints
  4. 规则触发 + 动作模型二次确认联合决策

## 8. 今日可执行 ToDo
- [ ] 把采集器与轨迹器代码真正落到 `main.py`
- [ ] 跑 3 段视频生成 3 份 session CSV
- [ ] 检查 CSV 的 `video_id/track_id/v_x/a_x` 连续性
- [ ] 实现半自动标注脚本（按键打标 + 自动生成片段）
- [ ] 完成首批标签定义与标注规范文档
- [ ] 增加触发式切窗与低置信窗口清洗逻辑
- [ ] 准备第一版 LSTM 训练脚本


session_id
本次推理会话 ID（每次 websocket 会话生成一次），用于区分不同运行批次。生成与传入见 main.py:470。

video_id
视频标识（由视频文件属性计算出的短哈希），用于区分不同视频来源。传入见 main.py:471。

ts_sec
当前样本在视频时间轴上的秒数，计算方式是 frame_count / video_fps。见 main.py:472 和写入 main.py:207。

frame_idx
帧序号（从 1 开始递增的处理帧计数）。见 main.py:473。

track_id
手部轨迹 ID，由最近邻匹配得到的“相对稳定”ID。赋值见 main.py:148。

person_id_raw
当前帧中人体检测/姿态结果的原始索引 p_idx，不是全局稳定 person ID。传入见 main.py:474。

person_track_id
跨帧稳定的人体轨迹 ID，由人物跟踪器按空间近邻持续分配，用于减少 `person_id_raw` 重排导致的串号。可把同一人的多帧样本关联到同一条人物时序。

hand_side
手的左右标记，取值为 left 或 right。见 main.py:138。

norm_x
手腕相对颈部的归一化横向坐标。计算为 (wrist_x - neck_x) / shoulder_width。见 main.py:134 和 main.py:136。

norm_y
手腕相对颈部的归一化纵向坐标。计算同上，用 shoulder_width 做尺度归一化。见 main.py:136。

score
该手腕关键点的置信度（来自姿态模型 keypoint score）。见 main.py:139。

v_x
归一化坐标在 x 方向的速度，近两帧差分除以时间间隔 dt。见 main.py:158。

v_y
归一化坐标在 y 方向的速度，计算方式与 v_x 相同。见 main.py:158。

a_x
归一化坐标在 x 方向的加速度，用三帧速度差分估计。见 main.py:165。

a_y
归一化坐标在 y 方向的加速度，计算方式与 a_x 相同。见 main.py:165。

is_in_box
是否进入任一标注框，多边形内判定结果。1 表示在框内，0 表示不在。见 main.py:194 和写入 main.py:219。

dist_to_box_center
手腕到目标框中心点的归一化距离（同样除以 shoulder_width）。见 main.py:192。

box_id
关联的框 ID。若在框内则是命中的框；否则是最近框；默认初值为 -1。见 main.py:180、main.py:196、main.py:202。