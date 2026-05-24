import GridPicker from './GridPicker';
import '../pages/AnnotatePage.css';

/** 标注工具栏（无画布），用于监控页抽屉 */
export default function AnnotateControls({ tool, embedded = false }) {
  const shelfReady = tool.shelfCornersReady;
  const gridReady = tool.gridGenerated;
  const selected = tool.selectedBoxCell;
  const activeShelf = tool.shelves?.find(
    (s) => String(s.shelf_code || '').trim() === String(tool.selectedShelfCode || '').trim(),
  );
  const activeShelfLabel =
    String(activeShelf?.shelf_name || '').trim() ||
    String(tool.selectedShelfCode || '').trim();

  return (
    <div className="annotate-controls-sidebar">
      {embedded && tool.selectedShelfCode ? (
        <div className="group annotate-shelf-panel">
          <div className="annotate-shelf-head">
            <span className="group-title">当前货架</span>
            <span className="annotate-shelf-name">{activeShelfLabel}</span>
          </div>
        </div>
      ) : null}

      <div className="group">
        <GridPicker
          rows={tool.gridRows}
          cols={tool.gridCols}
          onChange={tool.setGridDimensions}
        />
        <button
          type="button"
          className="annotate-btn-block annotate-btn-primary"
          disabled={!shelfReady}
          onClick={tool.confirmGenerateGrid}
        >
          {gridReady ? `重新生成 ${tool.gridRows}×${tool.gridCols}` : '确认生成货位'}
        </button>
      </div>

      {!embedded ? (
        <div className="group">
          <div className="group-title">画面</div>
          <label className="field">
            货架名称
            <input
              type="text"
              value={tool.cameraName}
              onChange={(e) => tool.setCameraName(e.target.value)}
              placeholder="保存标注时的货架标识"
            />
          </label>
          <button type="button" className="secondary annotate-btn-block" onClick={tool.captureFrame}>
            刷新画面
          </button>
          <p className="annotate-embedded-hint">在预览上标注；静止帧请点「刷新画面」。</p>
        </div>
      ) : null}

      {gridReady ? (
        <div className="annotate-box-panel group">
          <div className="group-title">货位编号</div>
          {selected ? (
            <>
              <p className="annotate-box-pos">
                第 {selected.row} 层 · 第 {selected.col} 列
              </p>
              <label className="field annotate-box-id-field">
                编号
                <input
                  type="text"
                  value={selected.value}
                  placeholder={selected.defaultId}
                  onChange={(e) =>
                    tool.onBoxIdChange(selected.rowIdx, selected.colIdx, e.target.value)
                  }
                />
              </label>
              <button
                type="button"
                className="annotate-btn-block annotate-btn-delete"
                onClick={() => {
                  if (
                    window.confirm(`确定删除第 ${selected.row} 层 · 第 ${selected.col} 列货位？`)
                  ) {
                    tool.deleteSelectedCell();
                  }
                }}
              >
                删除货位
              </button>
            </>
          ) : (
            <p className="annotate-box-empty-hint">点击画面中的货位进行编辑</p>
          )}
        </div>
      ) : null}
    </div>
  );
}
