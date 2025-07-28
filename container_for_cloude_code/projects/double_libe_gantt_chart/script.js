class GanttChart {
    constructor() {
        this.tasks = [];
        this.selectedTask = null;
        this.dayWidth = 60;
        this.startDate = new Date();
        this.endDate = new Date();
        this.viewStartDate = null;
        this.viewEndDate = null;
        this.isDragging = false;
        this.dragData = null;
        this.resizing = false;
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.loadSampleData();
    }
    
    bindEvents() {
        document.getElementById('addTask').addEventListener('click', () => this.showAddTaskModal());
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('saveData').addEventListener('click', () => this.saveData());
        
        // 表示範囲制御
        document.getElementById('applyRange').addEventListener('click', () => this.applyDateRange());
        document.getElementById('resetRange').addEventListener('click', () => this.resetDateRange());
        document.getElementById('zoomLevel').addEventListener('change', (e) => this.changeZoom(e.target.value));
        
        document.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        document.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        
        // モーダル関連のイベント
        document.querySelector('.close').addEventListener('click', () => this.hideAddTaskModal());
        document.getElementById('cancelBtn').addEventListener('click', () => this.hideAddTaskModal());
        document.getElementById('taskForm').addEventListener('submit', (e) => this.handleAddTask(e));
        
        // モーダルの外側をクリックした時に閉じる
        document.getElementById('taskModal').addEventListener('click', (e) => {
            if (e.target.id === 'taskModal') {
                this.hideAddTaskModal();
            }
        });
    }
    
    loadSampleData() {
        this.tasks = [
            {
                id: 1,
                name: "プロジェクト設計",
                planned: {
                    start: "2024-01-01",
                    end: "2024-01-05"
                },
                actual: {
                    start: "2024-01-01",
                    end: "2024-01-04"
                }
            },
            {
                id: 2,
                name: "開発フェーズ1",
                planned: {
                    start: "2024-01-06",
                    end: "2024-01-15"
                },
                actual: {
                    start: "2024-01-05",
                    end: "2024-01-16"
                }
            },
            {
                id: 3,
                name: "テスト実施",
                planned: {
                    start: "2024-01-16",
                    end: "2024-01-20"
                },
                actual: {
                    start: "2024-01-17",
                    end: "2024-01-21"
                }
            }
        ];
        
        this.calculateDateRange();
        this.initializeDateRangeControls();
        this.render();
    }
    
    calculateDateRange() {
        let minDate = new Date();
        let maxDate = new Date();
        
        this.tasks.forEach(task => {
            const plannedStart = new Date(task.planned.start);
            const plannedEnd = new Date(task.planned.end);
            const actualStart = new Date(task.actual.start);
            const actualEnd = new Date(task.actual.end);
            
            if (plannedStart < minDate) minDate = plannedStart;
            if (actualStart < minDate) minDate = actualStart;
            if (plannedEnd > maxDate) maxDate = plannedEnd;
            if (actualEnd > maxDate) maxDate = actualEnd;
        });
        
        this.startDate = new Date(minDate.getTime() - 2 * 24 * 60 * 60 * 1000);
        this.endDate = new Date(maxDate.getTime() + 2 * 24 * 60 * 60 * 1000);
    }
    
    render() {
        this.renderTaskList();
        this.renderTimeline();
        this.renderChart();
    }
    
    renderTaskList() {
        const taskList = document.getElementById('taskList');
        taskList.innerHTML = '';
        
        this.tasks.forEach((task, index) => {
            const taskElement = document.createElement('div');
            taskElement.className = 'task-item';
            taskElement.dataset.taskId = task.id;
            
            taskElement.innerHTML = `
                <div class="task-name">${task.name}</div>
                <div class="task-info">
                    予定: ${task.planned.start} ～ ${task.planned.end}<br>
                    実績: ${task.actual.start} ～ ${task.actual.end}
                </div>
            `;
            
            taskElement.addEventListener('click', () => this.selectTask(task.id));
            taskList.appendChild(taskElement);
        });
    }
    
    renderTimeline() {
        const timelineHeader = document.getElementById('timelineHeader');
        timelineHeader.innerHTML = '';
        
        const displayStart = this.viewStartDate || this.startDate;
        const displayEnd = this.viewEndDate || this.endDate;
        const totalDays = Math.ceil((displayEnd - displayStart) / (1000 * 60 * 60 * 24));
        
        for (let i = 0; i < totalDays; i++) {
            const date = new Date(displayStart.getTime() + i * 24 * 60 * 60 * 1000);
            const dateElement = document.createElement('div');
            dateElement.className = 'date-cell';
            dateElement.style.width = this.dayWidth + 'px';
            dateElement.textContent = this.formatDateDisplay(date);
            timelineHeader.appendChild(dateElement);
        }
    }
    
    renderChart() {
        const chartCanvas = document.getElementById('chartCanvas');
        chartCanvas.innerHTML = '';
        
        const displayStart = this.viewStartDate || this.startDate;
        const displayEnd = this.viewEndDate || this.endDate;
        const totalDays = Math.ceil((displayEnd - displayStart) / (1000 * 60 * 60 * 24));
        const chartWidth = totalDays * this.dayWidth;
        chartCanvas.style.width = chartWidth + 'px';
        
        this.renderGridLines(chartCanvas, totalDays);
        
        this.tasks.forEach((task, index) => {
            const rowElement = document.createElement('div');
            rowElement.className = 'gantt-row';
            rowElement.style.height = '50px';
            
            const plannedBar = this.createGanttBar(task, 'planned', index, displayStart);
            const actualBar = this.createGanttBar(task, 'actual', index, displayStart);
            
            if (plannedBar) rowElement.appendChild(plannedBar);
            if (actualBar) rowElement.appendChild(actualBar);
            
            chartCanvas.appendChild(rowElement);
        });
    }
    
    renderGridLines(container, totalDays) {
        for (let i = 0; i <= totalDays; i++) {
            const gridLine = document.createElement('div');
            gridLine.className = 'grid-line';
            gridLine.style.left = (i * this.dayWidth) + 'px';
            container.appendChild(gridLine);
        }
    }
    
    createGanttBar(task, type, rowIndex, displayStart) {
        displayStart = displayStart || this.startDate;
        const bar = document.createElement('div');
        bar.className = `gantt-bar ${type}-bar`;
        bar.dataset.taskId = task.id;
        bar.dataset.type = type;
        
        const startDate = new Date(task[type].start);
        const endDate = new Date(task[type].end);
        const displayEnd = this.viewEndDate || this.endDate;
        
        // 表示範囲外のタスクは表示しない
        if (endDate < displayStart || startDate > displayEnd) {
            return null;
        }
        
        const startOffset = Math.floor((startDate - displayStart) / (1000 * 60 * 60 * 24));
        const duration = Math.ceil((endDate - startDate) / (1000 * 60 * 60 * 24)) + 1;
        
        // 表示範囲内にクリップ
        const clippedStartOffset = Math.max(0, startOffset);
        const clippedEndOffset = Math.min(
            startOffset + duration, 
            Math.ceil((displayEnd - displayStart) / (1000 * 60 * 60 * 24))
        );
        const clippedDuration = clippedEndOffset - clippedStartOffset;
        
        if (clippedDuration <= 0) return null;
        
        bar.style.left = (clippedStartOffset * this.dayWidth) + 'px';
        bar.style.width = (clippedDuration * this.dayWidth - 4) + 'px';
        
        if (type === 'actual') {
            bar.style.top = '18px';
        } else {
            bar.style.top = '8px';
        }
        
        const label = document.createElement('span');
        label.className = 'bar-label';
        label.textContent = type === 'planned' ? '予定' : '実績';
        bar.appendChild(label);
        
        // リサイズハンドル
        const leftHandle = document.createElement('div');
        leftHandle.className = 'resize-handle left-handle';
        bar.appendChild(leftHandle);
        
        const rightHandle = document.createElement('div');
        rightHandle.className = 'resize-handle right-handle';
        bar.appendChild(rightHandle);
        
        bar.addEventListener('click', (e) => {
            e.stopPropagation();
            this.selectTask(task.id);
        });
        
        bar.addEventListener('mousedown', (e) => this.handleBarMouseDown(e, task, type));
        leftHandle.addEventListener('mousedown', (e) => this.handleResizeStart(e, task, type, 'left'));
        rightHandle.addEventListener('mousedown', (e) => this.handleResizeStart(e, task, type, 'right'));
        
        return bar;
    }
    
    selectTask(taskId) {
        this.selectedTask = taskId;
        
        document.querySelectorAll('.task-item').forEach(item => {
            item.classList.remove('selected');
        });
        
        document.querySelectorAll('.gantt-bar').forEach(bar => {
            bar.classList.remove('selected');
        });
        
        const selectedTaskElement = document.querySelector(`[data-task-id="${taskId}"]`);
        if (selectedTaskElement) {
            selectedTaskElement.classList.add('selected');
        }
        
        document.querySelectorAll(`[data-task-id="${taskId}"]`).forEach(element => {
            element.classList.add('selected');
        });
    }
    
    loadData() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    try {
                        this.tasks = JSON.parse(e.target.result);
                        this.calculateDateRange();
                        this.render();
                        alert('データを読み込みました');
                    } catch (error) {
                        alert('JSONファイルの読み込みに失敗しました');
                    }
                };
                reader.readAsText(file);
            }
        };
        
        input.click();
    }
    
    saveData() {
        const dataStr = JSON.stringify(this.tasks, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = 'gantt_data.json';
        link.click();
        
        alert('データを保存しました');
    }
    
    handleBarMouseDown(e, task, type) {
        if (e.target.classList.contains('resize-handle')) return;
        
        e.preventDefault();
        e.stopPropagation();
        
        this.isDragging = true;
        this.dragData = {
            task: task,
            type: type,
            startX: e.clientX,
            startLeft: parseInt(e.target.style.left),
            originalStart: task[type].start,
            originalEnd: task[type].end
        };
        
        document.body.style.cursor = 'move';
    }
    
    handleResizeStart(e, task, type, direction) {
        e.preventDefault();
        e.stopPropagation();
        
        this.resizing = true;
        this.dragData = {
            task: task,
            type: type,
            direction: direction,
            startX: e.clientX,
            originalStart: task[type].start,
            originalEnd: task[type].end
        };
        
        document.body.style.cursor = direction === 'left' ? 'w-resize' : 'e-resize';
    }
    
    handleMouseMove(e) {
        if (!this.isDragging && !this.resizing) return;
        
        const deltaX = e.clientX - this.dragData.startX;
        const daysDelta = Math.round(deltaX / this.dayWidth);
        
        if (this.isDragging) {
            this.moveTask(daysDelta);
        } else if (this.resizing) {
            this.resizeTask(daysDelta);
        }
    }
    
    handleMouseUp(e) {
        if (this.isDragging || this.resizing) {
            this.isDragging = false;
            this.resizing = false;
            this.dragData = null;
            document.body.style.cursor = 'default';
            
            this.render();
        }
    }
    
    moveTask(daysDelta) {
        if (daysDelta === 0) return;
        
        const { task, type } = this.dragData;
        const startDate = new Date(this.dragData.originalStart);
        const endDate = new Date(this.dragData.originalEnd);
        
        startDate.setDate(startDate.getDate() + daysDelta);
        endDate.setDate(endDate.getDate() + daysDelta);
        
        task[type].start = this.formatDate(startDate);
        task[type].end = this.formatDate(endDate);
        
        this.updateTaskDisplay(task, type);
    }
    
    resizeTask(daysDelta) {
        if (daysDelta === 0) return;
        
        const { task, type, direction } = this.dragData;
        
        if (direction === 'left') {
            const startDate = new Date(this.dragData.originalStart);
            startDate.setDate(startDate.getDate() + daysDelta);
            
            const endDate = new Date(task[type].end);
            if (startDate <= endDate) {
                task[type].start = this.formatDate(startDate);
            }
        } else {
            const endDate = new Date(this.dragData.originalEnd);
            endDate.setDate(endDate.getDate() + daysDelta);
            
            const startDate = new Date(task[type].start);
            if (endDate >= startDate) {
                task[type].end = this.formatDate(endDate);
            }
        }
        
        this.updateTaskDisplay(task, type);
    }
    
    updateTaskDisplay(task, type) {
        const bar = document.querySelector(`[data-task-id="${task.id}"][data-type="${type}"]`);
        if (!bar) return;
        
        const startDate = new Date(task[type].start);
        const endDate = new Date(task[type].end);
        const displayStart = this.viewStartDate || this.startDate;
        const displayEnd = this.viewEndDate || this.endDate;
        
        // 表示範囲外の場合は非表示
        if (endDate < displayStart || startDate > displayEnd) {
            bar.style.display = 'none';
            return;
        }
        
        const startOffset = Math.floor((startDate - displayStart) / (1000 * 60 * 60 * 24));
        const duration = Math.ceil((endDate - startDate) / (1000 * 60 * 60 * 24)) + 1;
        
        // 表示範囲内にクリップ
        const clippedStartOffset = Math.max(0, startOffset);
        const clippedEndOffset = Math.min(
            startOffset + duration, 
            Math.ceil((displayEnd - displayStart) / (1000 * 60 * 60 * 24))
        );
        const clippedDuration = clippedEndOffset - clippedStartOffset;
        
        if (clippedDuration <= 0) {
            bar.style.display = 'none';
            return;
        }
        
        bar.style.display = 'block';
        bar.style.left = (clippedStartOffset * this.dayWidth) + 'px';
        bar.style.width = (clippedDuration * this.dayWidth - 4) + 'px';
        
        this.updateTaskInfo(task);
    }
    
    updateTaskInfo(task) {
        const taskElement = document.querySelector(`[data-task-id="${task.id}"].task-item`);
        if (taskElement) {
            const taskInfo = taskElement.querySelector('.task-info');
            taskInfo.innerHTML = `
                予定: ${task.planned.start} ～ ${task.planned.end}<br>
                実績: ${task.actual.start} ～ ${task.actual.end}
            `;
        }
    }
    
    formatDate(date) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    }
    
    showAddTaskModal() {
        const modal = document.getElementById('taskModal');
        modal.style.display = 'block';
        
        // 今日の日付をデフォルトに設定
        const today = new Date();
        const tomorrow = new Date(today);
        tomorrow.setDate(tomorrow.getDate() + 1);
        
        document.getElementById('plannedStart').value = this.formatDate(today);
        document.getElementById('plannedEnd').value = this.formatDate(tomorrow);
        document.getElementById('taskName').focus();
    }
    
    hideAddTaskModal() {
        const modal = document.getElementById('taskModal');
        modal.style.display = 'none';
        document.getElementById('taskForm').reset();
    }
    
    handleAddTask(e) {
        e.preventDefault();
        
        const taskName = document.getElementById('taskName').value;
        const plannedStart = document.getElementById('plannedStart').value;
        const plannedEnd = document.getElementById('plannedEnd').value;
        const actualStart = document.getElementById('actualStart').value;
        const actualEnd = document.getElementById('actualEnd').value;
        
        // バリデーション
        if (!taskName || !plannedStart || !plannedEnd) {
            alert('タスク名、予定開始日、予定終了日は必須です');
            return;
        }
        
        if (new Date(plannedEnd) < new Date(plannedStart)) {
            alert('予定終了日は開始日以降にしてください');
            return;
        }
        
        if (actualStart && actualEnd && new Date(actualEnd) < new Date(actualStart)) {
            alert('実績終了日は開始日以降にしてください');
            return;
        }
        
        // 新しいIDを生成
        const newId = Math.max(...this.tasks.map(t => t.id), 0) + 1;
        
        // 新しいタスクを作成
        const newTask = {
            id: newId,
            name: taskName,
            planned: {
                start: plannedStart,
                end: plannedEnd
            },
            actual: {
                start: actualStart || plannedStart,
                end: actualEnd || plannedEnd
            }
        };
        
        this.tasks.push(newTask);
        this.calculateDateRange();
        this.render();
        this.hideAddTaskModal();
        
        alert('タスクを追加しました');
    }
    
    initializeDateRangeControls() {
        document.getElementById('viewStart').value = this.formatDate(this.startDate);
        document.getElementById('viewEnd').value = this.formatDate(this.endDate);
    }
    
    applyDateRange() {
        const startValue = document.getElementById('viewStart').value;
        const endValue = document.getElementById('viewEnd').value;
        
        if (!startValue || !endValue) {
            alert('開始日と終了日を入力してください');
            return;
        }
        
        const startDate = new Date(startValue);
        const endDate = new Date(endValue);
        
        if (endDate < startDate) {
            alert('終了日は開始日以降にしてください');
            return;
        }
        
        this.viewStartDate = startDate;
        this.viewEndDate = endDate;
        this.render();
    }
    
    resetDateRange() {
        this.viewStartDate = null;
        this.viewEndDate = null;
        document.getElementById('viewStart').value = this.formatDate(this.startDate);
        document.getElementById('viewEnd').value = this.formatDate(this.endDate);
        this.render();
    }
    
    changeZoom(zoomValue) {
        this.dayWidth = parseInt(zoomValue);
        this.render();
    }
    
    formatDateDisplay(date) {
        if (this.dayWidth >= 50) {
            return `${date.getMonth() + 1}/${date.getDate()}`;
        } else if (this.dayWidth >= 25) {
            return `${date.getDate()}`;
        } else {
            // 週単位表示
            if (date.getDay() === 1) { // 月曜日のみ表示
                return `${date.getMonth() + 1}/${date.getDate()}`;
            }
            return '';
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new GanttChart();
});