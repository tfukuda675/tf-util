class GanttChart {
    constructor() {
        this.tasks = [];
        this.startDate = null;
        this.endDate = null;
        this.dateRange = [];
        this.expandedTasks = new Set();
    }

    async init() {
        await this.loadData();
        this.calculateDateRange();
        this.render();
        this.attachEventListeners();
    }

    async loadData() {
        try {
            const response = await fetch('tasks.json');
            const data = await response.json();
            this.tasks = data.tasks;
            this.expandedTasks = new Set(this.tasks.filter(t => t.children && t.children.length > 0).map(t => t.id));
        } catch (error) {
            console.error('データの読み込みに失敗しました:', error);
        }
    }

    calculateDateRange() {
        let minDate = new Date();
        let maxDate = new Date();

        const getAllDates = (tasks) => {
            tasks.forEach(task => {
                const planStart = new Date(task.planStart);
                const planEnd = new Date(task.planEnd);
                
                if (planStart < minDate) minDate = planStart;
                if (planEnd > maxDate) maxDate = planEnd;

                if (task.actualStart) {
                    const actualStart = new Date(task.actualStart);
                    if (actualStart < minDate) minDate = actualStart;
                }
                if (task.actualEnd) {
                    const actualEnd = new Date(task.actualEnd);
                    if (actualEnd > maxDate) maxDate = actualEnd;
                }

                if (task.children) {
                    getAllDates(task.children);
                }
            });
        };

        getAllDates(this.tasks);

        minDate.setDate(1);
        maxDate.setMonth(maxDate.getMonth() + 1);
        maxDate.setDate(0);

        this.startDate = minDate;
        this.endDate = maxDate;

        this.dateRange = [];
        let currentDate = new Date(minDate);
        while (currentDate <= maxDate) {
            this.dateRange.push(new Date(currentDate));
            currentDate.setDate(currentDate.getDate() + 1);
        }
    }

    render() {
        const container = document.getElementById('ganttChart');
        container.innerHTML = '';

        const ganttContainer = document.createElement('div');
        ganttContainer.className = 'gantt-container';

        const sidebar = this.renderSidebar();
        const timeline = this.renderTimeline();

        ganttContainer.appendChild(sidebar);
        ganttContainer.appendChild(timeline);
        container.appendChild(ganttContainer);
    }

    renderSidebar() {
        const sidebar = document.createElement('div');
        sidebar.className = 'gantt-sidebar';

        const header = document.createElement('div');
        header.className = 'gantt-header';
        const headerCell = document.createElement('div');
        headerCell.className = 'task-cell task-header-cell';
        headerCell.textContent = 'タスク名';
        headerCell.style.width = '300px';
        header.appendChild(headerCell);
        sidebar.appendChild(header);

        const renderTaskRows = (tasks, level = 0) => {
            tasks.forEach(task => {
                const row = document.createElement('div');
                row.className = 'gantt-row';
                row.dataset.taskId = task.id;
                
                const cell = document.createElement('div');
                cell.className = 'task-cell';
                cell.style.width = '300px';

                if (task.children && task.children.length > 0) {
                    const toggle = document.createElement('span');
                    toggle.className = 'task-toggle';
                    toggle.textContent = this.expandedTasks.has(task.id) ? '▼' : '▶';
                    toggle.onclick = () => this.toggleTask(task.id);
                    cell.appendChild(toggle);
                } else {
                    const spacer = document.createElement('span');
                    spacer.style.width = '16px';
                    spacer.style.display = 'inline-block';
                    cell.appendChild(spacer);
                }

                const nameSpan = document.createElement('span');
                nameSpan.className = `task-name ${level === 0 ? 'task-parent' : 'task-child'}`;
                if (level > 1) {
                    nameSpan.classList.add(`task-child-level-${level}`);
                }
                nameSpan.style.paddingLeft = `${level * 20}px`;
                
                if (task.url) {
                    const link = document.createElement('a');
                    link.href = task.url;
                    link.textContent = task.name;
                    link.className = 'task-link';
                    link.target = '_blank';
                    link.rel = 'noopener noreferrer';
                    nameSpan.appendChild(link);
                } else {
                    nameSpan.textContent = task.name;
                }
                
                cell.appendChild(nameSpan);

                row.appendChild(cell);
                sidebar.appendChild(row);

                if (task.children && task.children.length > 0 && this.expandedTasks.has(task.id)) {
                    renderTaskRows(task.children, level + 1);
                }
            });
        };

        renderTaskRows(this.tasks);

        return sidebar;
    }

    renderTimeline() {
        const timeline = document.createElement('div');
        timeline.className = 'gantt-timeline';

        const header = this.renderTimelineHeader();
        timeline.appendChild(header);

        const renderTaskTimeline = (tasks) => {
            tasks.forEach(task => {
                const row = document.createElement('div');
                row.className = 'gantt-row';
                row.dataset.taskId = task.id;

                const grid = document.createElement('div');
                grid.className = 'timeline-grid';

                const today = new Date();
                today.setHours(0, 0, 0, 0);
                let todayIndex = -1;

                this.dateRange.forEach((date, index) => {
                    const cell = document.createElement('div');
                    cell.className = 'timeline-cell';
                    
                    const cellDate = new Date(date);
                    cellDate.setHours(0, 0, 0, 0);
                    
                    if (cellDate.getTime() === today.getTime()) {
                        cell.classList.add('today');
                        todayIndex = index;
                    }
                    
                    if (date.getDay() === 0 || date.getDay() === 6) {
                        cell.classList.add('weekend');
                    }

                    if (date.getDate() === 1 && index > 0) {
                        cell.classList.add('month-separator');
                    }

                    grid.appendChild(cell);
                });

                if (todayIndex !== -1) {
                    const todayMarker = document.createElement('div');
                    todayMarker.className = 'today-marker';
                    todayMarker.style.left = `${todayIndex * 30 + 14}px`;
                    grid.appendChild(todayMarker);
                }

                this.renderBars(grid, task);

                row.appendChild(grid);
                timeline.appendChild(row);

                if (task.children && task.children.length > 0 && this.expandedTasks.has(task.id)) {
                    renderTaskTimeline(task.children);
                }
            });
        };

        renderTaskTimeline(this.tasks);

        return timeline;
    }

    renderTimelineHeader() {
        const header = document.createElement('div');
        header.className = 'gantt-header';

        const grid = document.createElement('div');
        grid.className = 'timeline-grid';

        const today = new Date();
        today.setHours(0, 0, 0, 0);

        this.dateRange.forEach((date, index) => {
            const cell = document.createElement('div');
            cell.className = 'timeline-header-cell';
            
            const cellDate = new Date(date);
            cellDate.setHours(0, 0, 0, 0);
            
            if (cellDate.getTime() === today.getTime()) {
                cell.classList.add('today');
            }
            
            if (date.getDate() === 1) {
                cell.textContent = `${date.getMonth() + 1}/${date.getDate()}`;
                cell.style.fontWeight = 'bold';
                if (index > 0) {
                    cell.classList.add('month-separator');
                }
            } else {
                cell.textContent = date.getDate();
            }

            grid.appendChild(cell);
        });

        header.appendChild(grid);
        return header;
    }

    renderBars(grid, task) {
        const planStart = new Date(task.planStart);
        const planEnd = new Date(task.planEnd);
        
        const planStartIndex = this.dateRange.findIndex(d => 
            d.getTime() === planStart.getTime()
        );
        const planEndIndex = this.dateRange.findIndex(d => 
            d.getTime() === planEnd.getTime()
        );

        if (planStartIndex !== -1 && planEndIndex !== -1) {
            const planBar = document.createElement('div');
            planBar.className = 'bar-container';
            planBar.style.left = `${planStartIndex * 30}px`;
            planBar.style.width = `${(planEndIndex - planStartIndex + 1) * 30}px`;

            const planBarInner = document.createElement('div');
            let planBarClass = `bar-plan ${task.children ? 'bar-parent' : ''}`;
            
            // 完了済みの場合はグレーに
            if (task.actualEnd) {
                planBarClass += ' completed';
            }
            
            planBarInner.className = planBarClass;
            planBarInner.style.width = '100%';
            
            const planLabel = document.createElement('span');
            planLabel.className = 'bar-label';
            planLabel.textContent = '予定';
            planBarInner.appendChild(planLabel);
            
            planBar.appendChild(planBarInner);

            if (task.actualStart) {
                const actualStart = new Date(task.actualStart);
                const actualStartIndex = this.dateRange.findIndex(d => 
                    d.getTime() === actualStart.getTime()
                );

                let actualEndIndex;
                let actualEnd;
                let isCompleted = false;
                let isInProgress = false;

                if (task.actualEnd) {
                    actualEnd = new Date(task.actualEnd);
                    actualEndIndex = this.dateRange.findIndex(d => 
                        d.getTime() === actualEnd.getTime()
                    );
                    isCompleted = true;
                } else {
                    const today = new Date();
                    today.setHours(0, 0, 0, 0);
                    actualEnd = today;
                    actualEndIndex = this.dateRange.findIndex(d => {
                        const cellDate = new Date(d);
                        cellDate.setHours(0, 0, 0, 0);
                        return cellDate.getTime() === today.getTime();
                    });
                    isInProgress = true;
                }

                if (actualStartIndex !== -1 && actualEndIndex !== -1) {
                    const actualBar = document.createElement('div');
                    let className = 'bar-actual';
                    
                    if (isCompleted) {
                        className += ' completed';
                        if (actualEnd > planEnd) {
                            className += ' bar-delay';
                        }
                    } else if (isInProgress) {
                        className += ' in-progress';
                    }
                    
                    actualBar.className = className;
                    actualBar.style.width = `${((actualEndIndex - actualStartIndex + 1) / (planEndIndex - planStartIndex + 1)) * 100}%`;
                    actualBar.style.marginLeft = `${((actualStartIndex - planStartIndex) / (planEndIndex - planStartIndex + 1)) * 100}%`;
                    
                    const actualLabel = document.createElement('span');
                    actualLabel.className = 'bar-label';
                    actualLabel.textContent = isCompleted ? '完了' : '進行中';
                    actualBar.appendChild(actualLabel);
                    
                    planBar.appendChild(actualBar);
                }
            }

            grid.appendChild(planBar);
        }
    }

    toggleTask(taskId) {
        if (this.expandedTasks.has(taskId)) {
            this.expandedTasks.delete(taskId);
        } else {
            this.expandedTasks.add(taskId);
        }
        this.render();
    }

    expandAll() {
        const addAllParents = (tasks) => {
            tasks.forEach(task => {
                if (task.children && task.children.length > 0) {
                    this.expandedTasks.add(task.id);
                    addAllParents(task.children);
                }
            });
        };
        addAllParents(this.tasks);
        this.render();
    }

    collapseAll() {
        this.expandedTasks.clear();
        this.render();
    }

    scrollToToday() {
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        const todayIndex = this.dateRange.findIndex(d => {
            const cellDate = new Date(d);
            cellDate.setHours(0, 0, 0, 0);
            return cellDate.getTime() === today.getTime();
        });
        
        if (todayIndex !== -1) {
            const timeline = document.querySelector('.gantt-timeline');
            timeline.scrollLeft = todayIndex * 30 - 200;
        }
    }

    attachEventListeners() {
        document.getElementById('expandAll').addEventListener('click', () => this.expandAll());
        document.getElementById('collapseAll').addEventListener('click', () => this.collapseAll());
        document.getElementById('todayBtn').addEventListener('click', () => this.scrollToToday());
    }
}

const gantt = new GanttChart();
gantt.init();
