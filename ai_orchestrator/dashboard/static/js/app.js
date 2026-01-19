/**
 * AI Orchestrator Dashboard - Main Application
 */

class Dashboard {
    constructor() {
        this.config = null;
        this.availableModels = {};
        this.reasoningLevels = [];
        this.agentReasoningLevels = {};
        this.workflows = {};
        this.ws = null;
        this.wsReconnectAttempts = 0;
        this.maxReconnectAttempts = 5;

        this.init();
    }

    async init() {
        // Load initial configuration
        await this.loadConfig();

        // Load available models for all providers
        await this.loadAvailableModels();

        // Load Codex reasoning levels
        await this.loadReasoningLevels();

        // Setup navigation
        this.setupNavigation();

        // Setup all event listeners
        this.setupEventListeners();

        // Setup WebSocket
        this.setupWebSocket();

        // Update UI with loaded config
        this.updateUI();

        // Load initial metrics
        await this.loadMetrics();
    }

    // === Configuration Management ===

    async loadConfig() {
        try {
            const response = await fetch('/api/config');
            if (!response.ok) throw new Error('Failed to load config');
            this.config = await response.json();
            console.log('Config loaded:', this.config);
        } catch (error) {
            console.error('Failed to load config:', error);
            this.showToast('error', 'Error', 'Failed to load configuration');
        }
    }

    async loadAvailableModels() {
        try {
            const response = await fetch('/api/models');
            if (!response.ok) throw new Error('Failed to load models');
            this.availableModels = await response.json();
            console.log('Available models loaded:', this.availableModels);
        } catch (error) {
            console.error('Failed to load available models:', error);
            this.showToast('error', 'Error', 'Failed to load available models');
        }
    }

    async loadReasoningLevels() {
        try {
            // Load available reasoning levels
            const response = await fetch('/api/codex/reasoning-levels');
            if (!response.ok) throw new Error('Failed to load reasoning levels');
            this.reasoningLevels = await response.json();

            // Load current reasoning level for Codex
            const levelResponse = await fetch('/api/agents/codex/reasoning-level');
            if (levelResponse.ok) {
                const data = await levelResponse.json();
                this.agentReasoningLevels['codex'] = data.reasoning_level;
            }

            console.log('Reasoning levels loaded:', this.reasoningLevels);
        } catch (error) {
            console.error('Failed to load reasoning levels:', error);
        }
    }

    async saveConfig() {
        try {
            const response = await fetch('/api/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config: this.config })
            });

            if (!response.ok) throw new Error('Failed to save config');

            this.showToast('success', 'Success', 'Configuration saved successfully');
        } catch (error) {
            console.error('Failed to save config:', error);
            this.showToast('error', 'Error', 'Failed to save configuration');
        }
    }

    async resetConfig() {
        if (!confirm('Are you sure you want to reset all settings to defaults?')) return;

        try {
            const response = await fetch('/api/config/reset', { method: 'POST' });
            if (!response.ok) throw new Error('Failed to reset config');

            const data = await response.json();
            this.config = data.config;
            this.updateUI();
            this.showToast('success', 'Success', 'Configuration reset to defaults');
        } catch (error) {
            console.error('Failed to reset config:', error);
            this.showToast('error', 'Error', 'Failed to reset configuration');
        }
    }

    async exportConfig() {
        try {
            const response = await fetch('/api/config/export');
            if (!response.ok) throw new Error('Failed to export config');

            const config = await response.json();
            const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = 'ai_orchestrator_config.json';
            a.click();

            URL.revokeObjectURL(url);
            this.showToast('success', 'Success', 'Configuration exported');
        } catch (error) {
            console.error('Failed to export config:', error);
            this.showToast('error', 'Error', 'Failed to export configuration');
        }
    }

    triggerImportConfig() {
        document.getElementById('configFileInput').click();
    }

    async importConfig(event) {
        const file = event.target.files[0];
        if (!file) return;

        try {
            const text = await file.text();
            const config = JSON.parse(text);

            // API expects {"config": <config>} wrapper
            const response = await fetch('/api/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config: config })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || 'Failed to import config');
            }

            const result = await response.json();
            // Response is {success, message, config}
            this.config = result.config || result;
            await this.loadAvailableModels();
            await this.loadReasoningLevels();
            this.updateUI();
            this.showToast('success', 'Success', 'Configuration imported');
        } catch (error) {
            console.error('Failed to import config:', error);
            this.showToast('error', 'Error', `Failed to import: ${error.message}`);
        } finally {
            // Reset file input so same file can be selected again
            event.target.value = '';
        }
    }

    // === Navigation ===

    setupNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const section = item.dataset.section;
                this.navigateTo(section);
            });
        });
    }

    navigateTo(section) {
        // Update nav items
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.toggle('active', item.dataset.section === section);
        });

        // Update sections
        document.querySelectorAll('.section').forEach(sec => {
            sec.classList.toggle('active', sec.id === `${section}-section`);
        });

        // Update page title
        const titles = {
            overview: ['Overview', 'Monitor and control your AI orchestration workflow'],
            agents: ['Agents', 'Configure AI agents and their roles'],
            workflow: ['Workflow', 'Control workflow behavior and iteration settings'],
            research: ['Research', 'Configure deep research and web search providers'],
            prompts: ['Prompt Settings', 'Configure prompt enhancement and amelioration'],
            timeouts: ['Timeouts', 'Configure timeout settings for all operations'],
            monitoring: ['Monitoring', 'View active workflows and metrics']
        };

        const [title, description] = titles[section] || ['Dashboard', ''];
        document.getElementById('pageTitle').textContent = title;
        document.getElementById('pageDescription').textContent = description;
    }

    // === Event Listeners ===

    setupEventListeners() {
        // Save/Export/Import buttons
        document.getElementById('saveConfigBtn')?.addEventListener('click', () => this.saveConfig());
        document.getElementById('exportConfigBtn')?.addEventListener('click', () => this.exportConfig());
        document.getElementById('importConfigBtn')?.addEventListener('click', () => this.triggerImportConfig());
        document.getElementById('configFileInput')?.addEventListener('change', (e) => this.importConfig(e));
        document.getElementById('resetConfigBtn')?.addEventListener('click', () => this.resetConfig());

        // Quick actions
        document.getElementById('newWorkflowBtn')?.addEventListener('click', () => this.openNewWorkflowModal());
        document.getElementById('testAgentsBtn')?.addEventListener('click', () => this.testAllAgents());

        // Modal controls
        document.getElementById('closeModalBtn')?.addEventListener('click', () => this.closeModal());
        document.getElementById('cancelWorkflowBtn')?.addEventListener('click', () => this.closeModal());
        document.getElementById('startWorkflowBtn')?.addEventListener('click', () => this.startWorkflow());

        // Directory browser controls - main page and modal
        document.getElementById('mainBrowseBtn')?.addEventListener('click', () => this.openDirectoryBrowser('main'));
        document.getElementById('browseProjectBtn')?.addEventListener('click', () => this.openDirectoryBrowser('modal'));
        document.getElementById('closeBrowserBtn')?.addEventListener('click', () => this.closeDirectoryBrowser());
        document.getElementById('cancelBrowseBtn')?.addEventListener('click', () => this.closeDirectoryBrowser());
        document.getElementById('selectDirectoryBtn')?.addEventListener('click', () => this.selectDirectory());

        // Range sliders
        document.getElementById('agreementThreshold')?.addEventListener('input', (e) => {
            document.getElementById('agreementThresholdValue').textContent = `${e.target.value}%`;
            if (this.config) {
                this.config.iteration.agreement_threshold = e.target.value / 100;
            }
        });

        // Timeout tabs
        document.querySelectorAll('.timeout-tab').forEach(tab => {
            tab.addEventListener('click', () => this.switchTimeoutTab(tab.dataset.agent));
        });

        // Toggle settings visibility
        document.getElementById('incrementalReviewEnabled')?.addEventListener('change', (e) => {
            document.getElementById('incrementalReviewSettings').style.opacity = e.target.checked ? '1' : '0.5';
        });

        document.getElementById('deepResearchEnabled')?.addEventListener('change', (e) => {
            document.getElementById('deepResearchSettings').style.opacity = e.target.checked ? '1' : '0.5';
        });

        document.getElementById('webSearchEnabled')?.addEventListener('change', (e) => {
            document.getElementById('webSearchSettings').style.opacity = e.target.checked ? '1' : '0.5';
        });

        document.getElementById('promptEnhancementEnabled')?.addEventListener('change', (e) => {
            document.getElementById('promptEnhancementSettings').style.opacity = e.target.checked ? '1' : '0.5';
        });

        // Setup form change handlers to update config
        this.setupFormHandlers();
    }

    setupFormHandlers() {
        // Iteration settings
        this.bindInput('maxIterations', 'iteration.max_iterations', 'number');
        this.bindInput('convergenceCheckAfter', 'iteration.convergence_check_after', 'number');
        this.bindInput('consecutiveCleanRequired', 'iteration.consecutive_clean_required', 'number');
        this.bindCheckbox('stopOnZeroCritical', 'iteration.stop_on_zero_critical');
        this.bindCheckbox('stopOnPlateau', 'iteration.stop_on_plateau');

        // Human-in-loop settings
        this.bindCheckbox('hilAfterPlanSynthesis', 'human_loop.after_plan_synthesis');
        this.bindCheckbox('hilAfterPostChecksFailure', 'human_loop.after_post_checks_failure');
        this.bindSelect('hilAfterFinalIteration', 'human_loop.after_final_iteration');
        this.bindSelect('hilBeforeImplementing', 'human_loop.before_implementing');
        this.bindInput('autoApproveTimeout', 'human_loop.auto_approve_timeout', 'number');

        // Post-checks
        this.bindCheckbox('postCheckStaticAnalysis', 'post_checks.static_analysis');
        this.bindCheckbox('postCheckUnitTests', 'post_checks.unit_tests');
        this.bindCheckbox('postCheckBuild', 'post_checks.build_check');
        this.bindCheckbox('postCheckSecurityScan', 'post_checks.security_scan');
        this.bindCheckbox('postCheckSmokeTest', 'post_checks.manual_smoke_test');

        // Incremental review
        this.bindCheckbox('incrementalReviewEnabled', 'incremental_review.enabled');
        this.bindSelect('reviewGranularity', 'incremental_review.granularity');
        this.bindInput('reviewThresholdLines', 'incremental_review.threshold_lines', 'number');
        this.bindSelect('incrementalReviewAgent', 'incremental_review.review_agent');

        // Resilience
        this.bindInput('maxConcurrentClis', 'resilience.max_concurrent_clis', 'number');
        this.bindInput('retryAttempts', 'resilience.retry_attempts', 'number');
        this.bindInput('circuitBreakerThreshold', 'resilience.circuit_breaker_threshold', 'number');
        this.bindInput('circuitBreakerResetTimeout', 'resilience.circuit_breaker_reset_timeout', 'number');
        this.bindCheckbox('enableFallback', 'resilience.enable_fallback');

        // Research settings (MCP-based, providers control sources/depth)
        this.bindCheckbox('deepResearchEnabled', 'research.enabled');
        this.bindResearchProviders();  // Multiple provider checkboxes
        this.bindInput('researchTimeout', 'research.timeout', 'number');

        // Web search settings
        this.bindCheckbox('webSearchEnabled', 'web_search.enabled');
        this.bindSelect('webSearchProvider', 'web_search.provider');
        this.bindInput('maxSearchResults', 'web_search.max_results', 'number');
        this.bindCheckbox('includeSnippets', 'web_search.include_snippets');
        this.bindCheckbox('safeSearch', 'web_search.safe_search');

        // Prompt enhancement
        this.bindCheckbox('promptEnhancementEnabled', 'prompt_enhancement.enabled');
        this.bindCheckbox('addReasoningSteps', 'prompt_enhancement.add_reasoning_steps');
        this.bindCheckbox('addVerificationPrompts', 'prompt_enhancement.add_verification_prompts');
        this.bindCheckbox('injectBestPractices', 'prompt_enhancement.inject_best_practices');
        this.bindCheckbox('includeCodeContext', 'prompt_enhancement.include_code_context');
        this.bindInput('maxContextFiles', 'prompt_enhancement.max_context_files', 'number');
        this.bindCheckbox('geminiDepthEnhancement', 'prompt_enhancement.gemini_depth_enhancement');
        this.bindCheckbox('planningModePrefix', 'prompt_enhancement.planning_mode_prefix');

        // Timeout settings
        this.bindInput('maxTotalWorkflow', 'timeouts.max_total_workflow', 'number');
        this.bindInput('maxSingleOperation', 'timeouts.max_single_operation', 'number');
        this.bindSelect('timeoutAction', 'timeouts.timeout_action');
        this.bindCheckbox('retryOnTimeout', 'timeouts.retry_on_timeout');
    }

    bindInput(elementId, configPath, type = 'text') {
        const element = document.getElementById(elementId);
        if (!element) return;

        element.addEventListener('change', () => {
            let value = element.value;
            if (type === 'number') {
                value = value ? parseInt(value, 10) : null;
            }
            this.setConfigValue(configPath, value);
        });
    }

    bindCheckbox(elementId, configPath) {
        const element = document.getElementById(elementId);
        if (!element) return;

        element.addEventListener('change', () => {
            this.setConfigValue(configPath, element.checked);
        });
    }

    bindSelect(elementId, configPath) {
        const element = document.getElementById(elementId);
        if (!element) return;

        element.addEventListener('change', () => {
            this.setConfigValue(configPath, element.value);
        });
    }

    bindResearchProviders() {
        const container = document.getElementById('researchProviders');
        if (!container) return;

        container.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                const selectedProviders = [];
                container.querySelectorAll('input[type="checkbox"]:checked').forEach(cb => {
                    selectedProviders.push(cb.value);
                });
                this.setConfigValue('research.providers', selectedProviders);
            });
        });
    }

    setConfigValue(path, value) {
        if (!this.config) return;

        const parts = path.split('.');
        let obj = this.config;

        for (let i = 0; i < parts.length - 1; i++) {
            if (!obj[parts[i]]) obj[parts[i]] = {};
            obj = obj[parts[i]];
        }

        obj[parts[parts.length - 1]] = value;
    }

    getConfigValue(path) {
        if (!this.config) return null;

        const parts = path.split('.');
        let obj = this.config;

        for (const part of parts) {
            if (obj === undefined || obj === null) return null;
            obj = obj[part];
        }

        return obj;
    }

    // === UI Updates ===

    updateUI() {
        if (!this.config) return;

        this.updateAgentsUI();
        this.updateOverviewUI();
        this.updateWorkflowUI();
        this.updateResearchUI();
        this.updatePromptsUI();
        this.updateTimeoutsUI();
    }

    updateOverviewUI() {
        // Update agent status grid
        const grid = document.getElementById('agentStatusGrid');
        if (!grid || !this.config.agents) return;

        const colors = {
            claude: '#f97316',
            codex: '#22c55e',
            gemini: '#3b82f6',
            kilocode: '#a855f7'
        };

        grid.innerHTML = Object.entries(this.config.agents).map(([name, agent]) => `
            <div class="agent-status-card">
                <div class="agent-avatar" style="background: ${colors[name]}20; color: ${colors[name]}">
                    ${name.charAt(0).toUpperCase()}
                </div>
                <div class="agent-info">
                    <div class="agent-name">${agent.display_name}</div>
                    <div class="agent-model">${this.getModelDisplayName(name, agent.model)}</div>
                </div>
                <span class="agent-badge ${agent.status}">${agent.status}</span>
            </div>
        `).join('');

        // Update enabled agents count
        const enabledCount = Object.values(this.config.agents).filter(a => a.enabled).length;
        document.getElementById('enabledAgents').textContent = enabledCount;
    }

    getModelDisplayName(agentName, modelId) {
        const models = this.availableModels[agentName] || [];
        const model = models.find(m => m.id === modelId);
        return model ? model.name : (modelId || 'default');
    }

    updateAgentsUI() {
        const list = document.getElementById('agentsList');
        if (!list || !this.config.agents) return;

        const colors = {
            claude: '#f97316',
            codex: '#22c55e',
            gemini: '#3b82f6',
            kilocode: '#a855f7'
        };

        list.innerHTML = Object.entries(this.config.agents).map(([name, agent]) => `
            <div class="agent-item">
                <div class="agent-avatar" style="background: ${colors[name]}20; color: ${colors[name]}; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600;">
                    ${name.charAt(0).toUpperCase()}
                </div>
                <div class="agent-info" style="flex: 1;">
                    <div class="agent-name" style="font-weight: 600;">${agent.display_name}</div>
                    <div class="agent-selectors" style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                        <select class="model-selector" data-agent="${name}" onchange="dashboard.changeAgentModel('${name}', this.value)">
                            ${this.renderModelOptions(name, agent.model)}
                        </select>
                        ${name === 'codex' ? `
                            <select class="reasoning-selector" data-agent="${name}" onchange="dashboard.changeReasoningLevel('${name}', this.value)">
                                ${this.renderReasoningOptions(name)}
                            </select>
                        ` : ''}
                    </div>
                </div>
                <div class="agent-roles" style="display: flex; gap: 0.25rem;">
                    ${agent.roles.map(role => `<span style="padding: 0.125rem 0.5rem; background: var(--bg-tertiary); border-radius: 12px; font-size: 0.7rem;">${role}</span>`).join('')}
                </div>
                <label class="toggle agent-toggle">
                    <input type="checkbox" ${agent.enabled ? 'checked' : ''} onchange="dashboard.toggleAgent('${name}', this.checked)">
                    <span class="toggle-slider"></span>
                </label>
                <div class="agent-actions">
                    <button class="btn btn-sm btn-secondary" onclick="dashboard.testAgent('${name}')">Test</button>
                    <button class="btn btn-sm btn-secondary" onclick="dashboard.resetAgentCircuit('${name}')">Reset</button>
                </div>
            </div>
        `).join('');

        // Update role selectors
        this.updateRoleSelectors();
        this.updateFallbackChains();
    }

    renderModelOptions(agentName, currentModel) {
        const models = this.availableModels[agentName] || [];
        if (models.length === 0) {
            return `<option value="${currentModel || 'default'}">${currentModel || 'No models available'}</option>`;
        }

        return models.map(model => {
            const selected = model.id === currentModel ? 'selected' : '';
            const tierBadge = model.tier ? ` [${model.tier}]` : '';
            return `<option value="${model.id}" ${selected}>${model.name}${tierBadge}</option>`;
        }).join('');
    }

    async changeAgentModel(agentName, modelId) {
        try {
            const response = await fetch(`/api/agents/${agentName}/model`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: modelId })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to update model');
            }

            // Update local config
            if (this.config.agents[agentName]) {
                this.config.agents[agentName].model = modelId;
            }

            this.showToast('success', 'Model Updated', `${agentName} now using ${modelId}`);
            this.updateOverviewUI();
        } catch (error) {
            console.error('Failed to change agent model:', error);
            this.showToast('error', 'Error', error.message);
            // Revert the dropdown to the original value
            this.updateAgentsUI();
        }
    }

    renderReasoningOptions(agentName) {
        if (this.reasoningLevels.length === 0) {
            return '<option value="medium">Medium (Balanced)</option>';
        }

        const currentLevel = this.agentReasoningLevels[agentName] || 'medium';
        return this.reasoningLevels.map(level => {
            const selected = level.id === currentLevel ? 'selected' : '';
            return `<option value="${level.id}" ${selected} title="${level.description}">${level.name}</option>`;
        }).join('');
    }

    async changeReasoningLevel(agentName, level) {
        try {
            const response = await fetch(`/api/agents/${agentName}/reasoning-level`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ reasoning_level: level })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to update reasoning level');
            }

            // Update local state
            this.agentReasoningLevels[agentName] = level;

            const levelName = this.reasoningLevels.find(l => l.id === level)?.name || level;
            this.showToast('success', 'Reasoning Updated', `${agentName} reasoning: ${levelName}`);
        } catch (error) {
            console.error('Failed to change reasoning level:', error);
            this.showToast('error', 'Error', error.message);
            // Revert the dropdown
            this.updateAgentsUI();
        }
    }

    updateRoleSelectors() {
        const roles = ['planner', 'reviewer', 'implementer', 'researcher'];
        const assignment = this.config.agent_assignment || {};

        roles.forEach(role => {
            const selector = document.getElementById(`${role}Selector`);
            if (!selector) return;

            const assigned = assignment[`${role}s`] || [];

            selector.innerHTML = Object.entries(this.config.agents).map(([name, agent]) => {
                const isSelected = assigned.includes(name);
                return `
                    <div class="agent-chip ${isSelected ? 'selected' : ''}"
                         onclick="dashboard.toggleRoleAssignment('${role}s', '${name}')">
                        <span class="chip-dot"></span>
                        ${agent.display_name}
                    </div>
                `;
            }).join('');
        });
    }

    updateFallbackChains() {
        const roles = ['planning', 'reviewing', 'implementing'];
        const chains = this.config.resilience?.fallback_chains || {};

        roles.forEach(role => {
            const container = document.getElementById(`${role}Fallback`);
            if (!container) return;

            const chain = chains[role] || [];

            container.innerHTML = chain.map((name, index) => `
                <div class="sortable-item" draggable="true" data-agent="${name}" data-role="${role}">
                    <span class="drag-handle">⋮⋮</span>
                    <span class="order-num">${index + 1}</span>
                    <span>${this.config.agents[name]?.display_name || name}</span>
                </div>
            `).join('');

            // Add drag/drop event listeners
            this.setupDragDrop(container, role);
        });
    }

    setupDragDrop(container, role) {
        let draggedItem = null;

        container.querySelectorAll('.sortable-item').forEach(item => {
            item.addEventListener('dragstart', (e) => {
                draggedItem = item;
                item.classList.add('dragging');
                e.dataTransfer.effectAllowed = 'move';
                e.dataTransfer.setData('text/plain', item.dataset.agent);
            });

            item.addEventListener('dragend', () => {
                item.classList.remove('dragging');
                draggedItem = null;
                container.querySelectorAll('.sortable-item').forEach(i => i.classList.remove('drag-over'));
            });

            item.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'move';
                if (draggedItem && draggedItem !== item) {
                    item.classList.add('drag-over');
                }
            });

            item.addEventListener('dragleave', () => {
                item.classList.remove('drag-over');
            });

            item.addEventListener('drop', (e) => {
                e.preventDefault();
                item.classList.remove('drag-over');

                if (draggedItem && draggedItem !== item) {
                    // Get all items and their agents
                    const items = Array.from(container.querySelectorAll('.sortable-item'));
                    const fromIndex = items.indexOf(draggedItem);
                    const toIndex = items.indexOf(item);

                    // Reorder the chain
                    const chain = [...(this.config.resilience?.fallback_chains?.[role] || [])];
                    const [moved] = chain.splice(fromIndex, 1);
                    chain.splice(toIndex, 0, moved);

                    // Update config
                    if (!this.config.resilience) this.config.resilience = {};
                    if (!this.config.resilience.fallback_chains) this.config.resilience.fallback_chains = {};
                    this.config.resilience.fallback_chains[role] = chain;

                    // Re-render
                    this.updateFallbackChains();
                    this.showToast('info', 'Reordered', `${role} fallback chain updated`);
                }
            });
        });
    }

    updateWorkflowUI() {
        // Iteration settings
        this.setInputValue('maxIterations', this.config.iteration?.max_iterations);
        this.setInputValue('convergenceCheckAfter', this.config.iteration?.convergence_check_after);
        this.setInputValue('consecutiveCleanRequired', this.config.iteration?.consecutive_clean_required);
        this.setCheckbox('stopOnZeroCritical', this.config.iteration?.stop_on_zero_critical);
        this.setCheckbox('stopOnPlateau', this.config.iteration?.stop_on_plateau);

        const threshold = (this.config.iteration?.agreement_threshold || 0.8) * 100;
        this.setInputValue('agreementThreshold', threshold);
        document.getElementById('agreementThresholdValue').textContent = `${threshold}%`;

        // Human-in-loop
        this.setCheckbox('hilAfterPlanSynthesis', this.config.human_loop?.after_plan_synthesis);
        this.setCheckbox('hilAfterPostChecksFailure', this.config.human_loop?.after_post_checks_failure);
        this.setSelectValue('hilAfterFinalIteration', this.config.human_loop?.after_final_iteration);
        this.setSelectValue('hilBeforeImplementing', this.config.human_loop?.before_implementing);
        this.setInputValue('autoApproveTimeout', this.config.human_loop?.auto_approve_timeout);

        // Post-checks
        this.setCheckbox('postCheckStaticAnalysis', this.config.post_checks?.static_analysis);
        this.setCheckbox('postCheckUnitTests', this.config.post_checks?.unit_tests);
        this.setCheckbox('postCheckBuild', this.config.post_checks?.build_check);
        this.setCheckbox('postCheckSecurityScan', this.config.post_checks?.security_scan);
        this.setCheckbox('postCheckSmokeTest', this.config.post_checks?.manual_smoke_test);

        // Incremental review
        this.setCheckbox('incrementalReviewEnabled', this.config.incremental_review?.enabled);
        this.setSelectValue('reviewGranularity', this.config.incremental_review?.granularity);
        this.setInputValue('reviewThresholdLines', this.config.incremental_review?.threshold_lines);
        this.setSelectValue('incrementalReviewAgent', this.config.incremental_review?.review_agent);

        // Resilience
        this.setInputValue('maxConcurrentClis', this.config.resilience?.max_concurrent_clis);
        this.setInputValue('retryAttempts', this.config.resilience?.retry_attempts);
        this.setInputValue('circuitBreakerThreshold', this.config.resilience?.circuit_breaker_threshold);
        this.setInputValue('circuitBreakerResetTimeout', this.config.resilience?.circuit_breaker_reset_timeout);
        this.setCheckbox('enableFallback', this.config.resilience?.enable_fallback);
    }

    updateResearchUI() {
        // Deep research
        this.setCheckbox('deepResearchEnabled', this.config.research?.enabled);

        // Set research providers checkboxes (multiple selection)
        const providers = this.config.research?.providers || [];
        const providerContainer = document.getElementById('researchProviders');
        if (providerContainer) {
            providerContainer.querySelectorAll('input[type="checkbox"]').forEach(cb => {
                cb.checked = providers.includes(cb.value);
            });
        }

        this.setInputValue('researchTimeout', this.config.research?.timeout);

        // Web search
        this.setCheckbox('webSearchEnabled', this.config.web_search?.enabled);
        this.setSelectValue('webSearchProvider', this.config.web_search?.provider);
        this.setInputValue('maxSearchResults', this.config.web_search?.max_results);
        this.setCheckbox('includeSnippets', this.config.web_search?.include_snippets);
        this.setCheckbox('safeSearch', this.config.web_search?.safe_search);
    }

    updatePromptsUI() {
        this.setCheckbox('promptEnhancementEnabled', this.config.prompt_enhancement?.enabled);
        this.setCheckbox('addReasoningSteps', this.config.prompt_enhancement?.add_reasoning_steps);
        this.setCheckbox('addVerificationPrompts', this.config.prompt_enhancement?.add_verification_prompts);
        this.setCheckbox('injectBestPractices', this.config.prompt_enhancement?.inject_best_practices);
        this.setCheckbox('includeCodeContext', this.config.prompt_enhancement?.include_code_context);
        this.setInputValue('maxContextFiles', this.config.prompt_enhancement?.max_context_files);
        this.setCheckbox('geminiDepthEnhancement', this.config.prompt_enhancement?.gemini_depth_enhancement);
        this.setCheckbox('planningModePrefix', this.config.prompt_enhancement?.planning_mode_prefix);

        this.setTextareaValue('customPrefix', this.config.prompt_enhancement?.custom_prefix);
        this.setTextareaValue('customSuffix', this.config.prompt_enhancement?.custom_suffix);
    }

    updateTimeoutsUI() {
        this.setInputValue('maxTotalWorkflow', this.config.timeouts?.max_total_workflow);
        this.setInputValue('maxSingleOperation', this.config.timeouts?.max_single_operation);
        this.setSelectValue('timeoutAction', this.config.timeouts?.timeout_action);
        this.setCheckbox('retryOnTimeout', this.config.timeouts?.retry_on_timeout);

        // Show first agent's timeouts
        this.switchTimeoutTab('claude');
    }

    switchTimeoutTab(agent) {
        // Update tab states
        document.querySelectorAll('.timeout-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.agent === agent);
        });

        // Update content
        const content = document.getElementById('timeoutContent');
        if (!content || !this.config.timeouts) return;

        const timeouts = this.config.timeouts[agent] || {};

        content.innerHTML = `
            <div class="timeout-grid">
                <div class="timeout-item">
                    <label>Planning Timeout</label>
                    <div class="timeout-input-group">
                        <input type="number" value="${timeouts.planning || 900}"
                               onchange="dashboard.updateAgentTimeout('${agent}', 'planning', this.value)">
                        <span class="unit">seconds</span>
                    </div>
                </div>
                <div class="timeout-item">
                    <label>Reviewing Timeout</label>
                    <div class="timeout-input-group">
                        <input type="number" value="${timeouts.reviewing || 600}"
                               onchange="dashboard.updateAgentTimeout('${agent}', 'reviewing', this.value)">
                        <span class="unit">seconds</span>
                    </div>
                </div>
                <div class="timeout-item">
                    <label>Implementing Timeout</label>
                    <div class="timeout-input-group">
                        <input type="number" value="${timeouts.implementing || 1800}"
                               onchange="dashboard.updateAgentTimeout('${agent}', 'implementing', this.value)">
                        <span class="unit">seconds</span>
                    </div>
                </div>
                <div class="timeout-item">
                    <label>Researching Timeout</label>
                    <div class="timeout-input-group">
                        <input type="number" value="${timeouts.researching || 1200}"
                               onchange="dashboard.updateAgentTimeout('${agent}', 'researching', this.value)">
                        <span class="unit">seconds</span>
                    </div>
                </div>
            </div>
        `;
    }

    updateAgentTimeout(agent, phase, value) {
        if (!this.config.timeouts[agent]) {
            this.config.timeouts[agent] = {};
        }
        this.config.timeouts[agent][phase] = parseInt(value, 10);
    }

    // === Helper Methods for Setting Values ===

    setInputValue(id, value) {
        const el = document.getElementById(id);
        if (el && value !== undefined && value !== null) {
            el.value = value;
        }
    }

    setCheckbox(id, value) {
        const el = document.getElementById(id);
        if (el) {
            el.checked = !!value;
        }
    }

    setSelectValue(id, value) {
        const el = document.getElementById(id);
        if (el && value !== undefined) {
            el.value = value;
        }
    }

    setTextareaValue(id, value) {
        const el = document.getElementById(id);
        if (el) {
            el.value = value || '';
        }
    }

    // === Agent Actions ===

    async toggleAgent(name, enabled) {
        try {
            const action = enabled ? 'enable' : 'disable';
            const response = await fetch(`/api/agents/${name}/action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ agent: name, action })
            });

            if (!response.ok) throw new Error('Failed to toggle agent');

            const data = await response.json();
            this.config.agents[name] = data.agent_status;
            this.updateOverviewUI();
            this.showToast('success', 'Success', data.message);
        } catch (error) {
            console.error('Failed to toggle agent:', error);
            this.showToast('error', 'Error', 'Failed to toggle agent');
        }
    }

    async testAgent(name) {
        try {
            const response = await fetch(`/api/agents/${name}/action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ agent: name, action: 'test' })
            });

            if (!response.ok) throw new Error('Agent test failed');

            const data = await response.json();
            this.showToast('success', 'Test Complete', data.message);
        } catch (error) {
            console.error('Agent test failed:', error);
            this.showToast('error', 'Test Failed', `Failed to test agent ${name}`);
        }
    }

    async testAllAgents() {
        const agents = Object.keys(this.config.agents).filter(name => this.config.agents[name].enabled);

        for (const name of agents) {
            await this.testAgent(name);
        }
    }

    async resetAgentCircuit(name) {
        try {
            const response = await fetch(`/api/agents/${name}/action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ agent: name, action: 'reset_circuit_breaker' })
            });

            if (!response.ok) throw new Error('Failed to reset circuit breaker');

            const data = await response.json();
            this.config.agents[name] = data.agent_status;
            this.updateAgentsUI();
            this.showToast('success', 'Success', data.message);
        } catch (error) {
            console.error('Failed to reset circuit breaker:', error);
            this.showToast('error', 'Error', 'Failed to reset circuit breaker');
        }
    }

    toggleRoleAssignment(role, agent) {
        if (!this.config.agent_assignment) {
            this.config.agent_assignment = {};
        }
        if (!this.config.agent_assignment[role]) {
            this.config.agent_assignment[role] = [];
        }

        const list = this.config.agent_assignment[role];
        const index = list.indexOf(agent);

        if (index >= 0) {
            list.splice(index, 1);
        } else {
            list.push(agent);
        }

        this.updateRoleSelectors();
    }

    // === Workflow Actions ===

    openNewWorkflowModal() {
        document.getElementById('newWorkflowModal').classList.add('active');
    }

    closeModal() {
        document.getElementById('newWorkflowModal').classList.remove('active');
        document.getElementById('workflowPrompt').value = '';
        document.getElementById('projectPath').value = '';
        document.getElementById('dryRun').checked = false;
        document.getElementById('planOnly').checked = false;
    }

    // === Directory Browser ===

    async openDirectoryBrowser(target = 'main') {
        this.directoryBrowserTarget = target;
        this.selectedDirectoryPath = null;
        document.getElementById('directoryBrowserModal').classList.add('active');
        document.getElementById('selectDirectoryBtn').disabled = true;
        document.getElementById('selectedPath').textContent = 'None';

        // Start from current path based on target
        const pathInput = target === 'main'
            ? document.getElementById('mainProjectPath')
            : document.getElementById('projectPath');
        const currentPath = pathInput?.value || '~';
        await this.browseDirectory(currentPath);
    }

    closeDirectoryBrowser() {
        document.getElementById('directoryBrowserModal').classList.remove('active');
        this.selectedDirectoryPath = null;
        this.directoryBrowserTarget = null;
    }

    async browseDirectory(path) {
        try {
            const response = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
            if (!response.ok) throw new Error('Failed to browse directory');

            const data = await response.json();
            this.renderDirectoryBrowser(data);
        } catch (error) {
            console.error('Failed to browse directory:', error);
            this.showToast('error', 'Error', 'Failed to browse directory');
        }
    }

    renderDirectoryBrowser(data) {
        // Render breadcrumb
        const breadcrumb = document.getElementById('pathBreadcrumb');
        const pathParts = data.current_path.split(/[/\\]/).filter(p => p);

        let breadcrumbHtml = '';
        let cumulativePath = data.current_path.startsWith('/') ? '/' : '';

        // Add home shortcut
        breadcrumbHtml += `
            <span class="breadcrumb-item" data-path="~">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                    <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/>
                </svg>
            </span>
            <span class="breadcrumb-separator">/</span>
        `;

        pathParts.forEach((part, index) => {
            cumulativePath += (cumulativePath.endsWith('/') || cumulativePath.endsWith('\\') ? '' : '/') + part;
            const isLast = index === pathParts.length - 1;

            breadcrumbHtml += `
                <span class="breadcrumb-item ${isLast ? 'current' : ''}" data-path="${cumulativePath}">${part}</span>
                ${!isLast ? '<span class="breadcrumb-separator">/</span>' : ''}
            `;
        });

        breadcrumb.innerHTML = breadcrumbHtml;

        // Add click handlers to breadcrumb items
        breadcrumb.querySelectorAll('.breadcrumb-item').forEach(item => {
            item.addEventListener('click', () => {
                this.browseDirectory(item.dataset.path);
            });
        });

        // Render directory list
        const directoryList = document.getElementById('directoryList');

        if (data.entries.length === 0) {
            directoryList.innerHTML = `
                <div class="directory-item" style="justify-content: center; color: var(--text-muted);">
                    <span>No subdirectories found</span>
                </div>
            `;
            return;
        }

        let listHtml = '';

        // Add parent directory option
        if (data.parent_path) {
            listHtml += `
                <div class="directory-item" data-path="${data.parent_path}" data-action="navigate">
                    <svg class="folder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                        <path d="M15 18l-6-6 6-6"/>
                    </svg>
                    <span class="folder-name">..</span>
                    <span class="folder-badge">Parent</span>
                </div>
            `;
        }

        data.entries.forEach(entry => {
            const iconClass = entry.project_type || (entry.is_git_repo ? 'git' : '');
            const badge = entry.project_type ?
                `<span class="folder-badge ${entry.project_type}">${entry.project_type}</span>` :
                (entry.is_git_repo ? '<span class="folder-badge git">git</span>' : '');

            listHtml += `
                <div class="directory-item" data-path="${entry.path}" data-name="${entry.name}" data-type="${entry.project_type || ''}">
                    <svg class="folder-icon ${iconClass}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                        <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/>
                    </svg>
                    <span class="folder-name">${entry.name}</span>
                    ${badge}
                    <svg class="go-into" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                        <path d="M9 18l6-6-6-6"/>
                    </svg>
                </div>
            `;
        });

        directoryList.innerHTML = listHtml;

        // Add click handlers
        directoryList.querySelectorAll('.directory-item').forEach(item => {
            item.addEventListener('click', (e) => {
                if (item.dataset.action === 'navigate') {
                    // Navigate to parent
                    this.browseDirectory(item.dataset.path);
                } else {
                    // Select or navigate
                    const isDoubleClick = item.classList.contains('selected');

                    // Clear previous selection
                    directoryList.querySelectorAll('.directory-item').forEach(i => i.classList.remove('selected'));

                    if (isDoubleClick) {
                        // Double click - navigate into
                        this.browseDirectory(item.dataset.path);
                    } else {
                        // Single click - select
                        item.classList.add('selected');
                        this.selectedDirectoryPath = item.dataset.path;
                        document.getElementById('selectedPath').textContent = item.dataset.path;
                        document.getElementById('selectDirectoryBtn').disabled = false;
                    }
                }
            });

            // Also handle double click
            item.addEventListener('dblclick', () => {
                if (item.dataset.action !== 'navigate') {
                    this.browseDirectory(item.dataset.path);
                }
            });
        });
    }

    selectDirectory() {
        if (!this.selectedDirectoryPath) return;

        // Update the correct input based on target
        const target = this.directoryBrowserTarget || 'main';
        const pathInput = target === 'main'
            ? document.getElementById('mainProjectPath')
            : document.getElementById('projectPath');

        if (pathInput) {
            pathInput.value = this.selectedDirectoryPath;
        }

        // Also sync both inputs for convenience
        const mainPath = document.getElementById('mainProjectPath');
        const modalPath = document.getElementById('projectPath');
        if (mainPath) mainPath.value = this.selectedDirectoryPath;
        if (modalPath) modalPath.value = this.selectedDirectoryPath;

        this.closeDirectoryBrowser();

        // Update project info display
        this.detectProject(this.selectedDirectoryPath, target);
    }

    async detectProject(path, target = 'main') {
        try {
            const response = await fetch(`/api/project/detect?path=${encodeURIComponent(path)}`);
            if (!response.ok) return;

            const data = await response.json();

            // Update both main and modal project info displays
            const infos = [
                document.getElementById('mainProjectInfo'),
                document.getElementById('projectInfo')
            ].filter(el => el);

            infos.forEach(projectInfo => {
                if (data.project_type || data.is_git_repo) {
                    const typeClass = data.project_type || (data.is_git_repo ? 'git' : '');
                    const typeName = data.project_type || (data.is_git_repo ? 'Git Repo' : '');

                    projectInfo.innerHTML = `
                        <span class="project-type ${typeClass}">${typeName}</span>
                        <span class="project-name">${data.name}</span>
                    `;
                    projectInfo.style.display = 'flex';
                } else {
                    projectInfo.style.display = 'none';
                }
            });
        } catch (error) {
            console.error('Failed to detect project:', error);
        }
    }

    async startWorkflow() {
        const prompt = document.getElementById('workflowPrompt').value.trim();
        if (!prompt) {
            this.showToast('warning', 'Required', 'Please enter a task description');
            return;
        }

        try {
            const response = await fetch('/api/workflow/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt,
                    project_path: document.getElementById('projectPath').value || null,
                    dry_run: document.getElementById('dryRun').checked,
                    plan_only: document.getElementById('planOnly').checked
                })
            });

            if (!response.ok) throw new Error('Failed to start workflow');

            const data = await response.json();
            this.closeModal();
            this.showToast('success', 'Workflow Started', `ID: ${data.workflow_id.slice(0, 8)}...`);

            // Navigate to monitoring
            this.navigateTo('monitoring');
        } catch (error) {
            console.error('Failed to start workflow:', error);
            this.showToast('error', 'Error', 'Failed to start workflow');
        }
    }

    // === Metrics ===

    async loadMetrics() {
        try {
            const response = await fetch('/api/metrics');
            if (!response.ok) throw new Error('Failed to load metrics');

            const metrics = await response.json();
            this.updateMetricsUI(metrics);
        } catch (error) {
            console.error('Failed to load metrics:', error);
        }
    }

    updateMetricsUI(metrics) {
        document.getElementById('activeWorkflows').textContent = metrics.active_workflows || 0;
        document.getElementById('completedWorkflows').textContent = metrics.completed_workflows || 0;
        document.getElementById('failedWorkflows').textContent = metrics.failed_workflows || 0;

        // Update agent metrics table
        const tbody = document.querySelector('#agentMetricsTable tbody');
        if (tbody && metrics.agents) {
            tbody.innerHTML = Object.entries(metrics.agents).map(([name, agent]) => `
                <tr>
                    <td><strong>${this.config.agents[name]?.display_name || name}</strong></td>
                    <td>${agent.invocations || 0}</td>
                    <td>${agent.invocations > 0 ? Math.round((agent.successes / agent.invocations) * 100) : 0}%</td>
                    <td>${agent.avg_duration_seconds?.toFixed(1) || 0}s</td>
                    <td>
                        <span class="circuit-status ${agent.circuit_breaker_state}">
                            ${agent.circuit_breaker_state}
                        </span>
                    </td>
                    <td>
                        <button class="btn btn-sm btn-secondary" onclick="dashboard.resetAgentCircuit('${name}')">Reset</button>
                    </td>
                </tr>
            `).join('');
        }
    }

    // === WebSocket ===

    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.wsReconnectAttempts = 0;
                this.updateConnectionStatus(true);
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                this.scheduleReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            this.ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleWebSocketMessage(message);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this.updateConnectionStatus(false);
        }
    }

    scheduleReconnect() {
        if (this.wsReconnectAttempts >= this.maxReconnectAttempts) {
            console.log('Max reconnection attempts reached');
            return;
        }

        const delay = Math.min(1000 * Math.pow(2, this.wsReconnectAttempts), 30000);
        this.wsReconnectAttempts++;

        console.log(`Reconnecting in ${delay}ms (attempt ${this.wsReconnectAttempts})`);
        setTimeout(() => this.setupWebSocket(), delay);
    }

    updateConnectionStatus(connected) {
        const status = document.getElementById('connectionStatus');
        if (status) {
            const dot = status.querySelector('.status-dot');
            const text = status.querySelector('span:last-child');

            if (connected) {
                dot.classList.remove('disconnected');
                dot.classList.add('connected');
                text.textContent = 'Connected';
            } else {
                dot.classList.remove('connected');
                dot.classList.add('disconnected');
                text.textContent = 'Disconnected';
            }
        }
    }

    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'initial_state':
                this.config = message.data.config;
                this.workflows = message.data.workflows;
                this.updateUI();
                break;

            case 'config_updated':
            case 'config_reset':
            case 'config_imported':
                this.config = message.data;
                this.updateUI();
                break;

            case 'agent_updated':
            case 'agent_action':
                this.loadConfig().then(() => this.updateUI());
                break;

            case 'workflow_started':
            case 'workflow_updated':
                this.workflows[message.data.workflow_id] = message.data;
                this.updateWorkflowsList();
                this.loadMetrics();
                break;

            case 'workflow_cancelled':
                if (this.workflows[message.data.workflow_id]) {
                    this.workflows[message.data.workflow_id].current_phase = 'failed';
                }
                this.updateWorkflowsList();
                this.loadMetrics();
                break;

            case 'heartbeat':
                // Keep-alive, no action needed
                break;

            default:
                console.log('Unknown WebSocket message type:', message.type);
        }

        // Log event
        this.logEvent(message);
    }

    updateWorkflowsList() {
        const list = document.getElementById('activeWorkflowsList');
        if (!list) return;

        const activeWorkflows = Object.values(this.workflows).filter(
            w => w.current_phase !== 'completed' && w.current_phase !== 'failed'
        );

        if (activeWorkflows.length === 0) {
            list.innerHTML = `
                <div class="empty-state">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="48" height="48">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12.01" y2="16"/>
                    </svg>
                    <p>No active workflows</p>
                </div>
            `;
            return;
        }

        list.innerHTML = activeWorkflows.map(w => {
            const phases = ['init', 'planning', 'reviewing', 'implementing', 'post_checks', 'completed'];
            const currentIdx = phases.indexOf(w.current_phase);
            const progress = Math.round((currentIdx / (phases.length - 1)) * 100);

            return `
                <div class="workflow-item">
                    <div class="workflow-info">
                        <div class="workflow-id">${w.workflow_id.slice(0, 8)}...</div>
                        <div class="workflow-prompt">${w.prompt.slice(0, 60)}${w.prompt.length > 60 ? '...' : ''}</div>
                    </div>
                    <span class="workflow-phase">${w.current_phase}</span>
                    <div class="workflow-progress">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${progress}%"></div>
                        </div>
                    </div>
                    <button class="btn btn-sm btn-danger" onclick="dashboard.cancelWorkflow('${w.workflow_id}')">Cancel</button>
                </div>
            `;
        }).join('');
    }

    async cancelWorkflow(workflowId) {
        if (!confirm('Are you sure you want to cancel this workflow?')) return;

        try {
            const response = await fetch(`/api/workflow/${workflowId}/cancel`, {
                method: 'POST'
            });

            if (!response.ok) throw new Error('Failed to cancel workflow');

            this.showToast('success', 'Success', 'Workflow cancelled');
        } catch (error) {
            console.error('Failed to cancel workflow:', error);
            this.showToast('error', 'Error', 'Failed to cancel workflow');
        }
    }

    logEvent(message) {
        const log = document.getElementById('eventLog');
        if (!log) return;

        const time = new Date(message.timestamp).toLocaleTimeString();
        const type = message.type.includes('error') || message.type.includes('fail') ? 'error' :
                     message.type.includes('success') || message.type.includes('complete') ? 'success' :
                     message.type.includes('warning') ? 'warning' : 'info';

        const entry = document.createElement('div');
        entry.className = `log-entry ${type}`;
        entry.innerHTML = `
            <span class="log-time">${time}</span>
            <span class="log-message">${message.type}: ${JSON.stringify(message.data).slice(0, 100)}</span>
        `;

        log.insertBefore(entry, log.firstChild);

        // Keep only last 100 entries
        while (log.children.length > 100) {
            log.removeChild(log.lastChild);
        }
    }

    // === Toast Notifications ===

    showToast(type, title, message) {
        const container = document.getElementById('toastContainer');
        if (!container) return;

        const icons = {
            success: '<svg viewBox="0 0 24 24" fill="none" stroke="#3fb950" stroke-width="2" width="20" height="20"><polyline points="20 6 9 17 4 12"/></svg>',
            error: '<svg viewBox="0 0 24 24" fill="none" stroke="#f85149" stroke-width="2" width="20" height="20"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
            warning: '<svg viewBox="0 0 24 24" fill="none" stroke="#d29922" stroke-width="2" width="20" height="20"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
            info: '<svg viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" width="20" height="20"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
        };

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <span class="toast-icon">${icons[type]}</span>
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close" onclick="this.parentElement.remove()">&times;</button>
        `;

        container.appendChild(toast);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            toast.remove();
        }, 5000);
    }
}

// Initialize dashboard
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new Dashboard();
});
