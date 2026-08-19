async function runAnalysis() {
    const btn = document.querySelector('.analyze-btn');
    const statusMsg = document.getElementById('status-message');
    const processingContainer = document.getElementById('processing-container');
    const resultsContainer = document.getElementById('results-container');
    
    // Hide previous messages
    statusMsg.className = 'status-message hidden';
    resultsContainer.classList.add('hidden');
    
    // Get files
    const flairFile = document.getElementById('flair').files[0];
    const t1File = document.getElementById('t1').files[0];
    const t1ceFile = document.getElementById('t1ce').files[0];
    const t2File = document.getElementById('t2').files[0];
    const segFile = document.getElementById('seg').files[0];
    const mode = document.querySelector('input[name="mode"]:checked').value;
    
    if (!flairFile || !t1File || !t1ceFile || !t2File) {
        showError('Please select all four MRI modalities (FLAIR, T1, T1CE, T2).');
        return;
    }
    
    if (mode === 'ground_truth' && !segFile) {
        showError('Please select a SEG (Ground Truth) file for Ground Truth mode.');
        return;
    }
    
    // Update UI for processing state
    btn.disabled = true;
    processingContainer.classList.remove('hidden');
    
    try {
        // STEP 1 & 2: Collect files and POST to /api/v1/upload
        // The backend strictly expects exactly 1 file attached to the "file" field.
        // We append the primary sequence to "file" and others are sent in the payload.
        const uploadFormData = new FormData();
        uploadFormData.append('mode', mode);
        uploadFormData.append('flair', flairFile); 
        uploadFormData.append('t1', t1File);
        uploadFormData.append('t1ce', t1ceFile);
        uploadFormData.append('t2', t2File);
        if (mode === 'ground_truth' && segFile) {
            uploadFormData.append('seg', segFile);
        }
        
        const uploadResponse = await fetch('/api/v1/upload', {
            method: 'POST',
            body: uploadFormData
        });
        
        if (!uploadResponse.ok) {
            const err = await uploadResponse.json();
            throw new Error(err.detail || 'Failed to upload files.');
        }
        
        // STEP 3: Receive upload response and extract upload_id
        const uploadData = await uploadResponse.json();
        const uploadId = uploadData.upload_id;
        
        // STEP 4: POST /api/v1/inference using the returned upload_id
        const inferenceResponse = await fetch('/api/v1/inference', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ upload_id: uploadId, mode: mode })
        });
        
        if (!inferenceResponse.ok) {
            const err = await inferenceResponse.json();
            throw new Error(err.detail || 'Inference failed.');
        }
        
        // STEP 6: When inference finishes, display success message and metadata
        const inferenceData = await inferenceResponse.json();
        
        processingContainer.classList.add('hidden');
        showSuccess(inferenceData.message || 'Analysis completed successfully!');
        
        // Display returned metadata
        const modeLabel = mode === 'ground_truth' ? 'Ground Truth' : 'AI Prediction';
        const sourceLabel = mode === 'ground_truth' ? 'Uploaded SEG Mask' : 'nnUNet Prediction';
        document.getElementById('res-mode').textContent = modeLabel;
        document.getElementById('res-source').textContent = sourceLabel;
        
        document.getElementById('res-request-id').textContent = inferenceData.request_id || 'N/A';
        document.getElementById('res-report-id').textContent = inferenceData.report_id || 'N/A';
        
        // Configure Download Report Button
        const downloadBtn = document.getElementById('download-report-btn');
        if (inferenceData.report_id && downloadBtn) {
            const downloadUrl = `/api/v1/reports/${inferenceData.report_id}/download`;
            downloadBtn.href = downloadUrl;
            downloadBtn.className = 'download-btn';
            downloadBtn.textContent = 'Download Report';
            downloadBtn.onclick = function() {
                setTimeout(() => {
                    downloadBtn.className = 'download-btn disabled';
                    downloadBtn.textContent = 'Report Downloaded & Purged';
                    downloadBtn.removeAttribute('href');
                    downloadBtn.onclick = (e) => e.preventDefault();
                }, 1000);
            };
        }
        
        resultsContainer.classList.remove('hidden');
        
    } catch (error) {
        processingContainer.classList.add('hidden');
        showError(error.message || 'An unexpected error occurred.');
    } finally {
        btn.disabled = false;
    }
}

function showError(message) {
    const statusDiv = document.getElementById('status-message');
    statusDiv.textContent = message;
    statusDiv.className = 'status-message error';
}

function showSuccess(message) {
    const statusDiv = document.getElementById('status-message');
    statusDiv.textContent = message;
    statusDiv.className = 'status-message success';
}

function toggleSegInput() {
    const mode = document.querySelector('input[name="mode"]:checked').value;
    const segGroup = document.getElementById('seg-group');
    if (mode === 'ground_truth') {
        segGroup.classList.remove('hidden');
    } else {
        segGroup.classList.add('hidden');
    }
}
