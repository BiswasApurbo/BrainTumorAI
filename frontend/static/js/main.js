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
    
    if (!flairFile || !t1File || !t1ceFile || !t2File) {
        showError('Please select all four MRI modalities (FLAIR, T1, T1CE, T2).');
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
        uploadFormData.append('file', flairFile); 
        uploadFormData.append('t1', t1File);
        uploadFormData.append('t1ce', t1ceFile);
        uploadFormData.append('t2', t2File);
        
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
            body: JSON.stringify({ upload_id: uploadId })
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
        document.getElementById('res-request-id').textContent = inferenceData.request_id || 'N/A';
        document.getElementById('res-report-id').textContent = inferenceData.report_id || 'N/A';
        
        const filesList = document.getElementById('res-files-list');
        filesList.innerHTML = ''; // clear previous
        
        const addFileToList = (name, path) => {
            if (path) {
                const li = document.createElement('li');
                li.innerHTML = `<strong>${name}:</strong> ${path}`;
                filesList.appendChild(li);
            }
        };
        
        addFileToList('Visualization', inferenceData.visualization_path);
        addFileToList('Tumor Mask', inferenceData.tumor_mask_path);
        addFileToList('Anatomy Mask', inferenceData.anatomy_mask_path);
        
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
