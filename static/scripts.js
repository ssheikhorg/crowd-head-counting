document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewContainer = document.getElementById('previewContainer');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorMessage = document.getElementById('errorMessage');
    const uploadForm = document.getElementById('uploadForm');

    // Constants
    const VALID_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
    const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB

    // Initialize
    setCurrentYear();
    setupEventListeners();

    function setCurrentYear() {
        document.getElementById('currentYear').textContent = new Date().getFullYear();
    }

    function setupEventListeners() {
        console.log('Setting up event listeners');
        // Drag and drop events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });

        dropArea.addEventListener('drop', handleDrop, false);
        fileInput.addEventListener('change', handleFileSelect);
        uploadForm.addEventListener('submit', handleFormSubmit);
    }

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        dropArea.classList.add('highlight');
    }

    function unhighlight() {
        dropArea.classList.remove('highlight');
    }

    function handleDrop(e) {
        console.log('File dropped');
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFileSelect() {
        console.log('File selected via input');
        handleFiles(this.files);
    }

    function handleFiles(files) {
        console.log('Handling files:', files);
        if (!files || files.length === 0) {
            console.log('No files selected');
            return;
        }

        const file = files[0];
        console.log('Selected file:', file.name, file.type, file.size);

        // Validate file
        if (!VALID_IMAGE_TYPES.includes(file.type)) {
            console.log('Invalid file type:', file.type);
            showError('Please select a JPEG, PNG, or WebP image');
            return;
        }

        if (file.size > MAX_FILE_SIZE) {
            console.log('File too large:', file.size);
            showError('Image must be smaller than 5MB');
            return;
        }

        const reader = new FileReader();

        reader.onloadstart = () => {
            console.log('File reading started');
            loadingSpinner.style.display = 'block';
        };

        reader.onload = (e) => {
            console.log('File read successfully');
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block'; // Override styles.css display: none
            previewContainer.style.display = 'block';
            analyzeBtn.disabled = false;
            console.log('Button visibility:', analyzeBtn.style.display);
            loadingSpinner.style.display = 'none';
        };

        reader.onerror = () => {
            console.log('Error reading file');
            showError('Error reading file');
            loadingSpinner.style.display = 'none';
        };

        reader.readAsDataURL(file);
    }

    async function handleFormSubmit(e) {
        e.preventDefault();
        console.log('Form submitted');

        if (!fileInput.files.length) {
            console.log('No file selected for submission');
            showError('Please select an image first');
            return;
        }

        const formData = new FormData(uploadForm);
        console.log('FormData prepared for submission');

        // UI updates
        loadingSpinner.style.display = 'block';
        errorMessage.style.display = 'none';
        analyzeBtn.disabled = true;

        try {
            console.log('Submitting image to /predict');
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
                headers: {
                    'Accept': 'application/json'
                }
            });

            console.log('Received response:', response.status, response.statusText);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                console.log('Error response data:', errorData);
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }

            const data = await response.json();
            console.log('Processing results:', data);

            if (!data.success) {
                console.log('Processing failed:', data.error);
                throw new Error(data.error || 'Image processing failed');
            }

            // Build redirect URL
            const params = new URLSearchParams({
                count: data.count,
                image: encodeURIComponent(data.image_url),
                density: data.density,
                confidence: data.avg_confidence,
                processing_time: data.processing_time
            });

            console.log('Redirecting to:', `/results?${params.toString()}`);
            window.location.href = `/results?${params.toString()}`;

        } catch (error) {
            console.error('Submission error:', error);
            showError(error.message || 'Failed to process image');
        } finally {
            loadingSpinner.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    }

    function showError(message) {
        console.log('Showing error:', message);
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';

        setTimeout(() => {
            errorMessage.style.display = 'none';
        }, 5000);
    }
});