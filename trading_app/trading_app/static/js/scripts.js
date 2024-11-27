function downloadData() {
    var ticker = $('#ticker').val();  // Get selected ticker
    $.ajax({
        url: '/download_data',  // Backend endpoint
        type: 'POST',
        data: { ticker: ticker },  // Send selected ticker to backend
        success: function (response) {
            console.log('Data downloaded successfully:', response);
            // trainModel();  
            alert('Data is downloaded');
        },
        error: function (xhr, status, error) {
            console.error('Error:', status, error);
        }
    });
}

function trainModel() {
    $.ajax({
        url: '/train_model',  // Backend endpoint for training model
        type: 'POST',
        success: function (response) {
            console.log(response.success);  // Log success status
            console.log(response.message);  // Log server message
            alert('Model is trained');
        },
        error: function (xhr, status, error) {
            console.error('Error:', status, error);
        }
    });
}

function visualizeData() {
    $.ajax({
        url: '/visualize_data',  // Backend endpoint for visualizing data
        type: 'POST',
        success: function (response) {
            console.log(response.success);  // Log success status
            console.log(response.message);  // Log server message
        },
        error: function (xhr, status, error) {
            console.error('Error:', status, error);
        }
    });
}


// Function to parse CSV and return data in required format
function parseCSVData(csv) {
    const parsedData = [];
    const rows = csv.split("\n").slice(1); // Skip header row

    rows.forEach(row => {
        const columns = row.split(",");
        if (columns.length === 6) { // Ensure the row has the correct number of columns
            const timestamp = new Date(columns[0]);
            const open = parseFloat(columns[1]);
            const high = parseFloat(columns[2]);
            const low = parseFloat(columns[3]);
            const close = parseFloat(columns[4]);
            parsedData.push({ x: timestamp, y: [open, high, low, close] });
        }
    });

    return parsedData;
}

// Function to load CSV file based on selected stock
function loadCSV(stockFile) {
    const path = `../static/csv/${stockFile}`;
    fetch(path)
        .then(response => response.text())
        .then(csv => {
            const data = parseCSVData(csv);
            chart.updateSeries([{ data }]);
        })
        .catch(error => console.error("Error loading CSV file:", error));
}

let options = {
    series: [{
        data: []
    }],
    chart: {
        type: 'candlestick',
        height: 350
    },
    title: {
        text: 'CandleStick Chart',
        align: 'left'
    },
    xaxis: {
        type: 'datetime'
    },
    yaxis: {
        tooltip: {
            enabled: true
        }
    }
};

// Render the initial chart with empty data
let chart = new ApexCharts(document.querySelector("#chart"), options);
chart.render();

// Event listener for dropdown change
document.getElementById('stock-dropdown').addEventListener('change', function () {
    const selectedStock = this.value;
    loadCSV(selectedStock);
});

// Load initial data for the first stock in the dropdown
loadCSV(document.getElementById('stock-dropdown').value);
