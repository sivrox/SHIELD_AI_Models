CREATE TABLE analyzed_reports (
    report_id SERIAL PRIMARY KEY,
    patient_id INTEGER NOT NULL,
    file_name TEXT NOT NULL,
    analysis TEXT NOT NULL,
    severity_level TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
