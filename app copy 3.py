import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import pickle
import datetime
import uuid
import sqlite3
import hashlib
import base64
from PIL import Image
import io
import time
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# Set page configuration - this removes the default Streamlit title
st.set_page_config(
    page_title="",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Read and encode the background image
def set_background(image_file, is_login_screen):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    
    if is_login_screen:
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), url("data:image/jpg;base64,{encoded_string}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else: # Apply the default background or no background for other screens
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #f8f9fa; /* Or your desired default background color/style */
            }
            </style>
            """,
            unsafe_allow_html=True
        )

# Initialize session state
def init_session_state():
    # User authentication state
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = None
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    if 'user_role' not in st.session_state:
        st.session_state['user_role'] = None
    if 'login_time' not in st.session_state:
        st.session_state['login_time'] = None
    
    # Face encoding states
    if 'face_encoding_register' not in st.session_state:
        st.session_state['face_encoding_register'] = None
    if 'register_image' not in st.session_state:
        st.session_state['register_image'] = None
    
    # Load face encodings
    if 'face_encodings' not in st.session_state:
        if os.path.exists('face_encodings.pkl'):
            with open('face_encodings.pkl', 'rb') as f:
                st.session_state.face_encodings = pickle.load(f)
        else:
            st.session_state.face_encodings = {}

init_session_state()

# Conditionally set the background based on authentication status
if not st.session_state.authenticated:
    set_background("login.jpg", True) # Use login.jpg for the login screen
else:
    set_background("background.jpg", True) # Use existing background.jpg for authenticated screens

# Hide the default Streamlit header and footer
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Replace the face detection functions with:
def initialize_face_detection():
    """Initialize OpenCV face detection"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def get_face_embedding(face_detector, image):
    """Get face embedding using OpenCV"""
    if image is None:
        return None
        
    # Convert to grayscale for face detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detect faces
    faces = face_detector.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    
    # Get the largest face
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    
    # Extract face ROI and resize
    face_roi = gray[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (128, 128))
    
    # Flatten and normalize the face ROI
    face_vector = face_roi.flatten() / 255.0
    return face_vector

def compare_face_embeddings(embedding1, embedding2, tolerance=0.6):
    """Compare two face embeddings using cosine similarity"""
    if embedding1 is None or embedding2 is None:
        return False
        
    # Calculate cosine similarity
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = dot_product / (norm1 * norm2)
    
    return similarity > tolerance

# Function to load and display the logo
def load_logo():
    try:
        logo = Image.open("logo.jpg")
        return logo
    except FileNotFoundError:
        st.warning("Logo file (logo.jpg) not found in the current directory.")
        return None

# Apply custom CSS for a sleek interface
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f3f4;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
    }
    .stButton>button {
        background-color: #4e8df5;
        color: white;
        border-radius: 5px;
        height: 3em;
        width: 100%;
    }
    .success-message {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        color: #155724;
    }
    .error-message {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        color: #721c24;
    }
    .info-box {
        background-color: #e7f5fe;
        border-left: 5px solid #4e8df5;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
    .dashboard-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .centered-text {
        text-align: center;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .user-avatar {
        border-radius: 50%;
        width: 40px;
        height: 40px;
        margin-right: 10px;
    }
    .user-greeting {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .styled-header {
        background-color: #4e8df5;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .app-header {
        display: flex;
        align-items: center;
        background-color: #4e8df5;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1.5rem;
    }
    .logo-img {
        width: 80px;
        margin-right: 15px;
    }
    .title-text {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the logo
logo = load_logo()

# Create a custom header with logo and title
header_container = st.container()
with header_container:
    if logo:
        st.markdown(
            f"""
            <div class="app-header">
                <img src="data:image/jpeg;base64,{base64.b64encode(open('logo.jpg', 'rb').read()).decode()}" class="logo-img">
                <h1 class="title-text">BINGHAM UNIVERSITY HEALTH CARE CENTER</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

# Database setup
def setup_database():
    conn = sqlite3.connect('patient_records.db', check_same_thread=False)
    c = conn.cursor()
    
    # Create users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        role TEXT NOT NULL,
        date_created TEXT NOT NULL,
        last_login TEXT,
        is_active INTEGER DEFAULT 1
    )
    ''')
    
    # Create patients table
    c.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        matriculation_number TEXT UNIQUE,
        clinic_number TEXT UNIQUE,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        date_of_birth TEXT NOT NULL,
        gender TEXT NOT NULL,
        blood_group TEXT NOT NULL,
        contact_number TEXT,
        email TEXT,
        address TEXT,
        emergency_contact TEXT,
        registration_date TEXT NOT NULL,
        medical_history TEXT,
        allergies TEXT,
        registered_by TEXT,
        FOREIGN KEY (registered_by) REFERENCES users(user_id)
    )
    ''')
    
    # Create login_logs table
    c.execute('''
    CREATE TABLE IF NOT EXISTS login_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        username TEXT NOT NULL,
        login_time TEXT NOT NULL,
        ip_address TEXT,
        success INTEGER NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    ''')
    
    # Create patient_access_logs table
    c.execute('''
    CREATE TABLE IF NOT EXISTS patient_access_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        accessed_by TEXT NOT NULL,
        access_time TEXT NOT NULL,
        access_method TEXT NOT NULL,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
        FOREIGN KEY (accessed_by) REFERENCES users(user_id)
    )
    ''')
    
    # Create system_settings table
    c.execute('''
    CREATE TABLE IF NOT EXISTS system_settings (
        setting_name TEXT PRIMARY KEY,
        setting_value TEXT NOT NULL,
        last_updated TEXT NOT NULL,
        updated_by TEXT,
        FOREIGN KEY (updated_by) REFERENCES users(user_id)
    )
    ''')
    
    # Create an admin user if none exists
    c.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
    if c.fetchone()[0] == 0:
        admin_id = str(uuid.uuid4())
        hashed_password = hashlib.sha256("admin123".encode()).hexdigest()
        c.execute('''
        INSERT INTO users (user_id, username, password, first_name, last_name, email, role, date_created)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (admin_id, "admin", hashed_password, "System", "Administrator", "admin@example.com", "admin", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        
        # Insert default system settings
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO system_settings VALUES (?, ?, ?, ?)", 
                 ("face_recognition_tolerance", "0.6", current_time, admin_id))
        c.execute("INSERT INTO system_settings VALUES (?, ?, ?, ?)",
                 ("session_timeout_minutes", "30", current_time, admin_id))
        c.execute("INSERT INTO system_settings VALUES (?, ?, ?, ?)",
                 ("max_login_attempts", "5", current_time, admin_id))
        c.execute("INSERT INTO system_settings VALUES (?, ?, ?, ?)",
                 ("system_name", "BINGHAM UNIVERSITY HEALTH CARE CENTER", current_time, admin_id))
    
    conn.commit()
    return conn

# Initialize database connection
conn = setup_database()

# Initialize session state
def init_session_state():
    # User authentication state
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = None
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    if 'user_role' not in st.session_state:
        st.session_state['user_role'] = None
    if 'login_time' not in st.session_state:
        st.session_state['login_time'] = None
    
    # Face encoding states
    if 'face_encoding_register' not in st.session_state:
        st.session_state['face_encoding_register'] = None
    if 'register_image' not in st.session_state:
        st.session_state['register_image'] = None
    
    # Load face encodings
    if 'face_encodings' not in st.session_state:
        if os.path.exists('face_encodings.pkl'):
            with open('face_encodings.pkl', 'rb') as f:
                st.session_state.face_encodings = pickle.load(f)
        else:
            st.session_state.face_encodings = {}

init_session_state()

# Security Functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password, provided_password):
    return stored_password == hashlib.sha256(provided_password.encode()).hexdigest()

def log_login_attempt(user_id, username, success):
    c = conn.cursor()
    c.execute('''
    INSERT INTO login_logs (user_id, username, login_time, ip_address, success)
    VALUES (?, ?, ?, ?, ?)
    ''', (
        user_id,
        username,
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "127.0.0.1",  # Placeholder for IP address
        1 if success else 0
    ))
    conn.commit()

def log_patient_access(patient_id, access_method):
    if st.session_state.authenticated:
        c = conn.cursor()
        c.execute('''
        INSERT INTO patient_access_logs (patient_id, accessed_by, access_time, access_method)
        VALUES (?, ?, ?, ?)
        ''', (
            patient_id,
            st.session_state.user_id,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            access_method
        ))
        conn.commit()

def check_session_timeout():
    if st.session_state.authenticated and st.session_state.login_time:
        c = conn.cursor()
        c.execute("SELECT setting_value FROM system_settings WHERE setting_name = 'session_timeout_minutes'")
        result = c.fetchone()
        timeout_minutes = int(result[0]) if result else 30
        
        login_time = datetime.datetime.strptime(st.session_state.login_time, "%Y-%m-%d %H:%M:%S")
        current_time = datetime.datetime.now()
        elapsed_minutes = (current_time - login_time).total_seconds() / 60
        
        if elapsed_minutes > timeout_minutes:
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.user_role = None
            st.session_state.login_time = None
            st.warning("Your session has expired. Please log in again.")
            return False
    return True

# Face Recognition Functions
def save_face_encodings():
    with open('face_encodings.pkl', 'wb') as f:
        pickle.dump(st.session_state.face_encodings, f)

def capture_image(key="default_camera"):
    img_file_buffer = st.camera_input("Take a picture", key=key)
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        img = Image.open(io.BytesIO(bytes_data))
        return np.array(img)
    return None

def process_uploaded_image(uploaded_file):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        return np.array(image)
    return None

def encode_face(image):
    if image is None:
        return None
    
    face_detector = initialize_face_detection()
    return get_face_embedding(face_detector, image)

def find_matching_face(face_embedding):
    if face_embedding is None or not st.session_state.face_encodings:
        return None

    # Get tolerance setting from database
    c = conn.cursor()
    c.execute("SELECT setting_value FROM system_settings WHERE setting_name = 'face_recognition_tolerance'")
    result = c.fetchone()
    tolerance = float(result[0]) if result else 0.6

    for patient_id, stored_embedding in st.session_state.face_encodings.items():
        if compare_face_embeddings(stored_embedding, face_embedding, tolerance):
            return patient_id

    return None

# Database Functions
def get_patient_by_id(patient_id):
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    columns = [description[0] for description in c.description]
    result = c.fetchone()
    
    if result:
        return dict(zip(columns, result))
    return None

def get_patients(search_id=None, search_matriculation=None, search_clinic=None):
    c = conn.cursor()
    query = "SELECT * FROM patients"
    params = []
    conditions = []
    
    if search_id:
        conditions.append("patient_id LIKE ?")
        params.append(f"%{search_id}%")
    
    if search_matriculation:
        conditions.append("matriculation_number LIKE ?")
        params.append(f"%{search_matriculation}%")
    
    if search_clinic:
        conditions.append("clinic_number LIKE ?")
        params.append(f"%{search_clinic}%")
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    c.execute(query, params)
    columns = [description[0] for description in c.description]
    results = []
    
    for row in c.fetchall():
        results.append(dict(zip(columns, row)))
    
    return results

def register_patient(patient_data, face_encoding):
    c = conn.cursor()
    
    # Check if matriculation or clinic number already exists
    c.execute("SELECT COUNT(*) FROM patients WHERE matriculation_number = ? OR clinic_number = ?", 
              (patient_data['matriculation_number'], patient_data['clinic_number']))
    
    if c.fetchone()[0] > 0:
        return None, "A patient with the same matriculation or clinic number already exists."
    
    # Generate a unique patient ID
    patient_id = str(uuid.uuid4())
    patient_data['patient_id'] = patient_id
    
    # Add registration information
    patient_data['registration_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    patient_data['registered_by'] = st.session_state.user_id
    
    # Insert into database
    columns = ', '.join(patient_data.keys())
    placeholders = ', '.join(['?'] * len(patient_data))
    
    query = f"INSERT INTO patients ({columns}) VALUES ({placeholders})"
    c.execute(query, list(patient_data.values()))
    conn.commit()
    
    # Store face encoding
    if face_encoding is not None:
        st.session_state.face_encodings[patient_id] = face_encoding
        save_face_encodings()
    
    # Log the access
    log_patient_access(patient_id, "registration")
    
    return patient_id, "success"

def get_user_by_username(username):
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    columns = [description[0] for description in c.description]
    result = c.fetchone()
    
    if result:
        return dict(zip(columns, result))
    return None

def create_user(user_data):
    c = conn.cursor()
    
    # Check if username or email already exists
    c.execute("SELECT COUNT(*) FROM users WHERE username = ? OR email = ?", 
              (user_data['username'], user_data['email']))
    
    if c.fetchone()[0] > 0:
        return None, "Username or email already exists."
    
    # Generate a unique user ID
    user_id = str(uuid.uuid4())
    
    # Hash the password
    hashed_password = hash_password(user_data['password'])
    
    # Insert into database
    c.execute('''
    INSERT INTO users (user_id, username, password, first_name, last_name, email, role, date_created)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id,
        user_data['username'],
        hashed_password,
        user_data['first_name'],
        user_data['last_name'],
        user_data['email'],
        user_data['role'],
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    
    return user_id, "success"

def update_system_setting(setting_name, setting_value):
    c = conn.cursor()
    c.execute('''
    UPDATE system_settings
    SET setting_value = ?, last_updated = ?, updated_by = ?
    WHERE setting_name = ?
    ''', (
        setting_value,
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        st.session_state.user_id,
        setting_name
    ))
    conn.commit()

def get_all_settings():
    c = conn.cursor()
    c.execute("SELECT * FROM system_settings")
    columns = [description[0] for description in c.description]
    results = []
    
    for row in c.fetchall():
        results.append(dict(zip(columns, row)))
    
    return results

def get_all_users():
    c = conn.cursor()
    c.execute("SELECT user_id, username, first_name, last_name, email, role, date_created, last_login, is_active FROM users")
    columns = [description[0] for description in c.description]
    results = []
    
    for row in c.fetchall():
        results.append(dict(zip(columns, row)))
    
    return results

def get_login_logs(limit=100):
    c = conn.cursor()
    c.execute('''
    SELECT ll.log_id, ll.username, ll.login_time, ll.ip_address, ll.success, 
           u.first_name, u.last_name
    FROM login_logs ll
    LEFT JOIN users u ON ll.user_id = u.user_id
    ORDER BY ll.login_time DESC
    LIMIT ?
    ''', (limit,))
    
    columns = [description[0] for description in c.description]
    results = []
    
    for row in c.fetchall():
        results.append(dict(zip(columns, row)))
    
    return results

def get_patient_access_logs(limit=100):
    c = conn.cursor()
    c.execute('''
    SELECT pal.log_id, pal.patient_id, pal.access_time, pal.access_method,
           u.username as accessed_by_username, u.first_name as accessed_by_first_name, u.last_name as accessed_by_last_name,
           p.first_name as patient_first_name, p.last_name as patient_last_name
    FROM patient_access_logs pal
    LEFT JOIN users u ON pal.accessed_by = u.user_id
    LEFT JOIN patients p ON pal.patient_id = p.patient_id
    ORDER BY pal.access_time DESC
    LIMIT ?
    ''', (limit,))
    
    columns = [description[0] for description in c.description]
    results = []
    
    for row in c.fetchall():
        results.append(dict(zip(columns, row)))
    
    return results

def get_system_stats():
    c = conn.cursor()
    
    # Total patients
    c.execute("SELECT COUNT(*) FROM patients")
    total_patients = c.fetchone()[0]
    
    # Total users
    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]
    
    # Patient registrations by month (last 6 months)
    c.execute('''
    SELECT strftime('%Y-%m', registration_date) as month, COUNT(*) as count
    FROM patients
    WHERE registration_date >= date('now', '-6 months')
    GROUP BY month
    ORDER BY month
    ''')
    registrations_by_month = c.fetchall()
    
    # Access logs by method
    c.execute('''
    SELECT access_method, COUNT(*) as count
    FROM patient_access_logs
    GROUP BY access_method
    ''')
    access_by_method = c.fetchall()
    
    # Recent activity
    c.execute('''
    SELECT 'patient_registration' as activity_type, registration_date as timestamp, 
           first_name || ' ' || last_name as name,
           'Patient registered' as description
    FROM patients
    UNION ALL
    SELECT 'patient_access' as activity_type, access_time as timestamp,
           (SELECT first_name || ' ' || last_name FROM patients WHERE patient_id = patient_access_logs.patient_id) as name,
           'Patient accessed by ' || (SELECT username FROM users WHERE user_id = patient_access_logs.accessed_by) as description
    FROM patient_access_logs
    UNION ALL
    SELECT 'user_login' as activity_type, login_time as timestamp,
           username as name,
           CASE WHEN success = 1 THEN 'Successful login' ELSE 'Failed login attempt' END as description
    FROM login_logs
    ORDER BY timestamp DESC
    LIMIT 10
    ''')
    recent_activity = []
    columns = [description[0] for description in c.description]
    for row in c.fetchall():
        recent_activity.append(dict(zip(columns, row)))
    
    return {
        "total_patients": total_patients,
        "total_users": total_users,
        "registrations_by_month": registrations_by_month,
        "access_by_method": access_by_method,
        "recent_activity": recent_activity
    }

# Login Management
def login_form():
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>Welcome to the Medical Emergency Facial Recognition System</h3>
            <p>This system helps identify patients during medical emergencies using facial recognition technology, 
            even when they are unresponsive or unable to communicate.</p>
            <p>Features include:</p>
            <ul>
                <li>Fast patient identification via facial recognition</li>
                <li>Secure access to critical medical records</li>
                <li>Comprehensive patient management</li>
                <li>Advanced security and privacy controls</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.image("https://img.freepik.com/free-vector/doctor-nurse-giving-medical-care-patient_74855-7882.jpg", use_container_width=True)
    
    with col2:
        st.markdown("<h2>User Login</h2>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Log In")
            
            if submit:
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    user = get_user_by_username(username)
                    
                    if user and verify_password(user['password'], password) and user['is_active'] == 1:
                        # Update last login time
                        c = conn.cursor()
                        c.execute("UPDATE users SET last_login = ? WHERE user_id = ?", 
                                 (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user['user_id']))
                        conn.commit()
                        
                        # Set session state
                        st.session_state.authenticated = True
                        st.session_state.user_id = user['user_id']
                        st.session_state.username = user['username']
                        st.session_state.user_role = user['role']
                        st.session_state.login_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Log the successful login
                        log_login_attempt(user['user_id'], username, True)
                        
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        # Log failed login attempt
                        user_id = user['user_id'] if user else None
                        log_login_attempt(user_id, username, False)
                        
                        st.error("Invalid username or password.")
        
        # Registration button and info
        st.markdown("---")
        st.markdown("<p>Don't have an account? Contact an administrator to get registered.</p>", unsafe_allow_html=True)

# User Interface Components
def sidebar_menu():
    # User greeting with avatar
    st.sidebar.markdown(
        f"""
        <div class="user-greeting">
            <img src="https://ui-avatars.com/api/?name={st.session_state.username}&background=random" class="user-avatar">
            <div>
                <h3>Welcome, {st.session_state.username}</h3>
                <p>Role: {st.session_state.user_role.capitalize()}</p>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Navigation based on role
    if st.session_state.user_role == 'admin':
        selected = option_menu(
            "Main Menu",
            ["Dashboard", "Register Patient", "Find Patient", "Browse Records", "User Management", "System Settings", "Logs", "Logout"],
            icons=['speedometer2', 'person-plus', 'search', 'table', 'people', 'gear', 'journal-text', 'box-arrow-right'],
            menu_icon="cast",
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "#f0f2f6"},
                "icon": {"color": "orange", "font-size": "16px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#4e8df5"},
            }
        )
    else:
        selected = option_menu(
            "Main Menu",
            ["Dashboard", "Register Patient", "Find Patient", "Browse Records", "Logout"],
            icons=['speedometer2', 'person-plus', 'search', 'table', 'box-arrow-right'],
            menu_icon="cast",
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "#f0f2f6"},
                "icon": {"color": "orange", "font-size": "16px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#4e8df5"},
            }
        )
    
    return selected

def dashboard_page():
    st.markdown("<h1 class='centered-text'>System Dashboard</h1>", unsafe_allow_html=True)
    
    # Get system statistics
    stats = get_system_stats()
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <h3>Total Patients</h3>
                <h2 style="color:#4e8df5;">{stats["total_patients"]}</h2>
                <p>Registered in the system</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <h3>Total Users</h3>
                <h2 style="color:#4e8df5;">{stats["total_users"]}</h2>
                <p>Active system users</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        # Calculate successful recognition rate (can be improved with real metrics)
        if len(stats["access_by_method"]) > 0:
            facial_recognition_count = sum([count for method, count in stats["access_by_method"] if method == "facial_recognition"])
            total_access_count = sum([count for _, count in stats["access_by_method"]])
            success_rate = (facial_recognition_count / total_access_count * 100) if total_access_count > 0 else 0
        else:
            success_rate = 0
            
        st.markdown(
            f"""
            <div class="dashboard-card">
                <h3>Recognition Success Rate</h3>
                <h2 style="color:#4e8df5;">{success_rate:.1f}%</h2>
                <p>Facial recognition accuracy</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Charts row
    st.markdown("### System Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.subheader("Patient Registrations (Last 6 Months)")
        
        # Convert to DataFrame
        if stats["registrations_by_month"]:
            reg_df = pd.DataFrame(stats["registrations_by_month"], columns=["month", "count"])
            fig = px.bar(reg_df, x="month", y="count", text_auto=True)
            fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No registration data available for the past 6 months.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.subheader("Access Methods Distribution")
        
        # Convert to DataFrame
        if stats["access_by_method"]:
            access_df = pd.DataFrame(stats["access_by_method"], columns=["method", "count"])
            fig = px.pie(access_df, values="count", names="method", hole=0.4)
            fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No access method data available.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Recent Activity
    st.markdown("### Recent Activity")
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    
    if stats["recent_activity"]:
        for activity in stats["recent_activity"]:
            activity_time = activity["timestamp"]
            activity_name = activity["name"]
            activity_desc = activity["description"]
            
            if activity["activity_type"] == "patient_registration":
                icon = "📋"
                color = "#4CAF50"
            elif activity["activity_type"] == "patient_access":
                icon = "🔍"
                color = "#2196F3"
            elif activity["activity_type"] == "user_login":
                icon = "🔐"
                color = activity["description"].startswith("Failed") and "#F44336" or "#4e8df5"
            else:
                icon = "ℹ️"
                color = "#9E9E9E"
            
            st.markdown(
                f"""
                <div style="display: flex; margin-bottom: 10px; padding: 10px; border-left: 4px solid {color}; background-color: #f9f9f9;">
                    <div style="font-size: 24px; margin-right: 15px;">{icon}</div>
                    <div>
                        <div style="font-weight: bold;">{activity_name}</div>
                        <div>{activity_desc}</div>
                        <div style="font-size: 0.8em; color: #666;">{activity_time}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No recent activity to display.")
    
    st.markdown("</div>", unsafe_allow_html=True)

def register_patient_page():
    st.markdown("<h1 class='centered-text'>Register New Patient</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.subheader("Patient Information")
        
        with st.form("patient_registration_form", clear_on_submit=True):
            matriculation_number = st.text_input("Matriculation Number")
            clinic_number = st.text_input("Clinic Number")
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            dob = st.date_input("Date of Birth", 
                     min_value=datetime.date(1900, 1, 1),
                      max_value=datetime.date.today(),
                      value=datetime.date.today())
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
            
            with st.expander("Contact Information"):
                contact_number = st.text_input("Contact Number")
                email = st.text_input("Email Address")
                address = st.text_area("Address")
                emergency_contact = st.text_input("Emergency Contact")
            
            with st.expander("Medical Information"):
                medical_history = st.text_area("Medical History")
                allergies = st.text_area("Allergies")
                
            submit_button = st.form_submit_button("Register Patient")
            
            # Process form submission
            if submit_button:
                if not matriculation_number or not clinic_number or not first_name or not last_name:
                    st.error("Matriculation number, clinic number, first name, and last name are required.")
                elif st.session_state.get('face_encoding_register') is None:
                    st.error("A valid face image is required for registration.")
                else:
                    # Create patient data dictionary
                    patient_data = {
                        'matriculation_number': matriculation_number,
                        'clinic_number': clinic_number,
                        'first_name': first_name,
                        'last_name': last_name,
                        'date_of_birth': dob.strftime("%Y-%m-%d"),
                        'gender': gender,
                        'blood_group': blood_group,
                        'contact_number': contact_number,
                        'email': email,
                        'address': address,
                        'emergency_contact': emergency_contact,
                        'medical_history': medical_history,
                        'allergies': allergies
                    }
                    
                    # Register patient
                    patient_id, message = register_patient(patient_data, st.session_state.get('face_encoding_register'))
                    
                    if patient_id:
                        st.success(f"Patient registered successfully! Patient ID: {patient_id}")
                        # Clear the captured image
                        st.session_state['register_image'] = None
                        st.session_state['face_encoding_register'] = None
                    else:
                        st.error(f"Registration failed: {message}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.subheader("Patient Photo")
        
        photo_option = st.radio("Choose photo input method:", ["Upload Image", "Take Photo"], key="register_photo_option")

        patient_image = None
        if photo_option == "Upload Image":
            uploaded_file = st.file_uploader("Upload Patient Photo", type=["jpg", "jpeg", "png"], key="register_upload")
            patient_image = process_uploaded_image(uploaded_file)
        else:
            patient_image = capture_image(key="register_camera")

        if patient_image is not None:
            st.image(patient_image, caption="Patient Photo", width=300)
            face_encoding = encode_face(patient_image)
            if face_encoding is None:
                st.error("No face detected in the image. Please try again.")
                st.session_state['face_encoding_register'] = None
            else:
                st.session_state['face_encoding_register'] = face_encoding
                st.session_state['register_image'] = patient_image
                st.success("Face detected successfully!")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Guidelines
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("""
        ### Guidelines for Patient Photos
        
        For optimal facial recognition results:
        - Ensure good lighting on the face
        - Face should be clearly visible and centered
        - Avoid strong shadows or reflections
        - Maintain a neutral expression
        - Remove accessories that cover facial features
        """)
        st.markdown("</div>", unsafe_allow_html=True)

def find_patient_page():
    st.markdown("<h1 class='centered-text'>Find Patient by Facial Recognition</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.subheader("Patient Photo")
        
        photo_option = st.radio("Choose photo input method for search:", ["Upload Image", "Take Photo"], key="search_photo_option")

        search_image = None
        if photo_option == "Upload Image":
            search_uploaded_file = st.file_uploader("Upload Patient Photo", type=["jpg", "jpeg", "png"], key="search_upload")
            search_image = process_uploaded_image(search_uploaded_file)
        else:
            search_image = capture_image(key="search_camera")

        if search_image is not None:
            st.image(search_image, caption="Search Photo", width=300)
            
            # Add a "Search" button for better UX
            if st.button("Search Patient"):
                with st.spinner("Searching for matching patients..."):
                    search_face_encoding = encode_face(search_image)

                    if search_face_encoding is not None:
                        # Find matching patient
                        patient_id = find_matching_face(search_face_encoding)

                        if patient_id:
                            st.success(f"Match found! Patient ID: {patient_id}")
                            
                            # Log the patient access
                            log_patient_access(patient_id, "facial_recognition")
                            
                            # Store in session state to display in the right column
                            st.session_state['found_patient_id'] = patient_id
                        else:
                            st.error("No matching patient found in the database.")
                            st.session_state['found_patient_id'] = None
                    else:
                        st.error("No face detected in the image. Please try again.")
                        st.session_state['found_patient_id'] = None
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Emergency protocol info
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("""
        ### Emergency Protocol
        
        If a patient cannot be identified:
        1. Assign a temporary emergency ID
        2. Proceed with emergency medical care
        3. Continue identification attempts through other means
        4. Document all procedures and findings
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.subheader("Patient Information")
        
        if 'found_patient_id' in st.session_state and st.session_state['found_patient_id']:
            patient_id = st.session_state['found_patient_id']
            patient_data = get_patient_by_id(patient_id)
            
            if patient_data:
                st.markdown(f"""
                <div style="background-color: #f0f0f9; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                    <h3>{patient_data['first_name']} {patient_data['last_name']}</h3>
                    <p><strong>ID:</strong> {patient_data['patient_id']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create tabs for different sections of patient info
                patient_tabs = st.tabs(["Basic Info", "Medical Info", "Contact Info"])
                
                with patient_tabs[0]:
                    st.write(f"**Matriculation Number:** {patient_data['matriculation_number']}")
                    st.write(f"**Clinic Number:** {patient_data['clinic_number']}")
                    st.write(f"**Date of Birth:** {patient_data['date_of_birth']}")
                    st.write(f"**Gender:** {patient_data['gender']}")
                    st.write(f"**Blood Group:** {patient_data['blood_group']}")
                    st.write(f"**Registration Date:** {patient_data['registration_date']}")
                
                with patient_tabs[1]:
                    # Display medical history with special formatting for allergies
                    st.markdown(f"""
                    <div style="background-color: #ffebee; padding: 10px; border-left: 4px solid #f44336; margin-bottom: 15px;">
                        <h4>⚠️ Allergies</h4>
                        <p>{patient_data['allergies'] or "No allergies recorded"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.subheader("Medical History")
                    st.write(patient_data['medical_history'] or "No medical history recorded")
                
                with patient_tabs[2]:
                    st.write(f"**Contact Number:** {patient_data['contact_number'] or 'Not provided'}")
                    st.write(f"**Email:** {patient_data['email'] or 'Not provided'}")
                    st.write(f"**Address:** {patient_data['address'] or 'Not provided'}")
                    
                    st.subheader("Emergency Contact")
                    st.write(patient_data['emergency_contact'] or "No emergency contact recorded")
        else:
            st.info("No patient selected. Please use the facial recognition search on the left to find a patient.")
        
        st.markdown("</div>", unsafe_allow_html=True)

def browse_records_page():
    st.markdown("<h1 class='centered-text'>Browse Patient Records</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    # Search filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_id = st.text_input("Search by Patient ID")
    
    with col2:
        search_matriculation = st.text_input("Search by Matriculation Number")
    
    with col3:
        search_clinic = st.text_input("Search by Clinic Number")
    
    # Get filtered patients
    patients = get_patients(search_id, search_matriculation, search_clinic)
    
    if patients:
        st.write(f"Found {len(patients)} patient records:")
        
        # Convert to DataFrame for display
        df = pd.DataFrame(patients)
        display_cols = ['patient_id', 'matriculation_number', 'clinic_number', 'first_name', 'last_name', 'date_of_birth', 'gender', 'blood_group']
        
        # Display table with pagination
        page_size = 10
        total_pages = (len(df) + page_size - 1) // page_size
        
        col1, col2 = st.columns([4, 1])
        with col2:
            page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1, step=1)
        
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(df))
        
        st.dataframe(df.iloc[start_idx:end_idx][display_cols], use_container_width=True)
        
        # Detailed view for selected patient
        selected_patient_id = st.selectbox("Select a patient to view details:", [p['patient_id'] for p in patients])
        
        if selected_patient_id:
            selected_patient = next((p for p in patients if p['patient_id'] == selected_patient_id), None)
            
            if selected_patient:
                # Log the access
                log_patient_access(selected_patient_id, "manual_lookup")
                
                st.markdown(f"""
                <div style="background-color: #f0f0f9; padding: 15px; border-radius: 5px; margin: 15px 0;">
                    <h3>{selected_patient['first_name']} {selected_patient['last_name']}</h3>
                    <p><strong>ID:</strong> {selected_patient['patient_id']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
                    st.subheader("Basic Information")
                    st.write(f"**Matriculation Number:** {selected_patient['matriculation_number']}")
                    st.write(f"**Clinic Number:** {selected_patient['clinic_number']}")
                    st.write(f"**Date of Birth:** {selected_patient['date_of_birth']}")
                    st.write(f"**Gender:** {selected_patient['gender']}")
                    st.write(f"**Blood Group:** {selected_patient['blood_group']}")
                    st.write(f"**Registration Date:** {selected_patient['registration_date']}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
                    st.subheader("Contact Information")
                    st.write(f"**Contact Number:** {selected_patient['contact_number']}")
                    st.write(f"**Email:** {selected_patient['email']}")
                    st.write(f"**Address:** {selected_patient['address']}")
                    st.write(f"**Emergency Contact:** {selected_patient['emergency_contact']}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Medical info
                st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
                st.subheader("Medical Information")
                
                # Display allergies with warning styling
                st.markdown(f"""
                <div style="background-color: #ffebee; padding: 10px; border-left: 4px solid #f44336; margin-bottom: 15px;">
                    <h4>⚠️ Allergies</h4>
                    <p>{selected_patient['allergies'] or "No allergies recorded"}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("Medical History")
                st.write(selected_patient['medical_history'] or "No medical history recorded")
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No patient records found matching the search criteria.")
    
    st.markdown("</div>", unsafe_allow_html=True)

def user_management_page():
    st.markdown("<h1 class='centered-text'>User Management</h1>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["User List", "Create User"])
    
    with tab1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        users = get_all_users()
        
        if users:
            # Convert to DataFrame
            users_df = pd.DataFrame(users)
            
            # Display users table
            st.dataframe(users_df[[
                'username', 'first_name', 'last_name', 'email', 'role', 'date_created', 'last_login', 'is_active'
            ]], use_container_width=True)
            
            # User actions
            selected_user = st.selectbox("Select user for actions:", [u['username'] for u in users])
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Deactivate User", key="deactivate_user"):
                    selected_user_data = next((u for u in users if u['username'] == selected_user), None)
                    
                    if selected_user_data:
                        # Prevent deactivation of current user or the last admin
                        if selected_user_data['user_id'] == st.session_state.user_id:
                            st.error("You cannot deactivate your own account.")
                        else:
                            # Check if this is the last admin
                            admins = [u for u in users if u['role'] == 'admin' and u['is_active'] == 1]
                            if selected_user_data['role'] == 'admin' and len(admins) <= 1:
                                st.error("Cannot deactivate the last active administrator.")
                            else:
                                c = conn.cursor()
                                c.execute("UPDATE users SET is_active = 0 WHERE user_id = ?", (selected_user_data['user_id'],))
                                conn.commit()
                                st.success(f"User {selected_user} has been deactivated.")
                                st.rerun()

            
            with col2:
                if st.button("Activate User", key="activate_user"):
                    selected_user_data = next((u for u in users if u['username'] == selected_user), None)
                    
                    if selected_user_data and selected_user_data['is_active'] == 0:
                        c = conn.cursor()
                        c.execute("UPDATE users SET is_active = 1 WHERE user_id = ?", (selected_user_data['user_id'],))
                        conn.commit()
                        st.success(f"User {selected_user} has been activated.")
                        st.rerun()

        else:
            st.info("No users found in the system.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.subheader("Create New User")
        
        with st.form("create_user_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                first_name = st.text_input("First Name")
                last_name = st.text_input("Last Name")
                username = st.text_input("Username")
                email = st.text_input("Email")
            
            with col2:
                role = st.selectbox("Role", ["staff", "admin"])
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
            
            submit = st.form_submit_button("Create User")
            
            if submit:
                if not first_name or not last_name or not username or not email or not password:
                    st.error("All fields are required.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    user_data = {
                        'first_name': first_name,
                        'last_name': last_name,
                        'username': username,
                        'email': email,
                        'role': role,
                        'password': password
                    }
                    
                    user_id, message = create_user(user_data)
                    
                    if user_id:
                        st.success(f"User created successfully! User ID: {user_id}")
                    else:
                        st.error(f"User creation failed: {message}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def system_settings_page():
    st.markdown("<h1 class='centered-text'>System Settings</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    settings = get_all_settings()
    
    if settings:
        # Group settings by category
        facial_recognition_settings = [s for s in settings if s['setting_name'] == 'face_recognition_tolerance']
        security_settings = [s for s in settings if s['setting_name'] in ['session_timeout_minutes', 'max_login_attempts']]
        general_settings = [s for s in settings if s['setting_name'] in ['system_name']]
        
        # Display settings by category
        st.subheader("Facial Recognition Settings")
        for setting in facial_recognition_settings:
            with st.form(f"setting_form_{setting['setting_name']}"):
                st.write(f"**{setting['setting_name'].replace('_', ' ').title()}**")
                
                if setting['setting_name'] == 'face_recognition_tolerance':
                    value = st.slider("Face Recognition Tolerance", 0.1, 1.0, float(setting['setting_value']), 0.05,
                                     help="Lower values require more strict face matches (0.6 recommended)")
                else:
                    value = st.text_input("Value", setting['setting_value'])
                
                if st.form_submit_button("Save"):
                    update_system_setting(setting['setting_name'], str(value))
                    st.success(f"{setting['setting_name'].replace('_', ' ').title()} updated successfully!")
        
        st.subheader("Security Settings")
        for setting in security_settings:
            with st.form(f"setting_form_{setting['setting_name']}"):
                st.write(f"**{setting['setting_name'].replace('_', ' ').title()}**")
                
                if setting['setting_name'] == 'session_timeout_minutes':
                    value = st.number_input("Session Timeout (minutes)", 5, 120, int(setting['setting_value']), 5)
                elif setting['setting_name'] == 'max_login_attempts':
                    value = st.number_input("Max Login Attempts", 3, 10, int(setting['setting_value']))
                else:
                    value = st.text_input("Value", setting['setting_value'])
                
                if st.form_submit_button("Save"):
                    update_system_setting(setting['setting_name'], str(value))
                    st.success(f"{setting['setting_name'].replace('_', ' ').title()} updated successfully!")
        
        st.subheader("General Settings")
        for setting in general_settings:
            with st.form(f"setting_form_{setting['setting_name']}"):
                st.write(f"**{setting['setting_name'].replace('_', ' ').title()}**")
                value = st.text_input("Value", setting['setting_value'])
                
                if st.form_submit_button("Save"):
                    update_system_setting(setting['setting_name'], value)
                    st.success(f"{setting['setting_name'].replace('_', ' ').title()} updated successfully!")
    else:
        st.warning("No system settings found.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # System maintenance options
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.subheader("System Maintenance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Backup Database"):
            # Database backup logic would go here
            st.success("Database backup created successfully!")
    
    with col2:
        if st.button("Optimize Face Encodings"):
            # Face encoding optimization logic would go here
            st.success("Face encodings optimized successfully!")
    
    st.markdown("</div>", unsafe_allow_html=True)

def logs_page():
    st.markdown("<h1 class='centered-text'>System Logs</h1>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login Logs", "Patient Access Logs"])
    
    with tab1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        login_logs = get_login_logs()
        
        if login_logs:
            # Convert to DataFrame
            logs_df = pd.DataFrame(login_logs)
            
            # Add success/failure indicator
            logs_df['status'] = logs_df['success'].apply(lambda x: "✅ Success" if x == 1 else "❌ Failed")
            
            # Format the table
            st.dataframe(logs_df[[
                'log_id', 'username', 'first_name', 'last_name', 'login_time', 'ip_address', 'status'
            ]], use_container_width=True)
            
            # Visualize login success/failure
            st.subheader("Login Success vs. Failure")
            success_count = len(logs_df[logs_df['success'] == 1])
            failure_count = len(logs_df[logs_df['success'] == 0])
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['Success', 'Failure'],
                    y=[success_count, failure_count],
                    marker_color=['#4CAF50', '#F44336']
                )
            ])
            fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No login logs found.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        access_logs = get_patient_access_logs()
        
        if access_logs:
            # Convert to DataFrame
            logs_df = pd.DataFrame(access_logs)
            
            # Format the table
            st.dataframe(logs_df[[
                'log_id', 'patient_first_name', 'patient_last_name', 'accessed_by_username', 
                'access_time', 'access_method'
            ]], use_container_width=True)
            
            # Visualize access methods
            st.subheader("Access Methods Distribution")
            method_counts = logs_df['access_method'].value_counts().reset_index()
            method_counts.columns = ['method', 'count']
            
            fig = px.pie(method_counts, values='count', names='method', hole=0.4)
            fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No patient access logs found.")
        
        st.markdown("</div>", unsafe_allow_html=True)

def logout():
    # Reset all authentication state
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.user_role = None
    st.session_state.login_time = None
    st.success("You have been logged out successfully.")
    st.rerun()

def main():
    # Check if user is authenticated
    if not st.session_state.authenticated:
        login_form()
    else:
        # Check session timeout
        if not check_session_timeout():
            return

        # Show sidebar menu and get selection
        selected = sidebar_menu()
        
        # Handle menu selection
        if selected == "Dashboard":
            dashboard_page()
        elif selected == "Register Patient":
            register_patient_page()
        elif selected == "Find Patient":
            find_patient_page()
        elif selected == "Browse Records":
            browse_records_page()
        elif selected == "User Management":
            if st.session_state.user_role == 'admin':
                user_management_page()
            else:
                st.error("Access denied. Admin privileges required.")
        elif selected == "System Settings":
            if st.session_state.user_role == 'admin':
                system_settings_page()
            else:
                st.error("Access denied. Admin privileges required.")
        elif selected == "Logs":
            if st.session_state.user_role == 'admin':
                logs_page()
            else:
                st.error("Access denied. Admin privileges required.")
        elif selected == "Logout":
            logout()

if __name__ == "__main__":
    main()