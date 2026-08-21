import hashlib
import streamlit as st

class AuthManager:
    # Pre-registered default accounts
    DEFAULT_ACCOUNTS = {
        "user@health.com": {
            "password_hash": hashlib.sha256("User@123".encode()).hexdigest(),
            "name": "Dundi",
            "role": "Patient / User",
            "patient_id": "PX-90412",
            "age": 28.0,
            "gender": "Male"
        },
        "doctor@hospital.com": {
            "password_hash": hashlib.sha256("Doctor@123".encode()).hexdigest(),
            "name": "Dr. Sarah Jenkins, M.D.",
            "role": "Doctor / Clinician",
            "license": "REG-MCI-889412",
            "department": "Department of Endocrinology & Metabolic Health"
        }
    }

    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    @classmethod
    def initialize_session(cls):
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        if "user_info" not in st.session_state:
            st.session_state.user_info = None
        if "users_db" not in st.session_state:
            st.session_state.users_db = cls.DEFAULT_ACCOUNTS.copy()

    @classmethod
    def login(cls, email: str, password: str, role: str) -> bool:
        email = email.strip().lower()
        users = st.session_state.users_db
        if email in users:
            account = users[email]
            if account["password_hash"] == cls.hash_password(password) and account["role"] == role:
                st.session_state.authenticated = True
                st.session_state.user_info = account.copy()
                st.session_state.user_info["email"] = email
                return True
        return False

    @classmethod
    def register(cls, email: str, password: str, full_name: str, role: str, extra_data: dict) -> bool:
        email = email.strip().lower()
        if email in st.session_state.users_db:
            return False
        
        user_record = {
            "password_hash": cls.hash_password(password),
            "name": full_name,
            "role": role
        }
        user_record.update(extra_data)
        st.session_state.users_db[email] = user_record
        return True

    @staticmethod
    def logout():
        st.session_state.authenticated = False
        st.session_state.user_info = None
        st.rerun()