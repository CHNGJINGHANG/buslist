import streamlit as st
from datetime import datetime, timedelta
from PIL import Image
import re
import base64
import io
from typing import List, Dict
import requests
import pandas as pd
import urllib.parse

class GeminiOCR:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.api_url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"
    
    def set_api_key(self, api_key: str):
        self.api_key = api_key

    def extract_text_from_image(self, image: Image.Image) -> str:
        """Extract all text from image using OCR"""
        if not self.api_key:
            raise ValueError("API key not set")
        
        try:
            image_base64 = self._image_to_base64(image)
            
            prompt = """Extract ALL text from this image exactly as it appears.
            Return only the raw text, maintaining the original formatting and line breaks.
            Do not add any commentary or interpretation."""
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
                    ]
                }],
                "generationConfig": {"temperature": 0, "maxOutputTokens": 2000}
            }
            
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    return result['candidates'][0]['content']['parts'][0]['text'].strip()
            else:
                raise Exception(f"API error: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"OCR failed: {str(e)}")
        
        return ""
    
    def extract_names_from_text(self, text: str) -> List[str]:
        """Extract names from text using improved pattern matching"""
        if not text.strip():
            return []
        
        lines = text.split('\n')
        names = []
        
        for line in lines:
            line = line.strip()
            
            # Remove common prefixes
            line = re.sub(r'^[\d\.\-\*\)\]]+\s*', '', line)
            
            # Skip lines that are clearly not names
            if self._is_noise(line):
                continue
            
            # Check if it's a valid name
            if self._is_valid_name(line):
                formatted = self._format_name(line)
                if formatted and formatted not in names:
                    names.append(formatted)
        
        return names
    
    def extract_bus_info(self, text: str) -> Dict[str, str]:
        """Extract bus plate and phone from text"""
        bus_plate = ""
        phone = ""
        
        # Bus plate patterns (Singapore format)
        plate_patterns = [
            r'\b[A-Z]{1,3}\s?\d{1,4}\s?[A-Z]\b',  # ABC1234X
            r'\b[SG]{2}\s?\d{4}\s?[A-Z]\b',       # SG1234X
        ]
        
        for pattern in plate_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                bus_plate = match.group().upper().replace(' ', '')
                break
        
        # Phone patterns (Singapore)
        phone_patterns = [
            r'\+65\s?[689]\d{7}',      # +65 91234567
            r'\b[689]\d{7}\b',         # 91234567
            r'\b\d{4}\s?\d{4}\b',      # 9123 4567
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                phone = match.group().replace(' ', '')
                if not phone.startswith('+65'):
                    phone = '+65' + phone if phone[0] in '689' else phone
                break
        
        return {"bus_plate": bus_plate, "phone": phone}
    
    def extract_datetime_info(self, text: str) -> Dict[str, str]:
        """Extract date and time information"""
        day = ""
        date = ""
        times = []
        
        # Day of week
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for d in days:
            if d.lower() in text.lower():
                day = d
                break
        
        # Date patterns
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # 15/01/2024
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date = match.group()
                break
        
        # Time patterns
        time_patterns = [
            r'\b\d{4}\s?hrs\b',        # 0735hrs
            r'\b\d{1,2}:\d{2}\s?[ap]m\b',  # 7:35am
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            times.extend(matches)
        
        return {
            "day": day,
            "date": date,
            "ntu_time": times[0] if len(times) > 0 else "",
            "je_time": times[1] if len(times) > 1 else ""
        }
    
    def _is_noise(self, text: str) -> bool:
        """Check if text is noise (not a name)"""
        noise_patterns = [
            r'^\d+$',  # Only numbers
            r'^[\W_]+$',  # Only special chars
            r'\d{4}[-/]\d{2}[-/]\d{2}',  # Dates
            r'\d{1,2}:\d{2}',  # Times
            r'\+?\d{8,}',  # Phone numbers
            r'^(ntu|je|jurong|bus|stop|hall|residence|time|date|phone|driver|location)$',
        ]
        
        text_lower = text.lower()
        
        for pattern in noise_patterns:
            if re.match(pattern, text_lower):
                return True
        
        return len(text) < 2 or len(text) > 50
    
    def _is_valid_name(self, text: str) -> bool:
        """Check if text is a valid name"""
        # Must contain at least one letter
        if not re.search(r'[A-Za-z]', text):
            return False
        
        # Must not be mostly numbers
        if len(re.findall(r'\d', text)) > len(text) // 2:
            return False
        
        # Must have reasonable length
        if len(text) < 2 or len(text) > 50:
            return False
        
        return True
    
    def _format_name(self, name: str) -> str:
        """Format name with proper capitalization"""
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        # Title case
        words = []
        for word in name.split():
            if word.isupper() and len(word) > 1:
                word = word.title()
            elif word.islower():
                word = word.title()
            words.append(word)
        
        return ' '.join(words)
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


def initialize_session_state():
    if 'bus_list' not in st.session_state:
        st.session_state.bus_list = {'NTU': [], 'Jurong East': []}
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'gemini_ocr' not in st.session_state:
        st.session_state.gemini_ocr = GeminiOCR()
    if 'schedule_data' not in st.session_state:
        st.session_state.schedule_data = []
    if 'recipient_emails' not in st.session_state:
        st.session_state.recipient_emails = []


def format_bus_info(settings):
    """Format the bus information"""
    output = f"Bus Information - {settings['day']}, {settings['date']}\n\n"
    
    if settings['bus_number']:
        output += f"Bus Plate: {settings['bus_number']}\n"
    if settings['driver_phone']:
        output += f"Phone: {settings['driver_phone']}\n"
    output += "\n---\n\n"
    
    section_num = 1
    
    if st.session_state.bus_list['NTU']:
        output += f"{section_num}. NTU ({settings['ntu_time']})\n"
        output += f"Location: {settings['ntu_location']}\n"
        for name in sorted(st.session_state.bus_list['NTU']):
            output += f" {name}\n"
        output += "\n"
        section_num += 1
    
    if st.session_state.bus_list['Jurong East']:
        output += f"{section_num}. Jurong East ({settings['je_time']})\n"
        output += f"Location: {settings['je_location']}\n"
        for name in sorted(st.session_state.bus_list['Jurong East']):
            output += f" {name}\n"
        output += "\n"
    
    total = len(st.session_state.bus_list['NTU']) + len(st.session_state.bus_list['Jurong East'])
    output += f"Total Passengers: {total}\n"
    output += f"NTU: {len(st.session_state.bus_list['NTU'])} | JE: {len(st.session_state.bus_list['Jurong East'])}\n"
    
    return output


def main():
    st.set_page_config(
        page_title="Bus List Manager",
        page_icon="🚌",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("Bus Passenger List Manager")
    
    tab = st.sidebar.radio("Navigation:", 
                          ["API Setup", "Input & Processing", "Bus Settings", "Output", "Schedule"])
    
    if tab == "API Setup":
        st.header("Gemini API Configuration")
        
        with st.expander("Instructions"):
            st.markdown("""
            1. Get API key from: https://makersuite.google.com/app/apikey
            2. Paste below and save
            """)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            api_key = st.text_input("API Key:", value=st.session_state.api_key, type="password")
        
        with col2:
            if st.button("Save Key"):
                if api_key.strip():
                    st.session_state.api_key = api_key.strip()
                    st.session_state.gemini_ocr.set_api_key(api_key.strip())
                    st.success("Saved")
                else:
                    st.warning("Enter valid key")
        
        if st.session_state.api_key:
            st.success("API Ready")
        else:
            st.warning("No API key configured")
    
    elif tab == "Input & Processing":
        st.header("Add Passengers")
        
        location = st.selectbox("Add to:", ["NTU", "Jurong East"])
        
        # Image upload
        uploaded_file = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, width=400)
            
            if st.button("Extract from Image"):
                if not st.session_state.api_key:
                    st.error("Configure API key first")
                else:
                    try:
                        with st.spinner("Processing..."):
                            # Get raw OCR text
                            raw_text = st.session_state.gemini_ocr.extract_text_from_image(image)
                            
                            # Show raw text
                            with st.expander("Raw OCR Text"):
                                st.text(raw_text)
                            
                            # Extract names
                            names = st.session_state.gemini_ocr.extract_names_from_text(raw_text)
                            
                            # Add to list
                            added = 0
                            for name in names:
                                if name not in st.session_state.bus_list[location]:
                                    st.session_state.bus_list[location].append(name)
                                    added += 1
                            
                            st.success(f"Added {added} names to {location}")
                            
                            # Try to extract bus info
                            bus_info = st.session_state.gemini_ocr.extract_bus_info(raw_text)
                            if bus_info['bus_plate'] or bus_info['phone']:
                                st.session_state.extracted_bus_info = bus_info
                                st.info(f"Bus info found: {bus_info}")
                            
                            # Try to extract date/time
                            datetime_info = st.session_state.gemini_ocr.extract_datetime_info(raw_text)
                            if any(datetime_info.values()):
                                st.session_state.extracted_datetime = datetime_info
                                st.info(f"Date/time found: {datetime_info}")
                            
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Text input
        st.subheader("Or paste text")
        text_input = st.text_area("Paste here:")
        
        if st.button("Extract from Text"):
            if text_input:
                try:
                    names = st.session_state.gemini_ocr.extract_names_from_text(text_input)
                    
                    added = 0
                    for name in names:
                        if name not in st.session_state.bus_list[location]:
                            st.session_state.bus_list[location].append(name)
                            added += 1
                    
                    st.success(f"Added {added} names")
                    
                    # Extract other info
                    bus_info = st.session_state.gemini_ocr.extract_bus_info(text_input)
                    if bus_info['bus_plate'] or bus_info['phone']:
                        st.session_state.extracted_bus_info = bus_info
                        st.info(f"Bus info: {bus_info}")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Manual add
        st.subheader("Manual Entry")
        manual_name = st.text_input("Add name:")
        if st.button("Add") and manual_name:
            formatted = manual_name.strip().title()
            if formatted not in st.session_state.bus_list[location]:
                st.session_state.bus_list[location].append(formatted)
                st.success(f"Added {formatted}")
                st.rerun()
        
        # Show lists
        st.subheader("Current Passengers")
        for loc in ["NTU", "Jurong East"]:
            with st.expander(f"{loc}: {len(st.session_state.bus_list[loc])}"):
                if st.session_state.bus_list[loc]:
                    for i, name in enumerate(st.session_state.bus_list[loc]):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.text(name)
                        with col2:
                            if st.button("Remove", key=f"rm_{loc}_{i}"):
                                st.session_state.bus_list[loc].remove(name)
                                st.rerun()
                else:
                    st.text("No passengers")
    
    elif tab == "Bus Settings":
        st.header("Bus Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Date & Time")
            selected_date = st.date_input("Date:", value=datetime.now() + timedelta(days=1))
            day = selected_date.strftime("%A")
            date = selected_date.strftime("%B %d, %Y")
            
            st.text(f"Day: {day}")
            st.text(f"Date: {date}")
            
            default_ntu = st.session_state.get('extracted_datetime', {}).get('ntu_time', '0735hrs')
            default_je = st.session_state.get('extracted_datetime', {}).get('je_time', '0750hrs')
            
            ntu_time = st.text_input("NTU Time:", value=default_ntu)
            je_time = st.text_input("JE Time:", value=default_je)
        
        with col2:
            st.subheader("Bus Info")
            default_bus = st.session_state.get('extracted_bus_info', {}).get('bus_plate', '')
            default_phone = st.session_state.get('extracted_bus_info', {}).get('phone', '')
            
            bus_number = st.text_input("Bus Plate:", value=default_bus)
            driver_phone = st.text_input("Phone:", value=default_phone)
        
        st.subheader("Locations")
        ntu_location = st.text_input("NTU:", value="Hall 8 & 9 Bus Stop")
        je_location = st.text_input("JE:", value="Venture Avenue")
        
        st.session_state.settings = {
            'day': day,
            'date': date,
            'ntu_time': ntu_time,
            'je_time': je_time,
            'bus_number': bus_number,
            'driver_phone': driver_phone,
            'ntu_location': ntu_location,
            'je_location': je_location
        }
    
    elif tab == "Output":
        st.header("Generated Output")
        
        total = len(st.session_state.bus_list['NTU']) + len(st.session_state.bus_list['Jurong East'])
        
        if total == 0:
            st.warning("No passengers added yet")
        else:
            if 'settings' not in st.session_state:
                tomorrow = datetime.now() + timedelta(days=1)
                st.session_state.settings = {
                    'day': tomorrow.strftime("%A"),
                    'date': tomorrow.strftime("%B %d, %Y"),
                    'ntu_time': "0735hrs",
                    'je_time': "0750hrs",
                    'bus_number': "",
                    'driver_phone': "",
                    'ntu_location': "Hall 8 & 9 Bus Stop",
                    'je_location': "Venture Avenue"
                }
            
            if st.button("Generate"):
                output = format_bus_info(st.session_state.settings)
                st.session_state.generated_output = output
            
            if 'generated_output' in st.session_state:
                st.code(st.session_state.generated_output)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        "Download Text",
                        data=st.session_state.generated_output,
                        file_name=f"bus_list_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    st.download_button(
                        "Download Markdown",
                        data=st.session_state.generated_output,
                        file_name=f"bus_list_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )
                
                st.metric("Total Passengers", total)
    
    elif tab == "Schedule":
        st.header("Schedule Management")
        st.info("Schedule feature - implementation simplified")
    
    # Sidebar status
    st.sidebar.markdown("---")
    st.sidebar.subheader("Status")
    
    if st.session_state.api_key:
        st.sidebar.success("API Ready")
    else:
        st.sidebar.error("No API Key")
    
    total = len(st.session_state.bus_list['NTU']) + len(st.session_state.bus_list['Jurong East'])
    st.sidebar.metric("Total Passengers", total)
    st.sidebar.text(f"NTU: {len(st.session_state.bus_list['NTU'])}")
    st.sidebar.text(f"JE: {len(st.session_state.bus_list['Jurong East'])}")


if __name__ == "__main__":
    main()
