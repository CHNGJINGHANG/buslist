import streamlit as st
from datetime import datetime, timedelta
from PIL import Image
import re
import base64
import io
from typing import List, Dict
import requests
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import urllib.parse

class GeminiOCR:
    """Gemini API-powered OCR and text extraction"""
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.vision_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        self.text_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    
    def set_api_key(self, api_key: str):
        """Set the Gemini API key"""
        self.api_key = api_key
    
    def extract_names_from_image(self, image: Image.Image) -> List[str]:
        """Extract passenger names from image using Gemini Vision API"""
        if not self.api_key:
            raise ValueError("Gemini API key not set")
        
        try:
            # Convert PIL image to base64
            image_base64 = self._image_to_base64(image)
            
            # Prepare the prompt
            prompt = """
            You are an AI assistant helping to extract passenger names from bus lists, chat screenshots, or any text content in this image.
            
            Please:
            1. Extract ONLY the passenger names from this image
            2. Ignore any dates, times, phone numbers, bus information, locations, or other metadata
            3. Format each name properly with correct capitalization
            4. Return each name on a separate line
            5. If you see duplicates, only include each name once
            6. If no names are visible, return "NO_NAMES_FOUND"
            
            Focus only on identifying actual human names that would be passengers on a bus.
            """
            
            # Prepare request payload
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png" if image.format == "PNG" else "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 1000
                }
            }
            
            # Make API request
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(
                f"{self.vision_api_url}?key={self.api_key}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    
                    if "NO_NAMES_FOUND" in content.upper():
                        return []
                    
                    # Process the extracted text
                    names = []
                    lines = content.strip().split('\n')
                    
                    for line in lines:
                        line = line.strip()
                        # Remove numbering, bullets, etc.
                        line = re.sub(r'^[0-9\.\-\*\•\s]*', '', line)
                        line = line.strip()
                        
                        if line and self._is_likely_name(line):
                            formatted_name = self._format_name(line)
                            if formatted_name and formatted_name not in names:
                                names.append(formatted_name)
                    
                    return names
                else:
                    raise Exception("No response from Gemini API")
            else:
                error_msg = f"Gemini API error: {response.status_code}"
                if response.text:
                    try:
                        error_data = response.json()
                        if 'error' in error_data:
                            error_msg += f" - {error_data['error'].get('message', 'Unknown error')}"
                    except:
                        error_msg += f" - {response.text[:200]}"
                raise Exception(error_msg)
                
        except Exception as e:
            raise Exception(f"Failed to process image with Gemini: {str(e)}")
        
    def extract_date_info_from_text(self, text: str) -> Dict[str, str]:
        """Extract date, day, and time information from text"""
        if not self.api_key:
            return {"day": "", "date": "", "ntu_time": "", "je_time": ""}
        
        try:
            prompt = f"""Extract date and time information from this text:
            Look for:
            1. Day of the week (Monday, Tuesday, etc.)
            2. Date in any format (e.g., "January 15, 2024", "15/01/2024", "15 Jan")
            3. Time for NTU pickup (look for times around 7:30-8:00 AM, format as XXXXhrs)
            4. Time for Jurong East/JE pickup (look for times around 7:45-8:15 AM, format as XXXXhrs)
            
            Text: {text}
            
            Return format:
            DAY: [day of week or NONE]
            DATE: [formatted date or NONE]
            NTU_TIME: [time in XXXXhrs format or NONE]
            JE_TIME: [time in XXXXhrs format or NONE]"""
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 500}
            }
            
            response = requests.post(f"{self.text_api_url}?key={self.api_key}", 
                                headers={"Content-Type": "application/json"}, 
                                json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    return self._parse_date_time_info(content)
            
            return {"day": "", "date": "", "ntu_time": "", "je_time": ""}
        except:
            return {"day": "", "date": "", "ntu_time": "", "je_time": ""}

    def _parse_date_time_info(self, content: str) -> Dict[str, str]:
        """Parse date and time info from API response"""
        day = ""
        date = ""
        ntu_time = ""
        je_time = ""
        
        for line in content.split('\n'):
            line = line.strip()
            if 'DAY:' in line:
                day = line.split('DAY:')[1].strip()
                if day.upper() == 'NONE':
                    day = ""
            elif 'DATE:' in line:
                date = line.split('DATE:')[1].strip()
                if date.upper() == 'NONE':
                    date = ""
            elif 'NTU_TIME:' in line:
                ntu_time = line.split('NTU_TIME:')[1].strip()
                if ntu_time.upper() == 'NONE':
                    ntu_time = ""
            elif 'JE_TIME:' in line:
                je_time = line.split('JE_TIME:')[1].strip()
                if je_time.upper() == 'NONE':
                    je_time = ""
        
        return {"day": day, "date": date, "ntu_time": ntu_time, "je_time": je_time}
    
    def extract_names_from_text(self, text: str) -> List[str]:
        """Extract names from text using Gemini API"""
        if not self.api_key:
            raise ValueError("Gemini API key not set")
        
        if not text.strip():
            return []
        
        try:
            prompt = f"""
            Extract ONLY the passenger names from the following text. This might be from a chat message, list, or any other format.
            
            Rules:
            1. Extract ONLY human names that would be bus passengers
            2. Ignore dates, times, phone numbers, bus numbers, locations
            3. Ignore words like "bus", "time", "location", "phone", etc.
            4. Format names with proper capitalization
            5. One name per line
            6. Remove duplicates
            7. If no names found, return "NO_NAMES_FOUND"
            
            Text to process:
            {text}
            """
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 1000
                }
            }
            
            response = requests.post(
                f"{self.text_api_url}?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    
                    if "NO_NAMES_FOUND" in content.upper():
                        return []
                    
                    # Process the extracted text
                    names = []
                    lines = content.strip().split('\n')
                    
                    for line in lines:
                        line = line.strip()
                        line = re.sub(r'^[0-9\.\-\*\•\s]*', '', line)
                        line = line.strip()
                        
                        if line and self._is_likely_name(line):
                            formatted_name = self._format_name(line)
                            if formatted_name and formatted_name not in names:
                                names.append(formatted_name)
                    
                    return names
                else:
                    raise Exception("No response from Gemini API")
            else:
                raise Exception(f"Gemini API error: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Failed to process text with Gemini: {str(e)}")
    
    def extract_bus_info_from_image(self, image: Image.Image) -> Dict[str, str]:
        """Extract bus plate and phone number from image"""
        if not self.api_key:
            raise ValueError("Gemini API key not set")
        
        try:
            image_base64 = self._image_to_base64(image)
            
            prompt = """Extract bus information from this image:
            1. Bus plate number (format: ABC1234X or similar)
            2. Phone number (Singapore format: +65 XXXXXXXX or 9XXXXXXX)
            
            Return in format:
            BUS_PLATE: [plate number or NONE]
            PHONE: [phone number or NONE]"""
            
            payload = {
                "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 500}
            }
            
            response = requests.post(f"{self.vision_api_url}?key={self.api_key}", 
                                   headers={"Content-Type": "application/json"}, 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    return self._parse_bus_info(content)
            
            return {"bus_plate": "", "phone": ""}
        except Exception as e:
            raise Exception(f"Failed to extract bus info: {str(e)}")

    def extract_bus_info_from_text(self, text: str) -> Dict[str, str]:
        """Extract bus plate and phone from text"""
        if not self.api_key:
            return {"bus_plate": "", "phone": ""}
        
        try:
            prompt = f"""Extract bus information from this text:
            1. Bus plate number
            2. Phone number
            
            Text: {text}
            
            Return format:
            BUS_PLATE: [plate or NONE]
            PHONE: [phone or NONE]"""
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 500}
            }
            
            response = requests.post(f"{self.text_api_url}?key={self.api_key}", 
                                   headers={"Content-Type": "application/json"}, 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    return self._parse_bus_info(content)
            
            return {"bus_plate": "", "phone": ""}
        except:
            return {"bus_plate": "", "phone": ""}

    def _parse_bus_info(self, content: str) -> Dict[str, str]:
        """Parse bus info from API response"""
        bus_plate = ""
        phone = ""
        
        for line in content.split('\n'):
            line = line.strip()
            if 'BUS_PLATE:' in line:
                bus_plate = line.split('BUS_PLATE:')[1].strip()
                if bus_plate.upper() == 'NONE':
                    bus_plate = ""
            elif 'PHONE:' in line:
                phone = line.split('PHONE:')[1].strip()
                if phone.upper() == 'NONE':
                    phone = ""
        
        return {"bus_plate": bus_plate, "phone": phone}
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        image_bytes = buffer.getvalue()
        
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def _is_likely_name(self, text: str) -> bool:
        """Check if text looks like a person's name"""
        if not text or len(text) < 2 or len(text) > 50:
            return False
        
        skip_patterns = [
            r'^(bus|date|time|location|phone|driver|passenger|list|name|number)s?$',
            r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # Dates
            r'^\d{1,2}:\d{2}',  # Times
            r'^\+?\d{8,}$',  # Phone numbers
            r'^[A-Z]{2,}\d+$',  # Bus plates
            r'^(ntu|je|jurong|east|hall|residence|venture|avenue|stop)$',
            r'^(no_names_found|not_found|none|nil)$'
        ]
        
        for pattern in skip_patterns:
            if re.match(pattern, text.lower()):
                return False
        
        if not re.search(r'[A-Za-z]', text):
            return False
        
        if len(re.findall(r'\d', text)) > len(text) // 2:
            return False
        
        return True
    
    def _format_name(self, name: str) -> str:
        """Format name with proper capitalization"""
        words = name.split()
        formatted_words = []
        
        for word in words:
            if word.upper() == word and len(word) > 1:
                word = word.title()
            elif word.lower() == word:
                word = word.title()
            formatted_words.append(word)
        
        return ' '.join(formatted_words)

class SmartTextProcessor:
    """Fallback text processing when Gemini API is not available"""
    
    @staticmethod
    def extract_names_from_text(text: str) -> List[str]:
        """Extract potential names from raw text using pattern recognition"""
        if not text.strip():
            return []
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        names = []
        
        for line in lines:
            line = re.sub(r'^[0-9]+[\.\)\-\s]*', '', line)
            line = re.sub(r'[^\w\s\-\.]', '', line)
            line = line.strip()
            
            if not line:
                continue
                
            if SmartTextProcessor._is_likely_name(line):
                potential_names = re.split(r'[,;]', line)
                for name in potential_names:
                    name = name.strip()
                    if name and SmartTextProcessor._is_likely_name(name):
                        names.append(SmartTextProcessor._format_name(name))
        
        return names
    
    @staticmethod
    def _is_likely_name(text: str) -> bool:
        """Determine if text is likely to be a person's name"""
        if not text or len(text) < 2:
            return False
            
        skip_patterns = [
            r'^(bus|date|time|location|phone|driver|passenger|list|name)s?$',
            r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            r'^\d{1,2}:\d{2}',
            r'^\+?\d{8,}$',
            r'^[A-Z]{2,}\d+$',
            r'^(ntu|je|jurong|east|hall|residence|venture|avenue|stop)$'
        ]
        
        for pattern in skip_patterns:
            if re.match(pattern, text.lower()):
                return False
        
        if not re.search(r'[A-Za-z]', text):
            return False
            
        if len(text) > 50:
            return False
            
        if len(re.findall(r'\d', text)) > len(text) // 3:
            return False
            
        return True
    
    @staticmethod
    def _format_name(name: str) -> str:
        """Format name with proper capitalization"""
        words = name.split()
        formatted_words = []
        
        for word in words:
            if word.upper() == word and len(word) > 1:
                word = word.title()
            elif word.lower() == word:
                word = word.title()
            formatted_words.append(word)
            
        return ' '.join(formatted_words)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'bus_list' not in st.session_state:
        st.session_state.bus_list = {'NTU': [], 'Jurong East': []}
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    
    if 'gemini_ocr' not in st.session_state:
        st.session_state.gemini_ocr = GeminiOCR()
    
    if 'fallback_processor' not in st.session_state:
        st.session_state.fallback_processor = SmartTextProcessor()
    if 'schedule_data' not in st.session_state:
        st.session_state.schedule_data = []

    if 'email_config' not in st.session_state:
        st.session_state.email_config = {
            'smtp_server': 'outlook.live.com',
            'smtp_port': 587,
            'email': '',
            'password': '',
            'recipient_list': []  # Changed to list for multiple recipients
        }

def format_bus_info(settings):
    """Format the bus information"""
    output = f"### Bus Information - {settings['day']}, {settings['date']}\n\n"
    
    if settings['bus_number']:
        output += f"Bus Plate Number: {settings['bus_number']}\n"
    if settings['driver_phone']:
        output += f"Phone Number: {settings['driver_phone']}\n"
    output += "\n---\n\n"
    
    section_num = 1
    
    # NTU section
    if st.session_state.bus_list['NTU']:
        output += f"### {section_num}. NTU ({settings['ntu_time']})\n\n"
        output += f"Location: {settings['ntu_location']}\n"
        for name in sorted(st.session_state.bus_list['NTU']):
            output += f" {name}\n"
        output += "\n---\n\n"
        section_num += 1
        
    # JE section
    if st.session_state.bus_list['Jurong East']:
        output += f"### {section_num}. Jurong East ({settings['je_time']})\n\n"
        output += f"Location: {settings['je_location']}\n"
        for name in sorted(st.session_state.bus_list['Jurong East']):
            output += f" {name}\n"
        output += "\n---\n\n"
    
    # Add summary
    total_passengers = len(st.session_state.bus_list['NTU']) + len(st.session_state.bus_list['Jurong East'])
    output += f"**Total Passengers: {total_passengers}**\n"
    output += f"• NTU: {len(st.session_state.bus_list['NTU'])} passengers\n"
    output += f"• Jurong East: {len(st.session_state.bus_list['Jurong East'])} passengers\n"
        
    return output

def create_schedule_table():
    """Create and manage bus schedule table"""
    st.subheader("Bus Schedule Management")
    
    # Form for adding new schedule entry
    with st.form("schedule_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date = st.date_input("Date")
            activity = st.text_input("Activity", value="Dragon Boat (M)")
            pickup_point = st.text_input("Pick-Up Point", value="NTU Hall of Residence 8 & 9 Bus Stop")
        
        with col2:
            departure_time = st.text_input("Departure Time", value="0800 hrs")
            bus_capacity = st.selectbox("Bus Capacity", ["1 x 20 seater bus", "1 x 40 seater bus", "2 x 20 seater bus"])
            return_time = st.text_input("Return Time", value="NIL")
        
        with col3:
            contact_name = st.text_input("Contact Name")
            contact_number = st.text_input("Contact Number")
            
        # Destinations section with unique keys
        st.subheader("Destinations")
        destinations = []

        # Add unique key to number_input
        num_destinations = st.number_input(
            "Number of Destinations", 
            min_value=1, 
            max_value=5, 
            value=1,
            key="num_dest_input"  # Added unique key
        )

        for i in range(num_destinations):
            col1, col2 = st.columns([3, 1])
            with col1:
                dest = st.text_input(f"Destination {i+1}", key=f"dest_{i}")
                destinations.append(dest) if dest else None
            with col2:
                if dest:
                    maps_url = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(dest)}"
                    st.markdown(f'<a href="{maps_url}" target="_blank">📍 Open Maps</a>', unsafe_allow_html=True)
        
        # Add form submit button
        submitted = st.form_submit_button("Add to Schedule")
        
        if submitted:
            new_entry = {
                'date': date.strftime("%d/%m/%Y"),
                'day': date.strftime("%A").upper(),
                'activity': activity,
                'pickup_point': pickup_point,
                'departure_time': departure_time,
                'destinations': destinations,
                'return_time': return_time,
                'contact_name': contact_name,
                'contact_number': contact_number,
                'bus_capacity': bus_capacity
            }
            st.session_state.schedule_data.append(new_entry)
            st.success("Schedule entry added!")
            st.rerun()

def display_schedule_table():
    """Display the schedule in table format"""
    if not st.session_state.schedule_data:
        st.info("No schedule entries yet.")
        return
    
    # Create DataFrame for better display
    display_data = []
    for entry in st.session_state.schedule_data:
        # Format destinations with Google Maps links
        destinations_str = ""
        for i, dest in enumerate(entry['destinations']):
            maps_url = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(dest)}"
            destinations_str += f"{i+1}. {dest} [📍]({maps_url})\n"
        
        display_data.append({
            'Date': f"{entry['date']}\n{entry['day']}",
            'Activity': entry['activity'],
            'Pick-Up Point': entry['pickup_point'],
            'Departure Time': entry['departure_time'],
            'Destination': destinations_str,
            'Return Time': entry['return_time'],
            'Name & Contact': f"{entry['contact_name']}\n{entry['contact_number']}",
            'Seats': entry['bus_capacity']
        })
    
    df = pd.DataFrame(display_data)
    
    # Display table with height based on number of rows (40px per row + 60px header)
    num_rows = len(display_data)
    table_height = min(400, max(150, (num_rows * 40) + 60))
    
    # Use st.dataframe to display the table
    st.dataframe(
        df,
        height=table_height,
        width=True,
        use_container_width=True
    )
    
    # Add removal functionality
    st.subheader("Remove Schedule Entry")
    if len(st.session_state.schedule_data) > 0:
        selected_index = st.selectbox(
            "Select entry to remove:",
            range(len(st.session_state.schedule_data)),
            format_func=lambda x: f"Entry {x+1}: {st.session_state.schedule_data[x]['date']} - {st.session_state.schedule_data[x]['activity']}"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"Selected: {st.session_state.schedule_data[selected_index]['date']} - {st.session_state.schedule_data[selected_index]['activity']}")
        with col2:
            if st.button("🗑️ Remove Entry", type="secondary"):
                st.session_state.schedule_data.pop(selected_index)
                st.success("Entry removed successfully!")
                st.rerun()
                
def generate_schedule_html():
    """Generate a well-formatted HTML table for email with borders"""
    if not st.session_state.schedule_data:
        return ""
    
    html = """
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
        }
        th, td {
            border: 1px solid black; /* Add borders to all cells */
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2; /* Light gray background for headers */
        }
    </style>
    <table>
        <tr>
            <th>Date (2025)</th>
            <th>Activity</th>
            <th>Pick-Up Point</th>
            <th>Departure Time</th>
            <th>Destination</th>
            <th>Return Time</th>
            <th>Name & Contact No. of I/C</th>
            <th>Seats</th>
            <th>Price</th>
        </tr>
    """
    
    for entry in st.session_state.schedule_data:
        destinations_html = "<br>".join([f"{i+1}. {dest}" for i, dest in enumerate(entry['destinations'])])
        
        html += f"""
        <tr>
            <td>{entry['date']}<br>{entry['day']}</td>
            <td>{entry['activity']}</td>
            <td>{entry['pickup_point']}</td>
            <td>{entry['departure_time']}</td>
            <td>{destinations_html}</td>
            <td>{entry['return_time']}</td>
            <td>{entry['contact_name']},<br>{entry['contact_number']}</td>
            <td>{entry['bus_capacity']}</td>
            <td></td>
        </tr>
        """
    
    html += "</table>"
    return html

def send_schedule_email():
    """Generate email content using mailto links for cross-platform compatibility"""
    st.subheader("Email via Mailto Link")
    
    # Add subject definition
    subject = "NTUDB(M) Bus Schedule"
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Show recipient selection
        if st.session_state.email_config['recipient_list']:
            selected_recipients = st.multiselect(
                "Select Recipients:", 
                st.session_state.email_config['recipient_list'],
                default=st.session_state.email_config['recipient_list']
            )
            recipient = ','.join(selected_recipients)  # Join multiple emails
        else:
            recipient = st.text_input("Recipient Email (or manage recipients below)")
            st.info("No saved recipients. Add some below or enter manually above.")
    
    with col2:
        cc_email = st.text_input("CC Email (Optional)")
    
    # Input for sender's name under "Best regards"
    sender_name = st.text_input("Your Name (for Best regards):", placeholder="Enter your name")
    
    if st.button("📧 Generate Mailto Link", type="primary"):
        if recipient:
            # Generate HTML table
            html_table = generate_schedule_html()
            
            # Create email body with better formatting
            email_body = f"""
BUS SCHEDULE NTUDB(M)

{html_table}

Best regards,
{sender_name if sender_name else '\n NTU Dragon Boat (M)'}
            """
            
            # Encode the mailto link
            mailto_link = f"mailto:{recipient}?subject={urllib.parse.quote(subject)}"
            if cc_email:
                mailto_link += f"&cc={urllib.parse.quote(cc_email)}"
            mailto_link += f"&body={urllib.parse.quote(email_body)}"
            
            # Display the mailto link
            st.markdown(f'[📧 Click here to open your email client](<{mailto_link}>)', unsafe_allow_html=True)
            st.success("Mailto link generated! Click the link above to open your email client.")
            
            # Add instructions to open the email app
            st.info("📱 After clicking the link, if the email app does not open automatically, please open your email app manually to send the message.")
        else:
            st.warning("Please enter a recipient email address.")

def manage_recipient_emails():
    """Manage recipient email list"""
    st.subheader("Manage Recipient Emails")
    
    # Add new recipient
    col1, col2 = st.columns([3, 1])
    with col1:
        new_email = st.text_input("Add Recipient Email:", placeholder="example@email.com")
    with col2:
        if st.button("Add Email"):
            if new_email and '@' in new_email:
                if new_email not in st.session_state.email_config['recipient_list']:
                    st.session_state.email_config['recipient_list'].append(new_email)
                    st.success(f"Added: {new_email}")
                    st.rerun()
                else:
                    st.warning("Email already exists in list")
            else:
                st.warning("Please enter a valid email address")
    
    # Display and manage existing recipients
    if st.session_state.email_config['recipient_list']:
        st.write("**Current Recipients:**")
        
        for i, email in enumerate(st.session_state.email_config['recipient_list']):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(f"{i+1}. {email}")
            with col2:
                if st.button("Remove", key=f"remove_email_{i}"):
                    st.session_state.email_config['recipient_list'].remove(email)
                    st.success(f"Removed: {email}")
                    st.rerun()
        
        # Clear all button
        if st.button("Clear All Recipients"):
            st.session_state.email_config['recipient_list'] = []
            st.success("All recipients cleared")
            st.rerun()
    else:
        st.info("No recipient emails added yet")

def main():
    st.set_page_config(
        page_title="Bus Passenger List Manager",
        page_icon="🚌",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    st.title("🚌 Enhanced Bus Passenger List Manager with Gemini AI")
    st.markdown("*AI-powered name extraction from images and text*")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Choose Section:", 
                          ["🔑 API Setup", "📝 Input & Processing", "⚙️ Bus Settings", "📋 Generated Output","📅Bus Schedule"])
    
    if tab == "🔑 API Setup":
        st.header("🔑 Gemini API Configuration")
        
        with st.expander("Instructions", expanded=True):
            st.markdown("""
            **To use Gemini AI for superior OCR and text processing:**

            1. Go to: https://makersuite.google.com/app/apikey
            2. Sign in with your Google account
            3. Create a new API key
            4. Copy and paste it below
            5. Click 'Save API Key AIzaSyDUwRw_QzcA0mT39hiAU6Gd4l0zftXxdII' 

            **Benefits of using Gemini:**
            • Much better text recognition from images
            • Understands context and can distinguish names from other text
            • Handles handwritten text and various fonts
            • More accurate than traditional OCR
            """)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            api_key = st.text_input("Gemini API Key:", 
                                  value=st.session_state.api_key, 
                                  type="password",
                                  help="Enter your Gemini API key here")
        
        with col2:
            if st.button("💾 Save API Key", type="primary"):
                if api_key.strip():
                    st.session_state.api_key = api_key.strip()
                    st.session_state.gemini_ocr.set_api_key(api_key.strip())
                    st.success("✅ API Key saved successfully!")
                else:
                    st.warning("Please enter a valid API key.")
            
            if st.button("🧪 Test API"):
                if st.session_state.api_key:
                    try:
                        result = st.session_state.gemini_ocr.extract_names_from_text("Test message: John Doe, Jane Smith")
                        if result:
                            st.success(f"✅ API test successful! Extracted: {', '.join(result)}")
                        else:
                            st.info("⚠️ API connected but no names extracted from test")
                    except Exception as e:
                        st.error(f"❌ API test failed: {str(e)}")
                else:
                    st.warning("Please save an API key first.")
        
        if st.session_state.api_key:
            st.success("✅ API Key configured - Ready to use Gemini AI!")
        else:
            st.info("❌ No API Key configured")
            
        st.info("📝 If no API key is provided, the app will use basic text processing (less accurate)")

    elif tab == "📝 Input & Processing":
        st.header("📝 Add Passengers")
        
        # Simple assignment selector at top
        location = st.selectbox("Add passengers to:", ["NTU", "Jurong East"], key="main_assign")
        
        # Image upload
        uploaded_file = st.file_uploader("Upload passenger list image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file and st.button("Process Image", type="primary"):
            image = Image.open(uploaded_file)
            try:
                if st.session_state.api_key:
                    names = st.session_state.gemini_ocr.extract_names_from_image(image)
                    for name in names:
                        if name not in st.session_state.bus_list[location]:
                            st.session_state.bus_list[location].append(name)
                    st.success(f"Added {len(names)} names to {location}")
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Text input
        text_input = st.text_area("Or paste names here:")
        if text_input and st.button("Process Text", type="primary"):
            try:
                if st.session_state.api_key:
                    names = st.session_state.gemini_ocr.extract_names_from_text(text_input)
                    date_info = st.session_state.gemini_ocr.extract_date_info_from_text(text_input)
                else:
                    names = st.session_state.fallback_processor.extract_names_from_text(text_input)
                    date_info = {"day": "", "date": ""}
                
                # Add names
                for name in names:
                    if name not in st.session_state.bus_list[location]:
                        st.session_state.bus_list[location].append(name)
                
                # Store date info if found
                if date_info.get("day") or date_info.get("date"):
                    if 'extracted_date_info' not in st.session_state:
                        st.session_state.extracted_date_info = {}
                    st.session_state.extracted_date_info.update(date_info)
                    st.info(f"Date info extracted: {date_info.get('day', '')} {date_info.get('date', '')}")
                
                st.success(f"Added {len(names)} names to {location}")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Bus info
        st.subheader("Bus Information")
        bus_text = st.text_area("Paste bus info (plate/phone):")
        bus_file = st.file_uploader("Or upload bus info image", type=['jpg', 'jpeg', 'png'], key="bus_img")
        
        if st.button("Extract Bus Info") and st.session_state.api_key:
            try:
                if bus_file:
                    bus_image = Image.open(bus_file)
                    bus_info = st.session_state.gemini_ocr.extract_bus_info_from_image(bus_image)
                elif bus_text:
                    bus_info = st.session_state.gemini_ocr.extract_bus_info_from_text(bus_text)
                else:
                    bus_info = {}
                
                if bus_info.get("bus_plate") or bus_info.get("phone"):
                    st.session_state.extracted_bus_info = bus_info
                    st.success("Bus info extracted! Check Bus Settings tab.")
                else:
                    st.warning("No bus info found")
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Show current lists
        st.subheader("Current Passengers")

        # Manual input
        manual_name = st.text_input("Add name manually:")
        if manual_name and st.button("Add Manual Name"):
            formatted_name = manual_name.strip().title()
            if formatted_name not in st.session_state.bus_list[location]:
                st.session_state.bus_list[location].append(formatted_name)
                st.success(f"Added {formatted_name} to {location}")
                st.rerun()
            else:
                st.warning("Name already exists")

        for loc in ["NTU", "Jurong East"]:
            with st.expander(f"{loc}: {len(st.session_state.bus_list[loc])} passengers"):
                if st.session_state.bus_list[loc]:
                    # Multiple selection for removal
                    selected_for_removal = st.multiselect(
                        f"Select names to remove from {loc}:", 
                        st.session_state.bus_list[loc],
                        key=f"remove_{loc}"
                    )
                    
                    if selected_for_removal and st.button(f"Remove Selected from {loc}", key=f"remove_btn_{loc}"):
                        for name in selected_for_removal:
                            st.session_state.bus_list[loc].remove(name)
                        st.success(f"Removed {len(selected_for_removal)} names from {loc}")
                        st.rerun()
                    
                    # Name editing
                    st.write("Edit names:")
                    for i, name in enumerate(st.session_state.bus_list[loc]):
                        col1, col2 = st.columns([3, 1])
                        new_name = col1.text_input(f"Edit:", value=name, key=f"edit_{loc}_{i}")
                        if col2.button("Update", key=f"update_{loc}_{i}"):
                            if new_name.strip() and new_name.strip() != name:
                                formatted_new = new_name.strip().title()
                                if formatted_new not in st.session_state.bus_list[loc]:
                                    idx = st.session_state.bus_list[loc].index(name)
                                    st.session_state.bus_list[loc][idx] = formatted_new
                                    st.success(f"Updated {name} to {formatted_new}")
                                    st.rerun()
                                else:
                                    st.warning("Name already exists")
                else:
                    st.write("No passengers added yet")

    elif tab == "⚙️ Bus Settings":
        st.header("⚙️ Bus Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Date & Time")
            
            # Date picker (replaces manual date/day input)
            selected_date = st.date_input(
                "Select Date:", 
                value=datetime.now() + timedelta(days=1),
                help="Choose the bus trip date"
            )
            
            # Auto-generate day and formatted date from the selected date
            day = selected_date.strftime("%A")  # Full weekday name
            date = selected_date.strftime("%B %d, %Y")  # Format: January 15, 2024
            
            # Show the generated day and date (read-only)
            st.info(f"Day: {day}")
            st.info(f"Formatted Date: {date}")
            
            # Time inputs remain the same
            default_ntu_time = st.session_state.get('extracted_date_info', {}).get('ntu_time', "0735hrs")
            default_je_time = st.session_state.get('extracted_date_info', {}).get('je_time', "0750hrs")
            
            ntu_time = st.text_input("NTU Time:", value=default_ntu_time)
            je_time = st.text_input("JE Time:", value=default_je_time)
            
        with col2:
            st.subheader("Bus Information")
            # Auto-populate from extracted info if available
            default_bus = st.session_state.get('extracted_bus_info', {}).get('bus_plate', '')
            default_phone = st.session_state.get('extracted_bus_info', {}).get('phone', '')

            bus_number = st.text_input("Bus Number:", value=default_bus, help="Bus plate number")
            driver_phone = st.text_input("Driver Phone:", value=default_phone, help="Driver's contact number")
            
        st.subheader("Pickup Locations")
        ntu_location = st.text_input("NTU Location:", 
                                value="Hall of Residence 8 & 9 Bus Stop")
        je_location = st.text_input("JE Location:", 
                                value="JE Venture Avenue")
        
        # Store settings in session state
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

    elif tab == "📋 Generated Output":
        st.header("📋 Generated Bus Information")
        
        # Check if we have passengers
        total_passengers = len(st.session_state.bus_list['NTU']) + len(st.session_state.bus_list['Jurong East'])
        
        if total_passengers == 0:
            st.warning("⚠️ No passengers added yet. Please add passengers in the 'Input & Processing' tab.")
        else:
            # Get settings
            if 'settings' not in st.session_state:
                # Use default settings if not configured
                tomorrow = datetime.now() + timedelta(days=1)
                st.session_state.settings = {
                    'day': tomorrow.strftime("%A"),
                    'date': tomorrow.strftime("%B %d, %Y"),
                    'ntu_time': "0735hrs",
                    'je_time': "0750hrs",
                    'bus_number': "",
                    'driver_phone': "",
                    'ntu_location': "Hall of Residence 8 & 9 Bus Stop",
                    'je_location': "JE Venture Avenue"
                }
            
            if st.button("🚌 Generate Bus Information", type="primary"):
                output = format_bus_info(st.session_state.settings)
                st.session_state.generated_output = output
            
            # Display generated output
            if 'generated_output' in st.session_state:
                st.subheader("Generated Output")
                st.code(st.session_state.generated_output, language="markdown")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📋 Copy to Clipboard"):
                        st.write("Copy the text above manually (Ctrl+C)")
                        st.info("📋 Select and copy the text from the box above")
                
                with col2:
                    st.download_button(
                        label="💾 Download as Text File",
                        data=st.session_state.generated_output,
                        file_name=f"bus_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                with col3:
                    st.download_button(
                        label="📄 Download as Markdown",
                        data=st.session_state.generated_output,
                        file_name=f"bus_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                # Show summary
                st.subheader("Summary")
                col_summary1, col_summary2, col_summary3 = st.columns(3)
                
                with col_summary1:
                    st.metric("Total Passengers", total_passengers)
                
                with col_summary2:
                    st.metric("NTU Passengers", len(st.session_state.bus_list['NTU']))
                
                with col_summary3:
                    st.metric("JE Passengers", len(st.session_state.bus_list['Jurong East']))

    elif tab == "📅Bus Schedule":
        st.header("Bus Schedule Management")
        
        tab_schedule = st.tabs(["Create Schedule", "View Table", "Send Email"])
        
        with tab_schedule[0]:
            create_schedule_table()
        
        with tab_schedule[1]:
            st.subheader("Current Schedule")
            display_schedule_table()
            
            if st.session_state.schedule_data:
                # Clear schedule button
                if st.button("Clear All Schedule", type="secondary"):
                    st.session_state.schedule_data = []
                    st.rerun()
        
        with tab_schedule[2]:
            # Add recipient management
            manage_recipient_emails()
            
            st.markdown("---")
            
            if st.session_state.schedule_data:
                send_schedule_email()
            else:
                st.warning("No schedule data to send. Please create schedule entries first.")
    # Sidebar status
    st.sidebar.markdown("---")
    st.sidebar.subheader("Status")
    
    # API status
    if st.session_state.api_key:
        st.sidebar.success("✅ Gemini AI Ready")
    else:
        st.sidebar.error("❌ No API Key")
    
    # Passenger count
    st.sidebar.info(f"🚌 Total Passengers: {len(st.session_state.bus_list['NTU']) + len(st.session_state.bus_list['Jurong East'])}")
    st.sidebar.info(f"📍 NTU: {len(st.session_state.bus_list['NTU'])}")
    st.sidebar.info(f"📍 JE: {len(st.session_state.bus_list['Jurong East'])}")

if __name__ == "__main__":
    main()
