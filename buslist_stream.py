import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import urllib.parse

def initialize_session_state():
    if 'bus_list' not in st.session_state:
        st.session_state.bus_list = {}
    if 'locations' not in st.session_state:
        st.session_state.locations = ['NTU', 'Jurong East']
    if 'schedule_data' not in st.session_state:
        st.session_state.schedule_data = []
    if 'recipient_emails' not in st.session_state:
        st.session_state.recipient_emails = []
    
    # Initialize bus_list for all locations
    for loc in st.session_state.locations:
        if loc not in st.session_state.bus_list:
            st.session_state.bus_list[loc] = []


def format_bus_info(settings):
    """Format the bus information in the new style"""
    output = f"Bus Information - {settings['day']}, {settings['date']}\n"
    output += "Take note of the bus timing\n"
    
    if settings['bus_number']:
        output += f"Bus Plate Number: {settings['bus_number']}\n"
    if settings['driver_phone']:
        output += f"Phone Number: {settings['driver_phone']}\n"
    
    section_num = 1
    
    for location in st.session_state.locations:
        if st.session_state.bus_list.get(location):
            time_key = location.lower().replace(' ', '_') + '_time'
            location_key = location.lower().replace(' ', '_') + '_location'
            
            time_val = settings.get(time_key, '')
            location_val = settings.get(location_key, '')
            
            output += f"𝟏. {location} ({time_val})\n" if section_num == 1 else f"𝟐. {location} ({time_val})\n"
            output += f"Location: {location_val}\n"
            
            for name in sorted(st.session_state.bus_list[location]):
                output += f"{name}\n"
            output += "\n"
            section_num += 1
    
    return output


def passenger_management():
    """Passenger List Management"""
    st.header("📝 Passenger List Management")
    
    # Location Management
    with st.expander("➕ Manage Locations"):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_location = st.text_input("Add new location:", placeholder="e.g., Clementi")
        with col2:
            if st.button("Add Location"):
                if new_location and new_location not in st.session_state.locations:
                    st.session_state.locations.append(new_location)
                    st.session_state.bus_list[new_location] = []
                    st.success(f"Added: {new_location}")
                    st.rerun()
        
        if len(st.session_state.locations) > 0:
            st.write("**Current Locations:**")
            for i, loc in enumerate(st.session_state.locations):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"• {loc}")
                with col2:
                    if st.button("Remove", key=f"rm_loc_{i}"):
                        if loc in st.session_state.bus_list:
                            del st.session_state.bus_list[loc]
                        st.session_state.locations.remove(loc)
                        st.rerun()
    
    st.markdown("---")
    
    # Add Passengers
    st.subheader("Add Passengers")
    
    location = st.selectbox("Select Location:", st.session_state.locations)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        manual_name = st.text_input("Enter passenger name:")
    with col2:
        st.write("")
        st.write("")
        if st.button("➕ Add", type="primary"):
            if manual_name:
                formatted = manual_name.strip().title()
                if formatted not in st.session_state.bus_list[location]:
                    st.session_state.bus_list[location].append(formatted)
                    st.success(f"✓ Added {formatted} to {location}")
                    st.rerun()
                else:
                    st.warning("Name already exists in this location")
    
    st.markdown("---")
    
    # Show Passenger Lists
    st.subheader("Current Passengers")
    
    for loc in st.session_state.locations:
        count = len(st.session_state.bus_list.get(loc, []))
        with st.expander(f"📍 {loc}: {count} passenger(s)"):
            if st.session_state.bus_list.get(loc):
                for i, name in enumerate(st.session_state.bus_list[loc]):
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.text(f"• {name}")
                    with col2:
                        if st.button("❌", key=f"rm_{loc}_{i}"):
                            st.session_state.bus_list[loc].remove(name)
                            st.rerun()
            else:
                st.info("No passengers added yet")
    
    # Summary
    total = sum(len(st.session_state.bus_list.get(loc, [])) for loc in st.session_state.locations)
    st.metric("Total Passengers", total)


def bus_settings():
    """Bus Settings"""
    st.header("⚙️ Bus Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Date & Time")
        selected_date = st.date_input("Date:", value=datetime.now() + timedelta(days=1))
        day = selected_date.strftime("%A")
        date = selected_date.strftime("%b %d, %Y").upper()
        
        st.info(f"**{day}, {date}**")
    
    with col2:
        st.subheader("🚌 Bus Information")
        bus_number = st.text_input("Bus Plate Number:", placeholder="e.g., PC8811T")
        driver_phone = st.text_input("Phone Number:", placeholder="e.g., 97740325")
    
    st.markdown("---")
    st.subheader("📍 Locations & Timings")
    
    settings = {
        'day': day,
        'date': date,
        'bus_number': bus_number,
        'driver_phone': driver_phone
    }
    
    for loc in st.session_state.locations:
        st.write(f"**{loc}**")
        col1, col2 = st.columns(2)
        
        with col1:
            time_val = st.text_input(f"Time:", value="0725hrs", key=f"time_{loc}")
        with col2:
            location_val = st.text_input(f"Location:", value=f"{loc} Bus Stop", key=f"loc_{loc}")
        
        settings[f"{loc.lower().replace(' ', '_')}_time"] = time_val
        settings[f"{loc.lower().replace(' ', '_')}_location"] = location_val
    
    st.session_state.settings = settings


def output_generation():
    """Output Generation"""
    st.header("📄 Generate Output")
    
    total = sum(len(st.session_state.bus_list.get(loc, [])) for loc in st.session_state.locations)
    
    if total == 0:
        st.warning("⚠️ No passengers added yet. Please add passengers first.")
        return
    
    if 'settings' not in st.session_state:
        tomorrow = datetime.now() + timedelta(days=1)
        st.session_state.settings = {
            'day': tomorrow.strftime("%A"),
            'date': tomorrow.strftime("%b %d, %Y").upper(),
            'bus_number': "",
            'driver_phone': ""
        }
        for loc in st.session_state.locations:
            st.session_state.settings[f"{loc.lower().replace(' ', '_')}_time"] = "0725hrs"
            st.session_state.settings[f"{loc.lower().replace(' ', '_')}_location"] = f"{loc} Bus Stop"
    
    if st.button("🔄 Generate Output", type="primary"):
        output = format_bus_info(st.session_state.settings)
        st.session_state.generated_output = output
        st.success("✓ Output generated successfully!")
    
    if 'generated_output' in st.session_state:
        st.subheader("Preview:")
        st.text_area("Output", st.session_state.generated_output, height=400)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📥 Download as TXT",
                data=st.session_state.generated_output,
                file_name=f"bus_list_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
        
        with col2:
            st.download_button(
                "📥 Download as MD",
                data=st.session_state.generated_output,
                file_name=f"bus_list_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
        
        with col3:
            st.metric("Total", total)


def create_schedule():
    """Create new schedule entries"""
    st.header("📅 Add Schedule Entry")
    
    today = datetime.now()
    upcoming_saturday = today + timedelta((5 - today.weekday()) % 7)
    
    with st.form("schedule_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date = st.date_input("Date", value=upcoming_saturday)
            activity = st.text_input("Activity", value="Dragon Boat (M)")
            pickup_point = st.text_input("Pick-Up Point", value="NTU Hall 8 & 9 Bus Stop")
        
        with col2:
            departure_time = st.text_input("Departure Time", value="0740 hrs")
            bus_capacity = st.selectbox("Bus Capacity", ["1 x 20 seater bus", "1 x 40 seater bus"])
            return_time = st.text_input("Return Time", value="NIL")
        
        with col3:
            contact_name = st.text_input("Contact Name", value="Jing Hang")
            contact_number = st.text_input("Contact Number", value="88479136")
        
        st.subheader("Destinations")
        num_destinations = st.number_input("Number of Destinations", min_value=1, max_value=5, value=2)
        
        destinations = []
        for i in range(num_destinations):
            dest = st.text_input(f"Destination {i+1}", 
                                value="Venture Ave (Jurong East)" if i == 0 else "SDBA" if i == 1 else "",
                                key=f"dest_{i}")
            if dest:
                destinations.append(dest)
        
        submitted = st.form_submit_button("➕ Add to Schedule", type="primary")
        
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
            
            duplicate_found = False
            for existing in st.session_state.schedule_data:
                if (existing['date'] == new_entry['date'] and 
                    existing['departure_time'] == new_entry['departure_time']):
                    duplicate_found = True
                    st.warning(f"Schedule already exists for {new_entry['date']} at {new_entry['departure_time']}")
                    break
            
            if not duplicate_found:
                st.session_state.schedule_data.append(new_entry)
                st.success(f"✓ Added schedule for {new_entry['date']} ({new_entry['day']})")
                st.rerun()


def view_schedule():
    """View and manage schedule"""
    st.header("📋 Current Schedule")
    
    if not st.session_state.schedule_data:
        st.info("No schedule entries yet. Create one in the 'Create Schedule' tab.")
        return
    
    display_data = []
    for entry in st.session_state.schedule_data:
        destinations_str = ", ".join([f"{i+1}. {dest}" for i, dest in enumerate(entry['destinations'])])
        
        display_data.append({
            'Date': f"{entry['date']}\n{entry['day']}",
            'Activity': entry['activity'],
            'Pick-Up': entry['pickup_point'],
            'Time': entry['departure_time'],
            'Destination': destinations_str,
            'Return': entry['return_time'],
            'Contact': f"{entry['contact_name']}\n{entry['contact_number']}",
            'Seats': entry['bus_capacity']
        })
    
    df = pd.DataFrame(display_data)
    st.dataframe(df, use_container_width=True, height=400)
    
    st.markdown("---")
    st.subheader("Remove Entry")
    
    if len(st.session_state.schedule_data) > 0:
        selected_index = st.selectbox(
            "Select entry to remove:",
            range(len(st.session_state.schedule_data)),
            format_func=lambda x: f"{st.session_state.schedule_data[x]['date']} - {st.session_state.schedule_data[x]['activity']}"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Remove Selected Entry"):
                st.session_state.schedule_data.pop(selected_index)
                st.success("Entry removed")
                st.rerun()
        
        with col2:
            if st.button("⚠️ Clear All Schedules"):
                st.session_state.schedule_data = []
                st.success("All schedules cleared")
                st.rerun()


def generate_schedule_html():
    """Generate HTML table for email"""
    if not st.session_state.schedule_data:
        return ""
    
    html = """<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;">
    <thead>
    <tr style="background-color: #4CAF50; color: white;">
        <th>Date (2025)</th>
        <th>Activity</th>
        <th>Pick-Up Point</th>
        <th>Departure Time</th>
        <th>Destination</th>
        <th>Return Time</th>
        <th>Name & Contact No.</th>
        <th>Seats</th>
        <th>Price</th>
    </tr>
    </thead>
    <tbody>"""
    
    for entry in st.session_state.schedule_data:
        destinations_html = "<br>".join([f"{i+1}. {dest}" for i, dest in enumerate(entry['destinations'])])
        
        html += f"""
    <tr>
        <td>{entry['date']}<br><strong>{entry['day']}</strong></td>
        <td>{entry['activity']}</td>
        <td>{entry['pickup_point']}</td>
        <td>{entry['departure_time']}</td>
        <td>{destinations_html}</td>
        <td>{entry['return_time']}</td>
        <td>{entry['contact_name']},<br>{entry['contact_number']}</td>
        <td>{entry['bus_capacity']}</td>
        <td></td>
    </tr>"""
    
    html += """
    </tbody>
    </table>"""
    return html


def send_schedule_email():
    """Generate mailto link for schedule email"""
    st.header("📧 Email Schedule")
    
    if not st.session_state.schedule_data:
        st.warning("⚠️ No schedule data available. Please create a schedule first.")
        return
    
    # Manage recipients
    with st.expander("👥 Manage Recipients"):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_email = st.text_input("Add recipient email:", placeholder="example@email.com")
        with col2:
            st.write("")
            st.write("")
            if st.button("Add"):
                if new_email and '@' in new_email:
                    if new_email not in st.session_state.recipient_emails:
                        st.session_state.recipient_emails.append(new_email)
                        st.success(f"✓ Added: {new_email}")
                        st.rerun()
                    else:
                        st.warning("Email already exists")
                else:
                    st.warning("Enter valid email")
        
        if st.session_state.recipient_emails:
            st.write("**Current recipients:**")
            for i, email in enumerate(st.session_state.recipient_emails):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.text(f"• {email}")
                with col2:
                    if st.button("❌", key=f"rm_email_{i}"):
                        st.session_state.recipient_emails.remove(email)
                        st.rerun()
    
    st.markdown("---")
    
    # Email composition
    st.subheader("✉️ Compose Email")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.recipient_emails:
            selected_recipients = st.multiselect(
                "Recipients:",
                st.session_state.recipient_emails,
                default=st.session_state.recipient_emails
            )
        else:
            selected_recipients = []
            manual_recipient = st.text_input("Recipient Email:")
            if manual_recipient:
                selected_recipients = [manual_recipient]
    
    with col2:
        cc_email = st.text_input("CC (Optional):")
    
    recipient_name = st.text_input("Recipient's Name:", placeholder="Ms. Ivyna")
    sender_name = st.text_input("Your Name:", placeholder="Your name")
    
    if st.button("📧 Open Email Client", type="primary"):
        if not selected_recipients:
            st.warning("Please enter at least one recipient email")
            return
        
        recipient = ','.join(selected_recipients)
        subject = "NTUDB(M) Bus Schedule"
        html_table = generate_schedule_html()
        
        # Create plain text version for email body
        text_schedule = "\n\nSchedule Details:\n"
        for entry in st.session_state.schedule_data:
            text_schedule += f"\n{entry['date']} ({entry['day']})\n"
            text_schedule += f"Activity: {entry['activity']}\n"
            text_schedule += f"Pick-Up: {entry['pickup_point']} at {entry['departure_time']}\n"
            text_schedule += f"Destinations: {', '.join(entry['destinations'])}\n"
            text_schedule += f"Contact: {entry['contact_name']} ({entry['contact_number']})\n"
            text_schedule += "-" * 50 + "\n"
        
        email_body = f"Dear {recipient_name if recipient_name else 'Recipient'},\n\n"
        email_body += "The Bus Schedule for NTU Dragon Boat (M) is as follows:\n"
        email_body += text_schedule
        email_body += f"\n\nThank you for your support!\n\n"
        email_body += f"Warm regards,\n{sender_name if sender_name else 'NTU Dragon Boat (M)'}"
        
        # Build mailto URL
        mailto_parts = [f"mailto:{recipient}"]
        params = []
        
        params.append(f"subject={urllib.parse.quote(subject)}")
        
        if cc_email:
            params.append(f"cc={urllib.parse.quote(cc_email)}")
        
        params.append(f"body={urllib.parse.quote(email_body)}")
        
        mailto_link = mailto_parts[0] + "?" + "&".join(params)
        
        # Display the link
        st.markdown(f'<a href="{mailto_link}" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">📧 Click here to open email client</a>', unsafe_allow_html=True)
        
        st.success("✓ Email link generated! Click the link above to open your email client.")
        st.info("💡 Tip: If your email client doesn't open automatically, copy the schedule text below and paste it into your email manually.")
        
        with st.expander("📋 Copy Schedule Text"):
            st.text_area("Schedule Text:", email_body, height=300)


def main():
    st.set_page_config(
        page_title="Bus List Manager",
        page_icon="🚌",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🚌 Bus Passenger List Manager")
    
    # Main tabs
    tab = st.sidebar.radio("📋 Navigation:", ["Passenger List", "Schedule Manager"])
    
    if tab == "Passenger List":
        subtabs = st.tabs(["📝 Add Passengers", "⚙️ Bus Settings", "📄 Generate Output"])
        
        with subtabs[0]:
            passenger_management()
        
        with subtabs[1]:
            bus_settings()
        
        with subtabs[2]:
            output_generation()
    
    elif tab == "Schedule Manager":
        subtabs = st.tabs(["➕ Create Schedule", "📋 View & Manage", "📧 Send Email"])
        
        with subtabs[0]:
            create_schedule()
        
        with subtabs[1]:
            view_schedule()
        
        with subtabs[2]:
            send_schedule_email()
    
    # Sidebar status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Status")
    
    total = sum(len(st.session_state.bus_list.get(loc, [])) for loc in st.session_state.locations)
    st.sidebar.metric("Total Passengers", total)
    
    for loc in st.session_state.locations:
        count = len(st.session_state.bus_list.get(loc, []))
        st.sidebar.text(f"{loc}: {count}")
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Quick Tips:**\n\n• Add locations in 'Manage Locations'\n• Input passengers manually\n• Configure settings before generating output")


if __name__ == "__main__":
    main()
