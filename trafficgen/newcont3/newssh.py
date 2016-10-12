import paramiko
import sys
import time

root_command = "whoami\n"
root_command_result = "root"

def send_string_and_wait(command, wait_time, should_print):
    # Send the su command
    shell.send(command)

    # Wait a bit, if necessary
    time.sleep(wait_time)

    # Flush the receive buffer
    receive_buffer = shell.recv(1024)

    # Print the receive buffer, if necessary
    if should_print:
        print receive_buffer

def send_string_and_wait_for_string(command, wait_string, should_print):
    # Send the su command
    shell.send(command)

    # Create a new receive buffer
    receive_buffer = ""

    while not wait_string in receive_buffer:
        # Flush the receive buffer
        receive_buffer += shell.recv(1024)

    # Print the receive buffer, if necessary
    if should_print:
        print receive_buffer

# Get the command-line arguments
system_ip = sys.argv[1]
system_username = sys.argv[2]
system_ssh_password = "adeola"
system_su_password = "adeola"
system_tcp_command = sys.argv[3]

# Create an SSH client
client = paramiko.SSHClient()

# Make sure that we add the remote server's SSH key automatically
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Connect to the client
client.connect(system_ip, username=system_username, password=system_ssh_password)

# Create a raw shell
shell = client.invoke_shell()

# Send the su command
send_string_and_wait(system_tcp_command+"\n", 1, True)

# Send the client's su password followed by a newline
send_string_and_wait(system_su_password + "\n", 1, True)

# Send the install command followed by a newline and wait for the done string
#send_string_and_wait_for_string(root_command, root_command_result, True)

# Close the SSH connection
client.close()

