import tkinter as tk
import random

def create_widgets():
    global buttons, canvas
    buttons = []
    # Create a 2x2 grid of buttons
    for i in range(2):
        row = []
        for j in range(2):
            # Initialize buttons with red color and bind them to the button_pressed function
            btn = tk.Button(root, bg='red', command=lambda i=i, j=j: button_pressed(i, j))
            btn.grid(row=i, column=j, sticky='nsew')
            row.append(btn)
        buttons.append(row)

    # Set row and column weights to evenly distribute space
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)


def generate_sequence():
    global sequence
    # Generate a random permutation of button positions
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    random.shuffle(positions)
    sequence = positions
    print("Sequence:", sequence)  # Output the shuffled sequence for debugging purposes

def light_up_next_button():
    global current_index
    # Light up the next button in the sequence
    if current_index < len(sequence):
        i, j = sequence[current_index]
        buttons[i][j].config(bg='yellow')  # Change button color to yellow
        root.update_idletasks()  # Update window to immediately reflect color change
    else:
        # If all buttons are green, display "Good Job"
        label = tk.Label(root, text="Good Job", font=("Arial", 48), bg='black', fg='white')
        label.grid(row=0, column=0, columnspan=2, rowspan=2, sticky='nsew')
        root.update_idletasks()  # Update window to immediately reflect color change

def button_pressed(i, j):
    global current_index
    # Triggered when a button is pressed
    if current_index < len(sequence) and sequence[current_index] == (i, j):
        buttons[i][j].config(bg='green')  # Change color to green for correctly pressed button
        root.update_idletasks()  # Update window to immediately reflect color change
        current_index += 1  # Move to the next button in the sequence
        light_up_next_button()  # Light up the next button

def quit_program(event):
    # Quit the application
    root.quit()

# Main program
root = tk.Tk()
root.attributes("-fullscreen", True)  # Set window to fullscreen
root.bind("<q>", quit_program)  # Bind 'q' key to quit program

buttons = []  # List to hold buttons
sequence = []  # Holds the sequence of lit buttons
current_index = 0  # Current index in the button sequence

create_widgets()  # Create buttons and canvas
generate_sequence()  # Generate a random sequence of button presses
light_up_next_button()  # Light up the first button in the sequence

root.mainloop()  # Start the main event loop
