# Optical_flow
Please run the code by executing:

    python3 main.py [list of arguments]

After completing each section, you can enable the flag for that part. For instance,
if you are done with depth, you should execute:

    python3 main.py --plot_flow

You can also pass your confidence threshold to the program. For example, if a confidence score of 5 is used, you should execute:

    python3 main.py --depth --threshmin 5
