import pandas as pd


# get file names from user
Entryfile = input("provide dd_mm_yy ")
Exitfile = Entryfile

# set file paths
entry_path = "C:/Users/Mathi Shankar/OneDrive/Desktop/Survilence of HR Department to regulate In and Out-main/Survilence of HR Department-entry/Entry//"
exit_path = "C:/Users/Mathi Shankar/OneDrive/Desktop/Survilence of HR Department to regulate In and Out-main/Survilence of HR Department - exit/Exit//"

# extension
extension = ".csv"

#constant prefix for file name
constant1 = "Entry-"
constant2 = "Exit-"

    # read the files
f1 = pd.read_csv(entry_path + constant1 + Entryfile + extension )
f2 = pd.read_csv(exit_path + constant2  + Exitfile + extension)

    # merge the files
f3 = f1[["Name", "EmpId", "EntryTime"]].merge(f2[["Name", "EmpId", "ExitTime"]], on=["Name", "EmpId"], how="left")

    # create a new file
f3.to_csv("entrydata.csv", index=False)
