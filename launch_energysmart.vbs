' Launches EnergySmart without a visible terminal window once everything is
' installed. The window is shown instead the very first time (so pip install
' progress is visible) or if Python can't be found (so the user sees why).

Dim shell, fso, scriptDir, batPath
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
batPath = scriptDir & "\run_app.bat"

If Not fso.FileExists(batPath) Then
    MsgBox "Could not find run_app.bat in:" & vbCrLf & scriptDir & vbCrLf & vbCrLf & _
           "Make sure launch_energysmart.vbs stays in the same folder as run_app.bat.", _
           vbCritical, "EnergySmart"
    WScript.Quit 1
End If

Dim pythonFound, streamlitFound
pythonFound = CommandSucceeds("where python")
If Not pythonFound Then
    MsgBox "Python was not found on this PC." & vbCrLf & vbCrLf & _
           "Install Python from https://www.python.org/downloads/" & vbCrLf & _
           "(tick ""Add python.exe to PATH"" during setup), then try again.", _
           vbExclamation, "EnergySmart"
    WScript.Quit 1
End If

streamlitFound = CommandSucceeds("python -m streamlit --version")

If streamlitFound Then
    ' Everything is already installed - launch completely hidden, no terminal flash.
    shell.CurrentDirectory = scriptDir
    shell.Run """" & batPath & """", 0, False
Else
    ' First run: packages still need installing, so show the window for that.
    shell.CurrentDirectory = scriptDir
    shell.Run """" & batPath & """", 1, False
End If

Function CommandSucceeds(cmd)
    On Error Resume Next
    Dim exec
    Set exec = shell.Exec("cmd /c " & cmd & " >nul 2>&1")
    Do While exec.Status = 0
        WScript.Sleep 50
    Loop
    CommandSucceeds = (exec.ExitCode = 0)
    On Error Goto 0
End Function
