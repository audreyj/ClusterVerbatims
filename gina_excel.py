import wx
import wx.lib.scrolledpanel as scrolled
import re
from sklearn.externals import joblib


########################################################################
class MyForm(wx.Frame):
    # ----------------------------------------------------------------------
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Category-Maker", size=(1500, 800))
        self.panel_list = []
        self.original_verbatims = []

        # Add a panel so it looks the correct on all platforms
        self.panel = wx.Panel(self, wx.ID_ANY)
        # --------------------
        # Scrolled panel stuff
        self.scrolled_panel = scrolled.ScrolledPanel(self.panel, -1, style=wx.TAB_TRAVERSAL, name="panel1")
        self.scrolled_panel.SetAutoLayout(1)
        self.scrolled_panel.SetupScrolling()
        self.spSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.scrolled_panel.SetSizer(self.spSizer)
        # --------------------
        # Button panel stuff

        button_panel = wx.Panel(self.panel)
        load_button = wx.Button(button_panel, label="Load Data")
        load_button.Bind(wx.EVT_BUTTON, self.onLoad)
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.Add(load_button)
        button_panel.SetSizer(button_sizer)

        panelSizer = wx.BoxSizer(wx.VERTICAL)
        panelSizer.AddSpacer(20)
        panelSizer.Add(self.scrolled_panel, 1, wx.EXPAND)
        panelSizer.Add(button_panel)
        self.panel.SetSizer(panelSizer)

    # ----------------------------------------------------------------------
    def onLoad(self, event):
        self.original_verbatims = []
        with wx.FileDialog(self, "Open Data File", style=wx.FD_OPEN) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            pathname = fileDialog.GetPath()
            try:
                with open(pathname, 'r', encoding='utf-8') as file:
                    for line in file:
                        self.original_verbatims.append(line)
                    print('utf-8')
            except:
                with open(pathname, 'r', encoding='ANSI') as file:
                    for line in file:
                        self.original_verbatims.append(line)
                    print('ansi')
        text_box = self.panel_list[0].FindWindow(4)
        text_box.Clear()
        for v in self.original_verbatims:
            text_box.AppendText(v)
        lines_text = self.panel_list[0].FindWindow(3)
        lines_text.SetLabel("Lines = %d" % text_box.GetNumberOfLines())
        self.panel_list[0].Layout()


# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm().Show()
    app.MainLoop()
