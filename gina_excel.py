import wx
import wx.lib.scrolledpanel as scrolled
import xlsxwriter
from sklearn.externals import joblib
from sklearn.neighbors import typedefs
from sklearn import svm
from sklearn import feature_extraction
import string
from collections import Counter

########################################################################
class MyForm(wx.Frame):
    # ----------------------------------------------------------------------
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Category-Maker", size=(400, 400))
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
            clf = joblib.load('data/nps_model_file.pkl')
            vectorizer = joblib.load('data/nps_vectorizer.pkl')

            total_verbatim_count = 0
            output_predictions = []
            prediction_counter = Counter()
            for v in self.original_verbatims:
                total_verbatim_count += 1
                verbatim_one = v.split('\n')[0]
                verbatim_two = verbatim_one.lower()
                verbatim = ''.join([l for l in verbatim_two if l not in string.punctuation])
                x_test = vectorizer.transform([verbatim])
                prediction = clf.predict(x_test)[0]
                # print(verbatim, prediction)
                output_predictions.append(prediction)
                prediction_counter[prediction] += 1

        with wx.FileDialog(self, "Save Excel File", wildcard="xlsx files (*.xlsx)|*.xlsx",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            out_pathname = fileDialog.GetPath()
            workbook = xlsxwriter.Workbook(out_pathname)
            worksheet = workbook.add_worksheet()
            row = 0
            col = 0
            for e, s in enumerate(self.original_verbatims):
                worksheet.write(row, col, output_predictions[e])
                worksheet.write(row, col+1, s)
                row += 1

            worksheet2 = workbook.add_worksheet()
            row = 0
            col = 0
            for m in prediction_counter.most_common():
                worksheet2.write(row, col, m[0])
                worksheet2.write(row, col+1, m[1])
                row += 1


# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm().Show()
    app.MainLoop()
