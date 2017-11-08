import wx
import wx.lib.scrolledpanel as scrolled
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


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
        new_panel = self.add_panel(self.scrolled_panel, 'Loaded Verbatims')
        self.spSizer.Add(new_panel)
        self.scrolled_panel.SetSizer(self.spSizer)
        # --------------------
        # Button panel stuff

        button_panel = wx.Panel(self.panel)
        load_button = wx.Button(button_panel, label="Load Data")
        load_button.Bind(wx.EVT_BUTTON, self.onLoad)
        add_button = wx.Button(button_panel, label="Add Category")
        add_button.Bind(wx.EVT_BUTTON, self.onAdd)
        delete_button = wx.Button(button_panel, label="Delete Category")
        delete_button.Bind(wx.EVT_BUTTON, self.onDelete)
        go_button = wx.Button(button_panel, label="GO")
        go_button.Bind(wx.EVT_BUTTON, self.recalculate)
        lda_button = wx.Button(button_panel, label="LDA!")
        lda_button.Bind(wx.EVT_BUTTON, self.lda)
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.Add(load_button)
        button_sizer.Add(add_button)
        button_sizer.Add(delete_button)
        button_sizer.Add(lda_button)
        button_sizer.Add(go_button)
        button_panel.SetSizer(button_sizer)

        panelSizer = wx.BoxSizer(wx.VERTICAL)
        panelSizer.AddSpacer(20)
        panelSizer.Add(self.scrolled_panel, 1, wx.EXPAND)
        panelSizer.Add(button_panel)
        self.panel.SetSizer(panelSizer)

    # ----------------------------------------------------------------------
    def add_panel(self, add_to_what, category_label):
        this_panel = wx.Panel(add_to_what)
        panel_title = wx.TextCtrl(this_panel, id=1, value=category_label, size=(300, 20))
        line_count = wx.StaticText(this_panel, id=3, label="Lines = 0")
        keyword_box = wx.TextCtrl(this_panel, id=2, style=wx.TE_MULTILINE, size=(300, 50))
        text_box = wx.TextCtrl(this_panel, id=4, style=wx.TE_MULTILINE | wx.HSCROLL, size=(300, 500))
        internal_sizer = wx.BoxSizer(wx.VERTICAL)
        internal_sizer.Add(panel_title)
        internal_sizer.Add(keyword_box)
        internal_sizer.Add(line_count)
        internal_sizer.Add(text_box)
        this_panel.SetSizer(internal_sizer)
        self.panel_list.append(this_panel)
        return this_panel

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
                    # print('utf-8')
            except:
                with open(pathname, 'r', encoding='ANSI') as file:
                    for line in file:
                        self.original_verbatims.append(line)
                    # print('ansi')
        text_box = self.panel_list[0].FindWindow(4)
        text_box.Clear()
        for v in self.original_verbatims:
            text_box.AppendText(v)
        lines_text = self.panel_list[0].FindWindow(3)
        lines_text.SetLabel("Lines = %d" % text_box.GetNumberOfLines())
        self.panel_list[0].Layout()

    def onAdd(self, event):
        new_panel = self.add_panel(self.scrolled_panel, "Category %d" % len(self.panel_list))
        self.spSizer.Add(new_panel)
        self.scrolled_panel.Layout()
        self.scrolled_panel.SetupScrolling()

    def onDelete(self, event):
        # print("in onDelete")
        if len(self.panel_list) == 1:
            return 'nope'
        for p in self.panel_list[-1].GetChildren():
            p.Destroy()
        take_out = self.panel_list.pop(-1)

    def recalculate(self, event):
        original_list = []
        for p in self.panel_list:
            title_text = p.FindWindow(1)
            panel_title = title_text.GetLabel()
            text_box = p.FindWindow(4)
            if not original_list:
                original_list = self.original_verbatims[:]
                continue
            keyword_box = p.FindWindow(2)
            keyword_list = [f.lower().strip() for f in keyword_box.GetValue().split(',') if f not in ['', ' ']]
            # print(keyword_list)
            keyword_long = [m for m in keyword_list if len(m) > 3]
            keyword_short = [m for m in keyword_list if len(m) <= 3]
            text_box.Clear()
            transfer_list = []
            for t in original_list:
                if any(word in t.lower() for word in keyword_long):
                    transfer_list.append(t)
                elif any(re.search(r"\b%s\b" % word, t.lower()) for word in keyword_short):
                    transfer_list.append(t)
            # transfer_list = list(set(transfer_list))
            for verbatim in transfer_list:
                original_list.remove(verbatim)
                text_box.AppendText(verbatim)
            num_lines = text_box.GetNumberOfLines()
            lines_text = p.FindWindow(3)
            lines_text.SetLabel("Lines = %d" % num_lines)
            p.Layout()
        orig_text_box = self.panel_list[0].FindWindow(4)
        orig_text_box.Clear()
        for remaining_verbatim in original_list:
            orig_text_box.AppendText(remaining_verbatim)
        lines_text = self.panel_list[0].FindWindow(3)
        lines_text.SetLabel("Lines = %d" % orig_text_box.GetNumberOfLines())
        no_cat_title = self.panel_list[0].FindWindow(2)
        no_cat_title.Clear()
        no_cat_title.AppendText("<List of Verbatims with No Category Match>")
        self.panel_list[0].Layout()

    def lda(self, event):
        tf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        tf = tf_vectorizer.fit_transform(self.original_verbatims)
        lda = LatentDirichletAllocation(max_iter=150, learning_method='online', learning_offset=50., random_state=0,
                                        n_topics=len(self.panel_list)-1)
        # print(len(self.panel_list))
        lda.fit(tf)
        tf_feature_names = tf_vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(lda.components_):
            message = ", ".join([tf_feature_names[i] for i in topic.argsort()[:-10 - 1:-1]])
            keyword_box = self.panel_list[topic_idx+1].FindWindow(2)
            keyword_box.Clear()
            keyword_box.AppendText(message)
        self.recalculate(wx.EVT_BUTTON)


# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm().Show()
    app.MainLoop()
