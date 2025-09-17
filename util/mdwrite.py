from markdown import markdown
from os.path import join
import numpy as np
import pdfkit
import os


def md2pdf(src, dst=None):
    if dst is None:
        dst = src.replace('.md', '.pdf')
    with open(src, 'r') as f:
        t = markdown(f.read(), extensions=['markdown.extensions.tables'])
        pdfkit.from_string(t, dst)


def md2html(src, dst=None):
    if dst is None:
        dst = src.replace('.md', '.html')
    with open(src, 'r') as f:
        t = markdown(f.read(), extensions=['markdown.extensions.tables'])
    with open(dst, 'w') as f:
        f.write(t)


def html2pdf(src, dst=None):
    if dst is None:
        dst = src.replace('.html', '.html')
    with open(src, 'r') as f:
        pdfkit.from_string(f.read(), dst)


def strf(v):
    assert (not isinstance(v, dict))
    if isinstance(v, list): return "[" + ",".join([strf(a) for a in v]) + "]"
    elif isinstance(v, np.ndarray): return strf(v.tolist())
    else: return str(v)


class MDWrite:
    def __init__(self):
        self.root_path = os.getcwd()
        self.nume_curr_tab = -1
        self.curr_tab = 0
        self.reset()
        self.add_header = lambda level, text, tab=0: self.md_append(
            '\n\n' + '#' * level + ' ' + text + '\n', tab)
        self.add_item = lambda n_tab, text: self.md_append(
            '- ' + text + '\n', n_tab)
        self.add_nume = lambda n_tab, text: self.md_append(
            '%d. ' % self.nume_count(n_tab) + text + '\n', n_tab)
        self.add_text = lambda text, n_tab=None: self.md_append(
            text + '\n', n_tab)
        self.add_fig = lambda fig, size=600, align="center": self.md_append(
            '\n<p align="%s"><img src="file:///%s" width="%dpx"/></p>\n\n' % (
                align, join(self.root_path, fig), size), 0)

    def add_item_d(self, n_tab, dx, unroll_list=False):
        if isinstance(dx, dict):
            for k, v in dx.items():
                if isinstance(v, dict):
                    self.add_item(n_tab, k)
                    self.add_item_d(n_tab + 1, v, unroll_list)
                elif unroll_list and isinstance(v, list):
                    self.add_item(n_tab, k)
                    for a in v:
                        self.add_item_d(n_tab + 1, a, unroll_list)
                else:
                    self.add_item(n_tab, k + " : " + strf(v))
        elif isinstance(dx, list) and unroll_list:
            for a in dx:
                self.add_item_d(n_tab + 1, a, unroll_list)
        else:
            self.add_item(n_tab, strf(dx))

    def nume_count(self, tab):
        if self.nume_curr_tab != tab:
            self.nume_cnt = 1
        else:
            self.nume_cnt = self.nume_cnt + 1
        self.nume_curr_tab = tab
        return self.nume_cnt

    def md_append(self, text, tab=None):
        if tab is not None:
            self.curr_tab = tab
        else:
            tab = self.curr_tab
        self.md = self.md + ' ' * 2 * (tab) + text

    def reset(self):
        self.md = ""

    def add_table(self, table, n_tab=None, title=True):
        def table_row(a):
            out = "|"
            for x in a:
                out += '   {}   |'.format(x)
            return out + '\n'

        t = table.copy()
        for n in range(len(table) - 1):
            assert (len(table[n]) == len(table[n + 1]))
        if title == True: t.insert(1, [':-:'] * len(table[0]))
        self.add_text('')
        for a in t:
            self.md_append(table_row(a), n_tab)
        self.add_text('')

    def get_md_str(self):
        return self.md

    def save_md(self, fn):
        with open(fn, 'w') as f:
            f.write(self.md)

    def save_html(self, fn):
        fn_md = fn.replace('.html', '.md')
        self.save_md(fn_md)
        md2html(fn_md)

    def save_pdf(self, fn):
        fn_md = fn.replace('.pdf', '.md')
        self.save_md(fn_md)
        md2pdf(fn_md)

    def save(self, fn):
        self.save_html(fn)
        self.save_pdf(fn)


def test_mdw():
    mdw = MDWrite()
    t = [['a', 'b', 'c']] * 5
    mdw.add_header(1, "header1")
    mdw.add_text("this is a test XXXXXXXXXXXXXXXXX")
    mdw.add_text("this is a test ZZZZZZ")
    mdw.add_header(2, "header1.1")
    mdw.add_text("this is a test WWWW")
    mdw.add_item(0, "item1")
    mdw.add_item(0, "item2")
    mdw.add_item(1, "item2.1")
    mdw.add_item(1, "item2.2")
    mdw.add_item(2, "item2.2.1")
    mdw.add_item(2, "item2.2.2")
    mdw.add_nume(3, "item2.2.2.1")
    mdw.add_text("this is a test")
    mdw.add_text("this is a test2")
    mdw.add_nume(3, "item2.2.2.2")
    mdw.add_nume(3, "item2.2.2.3")
    mdw.add_table(t)
    mdw.add_nume(1, "item3")
    mdw.add_nume(1, "item3")
    mdw.add_table(t)
    mdw.add_fig("aaa.png", 200)
    mdw.add_fig("bbb.png", 300, 'left')
    mdw.add_header(2, "header1.2")
    mdw.save("my.md")
    print(mdw.get_md_str())


if __name__ == "__main__":
    test_mdw()