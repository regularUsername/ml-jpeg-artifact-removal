from IPython.core.magic import Magics, magics_class, cell_magic

@magics_class
class MyMagics(Magics):
    
    @cell_magic
    def skip(self, x,y):
        raw_code = 'print("skipping cell!")'
        # Comment out this line if you actually want to run the code.
        self.shell.run_cell(raw_code, store_history=False)

    @cell_magic
    def comment(self,arg,cell):
        self.shell.set_next_input("\n".join("# "+l for l in cell.splitlines()), replace=True)
    @cell_magic
    def uncomment(self,arg,cell):
        pass

ip = get_ipython()
ip.register_magics(MyMagics)