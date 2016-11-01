%.pdf: %.pmd
	@TEXINPUTS="../:" pandoc -s -S $< -o $@

