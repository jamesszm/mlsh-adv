merge:
	git checkout master && git merge --no-edit --no-ff bohan && git push \
	origin master && git checkout bohan