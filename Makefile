merge:
	git checkout master && git merge bohan --no-ff && git push origin master \
	&& git checkout bohan