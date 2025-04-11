a = ["a", "b"]
# Text_Params[0] + ": %{customdata[0]}<br>" for i in range(len(Text_Params))
print(str.join("",Text_Params[0] + ": %{customdata[0]}<br>" for i in range(len(Text_Params))))
