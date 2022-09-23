def Check_Type(y_pred_label,y_test_label,df,test_index,filename,dk):
    i=0
    type1=[]
    type2=[]
    true_index=[]#跌倒预测为跌倒的索引
    before_fall=[]#原跌倒
    after_fall=[]#后跌倒
    adl=[]#'将原始跌倒标签转换为非跌倒标签后预测为非跌倒的样本个数
    while(i<len(test_index)):
        if(y_test_label[i]==0 and y_pred_label[i]==1):
            type1.append(df['annotation_1'][test_index[i]])
            type2.append(df['annotation_2'][test_index[i]])
        if(y_test_label[i]==1 and y_pred_label[i]==1):#预测正确
            true_index.append(test_index[i])
            if(df['annotation_1'][test_index[i]]==1 and dk['annotation_1'][test_index[i]]==1):
                before_fall.append(test_index[i])
        if(y_test_label[i]==0 and y_pred_label[i]==1):
            if(df['annotation_1'][test_index[i]]==0 and dk['annotation_1'][test_index[i]]==1):
                after_fall.append(test_index[i])
        if(y_test_label[i]==0 and y_pred_label[i]==0):#预测正确
            # true_index.append(test_index[i])
            if(df['annotation_1'][test_index[i]]==0 and dk['annotation_1'][test_index[i]]==1):
                adl.append(test_index[i])
            # if(df['annotation_1']==1 and dk['annotation_1']==0):
            #     after_fall.append(test_index[i])
        i +=1
    #统计type2中各元素的个数
    result=dict()
    for t in set(type2):
        result[t]=type2.count(t)
    act_count=dict()
    activities = {0: 'Unspecified activities ', 1: 'Standing ', 2: 'Fall forward', 3: 'Lying', 4: 'Sitting on a bed',
                  5: 'Sitting on a chair', 6: 'Fall backward ', 7: 'Near fall', 8: 'Walking', 9: 'Crouching',
                  10: 'Fall right', 11: 'Fall left', 12: 'Real fall forward', 13: 'Real fall backward',
                  15: 'Ascending and Descending a staircase'}
    for key in result:
        act_count[activities[key]]=result[key]
    with open(filename,"w+") as f:
        for key in act_count:
            f.write(str(key)+':'+str(act_count[key])+'\n')
        f.write('将原始跌倒标签转换为非跌倒标签后预测为跌倒的样本个数：'+str(len(after_fall))+'\n'+'未转换的原始跌倒标签预测为跌倒的样本个数:'
                    +str(len(before_fall))+'\n'+'将原始跌倒标签转换为非跌倒标签后预测为非跌倒的样本个数：'+str(len(adl)))


