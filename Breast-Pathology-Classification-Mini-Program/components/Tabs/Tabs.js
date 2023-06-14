// components/Tabs/Tabs.js
Component({
  /**
   * 组件的属性列表
   */
  properties: {
    showView: false,
  },

  /**
   * 组件的初始数据
   */
  data: {
  
  },

  /**
   * 组件的方法列表
   */
  methods: {
    Selected:function(){
      wx.showModal({
       title: '挂号须知',
       content: '1.全程自费',
       success:function(res){
         if(res.confirm){
           wx.navigateTo({
             url: 'url',
           })
         }
       }
      })
    },
    showhide: function(){
      this.setData({
        showView:(!this.data.showView)
      })
    },
  }
})
