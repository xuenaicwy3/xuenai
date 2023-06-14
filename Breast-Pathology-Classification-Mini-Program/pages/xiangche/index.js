// pages/xiangche/index.js
const app = getApp()
Page({

  /**
   * 页面的初始数据
   */
  data: {
    tempFilePaths:'',
  },

  upload: function() {
    //获取图片地址
    var tempFilePaths = app.globalData.tempFilePaths;
    //显示进度弹框
    wx.showLoading({
      title: '处理中,请耐心等待',
      mask: true
    })
    wx.uploadFile({
      //  url: 'http://127.0.0.1:5000/predict',
      // url: 'http://127.0.0.1:5000/user/predict',
      url: 'https://xxxxxx/user/predict', //(这些填入服务器域名)
      // url: 'https://xxxxxx/predict',
      filePath: tempFilePaths,
      name: 'file',
      // name: 'photo',
      success(res) {
        //关闭进度弹框
        wx.hideLoading()
        //保存结果
        const result = res.data
        // result: json["result"],
        console.log('======>',result)
        app.globalData.result = result
        //界面跳转
        wx.navigateTo({
          url: '../predict/index'
        })
      },
      fail() {
        //关闭进度弹框
        wx.hideLoading()
        //显示失败弹框
        wx.showToast({
          title: '处理失败',
          icon: 'none',
          duration: 2000
        })
      }
    })
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad(options) {
    var tempFilePaths = app.globalData.tempFilePaths;
    this.setData({
      tempFilePaths: tempFilePaths
     })
  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady() {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow() {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide() {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload() {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh() {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom() {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage() {

  }
})