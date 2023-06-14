Page({

  /**
   * 页面的初始数据
   */
  data: {
    height:null,
    weight:null,
    bmi:'BMI',
    info:"健康提示"
  },
  heightInput(e){
    this.setData({
      height:e.detail.value
    })
},
  weightInput(e){
    this.setData({
      weight:e.detail.value
    })
  },
  bmi:function(event){
    var weightnum = parseFloat(this.data.weight)
    var heightnum = parseFloat(this.data.height)
    // BMI体重指数计算公式：
    var bmi = (weightnum/(heightnum/100)/(heightnum/100)).toFixed(2)
    var info=""
    if(bmi<18.5){
      info="过轻，加强营养！"
    }else if (bmi>=18.5&&bmi<23.9){
      info="正常，继续保持！"
    }else if(bmi>=23.9&&bmi<27.9){
      info="超重，加强锻炼！"
    }else if(bmi>=27.9){
      info="肥胖，提高警惕！"
    }
    console.log(bmi)
    console.log(info)
    this.setData({
      bmi:bmi,
      info:info
    })
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad(options) {

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