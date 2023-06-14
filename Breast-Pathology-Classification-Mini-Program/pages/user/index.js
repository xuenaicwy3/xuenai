// pages/user/index.js
var app = getApp();
let register = false;
// const app = getApp();
const defaultAvatarUrl = "https://mmbiz.qpic.cn/mmbiz/icTdbqWNOwNRna42FI242Lcia07jQodd2FJGIYQfG0LAJGFxM4FbnQP6yfMxBgJ0F3YRqJCJ1aPAK2dQagdusBZg/0";
Page({

  /**
   * 页面的初始数据
   */
  data: {
    // 默认选中用户端
    activeIndex:0,
    // 小程序版本
    version: "1.0.0",

    // 用户信息
    // userInfo: ""
    avatarUrl: defaultAvatarUrl,
    theme: wx.getSystemInfoSync().theme,
  },
  
  setTabIndex(e) {
    console.log(e)
    let activeIndex = e.currentTarget.dataset.i;
    console.log(activeIndex)
    this.setData({
      activeIndex
    })
  },

    // 病理疾病诊断
    uploadImage:function(){
      let that = this;
      wx.showActionSheet({
        itemList: ['从相册中选择', '拍照'],
        itemColor: "#00ff20",
        success: function(res) {
          if (!res.cancel) {
            if (res.tapIndex == 0) {
              that.chooseWxImage('album');
            } else if (res.tapIndex == 1) {
              that.chooseWxImage('camera');
            }
          }
        }
      })
    },
   
   /*打开相册、相机 */
     chooseWxImage: function(type) {
      console.log(type)
      let that = this;
      let _this = this;
      wx.chooseImage({
        count: that.data.countIndex,
        sizeType: ['original', 'compressed'],
        sourceType: [type],
        success: function(res) {
          // 选择图片后的完成确认操作
          const tempFilePaths = res.tempFilePaths[0]
          console.log(tempFilePaths)
          app.globalData.tempFilePaths = tempFilePaths
          that.setData({
            aimgurl: res.tempFilePaths
          });
          console.log(that.data.aimgurl);
          // _this.upload();
          wx.navigateTo({
            url: '/pages/xiangche/index',
          })
        }
      })
    },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad(options) {
    // app.bar({
    //   title: "我的",
    //   bgColor: "#b5cfff"
    // });
    wx.onThemeChange((result) => {
      this.setData({
        theme: result.theme
      })
    })

    // 初始化版本
    this.setData({
      version:app.globalData.version
    });

    // 监听数据 同步全局
    Object.defineProperty(this.data, "userInfo", {
      set: data => {
        app.globalData.userInfo = data;
      }
    });
  },

  onChooseAvatar(e) {
    const {avatarUrl} = e.detail
    console.log(e)
    this.setData ({
      avatarUrl: e.detail.avatarUrl
    })
  },
  formSubmit(e){
    console.log('昵称：', e.detail.value.nickname)
  },

   

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady() {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  getUserProfile(e){
    console.log("点击了按钮")
    wx.getUserProfile({
      desc: '获取用户的信息',
      success: (res)=> {
        console.log(res)
        let user = res.userInfo
        wx.setStorageSync("userinfo", user)
        console.log("成功",res)
        this.setData({
          userInfo:user,
        })
   
      }
    })
  },
  onShow() {
    const userinfo=wx.getStorageSync("userinfo");
    this.setData({userinfo});
    if (!Boolean(app.globalData.userInfo + []) ||
      !app.globalData.userInfo.bind_account) {
        wx.getSetting({
          success: setting => {
            if (setting.authSetting["scope.userInfo"]) {
              wx.getUserInfo({
                success: res => {
                  this.settingAccount(res, true);
                }
              });
            }
          }
        })
      } 
    else {
        this.setData({
          userInfo: app.globalData.userInfo,
          doctor: !!app.globalData.userInfo.bind_account.ysdm
        });
      }
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