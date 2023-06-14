// pages/demo10/demo10.js
const app = getApp()
Page({

  /**
   * 页面的初始数据
   */

  data: {
    tempFilePaths:'',//上传图片路径
    src2:'',
  },

  // BMI健康指数计算
  BMI:function(e) {
    wx.navigateTo({
      url:"/pages/bmi/bmi"
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
 // 加载病理图像并预测  
  // upload: function() {
  //   //获取图片地址
  //   var tempFilePaths = app.globalData.tempFilePaths;
  //   //显示进度弹框
  //   wx.showLoading({
  //     title: '处理中,请耐心等待',
  //     mask: true
  //   })
  //   wx.uploadFile({
  //      url: 'http://127.0.0.1:5000/predict',
  //     // url: 'http://1.13.20.13:5000/predict',
  //     // url: 'https://wangljcn.com.cn/predict',
  //     filePath: tempFilePaths,
  //     name: 'file',
  //     success(res) {
  //       //关闭进度弹框
  //       wx.hideLoading()
  //       //保存结果
  //       const result = res.data
  //       console.log(result)
  //       app.globalData.result = result
  //       //界面跳转
  //       wx.navigateTo({
  //         url: '../predict/index'
  //       })
  //     },
  //     fail() {
  //       //关闭进度弹框
  //       wx.hideLoading()
  //       //显示失败弹框
  //       wx.showToast({
  //         title: '处理失败',
  //         icon: 'none',
  //         duration: 2000
  //       })
  //     }
  //   })
  // },






  // uploadPictures(e) {
  //   console.log(e)
  //   let pictype = e.currentTarget.dataset.type;
  // },
  // uploadImage:function(e) {
  //   var vm = this
  //   wx.showActionSheet({
  //     itemList: ['从相册中选择', '拍照'],
  //     success: function(res){
  //       console.log(res)
  //       if (res.tapIndex == 0) {
  //         vm.chooseUploadImage('album')
  //       } else if (res.tapIndex == 1) {
  //         vm.chooseUploadImage('camera')
  //       }
  //     },
  //     fail: function(res) {
  //       console.log(res.errMsg)
  //     }
  //   })
  // },

  // chooseUploadImage(type) {
  //   console.log(type)
  //   var vm = this;
  //   console.log(vm.data.imageList)
  //   var row = vm.data.imageList.length
  //   if(row < 4) {
  //     wx.chooseImage({
  //       count: 4-row,
  //       sizeType:['compressed'],
  //       sourceType: [type],
  //       success(res) {
  //         console.log(res)
  //         var array = res.tempFilePaths
  //         for(var i = 0; i<array.length; i++) {
  //           vm.data.imageList[row] = array[i]
  //           row++
  //         }
  //         vm.setData({
  //           imageList:vm.data.imageList,
  //         })
  //       }
  //     })
  //   }else {
  //     wx.showToast({
  //       title: '最大上传4张图片',
  //       duration: 2000,
  //       icon: 'none'
  //     })
  //   }
  // },
  // 绑定点击选择医生事件
 
  setTabIndex(e) {
    console.log(e);
    this.onLoad();
    let activeIndex = e.currentTarget.dataset.i;
   
    this.setData({
      activeIndex
    })
    
  },
  /**
   * 搜索栏聚焦
   */
  searchFocus: function() {
    this.setData({
      searchClass: "inputFocus"
    });
  },
  /**
   * 搜索栏失焦
   */
  searchBlur: function() {
    this.setData({
      searchClass: ""
    })
  },
  /**
   * 搜索关键词
   */

  xinagche:function(e){
    var that=this;
    console.log(e)
    wx.chooseImage({
      count: 1, // 最多可以选择的图片张数，默认9
      sizeType: ['original', 'compressed'], // original 原图，compressed 压缩图，默认二者都有
      sourceType: ['album'], // album 从相册选图，camera 使用相机，默认二者都有
      success: function(res){
        console.log(e)
        let tempFilePaths = res.tempFilePaths[0];
        console.log(tempFilePaths)
        app.globalData.tempFilePaths = tempFilePaths;
        that.setData({
          tempFilePaths:tempFilePaths
        })
        wx.navigateTo({
          url:'/pages/xiangche/index'
        })
      },
      fail: function() {
        // fail
          //关闭进度弹框
          wx.hideLoading()
          //显示失败弹框
          wx.showToast({
            title: '处理失败',
            icon: 'none',
            duration: 2000
          })
        
      },
      complete: function() {
        // complete
      }
    })
  },

  // 相机按钮点击事件
  xiangji:function(e){
    var that = this;
    wx.chooseImage({
      count: 1,	// 默认为9
      // sizeType: ['original', 'compressed'],	// 指定原图或者压缩图
      sizeType: ['original'],
      sourceType: ['camera'],	// 指定图片来源
      success: function(res) {
        const tempFilePaths = res.tempFilePaths[0]
        console.log(tempFilePaths)
        app.globalData.tempFilePaths = tempFilePaths
        that.setData({
          src2:tempFilePaths
        })
        wx.navigateTo({
          url:"/pages/xiangche/index"
        })
      },
    })

  },

  BMI:function(e) {
    wx.navigateTo({
      url:"/pages/bmi/bmi"
    })
  },
  showMap:function(e){
    wx.navigateTo({
      url: "/pages/showMap/showMap"
    })
  },
  wenjuan:function(e){
    wx.navigateTo({
      url:"/pages/wenjuan/wenjuan"
    })
  },
  // hao:function(e){
  //   var that=this;
  //   console.log(e.currentTarget.dataset.id)
  //   const guan = e.currentTarget.dataset.id
  //   wx.chooseImage({
  //     count: 1,
  //     sizeType:['original', 'compressed'],
  //     sourceType:['album'],
  //     success:(res)=> {
  //        // tempFilePath可以作为img标签的src属性显示图片
  //       let tempFilePaths = res.tempFilePaths[0];
  //       // const tempFilePaths = res.tempFilePaths
  //       console.log(tempFilePaths)
  //       app.globalData.tempFilePaths = tempFilePaths
  //       that.setData({
  //         src:tempFilePaths
  //       })
  //       // const src = src;
  //       // app.globalData.src = src;
  //       wx.navigateTo({
  //         url: '../xiangche/index?info' + guan
  //       })
  //     }
  //   })

    
  // },
  // hao: function() {
  //   // console.log(e.currentTarget.dataset.id)
  //   // const guan = e.currentTarget.dataset.id
  //   wx.chooseImage({
  //     count: 1,
  //     sizeType: ['original', 'compressed'],
  //     sourceType: ['album'],
  //     success(res) {
  //       // tempFilePath可以作为img标签的src属性显示图片
  //       const tempFilePaths = res.tempFilePaths
  //       console.log(tempFilePaths)
  //       app.globalData.tempFilePaths = tempFilePaths
  //       wx.navigateTo({
  //         url: '../xiangche/index'
  //       })
  //     }
  //   })
  // },

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
