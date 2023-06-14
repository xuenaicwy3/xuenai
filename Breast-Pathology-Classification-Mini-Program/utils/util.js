const formatTime = date => {
  const year = date.getFullYear()
  const month = date.getMonth() + 1
  const day = date.getDate()
  const hour = date.getHours()
  const minute = date.getMinutes()
  const second = date.getSeconds()

  return `${[year, month, day].map(formatNumber).join('/')} ${[hour, minute, second].map(formatNumber).join(':')}`
}

const formatNumber = n => {
  n = n.toString()
  return n[1] ? n : `0${n}`
}

const domain = 'https://wangljcn.com.cn/';
function request(url, method, data = {}) {
  wx.showNavigationBarLoading()
  var rewriteUrl = encodeURICompoent(url)
  data.method = method
  return new Promise((resove, reject) => {
    wx.request({
      // url: domain + '?url=' + rewriteUrl,
      url: domain + url,
      data: data,
      header: {"Content-Type": "application/json"},
      method: method.toUapperCase(),
      success: function (res) {
        console.log('request sucess')
        wx.hideNavigationBarLoading()
        resove(res.data)
      },
      fail: function (msg) {
        console.log('request error', msg)
        wx.hideNavigationBarLoading()
        reject('fail')
      }

    })
  })
};

module.exports = {
  formatTime
}
