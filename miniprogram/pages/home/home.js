// pages/home/home.js

// 获取 app 实例
const app = getApp()

Page({

    /**
     * 页面的初始数据
     */
  data: {
    features: [
      { title: '拍照识花', desc: '拍照识别植物品种', url: '/pages/classify/classify' },
      { title: '添加新品种', desc: '训练扩展可识别品种', url: '/pages/train/train' },
      { title: '养护笔记', desc: '上传养护文档构建知识库', url: '/pages/knowledge/knowledge' },
      { title: '养护顾问', desc: 'AI 问答解答养护问题', url: '/pages/chat/chat' }
    ],
    modelCount: 0,
    isFinetuned: false
  },

  onShow() {
    this.loadStats()
  },

  loadStats() {
    wx.request({
      url: `${app.globalData.apiBase}/api/classify/stats`,
      method: 'GET',
      success: (res) => {
        if (res.statusCode === 200) {
          this.setData({
            modelCount: res.data.count || 0,
            isFinetuned: res.data.is_finetuned || false
          })
        }
      }
    })
  },

  navigateTo(e) {
    // 获取点击的 url
    const url = e.currentTarget.dataset.url
    if(!url)  return

    const tarBarPages = [
      '/pages/home/home',
      '/pages/classify/classify',
      '/pages/knowledge/knowledge',
      '/pages/train/train',
      '/pages/chat/chat'
    ]

    if (tarBarPages.includes(url)) {
        // 如果点击的是页面
        wx.switchTab({ url })
    } else {
        // 点击导航栏
        wx.navigateTo({ url })
    }
  }
})