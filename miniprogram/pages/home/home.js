// pages/home/home.js

// 获取 app 实例
const app = getApp()

Page({

    /**
     * 页面的初始数据
     */
  data: {
    features: [
      { title: '拍照识花', desc: '拍照或从相册选取，智能识别植物品种', url: '/pages/classify/classify', icon: '📷', gradient: 'linear-gradient(135deg, #E8F5E9, #C8E6C9)' },
      { title: '训练新品种', desc: '上传图片训练模型，扩展可识别范围', url: '/pages/train/train', icon: '🧠', gradient: 'linear-gradient(135deg, #E0F2F1, #B2DFDB)' },
      { title: '养护笔记', desc: '上传养护文档，构建专属知识库', url: '/pages/knowledge/knowledge', icon: '📖', gradient: 'linear-gradient(135deg, #FFF8E1, #FFECB3)' },
      { title: '养护顾问', desc: 'AI 智能问答，解答各类养护难题', url: '/pages/chat/chat', icon: '💬', gradient: 'linear-gradient(135deg, #E3F2FD, #BBDEFB)' }
    ],
    modelCount: 0,
    isFinetuned: false,
    classNames: []   // 可识别的品种名称列表
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
            isFinetuned: res.data.is_finetuned || false,
            classNames: res.data.class_names || []
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