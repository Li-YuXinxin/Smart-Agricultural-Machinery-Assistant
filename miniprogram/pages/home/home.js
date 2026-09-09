// home/home.js

// 获取 app 实例
const app = getApp()

Page({

    /**
     * 页面的初始数据
     */
  data: {
    features: [
      { title: '模型训练', url: '/pages/train/train' },
      { title: '稠密识别', url: '/pages/classify/classify' },
      { title: '知识库管理', url: '/pages/knowledge/knowledge' },
      { title: '智能问答', url: '/pages/chat/chat' }
    ]
  },

  navigateTo(e) {
    // 获取点击的 url
    const url = e.currentTarget.dataset.url
    if(!url)  return

    const tarBarPages = [
      "pages/home/home",
      "pages/classify/classify",
      "pages/knowledge/knowledge",
      "pages/train/train",
      "pages/chat/chat"
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