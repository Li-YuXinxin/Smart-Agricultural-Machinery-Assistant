// pages/classify/classify.js
const app = getApp()

// 获取用户数据目录路径（小程序持久化存储目录）
const USER_DATA_PATH = wx.env.USER_DATA_PATH
// 获取全局文件系统管理器，用于文件读写操作
const fs = wx.getFileSystemManager()

Page({

    /**
     * 页面的初始数据
     */
    data: {
      result: null,   // 识别结果
      isProcessing: false,    // 是否正在处理中
      image: {
        path:'',
        name:'',
        size:0,
        url:''
      },     // 当前待识别的图片信息
      error: null, // 用来向页面传递错误信息
      history: []   // 识别历史
    },

    onLoad() {
      this.loadHistory()
    },

    loadHistory() {
      const history = wx.getStorageSync('classifyHistory') || []
      this.setData({ history })
    },

    saveHistory(item) {
      let history = wx.getStorageSync('classifyHistory') || []
      history.unshift(item)
      if (history.length > 20) history = history.slice(0, 20)
      wx.setStorageSync('classifyHistory', history)
      this.setData({ history })
    },

    clearHistory() {
      wx.showModal({
        title: '清空历史',
        content: '确定要清空识别历史吗？',
        success: (res) => {
          if (res.confirm) {
            wx.removeStorageSync('classifyHistory')
            this.setData({ history: [] })
          }
        }
      })
    },

    navigateToTrain() {
      wx.switchTab({ url: '/pages/train/train' })
    },

    /**
     * 拍照/选择图片
     * 调用摄像头拍摄图片，并将临时文件复制到用户目录持久化保存
     */
    takePhoto(){
      wx.chooseMedia({
        count: 1,
        mediaType: ['image'],
        sourceType: ['album', 'camera'],
        camera: 'back',
        sizeType: ['original'],
        success: (res) => {
          // 获取拍摄的临时文件信息
          const tempFile = res.tempFiles[0]
          // 复制临时文件到用户目录并保存到data中
          this.copyAndAddImage(tempFile.tempFilePath, tempFile.size)
        }
      })
    },

    /**
      * 复制图片到用户目录并添加到页面数据
      * @param {string} tempPath - 微信临时文件路径
      * @param {number} size - 文件大小（字节）
      */
    copyAndAddImage(tempPath, size) {
      //给临时文件起名,避免重名
      const fileName = `img_${Date.now()}_${Math.random().toString(36).slice(2, 6)}.jpg`
      const cachePath = `${USER_DATA_PATH}/${fileName}`

      //拷贝到目录
      try {
        // 将临时文件复制到用户目录下
        fs.copyFileSync(tempPath, cachePath)
        // 从临时路径中提取原始文件名
        const name = tempPath.split('/').pop() || 'image.jpg'
        // 更新页面数据
        this.setData({ image: { path: cachePath, name, size, url: cachePath } })
        console.log('复制图片成功')
      } catch (error) {
        console.error('复制图片失败:', error)
        wx.showToast({
          title: '复制图片失败',
          icon: 'none'
        })
      }

      // 开始识别
      this.startRecognize()
    },

    /**
     * 开始识别：
     * 1. 读取图片 image → base64
     * 2. 调用后端API
     * 3. 把结果显示在页面上
     */
    startRecognize(){
      const imagePath = this.data.image.path
      if (!imagePath) return
      this.setData({ isProcessing: true, result: null })
      wx.showLoading({ title: '识别中...', mask: true })

      // 1.读取图片为base64
      fs.readFile({
        filePath: imagePath,
        fail: () => {
          wx.hideLoading()
          this.setData({ isProcessing: false })
          wx.showToast({ title: '读取图片失败', icon: 'none' })
        },
        success: (res) => {
          const base64 = wx.arrayBufferToBase64(res.data)
          const ext = imagePath.split('.').pop() || 'jpg'
          const mineType = `image/${ext === 'jpg' ? 'jpeg' : ext}`
          const base64Data = `data:${mineType};base64,${base64}`

          // 2.调用后端分类API
          wx.request({
            url: `${app.globalData.apiBase}/api/classify/`,
            method: 'POST',
            data: { image: base64Data },
            header: {
              'Content-Type':'application/json'
            },
            // 请求时长不是很长的时候,可以设置超时
            timeout:(120 * 1000),
            success: (res) => {
              wx.hideLoading()
              this.setData({ isProcessing: false })
              if (res.statusCode === 200) {
                // 3.把结果显示在页面上，置信度转为数字并格式化
                const raw = res.data
                const conf = Number(raw.top1_confidence)
                // 给 top5 每项加上预处理好的置信度文本
                const top5 = (raw.top5 || []).map(item => ({
                  ...item,
                  confidenceText: (Number(item.confidence) * 100).toFixed(1) + '%'
                }))
                const finalResult = {
                  imagePath: imagePath,
                  error: null,
                  ...raw,
                  top1_confidence: conf,
                  confidenceText: isNaN(conf) ? '' : (conf * 100).toFixed(1) + '%',
                  top5: top5
                }
                this.setData({ result: finalResult })
                // 保存识别历史
                this.saveHistory({
                  name: raw.top1,
                  confidence: finalResult.confidenceText,
                  time: new Date().toLocaleString(),
                  imagePath: imagePath
                })
              }
            },
            fail: (err) => {
              wx.hideLoading()
              this.setData({
                isProcessing: false,
                error:'识别失败'
              })
              console.error('识别失败', err)
              wx.showToast({ title: '识别失败', icon: 'none' })
            }
          })
        }
      })
    }
})