// pages/classify/classify.js

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
      }     // 当前待识别的图片信息
    },

    /**
     * 拍照/选择图片
     * 调用摄像头拍摄图片，并将临时文件复制到用户目录持久化保存
     */
    takePhoto(){
      wx.chooseMedia({
        count: 1,
        mediaType: ['image'],
        sourceType: ['camera'],
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

      // 1.读取图片为base64
      fs.readFile({
        filePath: imagePath,
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
            success: (res) => {
              if (res.statusCode === 200) {
                // 3.把结果显示在页面上
                this.setData({
                  result: {
                    imagePath: imagePath,
                    ...res.data
                  }
                })
              } else {
                wx.showToast({ title: '识别失败', icon: 'none' })
              }
            },
            fail: () => {
              wx.showToast({ title: '网络错误', icon: 'none' })
            },
            complete: () => {
              this.setData({ isProcessing: false })
            }
          })
        },
        fail: () => {
          wx.showToast({ title: '读取图片失败', icon: 'none' })
          this.setData({ isProcessing: false })
        }
      })
    }
})