// pages/knowledge/knowledge.js
const app = getApp()
Page({

    data: {
      loading: false,
      result: null,
      documents: []  // 已上传文档列表
    },

    onLoad() {
      this.loadDocuments()
    },

    onShow() {
      this.loadDocuments()
    },

    /**
     * 加载已上传文档列表
     */
    loadDocuments() {
      wx.request({
        url: `${app.globalData.apiBase}/api/knowledge/list`,
        method: 'GET',
        success: (res) => {
          if (res.statusCode === 200) {
            const docs = (res.data.documents || []).map(doc => ({
              ...doc,
              sizeText: this.formatSize(doc.size)
            }))
            this.setData({ documents: docs })
          }
        }
      })
    },

    /**
     * 格式化文件大小
     */
    formatSize(bytes) {
      if (bytes < 1024) return bytes + 'B'
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + 'KB'
      return (bytes / (1024 * 1024)).toFixed(1) + 'MB'
    },

    /**
     * 选择文件上传
     */
    chooseFile() {
      wx.chooseMessageFile({
        count: 1,
        type: 'file',
        extension: ['txt', 'doc', 'docx', 'pdf'],
        success: (res) => {
          const file = res.tempFiles[0]
          // 手机端 file.name 可能是临时路径，提取真实文件名
          let originalName = file.name || ''
          // 如果是临时路径格式，取最后一段
          if (originalName.includes('/')) {
            originalName = originalName.split('/').pop()
          }
          // 去掉可能的 tmp_ 前缀
          originalName = originalName.replace(/^tmp_[a-f0-9]+_?/, '')
          // 如果处理后为空，用 file.path 提取
          if (!originalName) {
            originalName = file.path.split('/').pop() || '未知文件.docx'
          }
          wx.showLoading({ title: '上传中...' })
          wx.uploadFile({
            url: `${app.globalData.apiBase}/api/knowledge/upload`,
            filePath: file.path,
            name: 'file',
            formData: { originalName: originalName },
            success: (res) => {
              try {
                const result = JSON.parse(res.data)
                this.setData({ result: result })
                wx.hideLoading()
                wx.showToast({ title: '上传成功', icon: 'success' })
                this.loadDocuments()
              } catch (e) {
                this.setData({ result: null })
                wx.hideLoading()
                wx.showToast({ title: '上传失败', icon: 'none' })
              }
            },
            fail: () => {
              wx.hideLoading()
              wx.showToast({ title: '无法与服务器通信', icon: 'none' })
            }
          })
        }
      })
    },

    /**
     * 删除文档
     */
    deleteDoc(e) {
      const doc = e.currentTarget.dataset.doc
      wx.showModal({
        title: '确认删除',
        content: `确定要删除「${doc.name}」吗？`,
        success: (res) => {
          if (res.confirm) {
            wx.showLoading({ title: '删除中...' })
            wx.request({
              url: `${app.globalData.apiBase}/api/knowledge/delete`,
              method: 'POST',
              data: { path: doc.path },
              header: { 'Content-Type': 'application/json' },
              success: (res) => {
                wx.hideLoading()
                if (res.statusCode === 200) {
                  wx.showToast({ title: '删除成功', icon: 'success' })
                  this.loadDocuments()
                } else {
                  wx.showToast({ title: '删除失败', icon: 'none' })
                }
              },
              fail: () => {
                wx.hideLoading()
                wx.showToast({ title: '删除失败', icon: 'none' })
              }
            })
          }
        }
      })
    },

    /**
     * 预览文档
     */
    previewDoc(e) {
      const doc = e.currentTarget.dataset.doc
      const downloadUrl = `${app.globalData.apiBase}/api/knowledge/download?name=${encodeURIComponent(doc.name)}`
      wx.showLoading({ title: '加载中...' })
      wx.downloadFile({
        url: downloadUrl,
        success: (res) => {
          wx.hideLoading()
          if (res.statusCode === 200) {
            wx.openDocument({
              filePath: res.tempFilePath,
              showMenu: true,
              fail: () => {
                wx.showToast({ title: '无法预览此文件', icon: 'none' })
              }
            })
          } else {
            wx.showToast({ title: '下载失败', icon: 'none' })
          }
        },
        fail: () => {
          wx.hideLoading()
          wx.showToast({ title: '下载失败', icon: 'none' })
        }
      })
    }
})