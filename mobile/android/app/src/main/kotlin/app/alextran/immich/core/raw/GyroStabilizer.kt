package app.alextran.immich.core.raw

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Undoes the shake of a raw Insta360 video with the gyroscope the camera wrote in it.
 *
 * ## 为什么这么做
 *
 * 相机录制时把角速度写进了 INSV 的 IMU 记录（trailer 的 record 3）。把它积分成「每一时刻镜头指向哪里」，
 * 就知道画面在什么时候往哪个方向抖了一下。做法是 Gyroflow 的实时简化版：
 *
 * 1. 角速度积分 → 原始姿态 `q_raw(t)`（跟着手一起晃）
 * 2. 对姿态做低通平滑 → 理想轨迹 `q_smooth(t)`
 * 3. 校正量 `q_corr = q_smooth × q_raw⁻¹`
 * 4. 渲染时把 `uViewToLens` 左乘 `q_corr`，画面被拉回平滑轨迹
 *
 * 纯 CPU 预计算 + 每次查询两次插值，[correctionMatrixAt] 在 `drawFrame` 里每帧调用一次，开销可忽略。
 *
 * @param samples 扁平样本 `[t, gx, gy, gz, ...]`，[t] 相对首样本的毫秒数，角速度弧度每秒
 * @param smoothMs 平滑窗口（毫秒）。越大越稳，转向的迟滞感越强
 */
class GyroStabilizer(
  private val samples: FloatArray,
  private val smoothMs: Float = 320f,
) {

  private val count = if (samples.size % 4 == 0) samples.size / 4 else 0

  /** 每个样本的时间（毫秒，相对首样本） */
  private val times = FloatArray(count)

  /** 原始姿态，每条 4 个分量 x,y,z,w */
  private var pose = FloatArray(count * 4)

  /** 平滑后的姿态 */
  private var smooth = FloatArray(count * 4)

  val isValid: Boolean get() = count >= 2

  /** IMU 覆盖的时长（毫秒），用于判断某一帧是否还有陀螺仪数据 */
  val spanMs: Float
    get() = if (count >= 2) times[count - 1] - times[0] else 0f

  init {
    if (count >= 2) build()
  }

  private fun build() {
    for (i in 0 until count) times[i] = samples[i * 4]

    // ---- 1. 角速度积分成姿态 ----
    // 起点取单位四元数：防抖只关心相对变化，绝对朝向由标定的 viewToLens 负责
    var qx = 0f
    var qy = 0f
    var qz = 0f
    var qw = 1f
    for (i in 0 until count) {
      val dt = if (i == 0) 0f else ((times[i] - times[i - 1]) / 1000f).coerceIn(0f, 0.1f)
      if (dt > 0f) {
        // 这一小段转过的轴角
        val ax = samples[i * 4 + 1] * dt
        val ay = samples[i * 4 + 2] * dt
        val az = samples[i * 4 + 3] * dt
        val angle = sqrt(ax * ax + ay * ay + az * az)
        if (angle > 1e-6f) {
          val s = sin(angle / 2f) / angle
          val dx = ax * s
          val dy = ay * s
          val dz = az * s
          val dw = cos(angle / 2f)
          // q = q * dq
          val nx = qw * dx + qx * dw + qy * dz - qz * dy
          val ny = qw * dy - qx * dz + qy * dw + qz * dx
          val nz = qw * dz + qx * dy - qy * dx + qz * dw
          val nw = qw * dw - qx * dx - qy * dy - qz * dz
          val len = sqrt(nx * nx + ny * ny + nz * nz + nw * nw)
          if (len > 1e-6f) {
            qx = nx / len; qy = ny / len; qz = nz / len; qw = nw / len
          }
        }
      }
      pose[i * 4] = qx; pose[i * 4 + 1] = qy; pose[i * 4 + 2] = qz; pose[i * 4 + 3] = qw
    }

    // ---- 2. 一阶低通平滑 ----
    // alpha 由样本间隔与窗口决定，省掉滑动窗口的存储
    val avgDt = ((times[count - 1] - times[0]) / (count - 1).coerceAtLeast(1)).coerceAtLeast(1f)
    val alpha = (avgDt / smoothMs).coerceIn(0.02f, 1f)
    var sx = pose[0]; var sy = pose[1]; var sz = pose[2]; var sw = pose[3]
    for (i in 0 until count) {
      sx += (pose[i * 4] - sx) * alpha
      sy += (pose[i * 4 + 1] - sy) * alpha
      sz += (pose[i * 4 + 2] - sz) * alpha
      sw += (pose[i * 4 + 3] - sw) * alpha
      // 与原始姿态保持同一半球，避免符号翻转让画面突然对折
      val dot = sx * pose[i * 4] + sy * pose[i * 4 + 1] + sz * pose[i * 4 + 2] + sw * pose[i * 4 + 3]
      val sign = if (dot < 0f) -1f else 1f
      var ox = sx * sign; var oy = sy * sign; var oz = sz * sign; var ow = sw * sign
      val len = sqrt(ox * ox + oy * oy + oz * oz + ow * ow)
      if (len > 1e-6f) { ox /= len; oy /= len; oz /= len; ow /= len }
      smooth[i * 4] = ox; smooth[i * 4 + 1] = oy; smooth[i * 4 + 2] = oz; smooth[i * 4 + 3] = ow
    }
  }

  /**
   * [tMs] 时刻的校正旋转，写入 3×3 **列主序**矩阵（GLSL 的 mat3 直接可用）。
   *
   * @return false 表示该时刻超出 IMU 覆盖范围，此时不应校正（宁可几帧不抖，也不要拿边界姿态外推甩一下）
   */
  fun correctionMatrixAt(tMs: Float, out: FloatArray): Boolean {
    if (count < 2) return false
    val i = indexAt(tMs) ?: return false
    val j = (i + 1).coerceAtMost(count - 1)
    if (j == i) return false
    val span = times[j] - times[i]
    if (span <= 0f) return false
    val f = ((tMs - times[i]) / span).coerceIn(0f, 1f)

    val raw = nlerp(pose, i, j, f)
    val sm = nlerp(smooth, i, j, f)
    // q_corr = q_smooth * q_raw⁻¹
    val corr = mul(sm, conjugate(raw))
    quatToMat3ColumnMajor(corr, out)
    return true
  }

  private fun indexAt(tMs: Float): Int? {
    if (tMs < times[0] || tMs > times[count - 1]) return null
    var lo = 0
    var hi = count - 1
    while (lo <= hi) {
      val mid = (lo + hi) / 2
      when {
        times[mid] < tMs -> lo = mid + 1
        times[mid] > tMs -> hi = mid - 1
        else -> return mid
      }
    }
    return hi.coerceIn(0, count - 2)
  }

  /** 四元数线性插值后归一化（nlerp）：相邻样本间隔很小，够用且不会引入额外旋转 */
  private fun nlerp(arr: FloatArray, i: Int, j: Int, f: Float): FloatArray {
    var x = arr[i * 4] + (arr[j * 4] - arr[i * 4]) * f
    var y = arr[i * 4 + 1] + (arr[j * 4 + 1] - arr[i * 4 + 1]) * f
    var z = arr[i * 4 + 2] + (arr[j * 4 + 2] - arr[i * 4 + 2]) * f
    var w = arr[i * 4 + 3] + (arr[j * 4 + 3] - arr[i * 4 + 3]) * f
    val len = sqrt(x * x + y * y + z * z + w * w)
    if (len > 1e-6f) { x /= len; y /= len; z /= len; w /= len }
    return floatArrayOf(x, y, z, w)
  }

  private fun conjugate(q: FloatArray) = floatArrayOf(-q[0], -q[1], -q[2], q[3])

  private fun mul(a: FloatArray, b: FloatArray): FloatArray {
    val ax = a[0]; val ay = a[1]; val az = a[2]; val aw = a[3]
    val bx = b[0]; val by = b[1]; val bz = b[2]; val bw = b[3]
    return floatArrayOf(
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
      aw * bw - ax * bx - ay * by - az * bz,
    )
  }

  /**
   * 四元数 → 3×3 **列主序**矩阵（GLSL 读 mat3 的约定）。
   *
   * 数学上的行主序矩阵 M 存成列主序是 `out[col * 3 + row] = M[row][col]`，这里直接按这个写。
   */
  private fun quatToMat3ColumnMajor(q: FloatArray, out: FloatArray) {
    val x = q[0]; val y = q[1]; val z = q[2]; val w = q[3]
    val xx = x * x; val yy = y * y; val zz = z * z
    val xy = x * y; val xz = x * z; val yz = y * z
    val wx = w * x; val wy = w * y; val wz = w * z

    val m00 = 1 - 2 * (yy + zz); val m01 = 2 * (xy - wz); val m02 = 2 * (xz + wy)
    val m10 = 2 * (xy + wz); val m11 = 1 - 2 * (xx + zz); val m12 = 2 * (yz - wx)
    val m20 = 2 * (xz - wy); val m21 = 2 * (yz + wx); val m22 = 1 - 2 * (xx + yy)

    out[0] = m00; out[1] = m10; out[2] = m20
    out[3] = m01; out[4] = m11; out[5] = m21
    out[6] = m02; out[7] = m12; out[8] = m22
  }
}
