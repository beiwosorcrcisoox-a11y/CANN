# CANN
Hank
#include <algorithm>
#include <cstdint>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {

constexpr uint32_t kVectorTileN = 4096;
constexpr uint32_t kFloatBlock = 8;
constexpr uint16_t kCubeToVectorFlag = 8;
constexpr auto kInt8MatmulConfig = CFG_NORM;

inline uint32_t QmqHostCeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1U) / divisor;
}

__aicore__ inline uint32_t QmqCeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1U) / divisor;
}

__aicore__ inline uint32_t QmqMinValue(uint32_t left, uint32_t right)
{
    return left < right ? left : right;
}

__aicore__ inline uint32_t QmqAlignUp(uint32_t value, uint32_t alignment)
{
    return QmqCeilDiv(value, alignment) * alignment;
}

__aicore__ inline void QmqDiv(
    LocalTensor<float> dst,
    LocalTensor<float> src0,
    LocalTensor<float> src1,
    uint32_t count)
{
    const uint32_t fullRepeats = count / 64U;
    const uint32_t fullCount = fullRepeats * 64U;
    const uint32_t tailCount = count - fullCount;
    const BinaryRepeatParams params{1, 1, 1, 8, 8, 8};
    if (fullRepeats > 0) {
        AscendC::Div(
            dst,
            src0,
            src1,
            static_cast<uint64_t>(64),
            static_cast<uint8_t>(fullRepeats),
            params);
    }
    if (tailCount > 0) {
        AscendC::Div(
            dst[fullCount],
            src0[fullCount],
            src1[fullCount],
            static_cast<uint64_t>(tailCount),
            static_cast<uint8_t>(1),
            params);
    }
}

__aicore__ inline void QmqCorrectDiv(
    LocalTensor<float> dst,
    LocalTensor<float> dividend,
    LocalTensor<float> divisor,
    LocalTensor<float> residual,
    LocalTensor<float> scratch,
    LocalTensor<uint8_t> correctionMask,
    uint32_t count,
    uint32_t compareCount)
{
    QmqDiv(dst, dividend, divisor, count);
    PipeBarrier<PIPE_V>();
    Muls(divisor, divisor, -1.0f, count);
    PipeBarrier<PIPE_V>();
    DataCopy(residual, dividend, count);
    PipeBarrier<PIPE_V>();
    MulAddDst(residual, dst, divisor, count);
    PipeBarrier<PIPE_V>();
    Abs(residual, residual, count);
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> quotientBits = dst.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> scratchBits = scratch.ReinterpretCast<int32_t>();

    Adds(quotientBits, quotientBits, static_cast<int32_t>(-1), count);
    PipeBarrier<PIPE_V>();
    DataCopy(scratch, dividend, count);
    PipeBarrier<PIPE_V>();
    MulAddDst(scratch, dst, divisor, count);
    PipeBarrier<PIPE_V>();
    Abs(scratch, scratch, count);
    PipeBarrier<PIPE_V>();
    Compare(correctionMask, scratch, residual, CMPMODE::LT, compareCount);
    PipeBarrier<PIPE_V>();
    Select(
        residual,
        correctionMask,
        scratch,
        residual,
        SELMODE::VSEL_TENSOR_TENSOR_MODE,
        count);
    PipeBarrier<PIPE_V>();
    Adds(scratchBits, quotientBits, static_cast<int32_t>(1), count);
    PipeBarrier<PIPE_V>();
    Select(
        dst,
        correctionMask,
        dst,
        scratch,
        SELMODE::VSEL_TENSOR_TENSOR_MODE,
        count);
    PipeBarrier<PIPE_V>();

    Adds(quotientBits, quotientBits, static_cast<int32_t>(1), count);
    PipeBarrier<PIPE_V>();
    DataCopy(scratch, dividend, count);
    PipeBarrier<PIPE_V>();
    MulAddDst(scratch, dst, divisor, count);
    PipeBarrier<PIPE_V>();
    Abs(scratch, scratch, count);
    PipeBarrier<PIPE_V>();
    Compare(correctionMask, scratch, residual, CMPMODE::LT, compareCount);
    PipeBarrier<PIPE_V>();
    Adds(scratchBits, quotientBits, static_cast<int32_t>(-1), count);
    PipeBarrier<PIPE_V>();
    Select(
        dst,
        correctionMask,
        dst,
        scratch,
        SELMODE::VSEL_TENSOR_TENSOR_MODE,
        count);
    PipeBarrier<PIPE_V>();
}

} // namespace

__aicore__ inline void quant_matmul_relu_quant_cube(
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR workspace,
    __gm__ uint8_t* systemWorkspace,
    AscendC::tiling::TCubeTiling tiling,
    uint32_t groupIndex,
    uint32_t m,
    uint32_t n,
    uint32_t k)
{
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    using BiasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

    if (workspace == nullptr || systemWorkspace == nullptr || tiling.usedCoreNum == 0 ||
        tiling.singleCoreM == 0 || tiling.singleCoreN == 0 || m == 0 ||
        n == 0 || k == 0) {
        return;
    }

    const uint32_t firstMBlock = groupIndex * 2U;
    const uint32_t firstMOffset = firstMBlock * static_cast<uint32_t>(tiling.singleCoreM);
    if (firstMOffset >= m) {
        return;
    }

    GlobalTensor<int8_t> x1Gm;
    GlobalTensor<int8_t> x2Gm;
    GlobalTensor<int32_t> workspaceGm;
    x1Gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int8_t*>(x1),
        static_cast<uint64_t>(m) * k);
    x2Gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int8_t*>(x2),
        static_cast<uint64_t>(n) * k);
    workspaceGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(workspace),
        static_cast<uint64_t>(m) * n);

    TPipe pipe;
    matmul::Matmul<AType, BType, CType, BiasType, kInt8MatmulConfig> matmulObj;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj, &tiling);

    for (uint32_t localTile = 0; localTile < 2U; ++localTile) {
        const uint32_t mBlockIndex = firstMBlock + localTile;
        const uint32_t mOffset =
            mBlockIndex * static_cast<uint32_t>(tiling.singleCoreM);
        if (mOffset >= m) {
            break;
        }
        const uint32_t currentM = QmqMinValue(
            static_cast<uint32_t>(tiling.singleCoreM), m - mOffset);
        const uint64_t aOffset = static_cast<uint64_t>(mOffset) * k;
        const uint64_t cOffset = static_cast<uint64_t>(mOffset) * n;

        matmulObj.SetOrgShape(m, n, k, k);
        matmulObj.SetTail(static_cast<int32_t>(currentM), static_cast<int32_t>(n));
        matmulObj.SetTensorA(x1Gm[aOffset], false);
        matmulObj.SetTensorB(x2Gm[0], true);
        matmulObj.IterateAll(workspaceGm[cOffset]);
    }
    matmulObj.End();
}

__aicore__ inline void quant_matmul_relu_quant_vector(
    GM_ADDR workspace,
    GM_ADDR x1Scale,
    GM_ADDR x2Scale,
    GM_ADDR y,
    GM_ADDR yScale,
    uint32_t m,
    uint32_t n,
    uint32_t rowStart,
    uint32_t rows)
{
    if (workspace == nullptr || x1Scale == nullptr || x2Scale == nullptr ||
        y == nullptr || yScale == nullptr || m == 0 || n == 0 || rowStart >= m) {
        return;
    }

    rows = QmqMinValue(rows, m - rowStart);
    if (rows == 0) {
        return;
    }
    const uint32_t alignedRows = QmqAlignUp(rows, kFloatBlock);

    GlobalTensor<int32_t> workspaceGm;
    GlobalTensor<float> x1ScaleGm;
    GlobalTensor<float> x2ScaleGm;
    GlobalTensor<int8_t> yGm;
    GlobalTensor<float> yScaleGm;
    workspaceGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(workspace), static_cast<uint64_t>(m) * n);
    x1ScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x1Scale), m);
    x2ScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x2Scale), n);
    yGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int8_t*>(y), static_cast<uint64_t>(m) * n);
    yScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(yScale), m);

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> workspaceQueue;
    TQue<QuePosition::VECIN, 1> x2ScaleQueue;
    TQue<QuePosition::VECIN, 1> x1ScaleQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
    TQue<QuePosition::VECOUT, 1> yScaleQueue;
    TBuf<TPosition::VECCALC> fp32Buffer;
    TBuf<TPosition::VECCALC> halfBuffer;
    TBuf<TPosition::VECCALC> rowMaxBuffer;
    TBuf<TPosition::VECCALC> rowScaleBuffer;
    TBuf<TPosition::VECCALC> reduceBuffer;
    TBuf<TPosition::VECCALC> correctionBuffer;
    TBuf<TPosition::VECCALC> scalarMaxBuffer;
    TBuf<TPosition::VECCALC> correctionMaskBuffer;

    pipe.InitBuffer(workspaceQueue, 1, kVectorTileN * sizeof(int32_t));
    pipe.InitBuffer(x2ScaleQueue, 1, kVectorTileN * sizeof(float));
    pipe.InitBuffer(x1ScaleQueue, 1, alignedRows * sizeof(float));
    pipe.InitBuffer(outputQueue, 1, kVectorTileN * sizeof(int8_t));
    pipe.InitBuffer(yScaleQueue, 1, alignedRows * sizeof(float));
    pipe.InitBuffer(fp32Buffer, kVectorTileN * sizeof(float));
    pipe.InitBuffer(halfBuffer, kVectorTileN * sizeof(float));
    pipe.InitBuffer(rowMaxBuffer, alignedRows * sizeof(float));
    pipe.InitBuffer(rowScaleBuffer, alignedRows * sizeof(float));
    pipe.InitBuffer(reduceBuffer, kVectorTileN * sizeof(float));
    pipe.InitBuffer(correctionBuffer, kVectorTileN * sizeof(float));
    pipe.InitBuffer(scalarMaxBuffer, kFloatBlock * sizeof(float));
    pipe.InitBuffer(correctionMaskBuffer, kVectorTileN / 8U);

    LocalTensor<float> x1ScaleLocal = x1ScaleQueue.AllocTensor<float>();
    DataCopyExtParams x1ScaleCopy{1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> x1ScalePad{
        true, 0, static_cast<uint8_t>(alignedRows - rows), 0.0f};
    DataCopyPad(x1ScaleLocal, x1ScaleGm[rowStart], x1ScaleCopy, x1ScalePad);
    x1ScaleQueue.EnQue(x1ScaleLocal);
    x1ScaleLocal = x1ScaleQueue.DeQue<float>();

    LocalTensor<float> rowMaxLocal = rowMaxBuffer.Get<float>();
    LocalTensor<float> rowScaleLocal = rowScaleBuffer.Get<float>();
    Duplicate(rowMaxLocal, 0.0f, alignedRows);
    PipeBarrier<PIPE_V>();

    for (uint32_t nOffset = 0; nOffset < n; nOffset += kVectorTileN) {
        const uint32_t currentN = QmqMinValue(kVectorTileN, n - nOffset);
        const uint32_t alignedN = QmqAlignUp(currentN, kFloatBlock);

        LocalTensor<float> x2ScaleLocal = x2ScaleQueue.AllocTensor<float>();
        DataCopyExtParams x2ScaleCopy{
            1, static_cast<uint32_t>(currentN * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> x2ScalePad{
            true, 0, static_cast<uint8_t>(alignedN - currentN), 0.0f};
        DataCopyPad(x2ScaleLocal, x2ScaleGm[nOffset], x2ScaleCopy, x2ScalePad);
        x2ScaleQueue.EnQue(x2ScaleLocal);
        x2ScaleLocal = x2ScaleQueue.DeQue<float>();

        for (uint32_t row = 0; row < rows; ++row) {
            const uint64_t offset = static_cast<uint64_t>(rowStart + row) * n + nOffset;
            LocalTensor<int32_t> inputLocal = workspaceQueue.AllocTensor<int32_t>();
            DataCopyExtParams inputCopy{
                1, static_cast<uint32_t>(currentN * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> inputPad{
                true, 0, static_cast<uint8_t>(alignedN - currentN), 0};
            DataCopyPad(inputLocal, workspaceGm[offset], inputCopy, inputPad);
            workspaceQueue.EnQue(inputLocal);
            inputLocal = workspaceQueue.DeQue<int32_t>();

            LocalTensor<float> fp32Local = fp32Buffer.Get<float>();
            LocalTensor<float> scalarMaxLocal = scalarMaxBuffer.Get<float>();
            LocalTensor<float> x1ScaleBroadcast = halfBuffer.Get<float>();
            Cast(fp32Local, inputLocal, RoundMode::CAST_RINT, alignedN);
            PipeBarrier<PIPE_V>();
            Duplicate(x1ScaleBroadcast, x1ScaleLocal.GetValue(row), alignedN);
            PipeBarrier<PIPE_V>();
            Mul(fp32Local, fp32Local, x1ScaleBroadcast, alignedN);
            PipeBarrier<PIPE_V>();
            Mul(fp32Local, fp32Local, x2ScaleLocal, alignedN);
            PipeBarrier<PIPE_V>();
            Maxs(fp32Local, fp32Local, 0.0f, alignedN);
            PipeBarrier<PIPE_V>();
            ReduceMax(scalarMaxLocal, fp32Local, reduceBuffer.Get<float>(), alignedN);
            PipeBarrier<PIPE_V>();

            const float tileMax = scalarMaxLocal.GetValue(0);
            const float oldMax = rowMaxLocal.GetValue(row);
            if (tileMax > oldMax) {
                rowMaxLocal.SetValue(row, tileMax);
            }
            workspaceQueue.FreeTensor(inputLocal);
        }
        x2ScaleQueue.FreeTensor(x2ScaleLocal);
    }

    LocalTensor<float> scaleDivisorLocal = halfBuffer.Get<float>();
    LocalTensor<float> scaleResidualLocal = reduceBuffer.Get<float>();
    LocalTensor<float> scaleScratchLocal = correctionBuffer.Get<float>();
    LocalTensor<uint8_t> scaleMaskLocal = correctionMaskBuffer.Get<uint8_t>();
    Duplicate(scaleDivisorLocal, 127.0f, alignedRows);
    PipeBarrier<PIPE_V>();
    QmqCorrectDiv(
        rowScaleLocal,
        rowMaxLocal,
        scaleDivisorLocal,
        scaleResidualLocal,
        scaleScratchLocal,
        scaleMaskLocal,
        alignedRows,
        QmqAlignUp(alignedRows, 64U));
    for (uint32_t row = 0; row < rows; ++row) {
        if (rowMaxLocal.GetValue(row) <= 0.0f) {
            rowScaleLocal.SetValue(row, 1.0f);
        }
    }
    PipeBarrier<PIPE_V>();

    LocalTensor<float> yScaleLocal = yScaleQueue.AllocTensor<float>();
    for (uint32_t row = 0; row < rows; ++row) {
        yScaleLocal.SetValue(row, rowScaleLocal.GetValue(row));
    }
    yScaleQueue.EnQue(yScaleLocal);
    yScaleLocal = yScaleQueue.DeQue<float>();
    DataCopyExtParams yScaleCopy{
        1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
    DataCopyPad(yScaleGm[rowStart], yScaleLocal, yScaleCopy);
    yScaleQueue.FreeTensor(yScaleLocal);

    for (uint32_t nOffset = 0; nOffset < n; nOffset += kVectorTileN) {
        const uint32_t currentN = QmqMinValue(kVectorTileN, n - nOffset);
        const uint32_t alignedN = QmqAlignUp(currentN, kFloatBlock);
        const uint32_t compareCount = QmqAlignUp(currentN, 64U);

        LocalTensor<float> x2ScaleLocal = x2ScaleQueue.AllocTensor<float>();
        DataCopyExtParams x2ScaleCopy{
            1, static_cast<uint32_t>(currentN * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> x2ScalePad{
            true, 0, static_cast<uint8_t>(alignedN - currentN), 0.0f};
        DataCopyPad(x2ScaleLocal, x2ScaleGm[nOffset], x2ScaleCopy, x2ScalePad);
        x2ScaleQueue.EnQue(x2ScaleLocal);
        x2ScaleLocal = x2ScaleQueue.DeQue<float>();

        for (uint32_t row = 0; row < rows; ++row) {
            const uint64_t offset = static_cast<uint64_t>(rowStart + row) * n + nOffset;
            LocalTensor<int32_t> inputLocal = workspaceQueue.AllocTensor<int32_t>();
            DataCopyExtParams inputCopy{
                1, static_cast<uint32_t>(currentN * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> inputPad{
                true, 0, static_cast<uint8_t>(alignedN - currentN), 0};
            DataCopyPad(inputLocal, workspaceGm[offset], inputCopy, inputPad);
            workspaceQueue.EnQue(inputLocal);
            inputLocal = workspaceQueue.DeQue<int32_t>();

            LocalTensor<float> fp32Local = fp32Buffer.Get<float>();
            LocalTensor<float> x1ScaleBroadcast = halfBuffer.Get<float>();
            Cast(fp32Local, inputLocal, RoundMode::CAST_RINT, alignedN);
            PipeBarrier<PIPE_V>();
            Duplicate(x1ScaleBroadcast, x1ScaleLocal.GetValue(row), alignedN);
            PipeBarrier<PIPE_V>();
            Mul(fp32Local, fp32Local, x1ScaleBroadcast, alignedN);
            PipeBarrier<PIPE_V>();
            Mul(fp32Local, fp32Local, x2ScaleLocal, alignedN);
            PipeBarrier<PIPE_V>();
            Maxs(fp32Local, fp32Local, 0.0f, alignedN);
            PipeBarrier<PIPE_V>();

            if (rowMaxLocal.GetValue(row) > 0.0f) {
                LocalTensor<float> activationLocal = inputLocal.ReinterpretCast<float>();
                LocalTensor<float> negScaleLocal = halfBuffer.Get<float>();
                LocalTensor<float> residualLocal = reduceBuffer.Get<float>();
                LocalTensor<float> scratchLocal = correctionBuffer.Get<float>();
                LocalTensor<uint8_t> correctionMaskLocal =
                    correctionMaskBuffer.Get<uint8_t>();
                DataCopy(activationLocal, fp32Local, alignedN);
                PipeBarrier<PIPE_V>();
                Duplicate(negScaleLocal, rowScaleLocal.GetValue(row), alignedN);
                PipeBarrier<PIPE_V>();
                QmqCorrectDiv(
                    fp32Local,
                    activationLocal,
                    negScaleLocal,
                    residualLocal,
                    scratchLocal,
                    correctionMaskLocal,
                    alignedN,
                    compareCount);
            } else {
                Duplicate(fp32Local, 0.0f, alignedN);
                PipeBarrier<PIPE_V>();
            }

            Maxs(fp32Local, fp32Local, 0.0f, alignedN);
            Mins(fp32Local, fp32Local, 127.0f, alignedN);
            PipeBarrier<PIPE_V>();
            Cast(inputLocal, fp32Local, RoundMode::CAST_RINT, alignedN);
            PipeBarrier<PIPE_V>();
            SetDeqScale(static_cast<half>(1.0f));
            LocalTensor<half> halfLocal = halfBuffer.Get<half>();
            Cast(halfLocal, inputLocal, RoundMode::CAST_NONE, alignedN);
            PipeBarrier<PIPE_V>();

            LocalTensor<int8_t> outputLocal = outputQueue.AllocTensor<int8_t>();
            Cast(outputLocal, halfLocal, RoundMode::CAST_RINT, alignedN);
            outputQueue.EnQue(outputLocal);
            outputLocal = outputQueue.DeQue<int8_t>();
            DataCopyExtParams outputCopy{
                1, static_cast<uint32_t>(currentN * sizeof(int8_t)), 0, 0, 0};
            DataCopyPad(yGm[offset], outputLocal, outputCopy);
            outputQueue.FreeTensor(outputLocal);
            workspaceQueue.FreeTensor(inputLocal);
        }
        x2ScaleQueue.FreeTensor(x2ScaleLocal);
    }

    x1ScaleQueue.FreeTensor(x1ScaleLocal);
    PipeBarrier<PIPE_ALL>();
}

__global__ __mix__(1, 2) void quant_matmul_relu_quant_fused(
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR workspace,
    __kfc_workspace__ __gm__ uint8_t* systemWorkspace,
    GM_ADDR x1Scale,
    GM_ADDR x2Scale,
    GM_ADDR y,
    GM_ADDR yScale,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    AscendC::tiling::TCubeTiling tiling,
    uint32_t groupCount)
{
    AscendC::InitSocState();

    if ASCEND_IS_AIC {
        AscendC::SetSysWorkspace(systemWorkspace);
        const uint32_t groupIdx = GetBlockIdx();
        if (groupIdx < groupCount) {
            quant_matmul_relu_quant_cube(
                x1, x2, workspace, systemWorkspace, tiling, groupIdx, m, n, k);
            CrossCoreSetFlag<2, PIPE_FIX>(kCubeToVectorFlag);
        }
    }

    if ASCEND_IS_AIV {
        const uint32_t aivIdx = GetBlockIdx();
        const uint32_t groupIdx = aivIdx / 2U;
        const uint32_t subIdx = GetSubBlockIdx();
        const uint32_t mBlockIndex = groupIdx * 2U + subIdx;
        const uint32_t groupRowStart = QmqMinValue(
            mBlockIndex * static_cast<uint32_t>(tiling.singleCoreM), m);
        const uint32_t groupRowEnd = QmqMinValue(
            groupRowStart + static_cast<uint32_t>(tiling.singleCoreM), m);
        const uint32_t groupRows = groupRowEnd - groupRowStart;

        CrossCoreWaitFlag(kCubeToVectorFlag);
        if (groupRows > 0) {
            quant_matmul_relu_quant_vector(
                workspace,
                x1Scale,
                x2Scale,
                y,
                yScale,
                m,
                n,
                groupRowStart,
                groupRows);
        }
    }
}

extern "C" void run_kernel(
    GM_ADDR x1,
    const TensorGroupInfo& info_x1,
    GM_ADDR x2,
    const TensorGroupInfo& info_x2,
    GM_ADDR x1Scale,
    const TensorGroupInfo& info_x1Scale,
    GM_ADDR x2Scale,
    const TensorGroupInfo& info_x2Scale,
    GM_ADDR y,
    const TensorGroupInfo& info_y,
    GM_ADDR yScale,
    const TensorGroupInfo& info_yScale,
    int64_t availableCoreNum,
    aclrtStream stream)
{
    constexpr int32_t kFloat32Dtype = 0;
    constexpr int32_t kInt8Dtype = 3;

    if (x1 == nullptr || x2 == nullptr || x1Scale == nullptr || x2Scale == nullptr ||
        y == nullptr || yScale == nullptr || stream == nullptr || availableCoreNum < 1 ||
        info_x1.numTensors != 1 || info_x2.numTensors != 1 ||
        info_x1Scale.numTensors != 1 || info_x2Scale.numTensors != 1 ||
        info_y.numTensors != 1 || info_yScale.numTensors != 1 ||
        info_x1.tensors == nullptr || info_x2.tensors == nullptr ||
        info_x1Scale.tensors == nullptr || info_x2Scale.tensors == nullptr ||
        info_y.tensors == nullptr || info_yScale.tensors == nullptr ||
        info_x1.tensors[0].shape == nullptr || info_x2.tensors[0].shape == nullptr ||
        info_x1Scale.tensors[0].shape == nullptr || info_x2Scale.tensors[0].shape == nullptr ||
        info_y.tensors[0].shape == nullptr || info_yScale.tensors[0].shape == nullptr) {
        return;
    }

    const TensorInfo& x1Info = info_x1.tensors[0];
    const TensorInfo& x2Info = info_x2.tensors[0];
    const TensorInfo& x1ScaleInfo = info_x1Scale.tensors[0];
    const TensorInfo& x2ScaleInfo = info_x2Scale.tensors[0];
    const TensorInfo& yInfo = info_y.tensors[0];
    const TensorInfo& yScaleInfo = info_yScale.tensors[0];
    if (x1Info.numDims != 2 || x2Info.numDims != 2 ||
        x1ScaleInfo.numDims != 1 || x2ScaleInfo.numDims != 1 ||
        yInfo.numDims != 2 || yScaleInfo.numDims != 1 ||
        x1Info.dtype != kInt8Dtype || x2Info.dtype != kInt8Dtype ||
        x1ScaleInfo.dtype != kFloat32Dtype || x2ScaleInfo.dtype != kFloat32Dtype ||
        yInfo.dtype != kInt8Dtype || yScaleInfo.dtype != kFloat32Dtype) {
        return;
    }

    const int64_t m64 = x1Info.shape[0];
    const int64_t k64 = x1Info.shape[1];
    const int64_t n64 = x2Info.shape[0];
    if (m64 <= 0 || n64 <= 0 || k64 <= 0 ||
        m64 > static_cast<int64_t>(INT32_MAX) ||
        n64 > static_cast<int64_t>(INT32_MAX) ||
        k64 > static_cast<int64_t>(INT32_MAX) ||
        x2Info.shape[1] != k64 || x1ScaleInfo.shape[0] != m64 ||
        x2ScaleInfo.shape[0] != n64 || yInfo.shape[0] != m64 ||
        yInfo.shape[1] != n64 || yScaleInfo.shape[0] != m64) {
        return;
    }

    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        return;
    }

    const uint32_t m = static_cast<uint32_t>(m64);
    const uint32_t n = static_cast<uint32_t>(n64);
    const uint32_t k = static_cast<uint32_t>(k64);
    const uint32_t usableCoreNum = static_cast<uint32_t>(std::min<int64_t>(
        availableCoreNum, static_cast<int64_t>(UINT32_MAX)));
    const uint32_t aivCandidate = static_cast<uint32_t>(std::min<uint64_t>(
        m, 2ULL * static_cast<uint64_t>(usableCoreNum)));
    const uint32_t balancedTileM = QmqHostCeilDiv(m, aivCandidate);
    const uint32_t tileM = std::min<uint32_t>(m, std::max<uint32_t>(16U, balancedTileM));
    matmul_tiling::MultiCoreMatmulTiling tilingApi(*platform);
    if (tilingApi.SetDim(1) == -1) {
        return;
    }
    tilingApi.SetAType(
        matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT8,
        false);
    tilingApi.SetBType(
        matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT8,
        true);
    tilingApi.SetCType(
        matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT32);
    tilingApi.SetBiasType(
        matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT32);
    tilingApi.SetOrgShape(
        static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k));
    tilingApi.SetShape(
        static_cast<int32_t>(tileM),
        static_cast<int32_t>(n),
        static_cast<int32_t>(k));
    tilingApi.EnableMultiCoreSplitK(false);
    tilingApi.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTM);
    tilingApi.EnableBias(false);
    tilingApi.SetBufferSpace(-1, -1, -1);

    AscendC::tiling::TCubeTiling tiling;
    if (tilingApi.GetTiling(tiling) == -1 || tiling.usedCoreNum == 0 ||
        tiling.M == 0 || tiling.N == 0 ||
        tiling.Ka == 0 || tiling.Kb == 0 || tiling.singleCoreM == 0 ||
        tiling.singleCoreN == 0 || tiling.singleCoreN != n) {
        return;
    }

    const uint32_t activeAivCount = QmqHostCeilDiv(
        m, static_cast<uint32_t>(tiling.singleCoreM));
    if (activeAivCount == 0 || activeAivCount > aivCandidate) {
        return;
    }
    const uint32_t activeGroupCount = QmqHostCeilDiv(activeAivCount, 2U);
    if (activeGroupCount > usableCoreNum) {
        return;
    }

    const uint64_t elementCount = static_cast<uint64_t>(m) * n;
    if (elementCount > SIZE_MAX / sizeof(int32_t)) {
        return;
    }
    const size_t resultWorkspaceSize = static_cast<size_t>(elementCount * sizeof(int32_t));
    const size_t requestedSystemWorkspaceSize =
        static_cast<size_t>(platform->GetLibApiWorkSpaceSize());
    const size_t systemWorkspaceSize = std::max<size_t>(requestedSystemWorkspaceSize, 1U);

    uint8_t* resultWorkspace = nullptr;
    uint8_t* systemWorkspace = nullptr;
    if (aclrtMalloc(
            reinterpret_cast<void**>(&resultWorkspace),
            resultWorkspaceSize,
            ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return;
    }
    if (aclrtMalloc(
            reinterpret_cast<void**>(&systemWorkspace),
            systemWorkspaceSize,
            ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        aclrtFree(resultWorkspace);
        return;
    }
    quant_matmul_relu_quant_fused<<<activeGroupCount, 0, stream>>>(
        x1,
        x2,
        resultWorkspace,
        systemWorkspace,
        x1Scale,
        x2Scale,
        y,
        yScale,
        m,
        n,
        k,
        tiling,
        activeGroupCount);
    (void)aclrtSynchronizeStream(stream);
    aclrtFree(systemWorkspace);
    aclrtFree(resultWorkspace);
}
