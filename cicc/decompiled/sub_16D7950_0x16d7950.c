// Function: sub_16D7950
// Address: 0x16d7950
//
__int64 __fastcall sub_16D7950(__int64 a1)
{
  double v2; // xmm2_8
  __int64 result; // rax
  double v4; // xmm0_8
  double v5; // xmm1_8
  double v6; // xmm0_8
  double v7[5]; // [rsp+0h] [rbp-30h] BYREF

  *(_BYTE *)(a1 + 128) = 0;
  sub_16D7810(v7, 0);
  v2 = v7[0] + *(double *)a1 - *(double *)(a1 + 32);
  result = *(_QWORD *)(a1 + 24) + *(_QWORD *)&v7[3] - *(_QWORD *)(a1 + 56);
  v4 = *(double *)(a1 + 16);
  v5 = v7[1] + *(double *)(a1 + 8) - *(double *)(a1 + 40);
  *(_QWORD *)(a1 + 24) = result;
  v6 = v4 + v7[2] - *(double *)(a1 + 48);
  *(double *)a1 = v2;
  *(double *)(a1 + 8) = v5;
  *(double *)(a1 + 16) = v6;
  return result;
}
