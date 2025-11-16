// Function: sub_C9E2A0
// Address: 0xc9e2a0
//
__int64 __fastcall sub_C9E2A0(__int64 a1)
{
  __int64 v2; // rdx
  double v3; // rax
  double v4; // xmm1_8
  double v5; // xmm0_8
  __int64 result; // rax
  double v7[7]; // [rsp+0h] [rbp-40h] BYREF

  *(_BYTE *)(a1 + 144) = 0;
  sub_C9E0E0(v7, 0);
  v2 = *(_QWORD *)(a1 + 24) + *(_QWORD *)&v7[3];
  v3 = v7[4];
  v4 = v7[1] + *(double *)(a1 + 8) - *(double *)(a1 + 48);
  v5 = *(double *)(a1 + 16) + v7[2] - *(double *)(a1 + 56);
  *(double *)a1 = v7[0] + *(double *)a1 - *(double *)(a1 + 40);
  result = *(_QWORD *)(a1 + 32) + *(_QWORD *)&v3 - *(_QWORD *)(a1 + 72);
  *(_QWORD *)(a1 + 24) = v2 - *(_QWORD *)(a1 + 64);
  *(_QWORD *)(a1 + 32) = result;
  *(double *)(a1 + 8) = v4;
  *(double *)(a1 + 16) = v5;
  return result;
}
