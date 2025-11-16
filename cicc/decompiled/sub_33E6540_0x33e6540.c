// Function: sub_33E6540
// Address: 0x33e6540
//
unsigned __int64 __fastcall sub_33E6540(_QWORD *a1, int a2, int a3, __int64 *a4, __int64 *a5)
{
  __int64 v8; // rsi
  __int64 v9; // r8
  int v10; // r15d
  unsigned __int64 v11; // r12
  unsigned __int8 *v12; // rsi
  __int64 v14; // rcx
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // [rsp+8h] [rbp-48h]
  __int64 v18[7]; // [rsp+18h] [rbp-38h] BYREF

  v8 = *a4;
  v18[0] = v8;
  if ( v8 )
    sub_B96E90((__int64)v18, v8, 1);
  v9 = *a5;
  v10 = *((_DWORD *)a5 + 2);
  v11 = a1[52];
  if ( v11 )
  {
    a1[52] = *(_QWORD *)v11;
LABEL_5:
    *(_QWORD *)v11 = 0;
    v12 = (unsigned __int8 *)v18[0];
    *(_QWORD *)(v11 + 8) = 0;
    *(_QWORD *)(v11 + 16) = 0;
    *(_DWORD *)(v11 + 24) = a2;
    *(_DWORD *)(v11 + 28) = 0;
    *(_WORD *)(v11 + 34) = -1;
    *(_DWORD *)(v11 + 36) = -1;
    *(_QWORD *)(v11 + 40) = 0;
    *(_QWORD *)(v11 + 48) = v9;
    *(_QWORD *)(v11 + 56) = 0;
    *(_DWORD *)(v11 + 64) = 0;
    *(_DWORD *)(v11 + 68) = v10;
    *(_DWORD *)(v11 + 72) = a3;
    *(_QWORD *)(v11 + 80) = v12;
    if ( v12 )
      sub_B976B0((__int64)v18, v12, v11 + 80);
    *(_QWORD *)(v11 + 88) = 0xFFFFFFFFLL;
    *(_WORD *)(v11 + 32) = 0;
    return v11;
  }
  v14 = a1[53];
  a1[63] += 120LL;
  v15 = (v14 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[54] < v15 + 120 || !v14 )
  {
    v17 = v9;
    v16 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    v9 = v17;
    v11 = v16;
    goto LABEL_5;
  }
  a1[53] = v15 + 120;
  if ( v15 )
  {
    v11 = (v14 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_5;
  }
  if ( v18[0] )
    sub_B91220((__int64)v18, v18[0]);
  return v11;
}
