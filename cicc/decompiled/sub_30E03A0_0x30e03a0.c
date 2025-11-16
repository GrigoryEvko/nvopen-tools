// Function: sub_30E03A0
// Address: 0x30e03a0
//
__int64 __fastcall sub_30E03A0(__int64 a1, __int64 a2, __int64 a3)
{
  const __m128i *v4; // rax
  __m128i v5; // xmm1
  int v6; // eax
  __int64 v7; // rdi
  __int64 v8; // rax
  int v9; // ecx
  int v10; // eax
  _OWORD v12[5]; // [rsp+8h] [rbp-3C0h] BYREF
  __int32 v13; // [rsp+58h] [rbp-370h]
  _QWORD v14[9]; // [rsp+68h] [rbp-360h] BYREF
  __int64 v15; // [rsp+B0h] [rbp-318h]
  __int64 v16; // [rsp+B8h] [rbp-310h]
  unsigned __int8 *v17; // [rsp+C8h] [rbp-300h]
  char v18; // [rsp+2F0h] [rbp-D8h]
  int v19; // [rsp+2F8h] [rbp-D0h]
  int v20; // [rsp+2FCh] [rbp-CCh]
  __int64 v21; // [rsp+300h] [rbp-C8h]
  int v22; // [rsp+328h] [rbp-A0h]
  int v23; // [rsp+334h] [rbp-94h]

  v4 = *(const __m128i **)(a1 + 664);
  v12[0] = _mm_loadu_si128(v4);
  v5 = _mm_loadu_si128(v4 + 1);
  LODWORD(v12[0]) = 100;
  v12[1] = v5;
  v12[2] = _mm_loadu_si128(v4 + 2);
  v12[3] = _mm_loadu_si128(v4 + 3);
  v12[4] = _mm_loadu_si128(v4 + 4);
  v13 = v4[5].m128i_i32[0];
  sub_30D4900(
    (__int64)v14,
    a2,
    a3,
    (int *)v12,
    *(_QWORD *)(a1 + 8),
    *(_QWORD *)(a1 + 64),
    *(_QWORD *)(a1 + 16),
    *(_QWORD *)(a1 + 24),
    *(_QWORD *)(a1 + 32),
    *(_QWORD *)(a1 + 40),
    *(_QWORD *)(a1 + 48),
    *(_QWORD *)(a1 + 56),
    *(_QWORD *)(a1 + 88),
    0,
    0);
  sub_30D2590((__int64)v14, (__int64)v17, v15);
  v22 += v20 + v19;
  v6 = sub_30D4FE0((__int64 *)v14[1], v17, v16);
  v7 = v15;
  v8 = v23 + (__int64)-v6;
  if ( v8 > 0x7FFFFFFF )
    v8 = 0x7FFFFFFF;
  if ( v8 < (__int64)0xFFFFFFFF80000000LL )
    LODWORD(v8) = 0x80000000;
  v23 = v8;
  v9 = v8;
  if ( ((*(_WORD *)(v15 + 2) >> 4) & 0x3FF) == 9 )
  {
    v9 = v8 + 2000;
    v23 = v8 + 2000;
  }
  if ( v22 > v9 || v18 )
  {
    if ( !*(_BYTE *)(v21 + 66) )
    {
      if ( sub_B2DCC0(v15) )
        return sub_30D30A0((__int64)v14);
      v7 = v15;
    }
    if ( v7 + 72 == (*(_QWORD *)(v7 + 72) & 0xFFFFFFFFFFFFFFF8LL) || !sub_30DC7E0(v14) )
    {
      v10 = v22 - v23;
      if ( v22 - v23 < 0 )
        v10 = 0;
      *(_DWORD *)(a1 + 716) -= v10;
    }
  }
  return sub_30D30A0((__int64)v14);
}
