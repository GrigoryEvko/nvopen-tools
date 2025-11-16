// Function: sub_18B71D0
// Address: 0x18b71d0
//
__int64 __fastcall sub_18B71D0(
        __int64 a1,
        __int64 a2,
        _BYTE *a3,
        __int64 a4,
        __int64 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  const __m128i *v15; // rbx
  _QWORD *v16; // r15
  __int64 v17; // r14
  _QWORD *v18; // rdi
  __m128 v19; // xmm0
  __int64 v20; // r9
  __int64 v21; // r14
  double v22; // xmm4_8
  double v23; // xmm5_8
  __int64 result; // rax
  const __m128i *v27; // [rsp+18h] [rbp-58h]
  __m128 v28; // [rsp+20h] [rbp-50h] BYREF
  _DWORD *v29; // [rsp+30h] [rbp-40h]

  v15 = *(const __m128i **)a2;
  v27 = *(const __m128i **)(a2 + 8);
  if ( *(const __m128i **)a2 != v27 )
  {
    do
    {
      v19 = (__m128)_mm_loadu_si128(v15);
      v28 = v19;
      v29 = (_DWORD *)v15[1].m128i_i64[0];
      v21 = sub_159C470(*(_QWORD *)(v19.m128_u64[1] & 0xFFFFFFFFFFFFFFF8LL), a5, 0);
      if ( *(_BYTE *)(a1 + 80) )
        sub_18B6C20(
          (__int64)&v28,
          "uniform-ret-val",
          15,
          a3,
          a4,
          v20,
          *(__int64 (__fastcall **)(__int64, __int64))(a1 + 88),
          *(_QWORD *)(a1 + 96));
      sub_164D160(v28.m128_u64[1] & 0xFFFFFFFFFFFFFFF8LL, v21, v19, a7, a8, a9, v22, v23, a12, a13);
      v16 = (_QWORD *)(v28.m128_u64[1] & 0xFFFFFFFFFFFFFFF8LL);
      if ( *(_BYTE *)((v28.m128_u64[1] & 0xFFFFFFFFFFFFFFF8LL) + 16) == 29 )
      {
        v17 = *(v16 - 6);
        v18 = sub_1648A60(56, 1u);
        if ( v18 )
          sub_15F8320((__int64)v18, v17, (__int64)v16);
        sub_157F2D0(*(v16 - 3), v16[5], 0);
        v16 = (_QWORD *)(v28.m128_u64[1] & 0xFFFFFFFFFFFFFFF8LL);
      }
      sub_15F20C0(v16);
      if ( v29 )
        --*v29;
      v15 = (const __m128i *)((char *)v15 + 24);
    }
    while ( v27 != v15 );
  }
  *(_BYTE *)(a2 + 24) = 1;
  result = *(_QWORD *)(a2 + 32);
  if ( result != *(_QWORD *)(a2 + 40) )
    *(_QWORD *)(a2 + 40) = result;
  return result;
}
