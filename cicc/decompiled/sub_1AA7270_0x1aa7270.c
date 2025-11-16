// Function: sub_1AA7270
// Address: 0x1aa7270
//
__int64 __fastcall sub_1AA7270(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned __int64 v11; // rax
  __int64 v12; // rbx
  unsigned int v13; // eax
  unsigned int v14; // r15d
  __int64 v15; // r13
  const __m128i *v16; // rsi
  __int64 v17; // rbx
  unsigned __int64 i; // r15
  __int64 v19; // rax
  double v20; // xmm4_8
  double v21; // xmm5_8
  _QWORD *v22; // rbx
  unsigned __int64 *v23; // rcx
  unsigned __int64 v24; // rdx
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 result; // rax
  int v28; // [rsp+Ch] [rbp-64h]
  __m128i v29; // [rsp+10h] [rbp-60h] BYREF
  const __m128i *v30; // [rsp+20h] [rbp-50h] BYREF
  __m128i *v31; // [rsp+28h] [rbp-48h]
  const __m128i *v32; // [rsp+30h] [rbp-40h]

  v11 = sub_157EBA0(a1);
  v30 = 0;
  v31 = 0;
  v12 = v11;
  v32 = 0;
  if ( a2 )
  {
    v13 = sub_15F4D60(v11);
    sub_1953AE0(&v30, v13);
  }
  v14 = 0;
  v28 = sub_15F4D60(v12);
  if ( v28 )
  {
    do
    {
      v15 = sub_15F4DF0(v12, v14);
      sub_157F2D0(v15, a1, 0);
      if ( a2 )
      {
        v29.m128i_i64[0] = a1;
        v16 = v31;
        v29.m128i_i64[1] = v15 | 4;
        if ( v31 == v32 )
        {
          sub_17F2860(&v30, v31, &v29);
        }
        else
        {
          if ( v31 )
          {
            a3 = (__m128)_mm_loadu_si128(&v29);
            *v31 = (__m128i)a3;
            v16 = v31;
          }
          v31 = (__m128i *)&v16[1];
        }
      }
      ++v14;
    }
    while ( v28 != v14 );
  }
  v17 = *(_QWORD *)(a1 + 40);
  for ( i = v17 & 0xFFFFFFFFFFFFFFF8LL; (v17 & 0xFFFFFFFFFFFFFFF8LL) != a1 + 40; i = v17 & 0xFFFFFFFFFFFFFFF8LL )
  {
    if ( !i )
      BUG();
    if ( *(_QWORD *)(i - 16) )
    {
      v19 = sub_1599EF0(*(__int64 ***)(i - 24));
      sub_164D160(i - 24, v19, a3, a4, a5, a6, v20, v21, a9, a10);
      v17 = *(_QWORD *)(a1 + 40);
    }
    v22 = (_QWORD *)(v17 & 0xFFFFFFFFFFFFFFF8LL);
    sub_157EA20(a1 + 40, (__int64)(v22 - 3));
    v23 = (unsigned __int64 *)v22[1];
    v24 = *v22 & 0xFFFFFFFFFFFFFFF8LL;
    *v23 = v24 | *v23 & 7;
    *(_QWORD *)(v24 + 8) = v23;
    *v22 &= 7uLL;
    v22[1] = 0;
    sub_164BEC0((__int64)(v22 - 3), (__int64)(v22 - 3), v24, (__int64)v23, a3, a4, a5, a6, v25, v26, a9, a10);
    v17 = *(_QWORD *)(a1 + 40);
  }
  if ( a2 )
  {
    sub_15CD9D0(a2, v30->m128i_i64, v31 - v30);
    result = (__int64)sub_15CD5A0(a2, a1);
  }
  else
  {
    result = sub_157F980(a1);
  }
  if ( v30 )
    return j_j___libc_free_0(v30, (char *)v32 - (char *)v30);
  return result;
}
