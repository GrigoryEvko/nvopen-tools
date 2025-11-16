// Function: sub_36F0560
// Address: 0x36f0560
//
__int64 __fastcall sub_36F0560(__int64 a1, __int64 a2, int a3, unsigned __int64 *a4, __m128i a5)
{
  __m128i *v7; // rsi
  const __m128i *v8; // r8
  __m128i *v9; // rsi
  __m128i v10; // [rsp+0h] [rbp-40h] BYREF
  __m128i v11; // [rsp+10h] [rbp-30h] BYREF

  v10.m128i_i64[0] = 0;
  v10.m128i_i32[2] = 0;
  v11.m128i_i64[0] = 0;
  v11.m128i_i32[2] = 0;
  if ( a3 != 4
    || !(unsigned __int8)sub_36DF750(a1, *(_QWORD *)a2, *(_QWORD *)(a2 + 8), (__int64)&v10, (__int64)&v11, a5) )
  {
    return 1;
  }
  v7 = (__m128i *)a4[1];
  v8 = (const __m128i *)a4[2];
  if ( v7 == v8 )
  {
    sub_33764F0(a4, v7, &v10);
    v9 = (__m128i *)a4[1];
    v8 = (const __m128i *)a4[2];
    if ( v9 != v8 )
    {
      if ( !v9 )
        goto LABEL_9;
      goto LABEL_8;
    }
  }
  else
  {
    if ( v7 )
    {
      *v7 = _mm_loadu_si128(&v10);
      v8 = (const __m128i *)a4[2];
      v7 = (__m128i *)a4[1];
    }
    v9 = v7 + 1;
    a4[1] = (unsigned __int64)v9;
    if ( v9 != v8 )
    {
LABEL_8:
      *v9 = _mm_loadu_si128(&v11);
      v9 = (__m128i *)a4[1];
LABEL_9:
      a4[1] = (unsigned __int64)&v9[1];
      return 0;
    }
  }
  sub_33764F0(a4, v8, &v11);
  return 0;
}
