// Function: sub_982320
// Address: 0x982320
//
__int64 __fastcall sub_982320(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 (__fastcall *a4)(__m128i *),
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9,
        __int128 a10)
{
  __int64 result; // rax
  __int64 i; // r15
  __int64 v12; // r14
  __m128i *v13; // r12
  __m128i *v14; // rax
  __int64 v15; // r15
  const __m128i *v16; // r13
  __m128i v17; // xmm5
  __m128i v18; // xmm6
  __m128i v19; // xmm7
  const __m128i *v20; // rax
  __int64 v22; // [rsp+8h] [rbp-98h]
  __int64 v23; // [rsp+10h] [rbp-90h]
  __m128i v25; // [rsp+30h] [rbp-70h] BYREF
  __m128i v26; // [rsp+40h] [rbp-60h] BYREF
  __m128i v27; // [rsp+50h] [rbp-50h] BYREF
  __m128i v28[4]; // [rsp+60h] [rbp-40h] BYREF

  result = (a3 - 1) / 2;
  v23 = result;
  v22 = a3 & 1;
  if ( a2 >= result )
  {
    v13 = (__m128i *)(a1 + (a2 << 6));
    if ( (a3 & 1) != 0 )
    {
      v25 = _mm_loadu_si128((const __m128i *)&a7);
      v26 = _mm_loadu_si128((const __m128i *)&a8);
      v27 = _mm_loadu_si128((const __m128i *)&a9);
      v28[0] = _mm_loadu_si128((const __m128i *)&a10);
      goto LABEL_13;
    }
    v12 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v12 )
  {
    v12 = 2 * (i + 1);
    v13 = (__m128i *)(a1 + ((i + 1) << 7));
    if ( a4(v13) )
      v13 = (__m128i *)(a1 + (--v12 << 6));
    v14 = (__m128i *)(a1 + (i << 6));
    *v14 = _mm_loadu_si128(v13);
    v14[1] = _mm_loadu_si128(v13 + 1);
    v14[2] = _mm_loadu_si128(v13 + 2);
    v14[3] = _mm_loadu_si128(v13 + 3);
    if ( v12 >= v23 )
      break;
  }
  if ( !v22 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v12 )
    {
      v12 = 2 * v12 + 1;
      v20 = (const __m128i *)(a1 + (v12 << 6));
      *v13 = _mm_loadu_si128(v20);
      v13[1] = _mm_loadu_si128(v20 + 1);
      v13[2] = _mm_loadu_si128(v20 + 2);
      v13[3] = _mm_loadu_si128(v20 + 3);
      v13 = (__m128i *)v20;
    }
  }
  result = v12 - 1;
  v25 = _mm_loadu_si128((const __m128i *)&a7);
  v26 = _mm_loadu_si128((const __m128i *)&a8);
  v27 = _mm_loadu_si128((const __m128i *)&a9);
  v15 = (v12 - 1) / 2;
  v28[0] = _mm_loadu_si128((const __m128i *)&a10);
  if ( v12 > a2 )
  {
    while ( 1 )
    {
      v16 = (const __m128i *)(a1 + (v15 << 6));
      result = ((__int64 (__fastcall *)(const __m128i *, __m128i *))a4)(v16, &v25);
      v13 = (__m128i *)(a1 + (v12 << 6));
      if ( !(_BYTE)result )
        break;
      v12 = v15;
      *v13 = _mm_loadu_si128(v16);
      v13[1] = _mm_loadu_si128(v16 + 1);
      result = (v15 - 1) / 2;
      v13[2] = _mm_loadu_si128(v16 + 2);
      v13[3] = _mm_loadu_si128(v16 + 3);
      if ( a2 >= v15 )
      {
        v13 = (__m128i *)(a1 + (v15 << 6));
        break;
      }
      v15 = (v15 - 1) / 2;
    }
  }
LABEL_13:
  v17 = _mm_loadu_si128(&v26);
  v18 = _mm_loadu_si128(&v27);
  v19 = _mm_loadu_si128(v28);
  *v13 = _mm_loadu_si128(&v25);
  v13[1] = v17;
  v13[2] = v18;
  v13[3] = v19;
  return result;
}
