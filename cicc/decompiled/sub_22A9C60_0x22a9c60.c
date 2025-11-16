// Function: sub_22A9C60
// Address: 0x22a9c60
//
void __fastcall sub_22A9C60(__int64 *a1, __m128i *a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v5; // rbx
  __int64 v6; // rax
  const __m128i *v7; // rdx
  __m128i v8; // xmm4
  __int64 v9; // rsi
  __int64 v10; // rax
  __m128i v11; // xmm5
  __int64 v12; // rcx
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  __int64 v16; // rcx
  unsigned __int64 v17; // rbx
  const __m128i *v18; // rax
  __m128i v19; // xmm6
  __m128i v20; // xmm7
  __m128i v21; // xmm6

  v3 = 0x249249249249249LL;
  if ( a3 <= 0x249249249249249LL )
    v3 = a3;
  *a1 = a3;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 > 0 )
  {
    while ( 1 )
    {
      v5 = 56 * v3;
      v6 = sub_2207800(56 * v3);
      v7 = (const __m128i *)v6;
      if ( v6 )
        break;
      v3 >>= 1;
      if ( !v3 )
        return;
    }
    v8 = _mm_loadu_si128(a2 + 1);
    v9 = v6 + v5;
    v10 = v6 + 56;
    v11 = _mm_loadu_si128(a2 + 2);
    v12 = a2[3].m128i_i64[0];
    *(__m128i *)(v10 - 56) = _mm_loadu_si128(a2);
    *(__m128i *)(v10 - 40) = v8;
    *(__m128i *)(v10 - 24) = v11;
    *(_QWORD *)(v10 - 8) = v12;
    if ( v9 == v10 )
    {
      v18 = v7;
    }
    else
    {
      do
      {
        v13 = _mm_loadu_si128((const __m128i *)(v10 - 56));
        v14 = _mm_loadu_si128((const __m128i *)(v10 - 40));
        v10 += 56;
        v15 = _mm_loadu_si128((const __m128i *)(v10 - 80));
        v16 = *(_QWORD *)(v10 - 64);
        *(__m128i *)(v10 - 56) = v13;
        *(__m128i *)(v10 - 40) = v14;
        *(__m128i *)(v10 - 24) = v15;
        *(_QWORD *)(v10 - 8) = v16;
      }
      while ( v9 != v10 );
      v17 = (0xDB6DB6DB6DB6DB7LL * ((unsigned __int64)(v5 - 112) >> 3)) & 0x1FFFFFFFFFFFFFFFLL;
      v18 = (const __m128i *)((char *)v7 + 56 * v17 + 56);
      v12 = v7[6].m128i_i64[7 * v17 + 1];
    }
    v19 = _mm_loadu_si128(v18 + 1);
    v20 = _mm_loadu_si128(v18 + 2);
    a2[3].m128i_i64[0] = v12;
    a1[2] = (__int64)v7;
    a2[1] = v19;
    v21 = _mm_loadu_si128(v18);
    a1[1] = v3;
    a2[2] = v20;
    *a2 = v21;
  }
}
