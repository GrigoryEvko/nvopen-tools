// Function: sub_1B2C540
// Address: 0x1b2c540
//
void __fastcall sub_1B2C540(__int64 *a1, __m128i *a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // rbx
  __int64 v5; // rax
  const __m128i *v6; // rdx
  __m128i v7; // xmm4
  __int64 v8; // rcx
  __int64 v9; // rax
  __m128i v10; // xmm5
  __m128i v11; // xmm0
  __m128i v12; // xmm1
  __m128i v13; // xmm2
  const __m128i *v14; // rax
  __m128i v15; // xmm6
  __m128i v16; // xmm7
  __m128i v17; // xmm6

  v3 = 0x2AAAAAAAAAAAAAALL;
  if ( a3 <= 0x2AAAAAAAAAAAAAALL )
    v3 = a3;
  *a1 = a3;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 > 0 )
  {
    while ( 1 )
    {
      v4 = 48 * v3;
      v5 = sub_2207800(48 * v3, &unk_435FF63);
      v6 = (const __m128i *)v5;
      if ( v5 )
        break;
      v3 >>= 1;
      if ( !v3 )
        return;
    }
    v7 = _mm_loadu_si128(a2 + 1);
    v8 = v5 + v4;
    v9 = v5 + 48;
    v10 = _mm_loadu_si128(a2 + 2);
    *(__m128i *)(v9 - 48) = _mm_loadu_si128(a2);
    *(__m128i *)(v9 - 32) = v7;
    *(__m128i *)(v9 - 16) = v10;
    if ( v8 == v9 )
    {
      v14 = v6;
    }
    else
    {
      do
      {
        v11 = _mm_loadu_si128((const __m128i *)(v9 - 48));
        v12 = _mm_loadu_si128((const __m128i *)(v9 - 32));
        v9 += 48;
        v13 = _mm_loadu_si128((const __m128i *)(v9 - 64));
        *(__m128i *)(v9 - 48) = v11;
        *(__m128i *)(v9 - 32) = v12;
        *(__m128i *)(v9 - 16) = v13;
      }
      while ( v8 != v9 );
      v14 = &v6[3 * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)(v4 - 96) >> 4)) & 0xFFFFFFFFFFFFFFFLL) + 3];
    }
    v15 = _mm_loadu_si128(v14);
    v16 = _mm_loadu_si128(v14 + 1);
    a1[2] = (__int64)v6;
    a1[1] = v3;
    *a2 = v15;
    v17 = _mm_loadu_si128(v14 + 2);
    a2[1] = v16;
    a2[2] = v17;
  }
}
