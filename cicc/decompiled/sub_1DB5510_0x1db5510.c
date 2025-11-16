// Function: sub_1DB5510
// Address: 0x1db5510
//
unsigned __int64 __fastcall sub_1DB5510(__m128i ***a1)
{
  __m128i *v1; // rsi
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  __m128i *v5; // rcx
  __m128i **v6; // rax
  __m128i **v7; // rdi
  __m128i *v8; // r10
  __m128i v9; // xmm1
  __m128i v10; // xmm0
  __int64 v11; // rdi

  v1 = (__m128i *)a1[2];
  v3 = *((unsigned int *)a1 + 10);
  v4 = 0xAAAAAAAAAAAAAAABLL * (((char *)a1[3] - (char *)v1) >> 3);
  if ( v4 > v3 )
    v4 = *((unsigned int *)a1 + 10);
  v5 = (__m128i *)((char *)v1 + 24 * v4);
  v6 = a1[4];
  v7 = &v6[3 * v3];
  v8 = **a1;
  a1[2] = (__m128i **)v5;
  if ( v1 != v5 )
  {
    do
    {
      while ( 1 )
      {
        v5 = (__m128i *)((char *)v5 - 24);
        if ( v8 == v1
          || (*(_DWORD *)((v1[-2].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
            | (unsigned int)(v1[-2].m128i_i64[1] >> 1) & 3) <= (*(_DWORD *)(((unsigned __int64)*(v7 - 3)
                                                                           & 0xFFFFFFFFFFFFFFF8LL)
                                                                          + 24)
                                                              | (unsigned int)((__int64)*(v7 - 3) >> 1) & 3) )
        {
          break;
        }
        v9 = _mm_loadu_si128((__m128i *)((char *)v1 - 24));
        v1 = (__m128i *)((char *)v1 - 24);
        *v5 = v9;
        v5[1].m128i_i64[0] = v1[1].m128i_i64[0];
        if ( v1 == v5 )
          goto LABEL_9;
      }
      v10 = _mm_loadu_si128((const __m128i *)(v7 - 3));
      v7 -= 3;
      *v5 = v10;
      v5[1].m128i_i64[0] = (__int64)v7[2];
    }
    while ( v1 != v5 );
LABEL_9:
    v6 = a1[4];
  }
  v11 = (char *)v7 - (char *)v6;
  *((_DWORD *)a1 + 10) = -1431655765 * (v11 >> 3);
  return 0xAAAAAAAAAAAAAAABLL;
}
