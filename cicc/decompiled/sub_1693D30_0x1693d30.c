// Function: sub_1693D30
// Address: 0x1693d30
//
void __fastcall sub_1693D30(__int64 a1, unsigned __int64 *a2)
{
  unsigned __int64 *v2; // rcx
  unsigned __int64 v5; // rdi
  unsigned __int64 *v6; // rdx
  __m128i v7; // xmm2
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rsi
  __m128i v10; // xmm0
  __m128i v11; // xmm3
  __m128i v12; // xmm4
  unsigned __int64 *v13; // rax
  __m128i v14; // xmm1
  __m128i v15; // xmm5
  __int64 v16; // [rsp-8h] [rbp-8h] BYREF

  if ( (unsigned __int64 *)a1 != a2 )
  {
    v2 = (unsigned __int64 *)(a1 + 24);
    while ( a2 != v2 )
    {
      v5 = *v2;
      v6 = v2;
      v2 += 3;
      if ( v5 >= *(_QWORD *)a1 )
      {
        v12 = _mm_loadu_si128((const __m128i *)(v2 - 3));
        *(&v16 - 2) = *(v2 - 1);
        v13 = v2 - 6;
        for ( *((__m128i *)&v16 - 2) = v12; v5 < *v13; *(__m128i *)(v13 + 7) = v14 )
        {
          v14 = _mm_loadu_si128((const __m128i *)(v13 + 1));
          v13[3] = *v13;
          v6 = v13;
          v13 -= 3;
        }
        v15 = _mm_loadu_si128((const __m128i *)(&v16 - 3));
        *v6 = v5;
        *(__m128i *)(v6 + 1) = v15;
      }
      else
      {
        v7 = _mm_loadu_si128((const __m128i *)(v2 - 3));
        *(&v16 - 2) = *(v2 - 1);
        *((__m128i *)&v16 - 2) = v7;
        v8 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v6 - a1) >> 3);
        if ( (__int64)v6 - a1 > 0 )
        {
          do
          {
            v9 = *(v6 - 3);
            v10 = _mm_loadu_si128((const __m128i *)v6 - 1);
            v6 -= 3;
            v6[3] = v9;
            *((__m128i *)v6 + 2) = v10;
            --v8;
          }
          while ( v8 );
        }
        v11 = _mm_loadu_si128((const __m128i *)(&v16 - 3));
        *(_QWORD *)a1 = v5;
        *(__m128i *)(a1 + 8) = v11;
      }
    }
  }
}
