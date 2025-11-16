// Function: sub_263F380
// Address: 0x263f380
//
void __fastcall sub_263F380(__int64 a1, unsigned __int64 *a2)
{
  unsigned __int64 *v3; // rsi
  unsigned __int64 v5; // rdi
  unsigned __int64 *v6; // rax
  __m128i v7; // xmm2
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  __m128i v10; // xmm0
  __m128i v11; // xmm3
  __m128i v12; // xmm4
  unsigned __int64 v13; // rcx
  unsigned __int64 *v14; // rdx
  __m128i v15; // xmm1
  __m128i v16; // xmm5
  __int64 v17; // [rsp-8h] [rbp-8h] BYREF

  if ( (unsigned __int64 *)a1 != a2 )
  {
    v3 = (unsigned __int64 *)(a1 + 24);
    while ( a2 != v3 )
    {
      v5 = *v3;
      v6 = v3;
      v3 += 3;
      if ( v5 >= *(_QWORD *)a1 )
      {
        v12 = _mm_loadu_si128((const __m128i *)(v3 - 3));
        v13 = *(v3 - 6);
        *(&v17 - 2) = *(v3 - 1);
        v14 = v3 - 6;
        *((__m128i *)&v17 - 2) = v12;
        if ( v5 < v13 )
        {
          do
          {
            v15 = _mm_loadu_si128((const __m128i *)(v14 + 1));
            v14[3] = v13;
            v6 = v14;
            v14 -= 3;
            *(__m128i *)(v14 + 7) = v15;
            v13 = *v14;
          }
          while ( v5 < *v14 );
        }
        v16 = _mm_loadu_si128((const __m128i *)(&v17 - 3));
        *v6 = v5;
        *(__m128i *)(v6 + 1) = v16;
      }
      else
      {
        v7 = _mm_loadu_si128((const __m128i *)(v3 - 3));
        *(&v17 - 2) = *(v3 - 1);
        *((__m128i *)&v17 - 2) = v7;
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
        v11 = _mm_loadu_si128((const __m128i *)(&v17 - 3));
        *(_QWORD *)a1 = v5;
        *(__m128i *)(a1 + 8) = v11;
      }
    }
  }
}
