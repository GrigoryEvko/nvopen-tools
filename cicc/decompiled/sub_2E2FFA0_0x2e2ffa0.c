// Function: sub_2E2FFA0
// Address: 0x2e2ffa0
//
void __fastcall sub_2E2FFA0(__int64 a1, unsigned int *a2)
{
  unsigned int *v2; // rcx
  unsigned int v4; // esi
  unsigned int *v5; // rdx
  __m128i v6; // xmm2
  unsigned __int64 v7; // rax
  unsigned int v8; // esi
  __m128i v9; // xmm0
  __m128i v10; // xmm3
  __m128i v11; // xmm4
  unsigned int *v12; // rax
  __m128i v13; // xmm1
  __m128i v14; // xmm5
  __int64 v15; // [rsp-8h] [rbp-8h] BYREF

  if ( (unsigned int *)a1 != a2 )
  {
    v2 = (unsigned int *)(a1 + 24);
    while ( a2 != v2 )
    {
      v4 = *v2;
      v5 = v2;
      v2 += 6;
      if ( v4 >= *(_DWORD *)a1 )
      {
        v11 = _mm_loadu_si128((const __m128i *)(v2 - 6));
        *(&v15 - 2) = *((_QWORD *)v2 - 1);
        v12 = v2 - 12;
        for ( *((__m128i *)&v15 - 2) = v11; v4 < *v12; *(__m128i *)(v12 + 14) = v13 )
        {
          v13 = _mm_loadu_si128((const __m128i *)(v12 + 2));
          v12[6] = *v12;
          v5 = v12;
          v12 -= 6;
        }
        v14 = _mm_loadu_si128((const __m128i *)(&v15 - 3));
        *v5 = v4;
        *(__m128i *)(v5 + 2) = v14;
      }
      else
      {
        v6 = _mm_loadu_si128((const __m128i *)(v2 - 6));
        *(&v15 - 2) = *((_QWORD *)v2 - 1);
        *((__m128i *)&v15 - 2) = v6;
        v7 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v5 - a1) >> 3);
        if ( (__int64)v5 - a1 > 0 )
        {
          do
          {
            v8 = *(v5 - 6);
            v9 = _mm_loadu_si128((const __m128i *)v5 - 1);
            v5 -= 6;
            v5[6] = v8;
            *((__m128i *)v5 + 2) = v9;
            --v7;
          }
          while ( v7 );
        }
        v10 = _mm_loadu_si128((const __m128i *)(&v15 - 3));
        *(_DWORD *)a1 = *((_DWORD *)&v15 - 8);
        *(__m128i *)(a1 + 8) = v10;
      }
    }
  }
}
