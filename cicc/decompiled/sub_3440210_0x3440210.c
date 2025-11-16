// Function: sub_3440210
// Address: 0x3440210
//
__int64 __fastcall sub_3440210(const __m128i *a1, const __m128i *a2, const __m128i *a3, const __m128i *a4, __int64 a5)
{
  __m128i v6; // xmm2
  __int64 v7; // rcx
  __int64 v8; // rax
  unsigned int v9; // ecx
  __m128i v10; // xmm3
  __int64 v11; // rax
  unsigned __int64 v12; // rsi
  __int64 v13; // rcx
  __m128i v14; // xmm0
  __int64 result; // rax
  __int64 v16; // r9
  unsigned __int64 v17; // rsi
  __int64 v18; // rcx
  __m128i v19; // xmm1

  if ( a1 != a2 && a3 != a4 )
  {
    do
    {
      v7 = a3[1].m128i_u32[0];
      v8 = a1[1].m128i_u32[0];
      if ( (unsigned int)v7 > 6 || (v9 = dword_44E2140[v7], (unsigned int)v8 > 6) )
        BUG();
      if ( v9 > dword_44E2140[v8] )
      {
        v6 = _mm_loadu_si128(a3);
        a5 += 24;
        a3 = (const __m128i *)((char *)a3 + 24);
        *(__m128i *)(a5 - 24) = v6;
        *(_DWORD *)(a5 - 8) = a3[-1].m128i_i32[2];
        if ( a1 == a2 )
          break;
      }
      else
      {
        v10 = _mm_loadu_si128(a1);
        a1 = (const __m128i *)((char *)a1 + 24);
        a5 += 24;
        *(__m128i *)(a5 - 24) = v10;
        *(_DWORD *)(a5 - 8) = a1[-1].m128i_i32[2];
        if ( a1 == a2 )
          break;
      }
    }
    while ( a3 != a4 );
  }
  v11 = (char *)a2 - (char *)a1;
  v12 = 0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)a1) >> 3);
  if ( v11 <= 0 )
  {
    result = a5;
  }
  else
  {
    v13 = a5;
    do
    {
      v14 = _mm_loadu_si128(a1);
      v13 += 24;
      a1 = (const __m128i *)((char *)a1 + 24);
      *(__m128i *)(v13 - 24) = v14;
      *(_DWORD *)(v13 - 8) = a1[-1].m128i_i32[2];
      --v12;
    }
    while ( v12 );
    result = a5 + v11;
  }
  v16 = (char *)a4 - (char *)a3;
  v17 = 0xAAAAAAAAAAAAAAABLL * (v16 >> 3);
  if ( v16 > 0 )
  {
    v18 = result;
    do
    {
      v19 = _mm_loadu_si128(a3);
      v18 += 24;
      a3 = (const __m128i *)((char *)a3 + 24);
      *(__m128i *)(v18 - 24) = v19;
      *(_DWORD *)(v18 - 8) = a3[-1].m128i_i32[2];
      --v17;
    }
    while ( v17 );
    result += v16;
  }
  return result;
}
