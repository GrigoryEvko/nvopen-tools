// Function: sub_22A72A0
// Address: 0x22a72a0
//
__int64 __fastcall sub_22A72A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 result; // rax
  __int64 v16; // r14
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx

  v7 = a3;
  v8 = a1;
  if ( a3 != a4 && a1 != a2 )
  {
    do
    {
      if ( (unsigned __int8)sub_22A71D0(v7, v8) )
      {
        v9 = *(_QWORD *)(v7 + 48);
        a5 += 56;
        v7 += 56;
        *(_QWORD *)(a5 - 8) = v9;
        *(__m128i *)(a5 - 40) = _mm_loadu_si128((const __m128i *)(v7 - 40));
        *(__m128i *)(a5 - 24) = _mm_loadu_si128((const __m128i *)(v7 - 24));
        *(__m128i *)(a5 - 56) = _mm_loadu_si128((const __m128i *)(v7 - 56));
        if ( v8 == a2 )
          break;
      }
      else
      {
        v10 = *(_QWORD *)(v8 + 48);
        v8 += 56;
        a5 += 56;
        *(_QWORD *)(a5 - 8) = v10;
        *(__m128i *)(a5 - 40) = _mm_loadu_si128((const __m128i *)(v8 - 40));
        *(__m128i *)(a5 - 24) = _mm_loadu_si128((const __m128i *)(v8 - 24));
        *(__m128i *)(a5 - 56) = _mm_loadu_si128((const __m128i *)(v8 - 56));
        if ( v8 == a2 )
          break;
      }
    }
    while ( v7 != a4 );
  }
  v11 = a2 - v8;
  v12 = 0x6DB6DB6DB6DB6DB7LL * ((a2 - v8) >> 3);
  if ( a2 - v8 <= 0 )
  {
    result = a5;
  }
  else
  {
    v13 = a5;
    do
    {
      v14 = *(_QWORD *)(v8 + 48);
      v13 += 56;
      v8 += 56;
      *(_QWORD *)(v13 - 8) = v14;
      *(__m128i *)(v13 - 40) = _mm_loadu_si128((const __m128i *)(v8 - 40));
      *(__m128i *)(v13 - 24) = _mm_loadu_si128((const __m128i *)(v8 - 24));
      *(__m128i *)(v13 - 56) = _mm_loadu_si128((const __m128i *)(v8 - 56));
      --v12;
    }
    while ( v12 );
    result = a5 + v11;
  }
  v16 = a4 - v7;
  v17 = 0x6DB6DB6DB6DB6DB7LL * (v16 >> 3);
  if ( v16 > 0 )
  {
    v18 = result;
    do
    {
      v19 = *(_QWORD *)(v7 + 48);
      v18 += 56;
      v7 += 56;
      *(_QWORD *)(v18 - 8) = v19;
      *(__m128i *)(v18 - 40) = _mm_loadu_si128((const __m128i *)(v7 - 40));
      *(__m128i *)(v18 - 24) = _mm_loadu_si128((const __m128i *)(v7 - 24));
      *(__m128i *)(v18 - 56) = _mm_loadu_si128((const __m128i *)(v7 - 56));
      --v17;
    }
    while ( v17 );
    result += v16;
  }
  return result;
}
