// Function: sub_1C13840
// Address: 0x1c13840
//
__int64 __fastcall sub_1C13840(__int64 a1, __int64 a2, __int64 a3, __int64 a4, const __m128i *a5)
{
  __int64 result; // rax

  if ( a1 )
  {
    *(_QWORD *)a1 = a2;
    *(_QWORD *)(a1 + 8) = a3;
    *(_QWORD *)(a1 + 16) = a4;
    *(_OWORD *)(a1 + 24) = 0;
    *(_OWORD *)(a1 + 40) = 0;
    *(_OWORD *)(a1 + 56) = 0;
    if ( a5 )
    {
      *(__m128i *)(a1 + 32) = _mm_loadu_si128(a5);
      *(__m128i *)(a1 + 48) = _mm_loadu_si128(a5 + 1);
      result = a5[2].m128i_i64[0];
      *(_QWORD *)(a1 + 64) = result;
    }
  }
  return result;
}
