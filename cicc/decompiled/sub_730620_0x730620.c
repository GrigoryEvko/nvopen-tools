// Function: sub_730620
// Address: 0x730620
//
__int64 __fastcall sub_730620(__int64 a1, const __m128i *a2)
{
  __int64 v2; // rax
  char v3; // cl
  int v4; // edx
  __int64 v5; // rsi
  __int64 result; // rax

  if ( (const __m128i *)a1 != a2 )
  {
    v2 = *(_QWORD *)(a1 + 16);
    v3 = *(_BYTE *)(a1 + 25);
    v4 = *(_DWORD *)(a1 + 24);
    *(__m128i *)a1 = _mm_loadu_si128(a2);
    *(__m128i *)(a1 + 16) = _mm_loadu_si128(a2 + 1);
    *(__m128i *)(a1 + 32) = _mm_loadu_si128(a2 + 2);
    *(__m128i *)(a1 + 48) = _mm_loadu_si128(a2 + 3);
    *(__m128i *)(a1 + 64) = _mm_loadu_si128(a2 + 4);
    v5 = a2[5].m128i_i64[0];
    *(_QWORD *)(a1 + 16) = v2;
    LODWORD(v2) = *(_DWORD *)(a1 + 24);
    *(_QWORD *)(a1 + 80) = v5;
    result = v4 & 0x100400 | (unsigned int)v2 & 0xFFEFFBFF;
    *(_DWORD *)(a1 + 24) = result;
    if ( (v3 & 4) != 0 )
      return sub_7304E0(a1);
  }
  return result;
}
