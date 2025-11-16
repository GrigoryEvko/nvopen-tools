// Function: sub_C7C7C0
// Address: 0xc7c7c0
//
__int64 __fastcall sub_C7C7C0(__int64 a1, const __m128i *a2, char a3, char a4)
{
  __m128i v4; // xmm0
  __m128i v5; // xmm1
  __int64 result; // rax
  bool v7; // zf

  if ( a2->m128i_i64[1] )
  {
    v4 = _mm_loadu_si128(a2);
    v5 = _mm_loadu_si128(a2 + 1);
    *(_BYTE *)(a1 + 32) = 1;
    *(__m128i *)a1 = v4;
    *(__m128i *)(a1 + 16) = v5;
  }
  else
  {
    *(_BYTE *)(a1 + 32) = 0;
  }
  result = 0;
  v7 = a2->m128i_i64[1] == 0;
  *(_BYTE *)(a1 + 40) = a4;
  *(_BYTE *)(a1 + 41) = a3;
  *(_DWORD *)(a1 + 44) = 1;
  if ( !v7 )
    result = a2->m128i_i64[0];
  *(_QWORD *)(a1 + 48) = result;
  *(_QWORD *)(a1 + 56) = 0;
  if ( a2->m128i_i64[1] )
  {
    if ( a3 )
      return sub_C7C5C0(a1);
    result = *(unsigned __int8 *)a2->m128i_i64[0];
    if ( (_BYTE)result != 10 && ((_BYTE)result != 13 || *(_BYTE *)(a2->m128i_i64[0] + 1) != 10) )
      return sub_C7C5C0(a1);
  }
  return result;
}
