// Function: sub_621340
// Address: 0x621340
//
__int64 __fastcall sub_621340(unsigned __int16 *a1, int a2, __int16 *a3, int a4, _BOOL4 *a5)
{
  const __m128i *v5; // rax
  __int64 result; // rax
  __m128i v9[3]; // [rsp+0h] [rbp-30h] BYREF

  v5 = (const __m128i *)a3;
  if ( a2 )
    v5 = (const __m128i *)a1;
  v9[0] = _mm_loadu_si128(v5);
  result = sub_621270(a1, a3, a2, a5);
  if ( a2 != a4 )
  {
    result = (int)sub_621000(v9[0].m128i_i16, 1, (__int16 *)a1, a2) > 0;
    *a5 = result;
  }
  return result;
}
