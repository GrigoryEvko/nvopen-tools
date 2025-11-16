// Function: sub_621710
// Address: 0x621710
//
__int64 __fastcall sub_621710(__int16 *a1, _BOOL4 *a2)
{
  __int64 result; // rax
  __m128i v3[3]; // [rsp+0h] [rbp-30h] BYREF

  sub_620D80(v3, 0);
  result = sub_6215F0((unsigned __int16 *)v3, a1, 1, a2);
  *(__m128i *)a1 = _mm_loadu_si128(v3);
  return result;
}
