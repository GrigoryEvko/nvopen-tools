// Function: sub_7700E0
// Address: 0x7700e0
//
__int64 __fastcall sub_7700E0(__int64 a1, __int64 a2, __int64 a3, _BYTE **a4, __m128i *a5)
{
  __int64 result; // rax

  result = 1;
  if ( **a4 == 37 )
    *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08280);
  else
    *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
  return result;
}
