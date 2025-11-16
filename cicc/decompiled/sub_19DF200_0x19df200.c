// Function: sub_19DF200
// Address: 0x19df200
//
__int64 *__fastcall sub_19DF200(__int64 *a1, __int64 a2, __m128i a3, __m128i a4, double a5)
{
  if ( *(_BYTE *)(a2 + 16) == 56 )
    return (__int64 *)sub_19DEFC0(a1, a2, a3, a4, a5);
  else
    return sub_19DDC30((__int64)a1, a2, a3, a4);
}
