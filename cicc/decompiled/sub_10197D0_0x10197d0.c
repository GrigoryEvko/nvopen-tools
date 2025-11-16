// Function: sub_10197D0
// Address: 0x10197d0
//
__int64 __fastcall sub_10197D0(unsigned __int64 a1, _BYTE *a2, _BYTE *a3, __m128i *a4)
{
  if ( (unsigned int)(a1 - 32) > 9 )
    return sub_1011B90(a1, a2, a3, 0, a4, 3u);
  else
    return sub_1012FB0(a1, a2, a3, a4->m128i_i64, 3u);
}
