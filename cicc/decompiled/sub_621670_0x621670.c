// Function: sub_621670
// Address: 0x621670
//
__int64 __fastcall sub_621670(__m128i *a1, int a2, __int16 *a3, int a4, _BOOL4 *a5)
{
  __int64 result; // rax
  _OWORD v9[4]; // [rsp+0h] [rbp-40h] BYREF

  v9[0] = _mm_loadu_si128(a1);
  result = sub_6215F0((unsigned __int16 *)a1, a3, a2, a5);
  if ( a2 != a4 )
  {
    if ( a4 && *a3 < 0 )
    {
      result = (int)sub_621000((__int16 *)v9, a2, a1->m128i_i16, a2) > 0;
      *a5 = result;
    }
    else
    {
      result = (int)sub_621000((__int16 *)v9, a2, a1->m128i_i16, a2) <= 0;
      *a5 = result;
    }
  }
  return result;
}
