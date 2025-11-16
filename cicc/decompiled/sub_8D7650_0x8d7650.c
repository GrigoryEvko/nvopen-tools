// Function: sub_8D7650
// Address: 0x8d7650
//
__int64 __fastcall sub_8D7650(__m128i *a1)
{
  __int8 v1; // al
  unsigned int v2; // r8d
  unsigned int v4; // r8d

  if ( !a1 )
    return 0;
  v1 = a1->m128i_i8[0];
  if ( (a1->m128i_i8[0] & 0x22) != 0 )
    return 0;
  if ( (v1 & 0x40) == 0 )
  {
    if ( (v1 & 4) == 0 )
    {
      v2 = 1;
      if ( (v1 & 1) == 0 )
        return a1->m128i_i64[1] == 0;
      return v2;
    }
    return 0;
  }
  sub_8955E0(a1, 0);
  v4 = 0;
  if ( (a1->m128i_i8[0] & 4) == 0 )
  {
    v4 = 1;
    if ( (a1->m128i_i8[0] & 1) == 0 )
      return a1->m128i_i64[1] == 0;
  }
  return v4;
}
