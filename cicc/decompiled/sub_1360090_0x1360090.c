// Function: sub_1360090
// Address: 0x1360090
//
__int64 __fastcall sub_1360090(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v3; // r12

  v2 = 4;
  v3 = a2 + 112;
  if ( (unsigned __int8)sub_1560180(a2 + 112, 36) )
    return v2;
  if ( (unsigned __int8)sub_1560180(v3, 36) || (unsigned __int8)sub_1560180(v3, 37) )
  {
    v2 = 61;
  }
  else if ( (unsigned __int8)sub_1560180(v3, 36) || (v2 = 63, (unsigned __int8)sub_1560180(v3, 57)) )
  {
    v2 = 62;
  }
  if ( (unsigned __int8)sub_1560180(v3, 4) )
    return v2 & 0xF;
  if ( !(unsigned __int8)sub_1560180(v3, 13) )
  {
    if ( (unsigned __int8)sub_1560180(v3, 14) )
      return v2 & 0x1F;
    return v2;
  }
  return v2 & 0x17;
}
