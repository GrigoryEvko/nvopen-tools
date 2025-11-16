// Function: sub_9959F0
// Address: 0x9959f0
//
__int64 __fastcall sub_9959F0(
        unsigned int a1,
        unsigned __int8 *a2,
        unsigned __int8 *a3,
        unsigned __int8 *a4,
        unsigned __int8 *a5)
{
  if ( a1 > 0x27 )
  {
    if ( a1 - 40 > 1
      || !(unsigned __int8)sub_995040(41, (__int64)a4, a2)
      || !(unsigned __int8)sub_995040(41, (__int64)a3, a5) )
    {
      return 0;
    }
  }
  else if ( a1 > 0x25 )
  {
    if ( !(unsigned __int8)sub_995040(41, (__int64)a2, a4) || !(unsigned __int8)sub_995040(41, (__int64)a5, a3) )
      return 0;
  }
  else if ( a1 <= 0x23 )
  {
    if ( a1 <= 0x21
      || !(unsigned __int8)sub_995040(37, (__int64)a2, a4)
      || !(unsigned __int8)sub_995040(37, (__int64)a5, a3) )
    {
      return 0;
    }
  }
  else if ( !(unsigned __int8)sub_995040(37, (__int64)a4, a2) || !(unsigned __int8)sub_995040(37, (__int64)a3, a5) )
  {
    return 0;
  }
  return 257;
}
