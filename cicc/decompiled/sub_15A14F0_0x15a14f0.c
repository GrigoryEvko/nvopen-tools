// Function: sub_15A14F0
// Address: 0x15a14f0
//
__int64 __fastcall sub_15A14F0(unsigned int a1, __int64 **a2, __int64 a3)
{
  __int64 v3; // rcx
  __int64 result; // rax

  v3 = a1;
  if ( a1 > 0x1C || ((1LL << a1) & 0x1C019800) == 0 )
  {
    if ( (_BYTE)a3 )
    {
      if ( a1 == 19 )
        return sub_15A10B0((__int64)a2, 1.0);
      if ( a1 > 0x13 )
      {
        v3 = a1 - 23;
        if ( (unsigned int)v3 <= 2 )
          return sub_15A06D0(a2, (__int64)a2, a3, v3);
      }
      else
      {
        if ( a1 > 0xE )
        {
          if ( a1 - 17 <= 1 )
            return sub_15A0680((__int64)a2, 1, 0);
          return 0;
        }
        if ( a1 > 0xC )
          return sub_15A06D0(a2, (__int64)a2, a3, v3);
      }
    }
    return 0;
  }
  switch ( a1 )
  {
    case 0xBu:
    case 0xDu:
    case 0xEu:
    case 0x11u:
    case 0x12u:
    case 0x13u:
    case 0x14u:
    case 0x15u:
    case 0x16u:
    case 0x17u:
    case 0x18u:
    case 0x19u:
    case 0x1Bu:
    case 0x1Cu:
      return sub_15A06D0(a2, (__int64)a2, a3, v3);
    case 0xCu:
      result = sub_15A1390((__int64)a2, (__int64)a2, a3, a1);
      break;
    case 0xFu:
      return sub_15A0680((__int64)a2, 1, 0);
    case 0x10u:
      return sub_15A10B0((__int64)a2, 1.0);
    case 0x1Au:
      result = sub_15A04A0(a2);
      break;
  }
  return result;
}
