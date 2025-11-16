// Function: sub_3215240
// Address: 0x3215240
//
__int64 __fastcall sub_3215240(__int64 *a1, unsigned int *a2, unsigned __int16 a3)
{
  __int64 result; // rax

  LOWORD(result) = sub_E0CC40(a3, *a2);
  if ( BYTE1(result) )
    return (unsigned __int8)result;
  if ( a3 <= 0x23u )
  {
    if ( a3 > 0xCu )
    {
      switch ( a3 )
      {
        case 0xDu:
          return sub_F03F10(*a1);
        case 0xFu:
        case 0x15u:
        case 0x1Au:
        case 0x1Bu:
        case 0x23u:
          return sub_F03EF0(*a1);
        default:
          break;
      }
    }
LABEL_9:
    BUG();
  }
  if ( (unsigned __int16)(a3 - 7937) > 1u )
    goto LABEL_9;
  return sub_F03EF0(*a1);
}
