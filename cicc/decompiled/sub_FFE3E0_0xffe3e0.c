// Function: sub_FFE3E0
// Address: 0xffe3e0
//
__int64 __fastcall sub_FFE3E0(unsigned int a1, _BYTE **a2, _BYTE **a3, __int64 *a4)
{
  _BYTE *v4; // r10
  _BYTE *v5; // r11
  __int64 v6; // r8
  __int64 result; // rax

  v4 = *a2;
  if ( **a2 > 0x15u )
    return 0;
  v5 = *a3;
  if ( **a3 > 0x15u )
  {
    if ( ((1LL << a1) & 0x70066000) != 0 )
    {
      *a2 = v5;
      *a3 = v4;
      return 0;
    }
    return 0;
  }
  switch ( a1 )
  {
    case 0xEu:
    case 0x10u:
    case 0x12u:
    case 0x15u:
    case 0x18u:
      v6 = a4[5];
      if ( !v6 )
        goto LABEL_6;
      result = sub_96F2E0(a1, (__int64)v4, v5, *a4, v6, 1);
      break;
    default:
LABEL_6:
      result = sub_96E6C0(a1, (__int64)v4, v5, *a4);
      break;
  }
  return result;
}
