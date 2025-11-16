// Function: sub_169DBD0
// Address: 0x169dbd0
//
__int64 __fastcall sub_169DBD0(_BYTE *a1, _BYTE *a2)
{
  __int64 result; // rax

  if ( (unsigned __int8)sub_169CC60(a1) || (unsigned __int8)sub_169CC60(a2) )
  {
LABEL_4:
    sub_16986F0(a1, 0, 0, 0);
    return 1;
  }
  else
  {
    switch ( (a2[18] & 7) + 4 * (a1[18] & 7) )
    {
      case 0:
      case 2:
      case 8:
        a1[18] &= 0xF8u;
        result = 0;
        break;
      case 1:
      case 9:
      case 0xD:
        a1[18] = a1[18] & 0xF0 | 1;
        sub_16985E0((__int64)a1, (__int64)a2);
        result = 0;
        break;
      case 3:
      case 0xC:
        goto LABEL_4;
      case 4:
      case 5:
      case 6:
      case 7:
        a1[18] &= ~8u;
        result = 0;
        break;
      case 0xA:
        result = 0;
        break;
      case 0xB:
      case 0xE:
      case 0xF:
        a1[18] = a1[18] & 0xF8 | 3;
        result = 0;
        break;
    }
  }
  return result;
}
