// Function: sub_C391B0
// Address: 0xc391b0
//
__int64 __fastcall sub_C391B0(_BYTE *a1, _BYTE *a2, char a3)
{
  __int64 result; // rax
  char v5; // al
  char v6; // bl

  switch ( (a2[20] & 7) + 4 * (a1[20] & 7) )
  {
    case 0:
      result = 0;
      if ( (((a2[20] ^ a1[20]) & 8) != 0) != a3 )
      {
        sub_C36070((__int64)a1, 0, 0, 0);
        return 1;
      }
      return result;
    case 1:
    case 9:
    case 0xD:
      sub_C33E20((__int64)a1, (__int64)a2);
      goto LABEL_3;
    case 2:
    case 3:
    case 0xB:
    case 0xF:
      return 0;
    case 4:
    case 5:
    case 6:
    case 7:
LABEL_3:
      if ( (unsigned __int8)sub_C35FD0(a1) )
      {
        sub_C39170((__int64)a1);
        result = 1;
      }
      else
      {
        result = (unsigned __int8)sub_C35FD0(a2);
      }
      break;
    case 8:
    case 0xC:
      v5 = a1[20] & 0xF0;
      a1[20] &= 0xF8u;
      v6 = v5 | (8 * (((a2[20] >> 3) ^ a3) & 1));
      a1[20] = v6;
      result = 0;
      break;
    case 0xA:
      result = 2;
      break;
    case 0xE:
      sub_C33E20((__int64)a1, (__int64)a2);
      a1[20] = a1[20] & 0xF7 | (8 * (((a2[20] >> 3) ^ a3) & 1));
      result = 0;
      break;
    default:
      BUG();
  }
  return result;
}
