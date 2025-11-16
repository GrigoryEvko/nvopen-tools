// Function: sub_C392E0
// Address: 0xc392e0
//
__int64 __fastcall sub_C392E0(_BYTE *a1, _BYTE *a2)
{
  unsigned __int8 v2; // cl
  char v3; // dl
  bool v4; // al
  __int64 result; // rax

  v2 = a2[20];
  switch ( (v2 & 7) + 4 * (a1[20] & 7) )
  {
    case 0:
    case 2:
    case 8:
      a1[20] &= 0xF8u;
      return 0;
    case 1:
    case 9:
    case 0xD:
      sub_C33E20((__int64)a1, (__int64)a2);
      v4 = 0;
      v3 = a1[20] & 0xF7;
      a1[20] = v3;
      v2 = a2[20];
      goto LABEL_3;
    case 3:
    case 0xC:
      sub_C36070((__int64)a1, 0, 0, 0);
      return 1;
    case 4:
    case 5:
    case 6:
    case 7:
      v3 = a1[20];
      v4 = (v3 & 8) != 0;
LABEL_3:
      a1[20] = v3 & 0xF7 | (8 * (((v2 >> 3) ^ v4) & 1));
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
    case 0xA:
      result = 0;
      break;
    case 0xB:
    case 0xE:
    case 0xF:
      a1[20] = a1[20] & 0xF8 | 3;
      result = 0;
      break;
    default:
      BUG();
  }
  return result;
}
