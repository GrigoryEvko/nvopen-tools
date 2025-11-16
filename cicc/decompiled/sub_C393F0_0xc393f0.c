// Function: sub_C393F0
// Address: 0xc393f0
//
__int64 __fastcall sub_C393F0(_BYTE *a1, _BYTE *a2)
{
  unsigned __int8 v2; // cl
  char v3; // dl
  __int64 result; // rax
  char v5; // al
  bool v6; // dl

  v2 = a2[20];
  v3 = a1[20];
  switch ( (v2 & 7) + 4 * (v3 & 7) )
  {
    case 0:
    case 0xF:
      sub_C36070((__int64)a1, 0, 0, 0);
      return 1;
    case 1:
    case 9:
    case 0xD:
      sub_C33E20((__int64)a1, (__int64)a2);
      v6 = 0;
      v5 = a1[20] & 0xF7;
      a1[20] = v5;
      v2 = a2[20];
      goto LABEL_4;
    case 2:
    case 3:
    case 0xA:
    case 0xC:
    case 0xE:
      return 0;
    case 4:
    case 5:
    case 6:
    case 7:
      v5 = a1[20];
      v6 = (v3 & 8) != 0;
LABEL_4:
      a1[20] = v5 & 0xF7 | (8 * (((v2 >> 3) ^ v6) & 1));
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
      a1[20] = a1[20] & 0xF8 | 3;
      result = 0;
      break;
    case 0xB:
      if ( *(_DWORD *)(*(_QWORD *)a1 + 16LL) == 1 )
        sub_C36070((__int64)a1, 0, (v3 & 8) != 0, 0);
      else
        a1[20] &= 0xF8u;
      result = 2;
      break;
    default:
      BUG();
  }
  return result;
}
