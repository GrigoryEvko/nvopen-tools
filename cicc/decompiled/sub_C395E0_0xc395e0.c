// Function: sub_C395E0
// Address: 0xc395e0
//
__int64 __fastcall sub_C395E0(_BYTE *a1, _BYTE *a2)
{
  __int64 result; // rax

  switch ( (a2[20] & 7) + 4 * (a1[20] & 7) )
  {
    case 0:
    case 2:
    case 3:
    case 0xB:
    case 0xF:
      sub_C36070((__int64)a1, 0, 0, 0);
      return 1;
    case 1:
    case 9:
    case 0xD:
      sub_C33E20((__int64)a1, (__int64)a2);
      goto LABEL_5;
    case 4:
    case 5:
    case 6:
    case 7:
LABEL_5:
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
    case 0xE:
      result = 0;
      break;
    case 0xA:
      result = 2;
      break;
    default:
      BUG();
  }
  return result;
}
