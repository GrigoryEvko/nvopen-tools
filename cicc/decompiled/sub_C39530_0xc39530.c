// Function: sub_C39530
// Address: 0xc39530
//
__int64 __fastcall sub_C39530(_BYTE *a1, _BYTE *a2)
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
      goto LABEL_3;
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
    case 0xA:
    case 0xC:
    case 0xE:
      result = 0;
      break;
    default:
      BUG();
  }
  return result;
}
