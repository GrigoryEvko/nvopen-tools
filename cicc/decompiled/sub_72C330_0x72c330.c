// Function: sub_72C330
// Address: 0x72c330
//
_QWORD *__fastcall sub_72C330(char a1)
{
  _QWORD *result; // rax

  switch ( a1 )
  {
    case 0:
      result = sub_72BA30(byte_4F068B0[0]);
      break;
    case 1:
      result = sub_72C270();
      break;
    case 2:
      result = sub_72C2A0();
      break;
    case 3:
      result = sub_72C2D0();
      break;
    case 4:
      result = sub_72C300();
      break;
    default:
      sub_721090();
  }
  return result;
}
