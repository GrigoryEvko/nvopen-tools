// Function: sub_726340
// Address: 0x726340
//
_BYTE *__fastcall sub_726340(char a1)
{
  _BYTE *result; // rax

  result = sub_7246D0(24);
  *result = a1;
  *((_QWORD *)result + 1) = 0;
  *((_QWORD *)result + 2) = 0;
  return result;
}
