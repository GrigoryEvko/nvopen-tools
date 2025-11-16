// Function: sub_727740
// Address: 0x727740
//
_QWORD *__fastcall sub_727740(char a1)
{
  _QWORD *result; // rax

  result = sub_7247C0(48);
  *((_BYTE *)result + 40) &= 0xF8u;
  *(_BYTE *)result = a1;
  result[1] = 0;
  result[2] = 0;
  result[3] = 0;
  result[4] = 0;
  return result;
}
