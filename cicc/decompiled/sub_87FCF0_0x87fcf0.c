// Function: sub_87FCF0
// Address: 0x87fcf0
//
_QWORD *sub_87FCF0()
{
  __int64 v0; // rsi
  _QWORD *result; // rax

  sub_87FB30("std", 0, &qword_4D049B8);
  v0 = qword_4D049B8[11];
  result = (_QWORD *)dword_4D041A8;
  *(_BYTE *)(v0 + 124) |= 0x10u;
  if ( (_DWORD)result )
    return sub_87FB30("meta", v0, (__int64 **)&qword_4D049B0);
  return result;
}
