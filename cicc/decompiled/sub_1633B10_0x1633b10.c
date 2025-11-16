// Function: sub_1633B10
// Address: 0x1633b10
//
_QWORD *__fastcall sub_1633B10(__int64 a1)
{
  _QWORD *result; // rax
  __int64 v2; // rdx

  result = (_QWORD *)sub_16328F0(a1, "Dwarf Version", 0xDu);
  if ( result )
  {
    v2 = result[17];
    result = *(_QWORD **)(v2 + 24);
    if ( *(_DWORD *)(v2 + 32) > 0x40u )
      return (_QWORD *)*result;
  }
  return result;
}
