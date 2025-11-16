// Function: sub_1683AB0
// Address: 0x1683ab0
//
_QWORD *__fastcall sub_1683AB0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *result; // rax

  v2 = sub_1689050();
  result = (_QWORD *)sub_1685080(*(_QWORD *)(v2 + 24), 16);
  if ( result )
  {
    result[1] = a1;
    *result = a2;
  }
  else
  {
    sub_1683C30();
    MEMORY[8] = a1;
    MEMORY[0] = a2;
    return 0;
  }
  return result;
}
