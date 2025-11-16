// Function: sub_16AF570
// Address: 0x16af570
//
_QWORD *__fastcall sub_16AF570(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax
  bool v3; // cf
  __int64 v4; // rsi

  result = a1;
  v3 = __CFADD__(*a1, a2);
  v4 = *a1 + a2;
  if ( v3 )
    *a1 = -1;
  else
    *a1 = v4;
  return result;
}
