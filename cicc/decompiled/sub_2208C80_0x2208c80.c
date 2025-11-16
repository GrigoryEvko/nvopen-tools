// Function: sub_2208C80
// Address: 0x2208c80
//
_QWORD *__fastcall sub_2208C80(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *result; // rax

  v2 = *(_QWORD *)(a2 + 8);
  *a1 = a2;
  a1[1] = v2;
  result = *(_QWORD **)(a2 + 8);
  *result = a1;
  *(_QWORD *)(a2 + 8) = a1;
  return result;
}
