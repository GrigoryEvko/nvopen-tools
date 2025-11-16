// Function: sub_2F4C150
// Address: 0x2f4c150
//
_QWORD *__fastcall sub_2F4C150(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // rsi

  result = *(_QWORD **)(a1 + 192);
  v3 = (_QWORD *)(a2 & 0xFFFFFFFFFFFFFFC0LL);
  *v3 = *result;
  *result = v3;
  return result;
}
