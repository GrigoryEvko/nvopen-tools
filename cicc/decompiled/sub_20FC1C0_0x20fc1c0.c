// Function: sub_20FC1C0
// Address: 0x20fc1c0
//
_QWORD *__fastcall sub_20FC1C0(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // rsi

  result = *(_QWORD **)(a1 + 200);
  v3 = (_QWORD *)(a2 & 0xFFFFFFFFFFFFFFC0LL);
  *v3 = *result;
  *result = v3;
  return result;
}
