// Function: sub_325D570
// Address: 0x325d570
//
_QWORD *__fastcall sub_325D570(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // rsi

  result = *(_QWORD **)(a1 + 144);
  v3 = (_QWORD *)(a2 & 0xFFFFFFFFFFFFFFC0LL);
  *v3 = *result;
  *result = v3;
  return result;
}
