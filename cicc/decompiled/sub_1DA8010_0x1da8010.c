// Function: sub_1DA8010
// Address: 0x1da8010
//
_QWORD *__fastcall sub_1DA8010(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // rsi

  result = *(_QWORD **)(a1 + 88);
  v3 = (_QWORD *)(a2 & 0xFFFFFFFFFFFFFFC0LL);
  *v3 = *result;
  *result = v3;
  return result;
}
