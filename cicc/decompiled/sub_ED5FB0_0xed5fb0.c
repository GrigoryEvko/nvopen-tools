// Function: sub_ED5FB0
// Address: 0xed5fb0
//
_QWORD *__fastcall sub_ED5FB0(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // rsi

  result = *(_QWORD **)(a1 + 104);
  v3 = (_QWORD *)(a2 & 0xFFFFFFFFFFFFFFC0LL);
  *v3 = *result;
  *result = v3;
  return result;
}
