// Function: sub_88F1C0
// Address: 0x88f1c0
//
_WORD *__fastcall sub_88F1C0(__int64 *a1, const __m128i *a2)
{
  __int64 v2; // r12
  char v3; // al
  _WORD *result; // rax
  __m128i v5[4]; // [rsp+0h] [rbp-40h] BYREF

  v2 = *a1;
  sub_7ADF70((__int64)v5, 1);
  v3 = sub_877F80(*(_QWORD *)v2);
  sub_88F140((__int64)a1, (unsigned __int64)v5, v3 == 1, (_DWORD *)(v2 + 48));
  sub_5EA710(a1[30], *(_QWORD *)v2, a2, v5);
  result = (_WORD *)a1[23];
  *result = 74;
  return result;
}
