// Function: sub_B2F930
// Address: 0xb2f930
//
_QWORD *__fastcall sub_B2F930(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  char v3; // r13
  __int64 v4; // r14
  unsigned __int64 v5; // rbx
  _BYTE *v6; // rax
  unsigned __int64 v7; // rdx

  v2 = *(_QWORD *)(a2 + 40);
  v3 = *(_BYTE *)(a2 + 32);
  v4 = *(_QWORD *)(v2 + 200);
  v5 = *(_QWORD *)(v2 + 208);
  v6 = (_BYTE *)sub_BD5D20(a2);
  sub_B2F7A0(a1, v6, v7, v3 & 0xF, v4, v5);
  return a1;
}
