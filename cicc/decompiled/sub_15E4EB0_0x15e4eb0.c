// Function: sub_15E4EB0
// Address: 0x15e4eb0
//
__int64 *__fastcall sub_15E4EB0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  char v3; // r13
  _BYTE *v4; // r14
  __int64 v5; // rbx
  _BYTE *v6; // rax
  __int64 v7; // rdx

  v2 = *(_QWORD *)(a2 + 40);
  v3 = *(_BYTE *)(a2 + 32);
  v4 = *(_BYTE **)(v2 + 208);
  v5 = *(_QWORD *)(v2 + 216);
  v6 = (_BYTE *)sub_1649960(a2);
  sub_15E4CF0(a1, v6, v7, v3 & 0xF, v4, v5);
  return a1;
}
