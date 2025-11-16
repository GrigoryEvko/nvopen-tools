// Function: sub_1688450
// Address: 0x1688450
//
_BYTE *__fastcall sub_1688450(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // rax
  int v6; // esi
  __int64 v7; // rdi
  int v8; // edx
  int v9; // ecx
  int v10; // r8d
  int v11; // r9d
  _QWORD *v12; // r12
  _QWORD *v13; // rdi
  char v15; // [rsp+0h] [rbp-20h]
  _BYTE *v16; // [rsp+8h] [rbp-18h] BYREF

  v4 = *(_QWORD *)(a1 + 8) + 1LL;
  v5 = sub_1689050(a1, a2, a3);
  v6 = v4;
  v7 = *(_QWORD *)(v5 + 24);
  v12 = sub_1685080(v7, v4);
  if ( !v12 )
    sub_1683C30(v7, v6, v8, v9, v10, v11, v15);
  v13 = *(_QWORD **)(a1 + 16);
  v16 = v12;
  sub_1683BE0(v13, (__int64 (__fastcall *)(_QWORD, __int64))sub_1688260, (__int64)&v16);
  *v16 = 0;
  return &v16[-*(_QWORD *)(a1 + 8)];
}
