// Function: sub_1F5DB50
// Address: 0x1f5db50
//
__int64 __fastcall sub_1F5DB50(__int64 a1, _QWORD **a2)
{
  __int64 v3; // r12
  __int64 v4; // r14
  _QWORD **v5; // rax
  _QWORD *v6; // rdi
  _QWORD *v8; // [rsp+18h] [rbp-A8h]
  _QWORD v9[12]; // [rsp+60h] [rbp-60h] BYREF

  v8 = *a2;
  v3 = sub_1643350(*a2);
  v4 = sub_16471D0(v8, 0);
  v5 = (_QWORD **)sub_1643350(v8);
  v6 = *v5;
  v9[0] = v5;
  v9[2] = v3;
  v9[1] = v4;
  *(_QWORD *)(a1 + 160) = sub_1645600(v6, v9, 3, 0);
  return 0;
}
