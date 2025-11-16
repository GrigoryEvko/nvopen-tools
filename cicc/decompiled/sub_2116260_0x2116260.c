// Function: sub_2116260
// Address: 0x2116260
//
__int64 __fastcall sub_2116260(_QWORD *a1, _QWORD **a2)
{
  __int64 *v3; // rbx
  __int64 *v4; // r13
  __int64 *v5; // rax
  __int64 v6; // rdx
  _QWORD *v7; // rdi
  _QWORD v9[12]; // [rsp+10h] [rbp-60h] BYREF

  v3 = (__int64 *)sub_16471D0(*a2, 0);
  v4 = (__int64 *)sub_1643350(*a2);
  a1[20] = sub_1645D80(v4, 4);
  v5 = sub_1645D80(v3, 5);
  v6 = a1[20];
  a1[21] = v5;
  v7 = (_QWORD *)*v3;
  v9[2] = v6;
  v9[5] = v5;
  v9[0] = v3;
  v9[3] = v3;
  v9[4] = v3;
  v9[1] = v4;
  a1[22] = sub_1645600(v7, v9, 6, 0);
  return 1;
}
