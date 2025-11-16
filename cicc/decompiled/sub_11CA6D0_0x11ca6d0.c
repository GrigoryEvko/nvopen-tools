// Function: sub_11CA6D0
// Address: 0x11ca6d0
//
__int64 __fastcall sub_11CA6D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 *v9; // r12
  __int64 v10; // rax
  unsigned int v11; // eax
  __int64 v12; // rax
  _QWORD v14[4]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v15[10]; // [rsp+30h] [rbp-50h] BYREF

  v9 = (__int64 *)sub_BCE3C0(*(__int64 **)(a4 + 72), 0);
  v10 = sub_AA4B30(*(_QWORD *)(a4 + 48));
  v11 = sub_97FA80(*a6, v10);
  v12 = sub_BCD140(*(_QWORD **)(a4 + 72), v11);
  v15[0] = a1;
  v15[1] = a2;
  v15[2] = a3;
  v14[0] = v9;
  v14[1] = v9;
  v14[2] = v12;
  return sub_11C9AF0(0x168u, v9, v14, 3, (int)v15, 3, a4, a6, 0);
}
