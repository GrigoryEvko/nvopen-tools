// Function: sub_11CA840
// Address: 0x11ca840
//
__int64 __fastcall sub_11CA840(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 *v9; // r13
  __int64 v10; // r14
  __int64 v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rax
  _QWORD v15[4]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v16[10]; // [rsp+30h] [rbp-50h] BYREF

  v9 = (__int64 *)sub_BCE3C0(*(__int64 **)(a4 + 72), 0);
  v10 = sub_BCD140(*(_QWORD **)(a4 + 72), *(_DWORD *)(*a6 + 172));
  v11 = sub_AA4B30(*(_QWORD *)(a4 + 48));
  v12 = sub_97FA80(*a6, v11);
  v13 = sub_BCD140(*(_QWORD **)(a4 + 72), v12);
  v16[1] = a2;
  v16[2] = a3;
  v15[0] = v9;
  v15[1] = v10;
  v16[0] = a1;
  v15[2] = v13;
  return sub_11C9AF0(0x169u, v9, v15, 3, (int)v16, 3, a4, a6, 0);
}
