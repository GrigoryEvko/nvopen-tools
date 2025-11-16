// Function: sub_11CAA80
// Address: 0x11caa80
//
__int64 __fastcall sub_11CAA80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 *v9; // r12
  __int64 v10; // r14
  __int64 v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rax
  _QWORD v16[4]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v17[10]; // [rsp+40h] [rbp-50h] BYREF

  v9 = (__int64 *)sub_BCE3C0(*(__int64 **)(a5 + 72), 0);
  v10 = sub_BCD140(*(_QWORD **)(a5 + 72), *(_DWORD *)(*a6 + 172));
  v11 = sub_AA4B30(*(_QWORD *)(a5 + 48));
  v12 = sub_97FA80(*a6, v11);
  v13 = sub_BCD140(*(_QWORD **)(a5 + 72), v12);
  v17[2] = a3;
  v17[3] = a4;
  v16[0] = v9;
  v16[1] = v9;
  v16[2] = v10;
  v17[0] = a1;
  v17[1] = a2;
  v16[3] = v13;
  return sub_11C9AF0(0x163u, v9, v16, 4, (int)v17, 4, a5, a6, 0);
}
