// Function: sub_11CA050
// Address: 0x11ca050
//
__int64 __fastcall sub_11CA050(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rax
  unsigned int v8; // eax
  __int64 *v9; // rax
  __int64 v11; // [rsp+8h] [rbp-28h] BYREF
  __int64 v12[3]; // [rsp+18h] [rbp-18h] BYREF

  v11 = a1;
  v5 = sub_BCE3C0(*(__int64 **)(a2 + 72), 0);
  v6 = *(_QWORD *)(a2 + 48);
  v12[0] = v5;
  v7 = sub_AA4B30(v6);
  v8 = sub_97FA80(*a4, v7);
  v9 = (__int64 *)sub_BCD140(*(_QWORD **)(a2 + 72), v8);
  return sub_11C9AF0(0x1D4u, v9, v12, 1, (int)&v11, 1, a2, a4, 0);
}
