// Function: sub_11CAE30
// Address: 0x11cae30
//
__int64 __fastcall sub_11CAE30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned int v10; // eax
  __int64 *v11; // rax
  _QWORD v13[4]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v14[10]; // [rsp+30h] [rbp-50h] BYREF

  v8 = sub_BCE3C0(*(__int64 **)(a4 + 72), 0);
  v9 = sub_AA4B30(*(_QWORD *)(a4 + 48));
  v10 = sub_97FA80(*a5, v9);
  v11 = (__int64 *)sub_BCD140(*(_QWORD **)(a4 + 72), v10);
  v14[0] = a1;
  v14[1] = a2;
  v14[2] = a3;
  v13[0] = v8;
  v13[1] = v8;
  v13[2] = v11;
  return sub_11C9AF0(0x1D3u, v11, v13, 3, (int)v14, 3, a4, a5, 0);
}
