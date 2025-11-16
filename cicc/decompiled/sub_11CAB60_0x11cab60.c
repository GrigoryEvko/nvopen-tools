// Function: sub_11CAB60
// Address: 0x11cab60
//
__int64 __fastcall sub_11CAB60(__int64 a1, __int64 a2, __int64 a3, char *a4, __int64 a5, __int64 a6, __int64 *a7)
{
  __int64 *v9; // r13
  __int64 v10; // rax
  unsigned int v11; // eax
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v17; // [rsp+18h] [rbp-B8h]
  __int64 v18; // [rsp+20h] [rbp-B0h]
  _QWORD v19[4]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD *v20; // [rsp+50h] [rbp-80h] BYREF
  __int64 v21; // [rsp+58h] [rbp-78h]
  _QWORD v22[3]; // [rsp+60h] [rbp-70h] BYREF
  char v23[88]; // [rsp+78h] [rbp-58h] BYREF

  v17 = sub_BCE3C0(*(__int64 **)(a6 + 72), 0);
  v9 = (__int64 *)sub_BCD140(*(_QWORD **)(a6 + 72), *(_DWORD *)(*a7 + 172));
  v10 = sub_AA4B30(*(_QWORD *)(a6 + 48));
  v11 = sub_97FA80(*a7, v10);
  v12 = sub_BCD140(*(_QWORD **)(a6 + 72), v11);
  v20 = v22;
  v18 = v12;
  v22[2] = a3;
  v22[0] = a1;
  v21 = 0x800000003LL;
  v22[1] = a2;
  sub_11C5120((__int64)&v20, v23, a4, &a4[8 * a5]);
  v19[0] = v17;
  v19[2] = v17;
  v19[1] = v18;
  v13 = sub_11C9AF0(0x1BEu, v9, v19, 3, (int)v20, v21, a6, a7, 1u);
  if ( v20 != v22 )
    _libc_free(v20, v9);
  return v13;
}
