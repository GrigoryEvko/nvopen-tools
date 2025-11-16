// Function: sub_11CACB0
// Address: 0x11cacb0
//
__int64 __fastcall sub_11CACB0(__int64 a1, __int64 a2, char *a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 *v14; // [rsp+18h] [rbp-98h]
  _QWORD v15[2]; // [rsp+20h] [rbp-90h] BYREF
  _QWORD *v16; // [rsp+30h] [rbp-80h] BYREF
  __int64 v17; // [rsp+38h] [rbp-78h]
  _QWORD v18[2]; // [rsp+40h] [rbp-70h] BYREF
  char v19[96]; // [rsp+50h] [rbp-60h] BYREF

  v9 = sub_BCE3C0(*(__int64 **)(a5 + 72), 0);
  v10 = sub_BCD140(*(_QWORD **)(a5 + 72), *(_DWORD *)(*a6 + 172));
  v16 = v18;
  v14 = (__int64 *)v10;
  v18[0] = a1;
  v18[1] = a2;
  v17 = 0x800000002LL;
  sub_11C5120((__int64)&v16, v19, a3, &a3[8 * a4]);
  v15[0] = v9;
  v15[1] = v9;
  v11 = sub_11C9AF0(0x1BFu, v14, v15, 2, (int)v16, v17, a5, a6, 1u);
  if ( v16 != v18 )
    _libc_free(v16, v14);
  return v11;
}
