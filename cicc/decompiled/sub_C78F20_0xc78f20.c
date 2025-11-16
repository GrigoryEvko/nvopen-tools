// Function: sub_C78F20
// Address: 0xc78f20
//
__int64 __fastcall sub_C78F20(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  _DWORD *v4; // rsi
  unsigned int v5; // r13d
  bool v6; // cc
  __int64 v9; // [rsp+10h] [rbp-B0h] BYREF
  int v10; // [rsp+18h] [rbp-A8h]
  __int64 v11; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v12; // [rsp+28h] [rbp-98h]
  __int64 v13; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v14; // [rsp+38h] [rbp-88h]
  __int64 v15; // [rsp+40h] [rbp-80h]
  unsigned int v16; // [rsp+48h] [rbp-78h]
  __int64 v17; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v18; // [rsp+58h] [rbp-68h]
  __int64 v19; // [rsp+60h] [rbp-60h]
  unsigned int v20; // [rsp+68h] [rbp-58h]
  __int64 v21; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v22; // [rsp+78h] [rbp-48h]
  __int64 v23; // [rsp+80h] [rbp-40h] BYREF
  unsigned int v24; // [rsp+88h] [rbp-38h]

  v4 = a2 + 4;
  v5 = *(v4 - 2);
  sub_C44830((__int64)&v21, v4, 2 * v5);
  sub_C44830((__int64)&v17, a2, 2 * v5);
  v14 = v18;
  v13 = v17;
  v16 = v22;
  v15 = v21;
  sub_C44830((__int64)&v21, a3 + 4, 2 * v5);
  sub_C44830((__int64)&v11, a3, 2 * v5);
  v18 = v12;
  v17 = v11;
  v20 = v22;
  v19 = v21;
  sub_C787D0((__int64)&v21, (__int64)&v13, (__int64)&v17, 0);
  sub_C440A0((__int64)&v11, &v23, v5, v5);
  sub_C440A0((__int64)&v9, &v21, v5, v5);
  v6 = v24 <= 0x40;
  *(_DWORD *)(a1 + 8) = v10;
  *(_QWORD *)a1 = v9;
  *(_DWORD *)(a1 + 24) = v12;
  *(_QWORD *)(a1 + 16) = v11;
  if ( !v6 && v23 )
    j_j___libc_free_0_0(v23);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  return a1;
}
