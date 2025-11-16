// Function: sub_AB2540
// Address: 0xab2540
//
__int64 __fastcall sub_AB2540(__int64 a1, __int64 a2)
{
  unsigned int v2; // ebx
  __int64 v3; // r15
  bool v4; // cc
  __int64 v6; // [rsp+10h] [rbp-E0h] BYREF
  unsigned int v7; // [rsp+18h] [rbp-D8h]
  __int64 v8; // [rsp+20h] [rbp-D0h] BYREF
  unsigned int v9; // [rsp+28h] [rbp-C8h]
  __int64 v10; // [rsp+30h] [rbp-C0h] BYREF
  unsigned int v11; // [rsp+38h] [rbp-B8h]
  __int64 v12; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v13; // [rsp+48h] [rbp-A8h]
  __int64 v14; // [rsp+50h] [rbp-A0h]
  unsigned int v15; // [rsp+58h] [rbp-98h]
  __int64 v16; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v17; // [rsp+68h] [rbp-88h]
  __int64 v18; // [rsp+70h] [rbp-80h]
  unsigned int v19; // [rsp+78h] [rbp-78h]
  __int64 v20; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v21; // [rsp+88h] [rbp-68h]
  __int64 v22; // [rsp+90h] [rbp-60h]
  int v23; // [rsp+98h] [rbp-58h]
  __int64 v24; // [rsp+A0h] [rbp-50h] BYREF
  unsigned int v25; // [rsp+A8h] [rbp-48h]
  __int64 v26; // [rsp+B0h] [rbp-40h]
  int v27; // [rsp+B8h] [rbp-38h]

  v7 = *(_DWORD *)(a2 + 8);
  v2 = v7;
  v3 = 1LL << ((unsigned __int8)v7 - 1);
  if ( v7 > 0x40 )
  {
    sub_C43690(&v6, 0, 0);
    v9 = v2;
    sub_C43690(&v8, 1, 0);
    v11 = v2;
    sub_C43690(&v10, 0, 0);
    if ( v11 <= 0x40 )
      v10 |= v3;
    else
      *(_QWORD *)(v10 + 8LL * ((v2 - 1) >> 6)) |= v3;
  }
  else
  {
    v6 = 0;
    v9 = v7;
    v8 = 1;
    v11 = v7;
    v10 = 1LL << ((unsigned __int8)v7 - 1);
    if ( v7 == 1 )
    {
      sub_AADB10((__int64)&v12, 1u, 0);
      goto LABEL_4;
    }
  }
  v25 = v11;
  if ( v11 > 0x40 )
  {
    sub_C43780(&v24, &v10);
    v21 = v9;
    if ( v9 <= 0x40 )
      goto LABEL_40;
  }
  else
  {
    v24 = v10;
    v21 = v9;
    if ( v9 <= 0x40 )
    {
LABEL_40:
      v20 = v8;
      goto LABEL_41;
    }
  }
  sub_C43780(&v20, &v8);
LABEL_41:
  sub_AADC30((__int64)&v12, (__int64)&v20, &v24);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
LABEL_4:
  v25 = v7;
  if ( v7 > 0x40 )
    sub_C43780(&v24, &v6);
  else
    v24 = v6;
  v21 = v11;
  if ( v11 > 0x40 )
    sub_C43780(&v20, &v10);
  else
    v20 = v10;
  sub_AADC30((__int64)&v16, (__int64)&v20, &v24);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  sub_AB2160((__int64)&v20, a2, (__int64)&v12, 0);
  sub_AB2160((__int64)&v24, a2, (__int64)&v16, 0);
  v4 = v19 <= 0x40;
  *(_DWORD *)(a1 + 8) = v21;
  *(_QWORD *)a1 = v20;
  *(_DWORD *)(a1 + 24) = v23;
  *(_QWORD *)(a1 + 16) = v22;
  *(_DWORD *)(a1 + 40) = v25;
  *(_QWORD *)(a1 + 32) = v24;
  *(_DWORD *)(a1 + 56) = v27;
  *(_QWORD *)(a1 + 48) = v26;
  if ( !v4 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  if ( v7 > 0x40 && v6 )
    j_j___libc_free_0_0(v6);
  return a1;
}
