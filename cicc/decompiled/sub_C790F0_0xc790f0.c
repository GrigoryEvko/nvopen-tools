// Function: sub_C790F0
// Address: 0xc790f0
//
__int64 __fastcall sub_C790F0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  unsigned int v4; // r15d
  __int64 v5; // rax
  bool v6; // cc
  unsigned int v8; // [rsp+4h] [rbp-CCh]
  unsigned __int64 v10; // [rsp+20h] [rbp-B0h] BYREF
  unsigned int v11; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v12; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v13; // [rsp+38h] [rbp-98h]
  unsigned __int64 v14; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v15; // [rsp+48h] [rbp-88h]
  __int64 v16; // [rsp+50h] [rbp-80h]
  unsigned int v17; // [rsp+58h] [rbp-78h]
  unsigned __int64 v18; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v19; // [rsp+68h] [rbp-68h]
  __int64 v20; // [rsp+70h] [rbp-60h]
  unsigned int v21; // [rsp+78h] [rbp-58h]
  __int64 v22; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v23; // [rsp+88h] [rbp-48h]
  __int64 v24; // [rsp+90h] [rbp-40h] BYREF
  unsigned int v25; // [rsp+98h] [rbp-38h]

  v3 = *(_DWORD *)(a2 + 8);
  v4 = 2 * v3;
  sub_C449B0((__int64)&v12, (const void **)a2, 2 * v3);
  v5 = a2;
  if ( v3 != v13 )
  {
    if ( v3 > 0x3F || v13 > 0x40 )
    {
      sub_C43C90(&v12, v3, v13);
      v5 = a2;
    }
    else
    {
      v12 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v3 - (unsigned __int8)v13 + 64) << v3;
    }
  }
  sub_C449B0((__int64)&v22, (const void **)(v5 + 16), v4);
  v19 = v13;
  if ( v13 > 0x40 )
  {
    sub_C43780((__int64)&v18, (const void **)&v12);
    v15 = v19;
    v14 = v18;
    v17 = v23;
    v16 = v22;
    if ( v13 > 0x40 && v12 )
      j_j___libc_free_0_0(v12);
  }
  else
  {
    v15 = v13;
    v14 = v12;
    v17 = v23;
    v16 = v22;
  }
  v8 = *(_DWORD *)(a3 + 8);
  sub_C449B0((__int64)&v10, (const void **)a3, v4);
  if ( v8 != v11 )
  {
    if ( v8 > 0x3F || v11 > 0x40 )
      sub_C43C90(&v10, v8, v11);
    else
      v10 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v8 - (unsigned __int8)v11 + 64) << v8;
  }
  sub_C449B0((__int64)&v22, (const void **)(a3 + 16), v4);
  v13 = v11;
  if ( v11 > 0x40 )
  {
    sub_C43780((__int64)&v12, (const void **)&v10);
    v19 = v13;
    v18 = v12;
    v21 = v23;
    v20 = v22;
    if ( v11 > 0x40 && v10 )
      j_j___libc_free_0_0(v10);
  }
  else
  {
    v19 = v11;
    v18 = v10;
    v21 = v23;
    v20 = v22;
  }
  sub_C787D0((__int64)&v22, (__int64)&v14, (__int64)&v18, 0);
  sub_C440A0((__int64)&v12, &v24, v3, v3);
  sub_C440A0((__int64)&v10, &v22, v3, v3);
  v6 = v25 <= 0x40;
  *(_DWORD *)(a1 + 8) = v11;
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 24) = v13;
  *(_QWORD *)(a1 + 16) = v12;
  if ( !v6 && v24 )
    j_j___libc_free_0_0(v24);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  return a1;
}
