// Function: sub_DBEFC0
// Address: 0xdbefc0
//
__int64 __fastcall sub_DBEFC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rbx
  __int64 v7; // rbx
  unsigned int v8; // eax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v13; // [rsp+18h] [rbp-F8h]
  __int64 v15; // [rsp+30h] [rbp-E0h] BYREF
  unsigned int v16; // [rsp+38h] [rbp-D8h]
  const void *v17; // [rsp+40h] [rbp-D0h] BYREF
  unsigned int v18; // [rsp+48h] [rbp-C8h]
  __int64 v19; // [rsp+50h] [rbp-C0h] BYREF
  unsigned int v20; // [rsp+58h] [rbp-B8h]
  const void *v21; // [rsp+60h] [rbp-B0h] BYREF
  unsigned int v22; // [rsp+68h] [rbp-A8h]
  __int64 v23; // [rsp+70h] [rbp-A0h] BYREF
  unsigned int v24; // [rsp+78h] [rbp-98h]
  const void *v25; // [rsp+80h] [rbp-90h] BYREF
  unsigned int v26; // [rsp+88h] [rbp-88h]
  __int64 v27; // [rsp+90h] [rbp-80h]
  unsigned int v28; // [rsp+98h] [rbp-78h]
  __int64 v29; // [rsp+A0h] [rbp-70h] BYREF
  unsigned int v30; // [rsp+A8h] [rbp-68h]
  __int64 v31; // [rsp+B0h] [rbp-60h]
  unsigned int v32; // [rsp+B8h] [rbp-58h]
  const void *v33; // [rsp+C0h] [rbp-50h] BYREF
  unsigned int v34; // [rsp+C8h] [rbp-48h]
  __int64 v35; // [rsp+D0h] [rbp-40h] BYREF
  unsigned int v36; // [rsp+D8h] [rbp-38h]

  v6 = sub_DBB9F0(a2, a3, 1u, 0);
  v18 = *(_DWORD *)(v6 + 8);
  if ( v18 > 0x40 )
    sub_C43780((__int64)&v17, (const void **)v6);
  else
    v17 = *(const void **)v6;
  v20 = *(_DWORD *)(v6 + 24);
  if ( v20 > 0x40 )
    sub_C43780((__int64)&v19, (const void **)(v6 + 16));
  else
    v19 = *(_QWORD *)(v6 + 16);
  v7 = sub_DBB9F0(a2, a4, 1u, 0);
  v22 = *(_DWORD *)(v7 + 8);
  if ( v22 > 0x40 )
    sub_C43780((__int64)&v21, (const void **)v7);
  else
    v21 = *(const void **)v7;
  v24 = *(_DWORD *)(v7 + 24);
  if ( v24 > 0x40 )
    sub_C43780((__int64)&v23, (const void **)(v7 + 16));
  else
    v23 = *(_QWORD *)(v7 + 16);
  sub_AB14C0((__int64)&v33, (__int64)&v21);
  sub_D94A80((__int64)&v25, &v33, (__int64)&v17, a5, 1);
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  sub_AB13A0((__int64)&v15, (__int64)&v21);
  sub_D94A80((__int64)&v29, &v15, (__int64)&v17, a5, 1);
  sub_AB3510((__int64)&v33, (__int64)&v25, (__int64)&v29, 0);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  v25 = v33;
  v8 = v34;
  v34 = 0;
  v26 = v8;
  if ( v28 > 0x40 && v27 )
  {
    j_j___libc_free_0_0(v27);
    v27 = v35;
    v28 = v36;
    if ( v34 > 0x40 && v33 )
    {
      j_j___libc_free_0_0(v33);
      if ( v32 <= 0x40 )
        goto LABEL_20;
      goto LABEL_66;
    }
  }
  else
  {
    v27 = v35;
    v28 = v36;
  }
  if ( v32 <= 0x40 )
    goto LABEL_20;
LABEL_66:
  if ( v31 )
    j_j___libc_free_0_0(v31);
LABEL_20:
  if ( v30 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  v9 = sub_DBB9F0(a2, a3, 0, 0);
  v34 = *(_DWORD *)(v9 + 8);
  if ( v34 > 0x40 )
  {
    v13 = v9;
    sub_C43780((__int64)&v33, (const void **)v9);
    v9 = v13;
  }
  else
  {
    v33 = *(const void **)v9;
  }
  v36 = *(_DWORD *)(v9 + 24);
  if ( v36 > 0x40 )
    sub_C43780((__int64)&v35, (const void **)(v9 + 16));
  else
    v35 = *(_QWORD *)(v9 + 16);
  v10 = sub_DBB9F0(a2, a4, 0, 0);
  sub_AB0910((__int64)&v15, v10);
  sub_D94A80((__int64)&v29, &v15, (__int64)&v33, a5, 0);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  if ( v36 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  sub_AB2160(a1, (__int64)&v25, (__int64)&v29, 0);
  if ( v32 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  if ( v30 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  return a1;
}
