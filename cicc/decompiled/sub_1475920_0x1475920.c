// Function: sub_1475920
// Address: 0x1475920
//
__int64 __fastcall sub_1475920(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // r13
  unsigned int v13; // eax
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v19; // [rsp+30h] [rbp-F0h] BYREF
  unsigned int v20; // [rsp+38h] [rbp-E8h]
  __int64 v21; // [rsp+40h] [rbp-E0h] BYREF
  unsigned int v22; // [rsp+48h] [rbp-D8h]
  __int64 v23; // [rsp+50h] [rbp-D0h] BYREF
  unsigned int v24; // [rsp+58h] [rbp-C8h]
  __int64 v25; // [rsp+60h] [rbp-C0h] BYREF
  unsigned int v26; // [rsp+68h] [rbp-B8h]
  __int64 v27; // [rsp+70h] [rbp-B0h] BYREF
  unsigned int v28; // [rsp+78h] [rbp-A8h]
  __int64 v29; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v30; // [rsp+88h] [rbp-98h]
  __int64 v31; // [rsp+90h] [rbp-90h] BYREF
  unsigned int v32; // [rsp+98h] [rbp-88h]
  __int64 v33; // [rsp+A0h] [rbp-80h]
  unsigned int v34; // [rsp+A8h] [rbp-78h]
  __int64 v35; // [rsp+B0h] [rbp-70h] BYREF
  unsigned int v36; // [rsp+B8h] [rbp-68h]
  __int64 v37; // [rsp+C0h] [rbp-60h]
  unsigned int v38; // [rsp+C8h] [rbp-58h]
  __int64 v39; // [rsp+D0h] [rbp-50h] BYREF
  unsigned int v40; // [rsp+D8h] [rbp-48h]
  __int64 v41; // [rsp+E0h] [rbp-40h] BYREF
  unsigned int v42; // [rsp+E8h] [rbp-38h]

  v8 = sub_1456040(a3);
  v9 = sub_14758B0(a2, a5, v8);
  v10 = sub_1477920(a2, v9, 0);
  sub_158A9F0(&v19, v10);
  v11 = sub_1477920(a2, a3, 1);
  v24 = *(_DWORD *)(v11 + 8);
  if ( v24 > 0x40 )
    sub_16A4FD0(&v23, v11);
  else
    v23 = *(_QWORD *)v11;
  v26 = *(_DWORD *)(v11 + 24);
  if ( v26 > 0x40 )
    sub_16A4FD0(&v25, v11 + 16);
  else
    v25 = *(_QWORD *)(v11 + 16);
  v12 = sub_1477920(a2, a4, 1);
  v28 = *(_DWORD *)(v12 + 8);
  if ( v28 > 0x40 )
    sub_16A4FD0(&v27, v12);
  else
    v27 = *(_QWORD *)v12;
  v30 = *(_DWORD *)(v12 + 24);
  if ( v30 > 0x40 )
    sub_16A4FD0(&v29, v12 + 16);
  else
    v29 = *(_QWORD *)(v12 + 16);
  sub_158ACE0(&v39, &v27);
  sub_14558A0((__int64)&v31, (__int64)&v39, (__int64)&v23, (__int64)&v19, a6, 1);
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  sub_158ABC0(&v21, &v27);
  sub_14558A0((__int64)&v35, (__int64)&v21, (__int64)&v23, (__int64)&v19, a6, 1);
  sub_158C3A0(&v39, &v31, &v35);
  if ( v32 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  v31 = v39;
  v13 = v40;
  v40 = 0;
  v32 = v13;
  if ( v34 > 0x40 && v33 )
  {
    j_j___libc_free_0_0(v33);
    v33 = v41;
    v34 = v42;
    if ( v40 > 0x40 && v39 )
    {
      j_j___libc_free_0_0(v39);
      if ( v38 <= 0x40 )
        goto LABEL_20;
      goto LABEL_69;
    }
  }
  else
  {
    v33 = v41;
    v34 = v42;
  }
  if ( v38 <= 0x40 )
    goto LABEL_20;
LABEL_69:
  if ( v37 )
    j_j___libc_free_0_0(v37);
LABEL_20:
  if ( v36 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  v14 = sub_1477920(a2, a3, 0);
  v40 = *(_DWORD *)(v14 + 8);
  if ( v40 > 0x40 )
    sub_16A4FD0(&v39, v14);
  else
    v39 = *(_QWORD *)v14;
  v42 = *(_DWORD *)(v14 + 24);
  if ( v42 > 0x40 )
    sub_16A4FD0(&v41, v14 + 16);
  else
    v41 = *(_QWORD *)(v14 + 16);
  v15 = sub_1477920(a2, a4, 0);
  sub_158A9F0(&v21, v15);
  sub_14558A0((__int64)&v35, (__int64)&v21, (__int64)&v39, (__int64)&v19, a6, 0);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  sub_158BE00(a1, &v31, &v35);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
  if ( v36 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
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
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  return a1;
}
