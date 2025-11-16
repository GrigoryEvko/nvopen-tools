// Function: sub_161DF70
// Address: 0x161df70
//
__int64 __fastcall sub_161DF70(__int64 *a1, __int64 a2, __int64 a3)
{
  int v4; // ebx
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rsi
  unsigned int v8; // eax
  __int64 v9; // rdx
  unsigned int v10; // eax
  unsigned int v11; // ebx
  __int64 v13; // r13
  __int64 v14; // [rsp+8h] [rbp-D8h]
  __int64 v15; // [rsp+10h] [rbp-D0h]
  __int64 v16; // [rsp+20h] [rbp-C0h] BYREF
  unsigned int v17; // [rsp+28h] [rbp-B8h]
  __int64 v18; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v19; // [rsp+38h] [rbp-A8h]
  __int64 v20; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v21; // [rsp+48h] [rbp-98h]
  __int64 v22; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v23; // [rsp+58h] [rbp-88h]
  __int64 v24; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v25; // [rsp+68h] [rbp-78h]
  __int64 v26; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v27; // [rsp+78h] [rbp-68h]
  __int64 v28; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v29; // [rsp+88h] [rbp-58h]
  __int64 v30; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v31; // [rsp+98h] [rbp-48h]
  __int64 v32; // [rsp+A0h] [rbp-40h] BYREF
  unsigned int v33; // [rsp+A8h] [rbp-38h]

  v31 = *(_DWORD *)(a3 + 32);
  if ( v31 > 0x40 )
    sub_16A4FD0(&v30, a3 + 24);
  else
    v30 = *(_QWORD *)(a3 + 24);
  v27 = *(_DWORD *)(a2 + 32);
  if ( v27 > 0x40 )
    sub_16A4FD0(&v26, a2 + 24);
  else
    v26 = *(_QWORD *)(a2 + 24);
  sub_15898E0((__int64)&v22, (__int64)&v26, &v30);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  v4 = *((_DWORD *)a1 + 2);
  v5 = *a1;
  v6 = *(_QWORD *)(*a1 + 8LL * (unsigned int)(v4 - 2));
  v15 = (unsigned int)(v4 - 2);
  v17 = *(_DWORD *)(v6 + 32);
  if ( v17 > 0x40 )
  {
    sub_16A4FD0(&v16, v6 + 24);
    v5 = *a1;
  }
  else
  {
    v16 = *(_QWORD *)(v6 + 24);
  }
  v7 = *(_QWORD *)(v5 + 8LL * (unsigned int)(v4 - 1));
  v14 = (unsigned int)(v4 - 1);
  v8 = *(_DWORD *)(v7 + 32);
  v19 = v8;
  if ( v8 <= 0x40 )
  {
    v9 = *(_QWORD *)(v7 + 24);
    v31 = v8;
    v18 = v9;
LABEL_15:
    v30 = v18;
    goto LABEL_16;
  }
  sub_16A4FD0(&v18, v7 + 24);
  v31 = v19;
  if ( v19 <= 0x40 )
    goto LABEL_15;
  sub_16A4FD0(&v30, &v18);
LABEL_16:
  v21 = v17;
  if ( v17 > 0x40 )
    sub_16A4FD0(&v20, &v16);
  else
    v20 = v16;
  sub_15898E0((__int64)&v26, (__int64)&v20, &v30);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  sub_158BE00((__int64)&v30, (__int64)&v22, (__int64)&v26);
  LOBYTE(v10) = sub_158A120((__int64)&v30);
  v11 = v10;
  if ( !(_BYTE)v10 )
  {
    v11 = 1;
    goto LABEL_53;
  }
  if ( v25 <= 0x40 )
  {
    if ( v24 == v26 )
      goto LABEL_53;
  }
  else
  {
    v11 = sub_16A5220(&v24, &v26);
    if ( (_BYTE)v11 )
      goto LABEL_53;
  }
  if ( v23 > 0x40 )
  {
    v11 = sub_16A5220(&v22, &v28);
    if ( v33 <= 0x40 )
      goto LABEL_29;
    goto LABEL_54;
  }
  LOBYTE(v11) = v22 == v28;
LABEL_53:
  if ( v33 <= 0x40 )
    goto LABEL_29;
LABEL_54:
  if ( v32 )
    j_j___libc_free_0_0(v32);
LABEL_29:
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  if ( (_BYTE)v11 )
  {
    sub_158C3A0((__int64)&v30, (__int64)&v26, (__int64)&v22);
    v13 = *(_QWORD *)a3;
    *(_QWORD *)(*a1 + 8 * v15) = sub_15A1070(v13, (__int64)&v30);
    *(_QWORD *)(*a1 + 8 * v14) = sub_15A1070(v13, (__int64)&v32);
    if ( v33 > 0x40 && v32 )
      j_j___libc_free_0_0(v32);
    if ( v31 > 0x40 && v30 )
      j_j___libc_free_0_0(v30);
  }
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  return v11;
}
