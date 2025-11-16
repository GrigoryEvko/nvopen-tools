// Function: sub_1590730
// Address: 0x1590730
//
__int64 __fastcall sub_1590730(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned int v7; // r13d
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // ecx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // [rsp+8h] [rbp-C8h]
  __int64 v14; // [rsp+10h] [rbp-C0h]
  __int64 v15; // [rsp+20h] [rbp-B0h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v17; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v18; // [rsp+38h] [rbp-98h]
  __int64 v19; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v20; // [rsp+48h] [rbp-88h]
  unsigned __int64 v21; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v22; // [rsp+58h] [rbp-78h]
  unsigned __int64 v23; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v24; // [rsp+68h] [rbp-68h]
  unsigned __int64 v25; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v26; // [rsp+78h] [rbp-58h]
  unsigned __int64 v27; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+88h] [rbp-48h]
  __int64 v29; // [rsp+90h] [rbp-40h] BYREF
  unsigned int v30; // [rsp+98h] [rbp-38h]

  if ( sub_158A120(a2) || sub_158A120(a3) )
  {
    sub_15897D0(a1, *(_DWORD *)(a2 + 8), 0);
    return a1;
  }
  sub_158ABC0((__int64)&v25, a2);
  sub_158AAD0((__int64)&v27, a3);
  v30 = v26;
  if ( v26 > 0x40 )
    sub_16A4FD0(&v29, &v25);
  else
    v29 = v25;
  sub_16A6020(&v29, &v27);
  sub_16A7490(&v29, 1);
  v16 = v30;
  v15 = v29;
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  sub_158ACE0((__int64)&v27, a2);
  sub_158A9F0((__int64)&v29, a3);
  v18 = v28;
  if ( v28 > 0x40 )
    sub_16A4FD0(&v17, &v27);
  else
    v17 = v27;
  sub_16A6020(&v17, &v29);
  if ( v30 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  sub_158ABC0((__int64)&v25, a2);
  sub_158A9F0((__int64)&v27, a3);
  v30 = v26;
  if ( v26 > 0x40 )
    sub_16A4FD0(&v29, &v25);
  else
    v29 = v25;
  sub_16A6020(&v29, &v27);
  sub_16A7490(&v29, 1);
  v20 = v30;
  v19 = v29;
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  sub_158ACE0((__int64)&v27, a2);
  sub_158AAD0((__int64)&v29, a3);
  v22 = v28;
  if ( v28 > 0x40 )
    sub_16A4FD0(&v21, &v27);
  else
    v21 = v27;
  sub_16A6020(&v21, &v29);
  if ( v30 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  v24 = 1;
  v23 = 0;
  v26 = 1;
  v25 = 0;
  sub_158ACE0((__int64)&v29, a2);
  v5 = 1LL << ((unsigned __int8)v30 - 1);
  if ( v30 > 0x40 )
  {
    v6 = *(_QWORD *)(v29 + 8LL * ((v30 - 1) >> 6)) & v5;
    if ( v29 )
    {
      v13 = v6;
      j_j___libc_free_0_0(v29);
      v6 = v13;
    }
  }
  else
  {
    v6 = v29 & v5;
  }
  if ( !v6 )
  {
    if ( v26 <= 0x40 && v18 <= 0x40 )
    {
      v26 = v18;
      v25 = v17 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v18);
    }
    else
    {
      sub_16A51C0(&v25, &v17);
    }
LABEL_43:
    if ( v24 <= 0x40 && v16 <= 0x40 )
    {
      v24 = v16;
      v23 = v15 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v16);
    }
    else
    {
      sub_16A51C0(&v23, &v15);
    }
    goto LABEL_46;
  }
  sub_158ABC0((__int64)&v29, a2);
  v11 = 1LL << ((unsigned __int8)v30 - 1);
  if ( v30 <= 0x40 )
  {
    v12 = v29 & v11;
  }
  else
  {
    v12 = *(_QWORD *)(v29 + 8LL * ((v30 - 1) >> 6)) & v11;
    if ( v29 )
    {
      v14 = v12;
      j_j___libc_free_0_0(v29);
      v12 = v14;
    }
  }
  if ( !v12 )
  {
    if ( v26 <= 0x40 && v22 <= 0x40 )
    {
      v26 = v22;
      v25 = v21 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v22);
    }
    else
    {
      sub_16A51C0(&v25, &v21);
    }
    goto LABEL_43;
  }
  if ( v26 <= 0x40 && v22 <= 0x40 )
  {
    v26 = v22;
    v25 = v21 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v22);
  }
  else
  {
    sub_16A51C0(&v25, &v21);
  }
  if ( v24 <= 0x40 && v20 <= 0x40 )
  {
    v24 = v20;
    v23 = v19 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v20);
  }
  else
  {
    sub_16A51C0(&v23, &v19);
  }
LABEL_46:
  v7 = v26;
  if ( v26 <= 0x40 )
  {
    v8 = v25;
    v9 = v23;
    if ( v25 != v23 )
    {
LABEL_49:
      v10 = v24;
      v29 = v9;
      v24 = 0;
      v30 = v10;
      v28 = v7;
      v27 = v8;
      v26 = 0;
      sub_15898E0(a1, (__int64)&v27, &v29);
      if ( v28 > 0x40 && v27 )
        j_j___libc_free_0_0(v27);
      if ( v30 > 0x40 && v29 )
        j_j___libc_free_0_0(v29);
      goto LABEL_55;
    }
  }
  else if ( !(unsigned __int8)sub_16A5220(&v25, &v23) )
  {
    v8 = v25;
    v9 = v23;
    goto LABEL_49;
  }
  sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
LABEL_55:
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
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  return a1;
}
