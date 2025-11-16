// Function: sub_AAE470
// Address: 0xaae470
//
__int64 __fastcall sub_AAE470(__int64 a1, __int64 *a2)
{
  unsigned int v2; // r12d
  unsigned __int64 v3; // rdx
  __int64 v4; // rax
  unsigned int v5; // r12d
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // eax
  unsigned int v10; // eax
  unsigned int v11; // eax
  __int64 v13; // rcx
  __int64 v14; // r8
  unsigned __int64 v15; // rax
  unsigned int v16; // eax
  unsigned int v17; // eax
  __int64 v18; // [rsp+0h] [rbp-A0h] BYREF
  unsigned int v19; // [rsp+8h] [rbp-98h]
  unsigned __int64 v20; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v21; // [rsp+18h] [rbp-88h]
  __int64 v22; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v23; // [rsp+28h] [rbp-78h]
  __int64 v24; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v25; // [rsp+38h] [rbp-68h]
  __int64 v26; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v27; // [rsp+48h] [rbp-58h]
  __int64 v28; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v29; // [rsp+58h] [rbp-48h]
  __int64 v30; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v31; // [rsp+68h] [rbp-38h]

  v2 = *((_DWORD *)a2 + 2);
  if ( v2 <= 0x40 )
  {
    if ( *a2 )
    {
      sub_986680((__int64)&v18, v2);
      v21 = v2;
      v3 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
      if ( !v2 )
        v3 = 0;
      v20 = v3;
      v4 = ~(1LL << ((unsigned __int8)v2 - 1));
      goto LABEL_6;
    }
LABEL_43:
    sub_AADB10(a1, v2, 1);
    return a1;
  }
  if ( v2 - (unsigned int)sub_C444A0(a2) <= 0x40 && !*(_QWORD *)*a2 )
    goto LABEL_43;
  sub_986680((__int64)&v18, v2);
  v21 = v2;
  sub_C43690(&v20, -1, 1);
  v4 = ~(1LL << ((unsigned __int8)v2 - 1));
  if ( v21 > 0x40 )
  {
    *(_QWORD *)(v20 + 8LL * ((v2 - 1) >> 6)) &= v4;
    v5 = *((_DWORD *)a2 + 2);
    if ( !v5 )
      goto LABEL_48;
    goto LABEL_7;
  }
LABEL_6:
  v5 = *((_DWORD *)a2 + 2);
  v20 &= v4;
  if ( !v5 )
    goto LABEL_48;
LABEL_7:
  if ( v5 > 0x40 )
  {
    if ( v5 != (unsigned int)sub_C445E0(a2) )
    {
      v6 = *a2;
      v23 = 1;
      v7 = 1LL << ((unsigned __int8)v5 - 1);
      v22 = 0;
      v25 = 1;
      v24 = 0;
      v8 = *(_QWORD *)(v6 + 8LL * ((v5 - 1) >> 6));
      goto LABEL_10;
    }
LABEL_48:
    sub_9865C0((__int64)&v30, (__int64)&v18);
    sub_9865C0((__int64)&v26, (__int64)&v20);
    if ( v27 > 0x40 )
    {
      sub_C43D10(&v26, &v20, v27, v13, v14);
    }
    else
    {
      v15 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v27) & ~v26;
      if ( !v27 )
        v15 = 0;
      v26 = v15;
    }
    sub_C46250(&v26);
    v16 = v27;
    v27 = 0;
    v29 = v16;
    v28 = v26;
    sub_AADC30(a1, (__int64)&v28, &v30);
    sub_969240(&v28);
    sub_969240(&v26);
    sub_969240(&v30);
    goto LABEL_36;
  }
  v8 = *a2;
  if ( *a2 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v5) )
    goto LABEL_48;
  v23 = 1;
  v22 = 0;
  v7 = 1LL << ((unsigned __int8)v5 - 1);
  v25 = 1;
  v24 = 0;
LABEL_10:
  if ( (v8 & v7) != 0 )
  {
    sub_C4CAA0(&v30, &v20, a2, 2);
    if ( v23 > 0x40 && v22 )
      j_j___libc_free_0_0(v22);
    v22 = v30;
    v9 = v31;
    v31 = 0;
    v23 = v9;
    sub_969240(&v30);
    sub_C4CAA0(&v30, &v18, a2, 0);
    if ( v25 <= 0x40 )
      goto LABEL_17;
  }
  else
  {
    sub_C4CAA0(&v30, &v18, a2, 2);
    if ( v23 > 0x40 && v22 )
      j_j___libc_free_0_0(v22);
    v22 = v30;
    v17 = v31;
    v31 = 0;
    v23 = v17;
    sub_969240(&v30);
    sub_C4CAA0(&v30, &v20, a2, 0);
    if ( v25 <= 0x40 )
      goto LABEL_17;
  }
  if ( v24 )
    j_j___libc_free_0_0(v24);
LABEL_17:
  v24 = v30;
  v10 = v31;
  v31 = 0;
  v25 = v10;
  sub_969240(&v30);
  v27 = v25;
  if ( v25 > 0x40 )
    sub_C43780(&v26, &v24);
  else
    v26 = v24;
  sub_C46A40(&v26, 1);
  v11 = v27;
  v27 = 0;
  v29 = v11;
  v28 = v26;
  v31 = v23;
  if ( v23 > 0x40 )
    sub_C43780(&v30, &v22);
  else
    v30 = v22;
  sub_9875E0(a1, &v30, &v28);
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
LABEL_36:
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  return a1;
}
