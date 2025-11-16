// Function: sub_C7BDB0
// Address: 0xc7bdb0
//
__int64 *__fastcall sub_C7BDB0(__int64 *a1, __int64 *a2)
{
  unsigned int v4; // ebx
  __int64 v5; // rdx
  __int64 v6; // rdx
  unsigned int v7; // edx
  unsigned __int64 v8; // r12
  __int64 v9; // r12
  __int64 v10; // r12
  unsigned int v11; // edx
  __int64 v12; // rcx
  __int64 v13; // rcx
  unsigned int v14; // esi
  unsigned __int64 v15; // rcx
  __int64 v16; // rcx
  __int64 v17; // r13
  bool v18; // cc
  __int64 v19; // rdi
  __int64 *v21; // [rsp+8h] [rbp-78h]
  unsigned int v22; // [rsp+8h] [rbp-78h]
  __int64 v23; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v24; // [rsp+18h] [rbp-68h]
  __int64 v25; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v26; // [rsp+28h] [rbp-58h]
  __int64 v27; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+38h] [rbp-48h]
  __int64 v29; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v30; // [rsp+48h] [rbp-38h]

  v4 = *((_DWORD *)a1 + 6);
  v21 = a2 + 2;
  v28 = v4;
  if ( v4 <= 0x40 )
  {
    v5 = a1[2];
LABEL_3:
    v6 = a2[2] & v5;
    v27 = v6;
    goto LABEL_4;
  }
  sub_C43780((__int64)&v27, (const void **)a1 + 2);
  v4 = v28;
  if ( v28 <= 0x40 )
  {
    v5 = v27;
    goto LABEL_3;
  }
  sub_C43B90(&v27, v21);
  v4 = v28;
  v6 = v27;
LABEL_4:
  v29 = v6;
  v7 = *((_DWORD *)a1 + 2);
  v30 = v4;
  v28 = 0;
  v24 = v7;
  if ( v7 <= 0x40 )
  {
    v8 = *a1;
LABEL_6:
    v9 = *a2 & v8;
    v23 = v9;
    goto LABEL_7;
  }
  sub_C43780((__int64)&v23, (const void **)a1);
  v7 = v24;
  if ( v24 <= 0x40 )
  {
    v8 = v23;
    v4 = v30;
    goto LABEL_6;
  }
  sub_C43B90(&v23, a2);
  v7 = v24;
  v9 = v23;
  v4 = v30;
LABEL_7:
  v26 = v7;
  v25 = v9;
  v24 = 0;
  if ( v4 > 0x40 )
  {
    sub_C43BD0(&v29, &v25);
    v4 = v30;
    v10 = v29;
    v7 = v26;
  }
  else
  {
    v10 = v29 | v9;
    v29 = v10;
  }
  v30 = 0;
  if ( v7 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  if ( v30 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  v11 = *((_DWORD *)a1 + 6);
  v28 = v11;
  if ( v11 <= 0x40 )
  {
    v12 = a1[2];
LABEL_23:
    v13 = *a2 & v12;
    v27 = v13;
    goto LABEL_24;
  }
  sub_C43780((__int64)&v27, (const void **)a1 + 2);
  v11 = v28;
  if ( v28 <= 0x40 )
  {
    v12 = v27;
    goto LABEL_23;
  }
  sub_C43B90(&v27, a2);
  v11 = v28;
  v13 = v27;
LABEL_24:
  v14 = *((_DWORD *)a1 + 2);
  v30 = v11;
  v29 = v13;
  v28 = 0;
  v24 = v14;
  if ( v14 <= 0x40 )
  {
    v15 = *a1;
LABEL_26:
    v16 = a2[2] & v15;
    v23 = v16;
    goto LABEL_27;
  }
  sub_C43780((__int64)&v23, (const void **)a1);
  v14 = v24;
  if ( v24 <= 0x40 )
  {
    v15 = v23;
    v11 = v30;
    goto LABEL_26;
  }
  sub_C43B90(&v23, v21);
  v14 = v24;
  v16 = v23;
  v11 = v30;
LABEL_27:
  v26 = v14;
  v25 = v16;
  v24 = 0;
  if ( v11 > 0x40 )
  {
    sub_C43BD0(&v29, &v25);
    v11 = v30;
  }
  else
  {
    v29 |= v16;
  }
  v17 = v29;
  v18 = *((_DWORD *)a1 + 6) <= 0x40u;
  v30 = 0;
  if ( !v18 )
  {
    v19 = a1[2];
    if ( v19 )
    {
      v22 = v11;
      j_j___libc_free_0_0(v19);
      v11 = v22;
    }
  }
  v18 = v26 <= 0x40;
  a1[2] = v17;
  *((_DWORD *)a1 + 6) = v11;
  if ( !v18 && v25 )
    j_j___libc_free_0_0(v25);
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  if ( v30 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( *((_DWORD *)a1 + 2) > 0x40u && *a1 )
    j_j___libc_free_0_0(*a1);
  *a1 = v10;
  *((_DWORD *)a1 + 2) = v4;
  return a1;
}
