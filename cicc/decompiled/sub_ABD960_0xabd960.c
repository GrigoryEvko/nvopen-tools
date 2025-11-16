// Function: sub_ABD960
// Address: 0xabd960
//
__int64 __fastcall sub_ABD960(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  __int64 v3; // rdx
  unsigned __int64 v4; // rdx
  int v5; // ebx
  unsigned int v6; // r12d
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  unsigned int v11; // edx
  __int64 v12; // r12
  unsigned int v13; // eax
  __int64 v14; // rdx
  unsigned __int64 v15; // rdx
  int v16; // ebx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  unsigned int v20; // [rsp+8h] [rbp-98h]
  __int64 v21; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v22; // [rsp+18h] [rbp-88h]
  __int64 v23; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v24; // [rsp+28h] [rbp-78h]
  __int64 v25; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v26; // [rsp+38h] [rbp-68h]
  __int64 v27; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v28; // [rsp+48h] [rbp-58h]
  unsigned __int64 v29; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v30; // [rsp+58h] [rbp-48h]
  unsigned __int64 v31; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v32; // [rsp+68h] [rbp-38h]

  if ( sub_AAF7D0(a1) || sub_AAF7D0(a2) )
    return 2;
  sub_AB0A00((__int64)&v21, a1);
  sub_AB0910((__int64)&v23, a1);
  sub_AB0A00((__int64)&v25, a2);
  sub_AB0910((__int64)&v27, a2);
  v2 = v26;
  v30 = v26;
  if ( v26 <= 0x40 )
  {
    v3 = v25;
LABEL_5:
    v32 = v2;
    v30 = 0;
    v4 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v2) & ~v3;
    if ( !v2 )
      v4 = 0;
    v29 = v4;
    v31 = v4;
    v5 = sub_C49970(&v21, &v31);
LABEL_8:
    v6 = 1;
    if ( v5 > 0 )
      goto LABEL_9;
    goto LABEL_29;
  }
  sub_C43780(&v29, &v25);
  v2 = v30;
  if ( v30 <= 0x40 )
  {
    v3 = v29;
    goto LABEL_5;
  }
  sub_C43D10(&v29, &v25, v8, v9, v10);
  v11 = v30;
  v12 = v29;
  v30 = 0;
  v32 = v11;
  v20 = v11;
  v31 = v29;
  v5 = sub_C49970(&v21, &v31);
  if ( v20 <= 0x40 )
    goto LABEL_8;
  if ( !v12 )
    goto LABEL_8;
  j_j___libc_free_0_0(v12);
  if ( v30 <= 0x40 || !v29 )
    goto LABEL_8;
  j_j___libc_free_0_0(v29);
  v6 = 1;
  if ( v5 > 0 )
    goto LABEL_9;
LABEL_29:
  v13 = v28;
  v30 = v28;
  if ( v28 <= 0x40 )
  {
    v14 = v27;
LABEL_31:
    v15 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v13) & ~v14;
    if ( !v13 )
      v15 = 0;
    v29 = v15;
    goto LABEL_34;
  }
  sub_C43780(&v29, &v27);
  v13 = v30;
  if ( v30 <= 0x40 )
  {
    v14 = v29;
    goto LABEL_31;
  }
  sub_C43D10(&v29, &v27, v17, v18, v19);
  v13 = v30;
  v15 = v29;
LABEL_34:
  v31 = v15;
  v32 = v13;
  v30 = 0;
  v16 = sub_C49970(&v23, &v31);
  sub_969240((__int64 *)&v31);
  sub_969240((__int64 *)&v29);
  v6 = (v16 <= 0) + 2;
LABEL_9:
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  return v6;
}
