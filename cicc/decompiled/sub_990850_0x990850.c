// Function: sub_990850
// Address: 0x990850
//
char __fastcall sub_990850(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r13d
  __int64 v4; // r14
  unsigned int v5; // r12d
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // r12
  unsigned int v9; // edx
  __int64 v10; // r15
  bool v11; // cc
  __int64 v12; // rdi
  unsigned int v13; // ecx
  unsigned int v15; // [rsp+4h] [rbp-DCh]
  unsigned int v16; // [rsp+4h] [rbp-DCh]
  unsigned int v17; // [rsp+4h] [rbp-DCh]
  unsigned int v18; // [rsp+8h] [rbp-D8h]
  unsigned int v19; // [rsp+8h] [rbp-D8h]
  unsigned int v20; // [rsp+8h] [rbp-D8h]
  __int64 *v21; // [rsp+10h] [rbp-D0h] BYREF
  __int64 *v22; // [rsp+18h] [rbp-C8h] BYREF
  __int64 v23; // [rsp+20h] [rbp-C0h] BYREF
  unsigned int v24; // [rsp+28h] [rbp-B8h]
  __int64 v25; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v26; // [rsp+38h] [rbp-A8h]
  __int64 v27; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v28; // [rsp+48h] [rbp-98h]
  __int64 v29; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v30; // [rsp+58h] [rbp-88h]
  __int64 v31; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v32; // [rsp+68h] [rbp-78h]
  __int64 v33; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v34; // [rsp+78h] [rbp-68h]
  __int64 v35; // [rsp+80h] [rbp-60h]
  unsigned int v36; // [rsp+88h] [rbp-58h]
  __int64 v37; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v38; // [rsp+98h] [rbp-48h]
  __int64 v39; // [rsp+A0h] [rbp-40h] BYREF
  unsigned int v40; // [rsp+A8h] [rbp-38h]

  LOBYTE(v2) = sub_990670(a1, &v21, &v22);
  if ( !(_BYTE)v2 )
    return v2;
  v24 = *((_DWORD *)v22 + 2);
  if ( v24 > 0x40 )
    sub_C43780(&v23, v22);
  else
    v23 = *v22;
  sub_C46A40(&v23, 1);
  v3 = v24;
  v24 = 0;
  v4 = v23;
  v5 = *((_DWORD *)v21 + 2);
  v26 = v3;
  v25 = v23;
  v28 = v5;
  if ( v5 > 0x40 )
  {
    sub_C43780(&v27, v21);
    v5 = v28;
    if ( v28 > 0x40 )
    {
      if ( (unsigned __int8)sub_C43C50(&v27, &v25) )
        goto LABEL_8;
      v6 = v27;
      goto LABEL_44;
    }
    v6 = v27;
  }
  else
  {
    v6 = *v21;
    v27 = *v21;
  }
  if ( v4 == v6 )
  {
LABEL_8:
    sub_AADB10(&v33, v5, 1);
    goto LABEL_9;
  }
LABEL_44:
  v32 = v3;
  v31 = v4;
  v26 = 0;
  v38 = v5;
  v37 = v6;
  v28 = 0;
  sub_AADC30(&v33, &v37, &v31);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
  if ( v32 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  v3 = 0;
LABEL_9:
  sub_AB0A90(&v37, &v33);
  v2 = *(_DWORD *)(a2 + 24);
  v30 = v2;
  if ( v2 > 0x40 )
  {
    sub_C43780(&v29, a2 + 16);
    v2 = v30;
    if ( v30 > 0x40 )
    {
      sub_C43BD0(&v29, &v39);
      v2 = v30;
      v8 = v29;
      goto LABEL_12;
    }
    v7 = v29;
  }
  else
  {
    v7 = *(_QWORD *)(a2 + 16);
  }
  v8 = v39 | v7;
  v29 = v8;
LABEL_12:
  v30 = 0;
  v9 = *(_DWORD *)(a2 + 8);
  v32 = v9;
  if ( v9 > 0x40 )
  {
    v16 = v2;
    sub_C43780(&v31, a2);
    v9 = v32;
    v2 = v16;
    if ( v32 <= 0x40 )
    {
      v13 = v30;
      v10 = v31 | v37;
    }
    else
    {
      sub_C43BD0(&v31, &v37);
      v9 = v32;
      v10 = v31;
      v13 = v30;
      v2 = v16;
    }
    if ( v13 > 0x40 && v29 )
    {
      v17 = v9;
      v20 = v2;
      j_j___libc_free_0_0(v29);
      v9 = v17;
      v2 = v20;
    }
  }
  else
  {
    v10 = v37 | *(_QWORD *)a2;
  }
  if ( *(_DWORD *)(a2 + 8) > 0x40u && *(_QWORD *)a2 )
  {
    v15 = v9;
    v18 = v2;
    j_j___libc_free_0_0(*(_QWORD *)a2);
    v9 = v15;
    v2 = v18;
  }
  v11 = *(_DWORD *)(a2 + 24) <= 0x40u;
  *(_QWORD *)a2 = v10;
  *(_DWORD *)(a2 + 8) = v9;
  if ( !v11 )
  {
    v12 = *(_QWORD *)(a2 + 16);
    if ( v12 )
    {
      v19 = v2;
      j_j___libc_free_0_0(v12);
      v2 = v19;
    }
  }
  v11 = v40 <= 0x40;
  *(_QWORD *)(a2 + 16) = v8;
  *(_DWORD *)(a2 + 24) = v2;
  if ( !v11 && v39 )
    LOBYTE(v2) = j_j___libc_free_0_0(v39);
  if ( v38 > 0x40 && v37 )
    LOBYTE(v2) = j_j___libc_free_0_0(v37);
  if ( v36 > 0x40 && v35 )
    LOBYTE(v2) = j_j___libc_free_0_0(v35);
  if ( v34 > 0x40 && v33 )
    LOBYTE(v2) = j_j___libc_free_0_0(v33);
  if ( v28 > 0x40 && v27 )
    LOBYTE(v2) = j_j___libc_free_0_0(v27);
  if ( v3 > 0x40 && v4 )
    LOBYTE(v2) = j_j___libc_free_0_0(v4);
  if ( v24 > 0x40 && v23 )
    LOBYTE(v2) = j_j___libc_free_0_0(v23);
  return v2;
}
