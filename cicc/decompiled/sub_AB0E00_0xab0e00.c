// Function: sub_AB0E00
// Address: 0xab0e00
//
__int64 __fastcall sub_AB0E00(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r15d
  unsigned int v6; // r12d
  unsigned int v7; // r12d
  unsigned int v8; // eax
  unsigned __int64 v9; // rax
  unsigned int v10; // ecx
  unsigned int v11; // eax
  __int64 v12; // rdx
  unsigned __int64 v13; // rdx
  unsigned int v14; // edx
  unsigned int v15; // eax
  __int64 v16; // rsi
  unsigned int v17; // eax
  unsigned __int64 v18; // rax
  unsigned int v19; // eax
  unsigned __int64 v20; // rdi
  unsigned int v21; // eax
  int v22; // ebx
  unsigned __int64 *v23; // rsi
  unsigned __int64 v24; // rdx
  unsigned int v25; // [rsp+18h] [rbp-C8h]
  unsigned int v27; // [rsp+20h] [rbp-C0h]
  unsigned int v28; // [rsp+20h] [rbp-C0h]
  unsigned int v29; // [rsp+20h] [rbp-C0h]
  unsigned int v30; // [rsp+2Ch] [rbp-B4h]
  unsigned int v31; // [rsp+2Ch] [rbp-B4h]
  char v32; // [rsp+3Fh] [rbp-A1h] BYREF
  unsigned __int64 v33; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v34; // [rsp+48h] [rbp-98h]
  unsigned __int64 v35; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v36; // [rsp+58h] [rbp-88h]
  unsigned __int64 v37; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v38; // [rsp+68h] [rbp-78h]
  unsigned __int64 v39; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v40; // [rsp+78h] [rbp-68h]
  __int64 v41; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v42; // [rsp+88h] [rbp-58h]
  __int64 v43; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v44; // [rsp+98h] [rbp-48h]
  unsigned __int64 v45; // [rsp+A0h] [rbp-40h] BYREF
  unsigned int v46; // [rsp+A8h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 8);
  sub_AB0A00((__int64)&v33, a2);
  sub_AB0A00((__int64)&v45, a3);
  v30 = v46;
  if ( v46 > 0x40 )
  {
    if ( v30 - (unsigned int)sub_C444A0(&v45) <= 0x40 )
    {
      v20 = v45;
      if ( (unsigned __int64)v4 >= *(_QWORD *)v45 )
      {
        v31 = *(_DWORD *)v45;
        goto LABEL_66;
      }
    }
    else
    {
      v20 = v45;
      if ( !v45 )
        goto LABEL_3;
    }
    v31 = v4;
LABEL_66:
    j_j___libc_free_0_0(v20);
    goto LABEL_4;
  }
  if ( v4 < v45 )
  {
LABEL_3:
    v31 = v4;
    goto LABEL_4;
  }
  v31 = v45;
LABEL_4:
  sub_C47B80(&v35, &v33, v31, &v32);
  if ( v32 )
  {
    sub_AADB10(a1, v4, 0);
    goto LABEL_6;
  }
  sub_AB0910((__int64)&v37, a2);
  sub_AB0910((__int64)&v45, a3);
  v6 = v46;
  if ( v46 > 0x40 )
  {
    if ( v6 - (unsigned int)sub_C444A0(&v45) <= 0x40 && (v7 = *(_QWORD *)v45, (unsigned __int64)v4 >= *(_QWORD *)v45)
      || (v7 = v4, v45) )
    {
      j_j___libc_free_0_0(v45);
      v40 = v36;
      if ( v36 <= 0x40 )
        goto LABEL_17;
      goto LABEL_87;
    }
  }
  else
  {
    v7 = v45;
    if ( v4 < v45 )
      v7 = v4;
  }
  v40 = v36;
  if ( v36 <= 0x40 )
  {
LABEL_17:
    v39 = v35;
    goto LABEL_18;
  }
LABEL_87:
  sub_C43780(&v39, &v35);
LABEL_18:
  v8 = v38;
  if ( v38 > 0x40 )
  {
    v8 = sub_C444A0(&v37);
  }
  else if ( v37 )
  {
    _BitScanReverse64(&v9, v37);
    v8 = v38 - 64 + (v9 ^ 0x3F);
  }
  if ( v31 <= v8 )
  {
    v10 = v8;
    if ( v7 <= v8 )
      v10 = v7;
    v25 = v8;
    v27 = v10;
    sub_9865C0((__int64)&v45, (__int64)&v37);
    v11 = v25;
    if ( v46 > 0x40 )
    {
      sub_C47690(&v45, v27);
      v11 = v25;
    }
    else
    {
      v12 = 0;
      if ( v46 != v27 )
        v12 = v45 << v27;
      v13 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v46) & v12;
      if ( !v46 )
        v13 = 0;
      v45 = v13;
    }
    if ( v40 > 0x40 && v39 )
    {
      v28 = v11;
      j_j___libc_free_0_0(v39);
      v11 = v28;
    }
    v29 = v11;
    v39 = v45;
    v14 = v46;
    v46 = 0;
    v40 = v14;
    sub_969240((__int64 *)&v45);
    v8 = v29;
  }
  v15 = v8 + 1;
  v16 = v31;
  if ( v15 >= v31 )
    v16 = v15;
  v17 = v34;
  if ( v34 <= 0x40 )
  {
    if ( v33 )
    {
      _BitScanReverse64(&v18, v33);
      v17 = v34 - 64 + (v18 ^ 0x3F);
    }
    if ( v7 <= v17 )
      v17 = v7;
    if ( v17 < (unsigned int)v16 )
      goto LABEL_42;
LABEL_70:
    v46 = v4;
    v22 = v16 - v4;
    if ( v4 > 0x40 )
    {
      sub_C43690(&v45, 0, 0);
      v4 = v46;
      v16 = v46 + v22;
    }
    else
    {
      v45 = 0;
    }
    if ( v4 != (_DWORD)v16 )
    {
      if ( (unsigned int)v16 > 0x3F || v4 > 0x40 )
        sub_C43C90(&v45, v16, v4);
      else
        v45 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v22 + 64) << v16;
    }
    v23 = &v45;
    if ( (int)sub_C49970(&v39, &v45) > 0 )
      v23 = &v39;
    if ( v40 <= 0x40 && *((_DWORD *)v23 + 2) <= 0x40u )
    {
      v24 = *v23;
      v40 = *((_DWORD *)v23 + 2);
      v39 = v24;
    }
    else
    {
      sub_C43990(&v39, v23);
    }
    sub_969240((__int64 *)&v45);
    v42 = v40;
    if ( v40 <= 0x40 )
      goto LABEL_43;
LABEL_82:
    sub_C43780(&v41, &v39);
    goto LABEL_44;
  }
  v21 = sub_C444A0(&v33);
  v16 = (unsigned int)v16;
  if ( v7 <= v21 )
    v21 = v7;
  if ( v21 >= (unsigned int)v16 )
    goto LABEL_70;
LABEL_42:
  v42 = v40;
  if ( v40 > 0x40 )
    goto LABEL_82;
LABEL_43:
  v41 = v39;
LABEL_44:
  sub_C46A40(&v41, 1);
  v19 = v42;
  v42 = 0;
  v44 = v19;
  v43 = v41;
  v46 = v36;
  if ( v36 > 0x40 )
    sub_C43780(&v45, &v35);
  else
    v45 = v35;
  sub_9875E0(a1, (__int64 *)&v45, &v43);
  if ( v46 > 0x40 && v45 )
    j_j___libc_free_0_0(v45);
  if ( v44 > 0x40 && v43 )
    j_j___libc_free_0_0(v43);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
LABEL_6:
  if ( v36 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  return a1;
}
