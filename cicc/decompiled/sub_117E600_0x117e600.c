// Function: sub_117E600
// Address: 0x117e600
//
__int64 __fastcall sub_117E600(__int64 a1, unsigned int **a2)
{
  __int64 v3; // r14
  __int64 v4; // r15
  __int64 v5; // r10
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 v9; // rdx
  __int64 v10; // rdi
  _BYTE *v11; // r9
  __int64 v12; // rax
  const void **v13; // rsi
  __int64 v14; // r10
  __int64 v15; // r9
  unsigned int v16; // r14d
  unsigned __int64 v17; // r11
  const void *v18; // r11
  bool v19; // cc
  bool v20; // al
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // r12
  __int64 v27; // rax
  char v28; // al
  const void **v29; // rsi
  unsigned int v30; // edx
  unsigned __int64 v31; // r9
  const void *v32; // r9
  bool v33; // al
  bool v34; // r12
  __int64 v35; // rax
  __int64 v36; // rax
  _BYTE *v37; // rax
  _BYTE *v38; // rax
  const void *v39; // [rsp+0h] [rbp-B0h]
  __int64 v40; // [rsp+8h] [rbp-A8h]
  const void *v41; // [rsp+8h] [rbp-A8h]
  bool v42; // [rsp+8h] [rbp-A8h]
  unsigned int v43; // [rsp+8h] [rbp-A8h]
  __int64 v44; // [rsp+8h] [rbp-A8h]
  __int64 v45; // [rsp+8h] [rbp-A8h]
  __int64 v46; // [rsp+10h] [rbp-A0h]
  __int64 v47; // [rsp+10h] [rbp-A0h]
  __int64 v48; // [rsp+10h] [rbp-A0h]
  __int64 v49; // [rsp+10h] [rbp-A0h]
  __int64 v50; // [rsp+10h] [rbp-A0h]
  __int64 v51; // [rsp+10h] [rbp-A0h]
  __int64 v52; // [rsp+10h] [rbp-A0h]
  __int64 v53; // [rsp+10h] [rbp-A0h]
  __int64 v54; // [rsp+10h] [rbp-A0h]
  __int64 v55; // [rsp+10h] [rbp-A0h]
  const void **v57; // [rsp+28h] [rbp-88h] BYREF
  const void *v58; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v59; // [rsp+38h] [rbp-78h]
  const void *v60; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v61; // [rsp+48h] [rbp-68h]
  const char *v62; // [rsp+50h] [rbp-60h] BYREF
  const void ***v63; // [rsp+58h] [rbp-58h] BYREF
  char v64; // [rsp+60h] [rbp-50h]
  __int16 v65; // [rsp+70h] [rbp-40h]

  v3 = *(_QWORD *)(a1 - 64);
  v4 = *(_QWORD *)(a1 - 96);
  v5 = *(_QWORD *)(a1 - 32);
  v6 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)v3 != 57 )
    goto LABEL_2;
  v24 = *(_QWORD *)(v3 - 64);
  if ( !v24 )
    goto LABEL_2;
  v25 = *(_QWORD *)(v3 - 32);
  v26 = v25 + 24;
  if ( *(_BYTE *)v25 != 17 )
  {
    v54 = *(_QWORD *)(v3 - 64);
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v25 + 8) + 8LL) - 17 > 1 )
      goto LABEL_2;
    if ( *(_BYTE *)v25 > 0x15u )
      goto LABEL_2;
    v44 = v5;
    v37 = sub_AD7630(v25, 0, v24);
    v5 = v44;
    if ( !v37 || *v37 != 17 )
      goto LABEL_2;
    v24 = v54;
    v26 = (__int64)(v37 + 24);
  }
  v62 = (const char *)v24;
  v63 = &v57;
  v64 = 0;
  v27 = *(_QWORD *)(v5 + 16);
  if ( !v27 )
    goto LABEL_2;
  if ( *(_QWORD *)(v27 + 8) )
    goto LABEL_2;
  if ( *(_BYTE *)v5 != 58 )
    goto LABEL_2;
  if ( v24 != *(_QWORD *)(v5 - 64) )
    goto LABEL_2;
  v50 = v5;
  v28 = sub_991580((__int64)&v63, *(_QWORD *)(v5 - 32));
  v5 = v50;
  if ( !v28 )
    goto LABEL_2;
  v29 = v57;
  v30 = *((_DWORD *)v57 + 2);
  v59 = v30;
  if ( v30 > 0x40 )
  {
    sub_C43780((__int64)&v58, v57);
    v30 = v59;
    v5 = v50;
    if ( v59 > 0x40 )
    {
      sub_C43D10((__int64)&v58);
      v30 = v59;
      v32 = v58;
      v5 = v50;
      goto LABEL_38;
    }
    v31 = (unsigned __int64)v58;
  }
  else
  {
    v31 = (unsigned __int64)*v57;
  }
  v32 = (const void *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v30) & ~v31);
  if ( !v30 )
    v32 = 0;
  v58 = v32;
LABEL_38:
  v59 = 0;
  v19 = *(_DWORD *)(v26 + 8) <= 0x40u;
  v61 = v30;
  v60 = v32;
  if ( v19 )
  {
    v34 = *(_QWORD *)v26 == (_QWORD)v32;
  }
  else
  {
    v29 = &v60;
    v39 = v32;
    v43 = v30;
    v51 = v5;
    v33 = sub_C43C50(v26, &v60);
    v32 = v39;
    v30 = v43;
    v5 = v51;
    v34 = v33;
  }
  if ( v30 > 0x40 )
  {
    if ( v32 )
    {
      v52 = v5;
      j_j___libc_free_0_0(v32);
      v5 = v52;
      if ( v59 > 0x40 )
      {
        if ( v58 )
        {
          j_j___libc_free_0_0(v58);
          v5 = v52;
        }
      }
    }
  }
  if ( v34 )
  {
    v53 = sub_AD6530(v6, (__int64)v29);
    v35 = sub_AD8D80(v6, (__int64)v57);
    v65 = 259;
    v62 = "masksel";
    v36 = sub_B36550(a2, v4, v53, v35, (__int64)&v62, a1);
    v65 = 257;
    return sub_B504D0(29, v3, v36, (__int64)&v62, 0, 0);
  }
LABEL_2:
  if ( *(_BYTE *)v5 != 57 )
    return 0;
  v9 = *(_QWORD *)(v5 - 64);
  if ( !v9 )
    return 0;
  v10 = *(_QWORD *)(v5 - 32);
  v11 = (_BYTE *)(v10 + 24);
  if ( *(_BYTE *)v10 != 17 )
  {
    v45 = *(_QWORD *)(v5 - 64);
    v55 = v5;
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v10 + 8) + 8LL) - 17 > 1 )
      return 0;
    if ( *(_BYTE *)v10 > 0x15u )
      return 0;
    v38 = sub_AD7630(v10, 0, v9);
    if ( !v38 || *v38 != 17 )
      return 0;
    v9 = v45;
    v5 = v55;
    v11 = v38 + 24;
  }
  v62 = (const char *)v9;
  v63 = &v57;
  v64 = 0;
  v12 = *(_QWORD *)(v3 + 16);
  if ( !v12 )
    return 0;
  v7 = *(_QWORD *)(v12 + 8);
  if ( v7 )
    return 0;
  if ( *(_BYTE *)v3 != 58 )
    return 0;
  if ( v9 != *(_QWORD *)(v3 - 64) )
    return 0;
  v40 = (__int64)v11;
  v46 = v5;
  if ( !(unsigned __int8)sub_991580((__int64)&v63, *(_QWORD *)(v3 - 32)) )
    return 0;
  v13 = v57;
  v14 = v46;
  v15 = v40;
  v59 = *((_DWORD *)v57 + 2);
  v16 = v59;
  if ( v59 <= 0x40 )
  {
    v17 = (unsigned __int64)*v57;
LABEL_14:
    v18 = (const void *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v16) & ~v17);
    if ( !v16 )
      v18 = 0;
    v58 = v18;
    goto LABEL_17;
  }
  sub_C43780((__int64)&v58, v57);
  v16 = v59;
  v14 = v46;
  v15 = v40;
  if ( v59 <= 0x40 )
  {
    v17 = (unsigned __int64)v58;
    goto LABEL_14;
  }
  sub_C43D10((__int64)&v58);
  v16 = v59;
  v18 = v58;
  v15 = v40;
  v14 = v46;
LABEL_17:
  v59 = 0;
  v19 = *(_DWORD *)(v15 + 8) <= 0x40u;
  v61 = v16;
  v60 = v18;
  if ( v19 )
  {
    v20 = *(_QWORD *)v15 == (_QWORD)v18;
  }
  else
  {
    v13 = &v60;
    v41 = v18;
    v47 = v14;
    v20 = sub_C43C50(v15, &v60);
    v18 = v41;
    v14 = v47;
  }
  if ( v16 > 0x40 )
  {
    if ( v18 )
    {
      v42 = v20;
      v48 = v14;
      j_j___libc_free_0_0(v18);
      v14 = v48;
      v20 = v42;
      if ( v59 > 0x40 )
      {
        if ( v58 )
        {
          j_j___libc_free_0_0(v58);
          v20 = v42;
          v14 = v48;
        }
      }
    }
  }
  v49 = v14;
  if ( v20 )
  {
    v21 = sub_AD6530(v6, (__int64)v13);
    v22 = sub_AD8D80(v6, (__int64)v57);
    v65 = 259;
    v62 = "masksel";
    v23 = sub_B36550(a2, v4, v22, v21, (__int64)&v62, a1);
    v65 = 257;
    return sub_B504D0(29, v49, v23, (__int64)&v62, 0, 0);
  }
  return v7;
}
