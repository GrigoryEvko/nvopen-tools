// Function: sub_13DD0F0
// Address: 0x13dd0f0
//
__int64 __fastcall sub_13DD0F0(__int64 a1, __int64 a2, __int64 *a3, int a4, unsigned int a5)
{
  _BYTE *v5; // r11
  unsigned int v9; // ebx
  __int64 **v10; // rax
  __int64 v12; // r9
  unsigned __int8 v13; // al
  unsigned int v14; // eax
  __int64 v15; // rdi
  unsigned int v16; // r14d
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // r9
  _BYTE *v20; // r11
  unsigned int v21; // esi
  unsigned __int64 v22; // rcx
  bool v23; // zf
  unsigned int v24; // ecx
  unsigned __int64 v25; // rdx
  unsigned int v26; // eax
  __int64 v27; // rax
  __int64 v28; // r11
  __int64 v29; // r14
  char v30; // al
  unsigned __int8 v31; // al
  _BYTE *v32; // r15
  unsigned int v33; // eax
  unsigned __int64 v34; // rsi
  unsigned int v35; // r14d
  int v36; // eax
  unsigned int v37; // ecx
  unsigned __int64 v38; // rdx
  unsigned int v39; // eax
  __int64 v40; // rdx
  unsigned __int64 v41; // rax
  unsigned int v42; // ecx
  unsigned __int64 v43; // rdx
  unsigned int v44; // eax
  __int64 v45; // r14
  __int64 **v46; // rax
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  __int64 v49; // rax
  unsigned __int64 v50; // rdx
  __int64 v51; // [rsp-78h] [rbp-78h]
  _BYTE *v52; // [rsp-70h] [rbp-70h]
  _BYTE *v53; // [rsp-70h] [rbp-70h]
  _BYTE *v54; // [rsp-70h] [rbp-70h]
  __int64 v55; // [rsp-70h] [rbp-70h]
  _BYTE *v56; // [rsp-70h] [rbp-70h]
  __int64 v57; // [rsp-68h] [rbp-68h]
  __int64 v58; // [rsp-68h] [rbp-68h]
  _BYTE *v59; // [rsp-68h] [rbp-68h]
  __int64 v60; // [rsp-68h] [rbp-68h]
  __int64 v61; // [rsp-68h] [rbp-68h]
  __int64 v62; // [rsp-68h] [rbp-68h]
  _BYTE *v63; // [rsp-68h] [rbp-68h]
  _BYTE *v64; // [rsp-68h] [rbp-68h]
  _BYTE *v65; // [rsp-68h] [rbp-68h]
  __int64 v66; // [rsp-60h] [rbp-60h]
  unsigned __int64 v67; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v68; // [rsp-50h] [rbp-50h]
  unsigned __int64 v69; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v70; // [rsp-40h] [rbp-40h]

  if ( !a4 )
    return 0;
  v5 = (_BYTE *)a2;
  v9 = a4 - 1;
  if ( !(_BYTE)a5 )
  {
    v10 = sub_13D9330(36, a1, a2, a3, v9);
    if ( v10 && *((_BYTE *)v10 + 16) <= 0x10u )
      return sub_1596070(v10);
    return 0;
  }
  v12 = a1 + 24;
  v66 = *(_QWORD *)a1;
  v13 = *(_BYTE *)(a1 + 16);
  if ( v13 != 13 )
  {
    if ( *(_BYTE *)(v66 + 8) != 16 )
      goto LABEL_35;
    if ( v13 > 0x10u )
      goto LABEL_35;
    v49 = sub_15A1020(a1);
    v5 = (_BYTE *)a2;
    if ( !v49 || *(_BYTE *)(v49 + 16) != 13 )
      goto LABEL_35;
    v12 = v49 + 24;
  }
  v14 = *(_DWORD *)(v12 + 8);
  v15 = *(_QWORD *)v12;
  v16 = v14 - 1;
  if ( v14 <= 0x40 )
  {
    if ( v15 == 1LL << v16 )
      goto LABEL_35;
  }
  else if ( (*(_QWORD *)(v15 + 8LL * (v16 >> 6)) & (1LL << v16)) != 0 )
  {
    v52 = v5;
    v57 = v12;
    v17 = sub_16A58A0(v12);
    v12 = v57;
    v5 = v52;
    if ( v17 == v16 )
      goto LABEL_35;
  }
  v53 = v5;
  v58 = v12;
  sub_13A3E40((__int64)&v69, v12);
  v18 = sub_15A1070(v66, &v69);
  v19 = v58;
  v51 = v18;
  v20 = v53;
  if ( v70 > 0x40 && v69 )
  {
    j_j___libc_free_0_0(v69);
    v20 = v53;
    v19 = v58;
  }
  v21 = *(_DWORD *)(v19 + 8);
  if ( v21 <= 0x40 )
    v22 = *(_QWORD *)v19;
  else
    v22 = *(_QWORD *)(*(_QWORD *)v19 + 8LL * ((v21 - 1) >> 6));
  v23 = (v22 & (1LL << ((unsigned __int8)v21 - 1))) == 0;
  v24 = *(_DWORD *)(v19 + 8);
  if ( !v23 )
  {
    v70 = *(_DWORD *)(v19 + 8);
    if ( v24 > 0x40 )
    {
      v65 = v20;
      sub_16A4FD0(&v69, v19);
      LOBYTE(v24) = v70;
      v20 = v65;
      if ( v70 > 0x40 )
      {
        sub_16A8F40(&v69);
        v20 = v65;
        goto LABEL_19;
      }
      v25 = v69;
    }
    else
    {
      v25 = *(_QWORD *)v19;
    }
    v69 = ~v25 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v24);
LABEL_19:
    v59 = v20;
    sub_16A7400(&v69);
    v24 = v70;
    v20 = v59;
    v68 = v70;
    v67 = v69;
    goto LABEL_20;
  }
  v68 = *(_DWORD *)(v19 + 8);
  if ( v24 <= 0x40 )
  {
    v50 = *(_QWORD *)v19;
LABEL_89:
    v67 = ~v50 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v24);
    goto LABEL_22;
  }
  v56 = v20;
  sub_16A4FD0(&v67, v19);
  v24 = v68;
  v20 = v56;
LABEL_20:
  if ( v24 <= 0x40 )
  {
    v50 = v67;
    goto LABEL_89;
  }
  v54 = v20;
  sub_16A8F40(&v67);
  v20 = v54;
LABEL_22:
  v60 = (__int64)v20;
  sub_16A7400(&v67);
  v26 = v68;
  v68 = 0;
  v70 = v26;
  v69 = v67;
  v27 = sub_15A1070(v66, &v69);
  v28 = v60;
  v29 = v27;
  if ( v70 > 0x40 && v69 )
  {
    j_j___libc_free_0_0(v69);
    v28 = v60;
  }
  if ( v68 > 0x40 && v67 )
  {
    v61 = v28;
    j_j___libc_free_0_0(v67);
    v28 = v61;
  }
  v62 = v28;
  if ( (unsigned __int8)sub_13DB920(40, v28, v29, a3, v9) )
    return a5;
  v30 = sub_13DB920(38, v62, v51, a3, v9);
  v5 = (_BYTE *)v62;
  if ( v30 )
    return a5;
LABEL_35:
  v31 = v5[16];
  v32 = v5 + 24;
  if ( v31 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) != 16 )
      return 0;
    if ( v31 > 0x10u )
      return 0;
    v64 = v5;
    v47 = sub_15A1020(v5);
    if ( !v47 || *(_BYTE *)(v47 + 16) != 13 )
      return 0;
    v5 = v64;
    v32 = (_BYTE *)(v47 + 24);
  }
  v33 = *((_DWORD *)v32 + 2);
  v34 = *(_QWORD *)v32;
  v35 = v33 - 1;
  if ( v33 > 0x40 )
  {
    v63 = v5;
    if ( (*(_QWORD *)(v34 + 8LL * (v35 >> 6)) & (1LL << v35)) == 0 )
      goto LABEL_38;
    v36 = sub_16A58A0(v32);
    v5 = v63;
    if ( v36 != v35 )
    {
LABEL_44:
      v37 = *((_DWORD *)v32 + 2);
      v70 = v37;
      if ( v37 > 0x40 )
      {
        sub_16A4FD0(&v69, v32);
        LOBYTE(v37) = v70;
        if ( v70 > 0x40 )
        {
          sub_16A8F40(&v69);
          goto LABEL_47;
        }
        v38 = v69;
      }
      else
      {
        v38 = *(_QWORD *)v32;
      }
      v69 = ~v38 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v37);
LABEL_47:
      sub_16A7400(&v69);
      v68 = v70;
      v67 = v69;
      goto LABEL_48;
    }
    return sub_13DB920(33, a1, (__int64)v5, a3, v9);
  }
  if ( 1LL << v35 == v34 )
    return sub_13DB920(33, a1, (__int64)v5, a3, v9);
  if ( ((1LL << v35) & v34) != 0 )
    goto LABEL_44;
LABEL_38:
  v68 = *((_DWORD *)v32 + 2);
  if ( v68 > 0x40 )
    sub_16A4FD0(&v67, v32);
  else
    v67 = *(_QWORD *)v32;
LABEL_48:
  v55 = sub_15A1070(v66, &v67);
  if ( v68 > 0x40 && v67 )
    j_j___libc_free_0_0(v67);
  v39 = *((_DWORD *)v32 + 2);
  v40 = 1LL << ((unsigned __int8)v39 - 1);
  if ( v39 > 0x40 )
    v41 = *(_QWORD *)(*(_QWORD *)v32 + 8LL * ((v39 - 1) >> 6));
  else
    v41 = *(_QWORD *)v32;
  v42 = *((_DWORD *)v32 + 2);
  if ( (v41 & v40) != 0 )
  {
    v70 = *((_DWORD *)v32 + 2);
    if ( v42 > 0x40 )
    {
      sub_16A4FD0(&v69, v32);
      LOBYTE(v42) = v70;
      if ( v70 > 0x40 )
      {
        sub_16A8F40(&v69);
        goto LABEL_57;
      }
      v43 = v69;
    }
    else
    {
      v43 = *(_QWORD *)v32;
    }
    v69 = ~v43 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v42);
LABEL_57:
    sub_16A7400(&v69);
    v42 = v70;
    v68 = v70;
    v67 = v69;
    goto LABEL_58;
  }
  v68 = *((_DWORD *)v32 + 2);
  if ( v42 <= 0x40 )
  {
    v48 = *(_QWORD *)v32;
LABEL_82:
    v67 = ~v48 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v42);
    goto LABEL_60;
  }
  sub_16A4FD0(&v67, v32);
  v42 = v68;
LABEL_58:
  if ( v42 <= 0x40 )
  {
    v48 = v67;
    goto LABEL_82;
  }
  sub_16A8F40(&v67);
LABEL_60:
  sub_16A7400(&v67);
  v44 = v68;
  v68 = 0;
  v70 = v44;
  v69 = v67;
  v45 = sub_15A1070(v66, &v69);
  if ( v70 > 0x40 && v69 )
    j_j___libc_free_0_0(v69);
  if ( v68 > 0x40 && v67 )
    j_j___libc_free_0_0(v67);
  v46 = sub_13D9330(38, a1, v45, a3, v9);
  if ( !v46 || *((_BYTE *)v46 + 16) > 0x10u || !(unsigned __int8)sub_1596070(v46) )
    return 0;
  return sub_13DB920(40, a1, v55, a3, v9);
}
