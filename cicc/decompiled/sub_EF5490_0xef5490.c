// Function: sub_EF5490
// Address: 0xef5490
//
__int64 __fastcall sub_EF5490(__int64 a1)
{
  unsigned int v2; // ebx
  __int64 v3; // r12
  __int64 v4; // r8
  __int64 v5; // r9
  _BYTE *v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // r8
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r9
  __int64 v13; // r14
  _BYTE *v14; // rax
  char v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  _QWORD *v20; // rax
  __int64 v21; // rdi
  __int64 *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // r12
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  _BYTE *v35; // rax
  unsigned __int64 *v36; // rax
  __int64 v37; // rdx
  __int64 v38; // r14
  char v39; // r12
  _QWORD *v40; // rax
  __int64 v41; // r14
  _BYTE *v42; // rax
  _BYTE *v43; // rcx
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  int v52; // r9d
  _BYTE *v53; // rax
  __int64 v54; // r9
  unsigned __int64 v55; // rdx
  _BYTE **v56; // rsi
  _QWORD *v57; // rax
  __int64 v58; // r8
  __int64 *v59; // rax
  __int64 v60; // rax
  __int64 v61; // rdi
  __int64 v62; // rax
  __int64 v63; // rsi
  __int64 v64; // rdi
  __int64 v65; // rax
  __int64 v66; // r8
  __int64 v67; // rax
  _BYTE *v68; // [rsp+0h] [rbp-100h]
  char v69; // [rsp+Fh] [rbp-F1h]
  char v70; // [rsp+10h] [rbp-F0h]
  unsigned __int64 *v71; // [rsp+20h] [rbp-E0h]
  unsigned int v72; // [rsp+20h] [rbp-E0h]
  _BYTE *v73; // [rsp+20h] [rbp-E0h]
  __int64 v74; // [rsp+28h] [rbp-D8h]
  _QWORD *v75; // [rsp+28h] [rbp-D8h]
  __int64 v76; // [rsp+28h] [rbp-D8h]
  _QWORD *v77; // [rsp+28h] [rbp-D8h]
  _QWORD *v78; // [rsp+28h] [rbp-D8h]
  __int64 *v79; // [rsp+38h] [rbp-C8h] BYREF
  _BYTE *v80; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v81; // [rsp+48h] [rbp-B8h]
  _BYTE v82[176]; // [rsp+50h] [rbp-B0h] BYREF

  v2 = sub_EE3340(a1);
  if ( (unsigned __int8)sub_EE3B50((const void **)a1, 2u, "Do") )
  {
    v3 = sub_EE68C0(a1 + 808, "noexcept");
    if ( v3 )
      goto LABEL_3;
    return 0;
  }
  if ( (unsigned __int8)sub_EE3B50((const void **)a1, 2u, "DO") )
  {
    v13 = sub_EEA9F0(a1);
    if ( !v13 )
      return 0;
    v14 = *(_BYTE **)a1;
    if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) || *v14 != 69 )
      return 0;
    *(_QWORD *)a1 = v14 + 1;
    v15 = *(_BYTE *)(a1 + 937);
    v80 = v82;
    v81 = 0x2000000000LL;
    sub_D953B0((__int64)&v80, 17, v10, v11, (__int64)v82, v12);
    sub_D953B0((__int64)&v80, v13, v16, v17, v18, v19);
    v20 = sub_C65B40(a1 + 904, (__int64)&v80, (__int64 *)&v79, (__int64)off_497B2F0);
    if ( v20 )
    {
      v21 = (__int64)v80;
      v3 = (__int64)(v20 + 1);
      if ( v80 != v82 )
LABEL_13:
        _libc_free(v21, &v80);
      goto LABEL_14;
    }
    if ( !v15 )
    {
      v61 = (__int64)v80;
      if ( v80 == v82 )
        goto LABEL_63;
      goto LABEL_55;
    }
    v67 = sub_CD1D40((__int64 *)(a1 + 808), 32, 3);
    *(_WORD *)(v67 + 16) = 16401;
    v63 = v67;
    v3 = v67 + 8;
    *(_QWORD *)v67 = 0;
    LOBYTE(v67) = *(_BYTE *)(v67 + 18);
    *(_QWORD *)(v63 + 24) = v13;
    *(_BYTE *)(v63 + 18) = v67 & 0xF0 | 5;
    *(_QWORD *)(v63 + 8) = &unk_49DF428;
    sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v63, v79, (__int64)off_497B2F0);
    v64 = (__int64)v80;
    if ( v80 == v82 )
      goto LABEL_66;
    goto LABEL_65;
  }
  v24 = 2;
  if ( !(unsigned __int8)sub_EE3B50((const void **)a1, 2u, &unk_3F7C044) )
  {
    v3 = 0;
    goto LABEL_3;
  }
  v29 = (__int64)(*(_QWORD *)(a1 + 24) - *(_QWORD *)(a1 + 16)) >> 3;
  while ( 1 )
  {
    v35 = *(_BYTE **)a1;
    if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) && *v35 == 69 )
      break;
    v30 = sub_EF1F20(a1, v24, v25, v26, v27, v28);
    v80 = (_BYTE *)v30;
    if ( !v30 )
      return v30;
    v24 = (__int64)&v80;
    sub_E18380(a1 + 16, (__int64 *)&v80, v31, v32, v33, v34);
  }
  *(_QWORD *)a1 = v35 + 1;
  v36 = (unsigned __int64 *)sub_EE6060((_QWORD *)a1, v29);
  v38 = v37;
  v80 = v82;
  v39 = *(_BYTE *)(a1 + 937);
  v71 = v36;
  v81 = 0x2000000000LL;
  sub_EE4780((__int64)&v80, 0x12u, v71, v37, (__int64)v71, (__int64)v82);
  v40 = sub_C65B40(a1 + 904, (__int64)&v80, (__int64 *)&v79, (__int64)off_497B2F0);
  if ( !v40 )
  {
    if ( !v39 )
    {
      v61 = (__int64)v80;
      if ( v80 == v82 )
      {
LABEL_63:
        *(_QWORD *)(a1 + 920) = 0;
        return 0;
      }
LABEL_55:
      _libc_free(v61, &v80);
      *(_QWORD *)(a1 + 920) = 0;
      return 0;
    }
    v62 = sub_CD1D40((__int64 *)(a1 + 808), 40, 3);
    *(_WORD *)(v62 + 16) = 16402;
    v63 = v62;
    v3 = v62 + 8;
    *(_QWORD *)v62 = 0;
    LOBYTE(v62) = *(_BYTE *)(v62 + 18);
    *(_QWORD *)(v63 + 24) = v71;
    *(_QWORD *)(v63 + 32) = v38;
    *(_BYTE *)(v63 + 18) = v62 & 0xF0 | 5;
    *(_QWORD *)(v63 + 8) = &unk_49DF488;
    sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v63, v79, (__int64)off_497B2F0);
    v64 = (__int64)v80;
    if ( v80 == v82 )
    {
LABEL_66:
      *(_QWORD *)(a1 + 920) = v3;
      goto LABEL_3;
    }
LABEL_65:
    _libc_free(v64, v63);
    goto LABEL_66;
  }
  v21 = (__int64)v80;
  v3 = (__int64)(v40 + 1);
  if ( v80 != v82 )
    goto LABEL_13;
LABEL_14:
  v80 = (_BYTE *)v3;
  v22 = sub_EE6840(a1 + 944, (__int64 *)&v80);
  if ( v22 )
  {
    v23 = v22[1];
    if ( v23 )
      v3 = v23;
  }
  if ( *(_QWORD *)(a1 + 928) == v3 )
    *(_BYTE *)(a1 + 936) = 1;
LABEL_3:
  sub_EE3B50((const void **)a1, 2u, &unk_3F7C047);
  v6 = *(_BYTE **)a1;
  v7 = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)a1 == v7 || *v6 != 70 )
    return 0;
  *(_QWORD *)a1 = v6 + 1;
  if ( (_BYTE *)v7 != v6 + 1 && v6[1] == 89 )
    *(_QWORD *)a1 = v6 + 2;
  v41 = sub_EF1F20(a1, 2, v7, (__int64)(v6 + 1), v4, v5);
  if ( !v41 )
    return 0;
  v42 = *(_BYTE **)a1;
  v43 = *(_BYTE **)(a1 + 8);
  v74 = (__int64)(*(_QWORD *)(a1 + 24) - *(_QWORD *)(a1 + 16)) >> 3;
  while ( 1 )
  {
    if ( v43 == v42 )
      goto LABEL_37;
LABEL_35:
    if ( *v42 == 69 )
    {
      v69 = 0;
      v52 = 0;
      *(_QWORD *)a1 = v42 + 1;
      goto LABEL_43;
    }
    if ( *v42 != 118 )
      break;
    *(_QWORD *)a1 = ++v42;
  }
  while ( 1 )
  {
LABEL_37:
    if ( (unsigned __int8)sub_EE3B50((const void **)a1, 2u, "RE") )
    {
      v69 = 1;
      v52 = 1;
      goto LABEL_43;
    }
    if ( (unsigned __int8)sub_EE3B50((const void **)a1, 2u, "OE") )
      break;
    v30 = sub_EF1F20(a1, 2, v44, v45, v46, v47);
    v80 = (_BYTE *)v30;
    if ( !v30 )
      return v30;
    sub_E18380(a1 + 16, (__int64 *)&v80, v48, v49, v50, v51);
    v42 = *(_BYTE **)a1;
    v43 = *(_BYTE **)(a1 + 8);
    if ( v43 != *(_BYTE **)a1 )
      goto LABEL_35;
  }
  v69 = 2;
  v52 = 2;
LABEL_43:
  v72 = v52;
  v53 = sub_EE6060((_QWORD *)a1, v74);
  v54 = v72;
  v73 = (_BYTE *)v55;
  v70 = *(_BYTE *)(a1 + 937);
  v81 = 0x2000000000LL;
  v80 = v82;
  v68 = v53;
  sub_EE45C0((__int64)&v80, v41, (__int64)v53, v55, v2, v54, v3);
  v56 = &v80;
  v57 = sub_C65B40(a1 + 904, (__int64)&v80, (__int64 *)&v79, (__int64)off_497B2F0);
  v8 = v57;
  if ( v57 )
  {
    v58 = (__int64)(v57 + 1);
    if ( v80 != v82 )
    {
      v75 = v57 + 1;
      _libc_free(v80, &v80);
      v58 = (__int64)v75;
    }
    v80 = (_BYTE *)v58;
    v76 = v58;
    v59 = sub_EE6840(a1 + 944, (__int64 *)&v80);
    v8 = (_QWORD *)v76;
    if ( v59 )
    {
      v60 = v59[1];
      if ( v60 )
        v8 = (_QWORD *)v60;
    }
    if ( *(_QWORD **)(a1 + 928) == v8 )
      *(_BYTE *)(a1 + 936) = 1;
  }
  else
  {
    if ( v70 )
    {
      v65 = sub_CD1D40((__int64 *)(a1 + 808), 64, 3);
      *(_QWORD *)v65 = 0;
      v56 = (_BYTE **)v65;
      v66 = v65 + 8;
      *(_WORD *)(v65 + 16) = 16;
      LOBYTE(v65) = *(_BYTE *)(v65 + 18);
      v56[3] = (_BYTE *)v41;
      v56[5] = v73;
      *((_DWORD *)v56 + 12) = v2;
      *((_BYTE *)v56 + 18) = v65 & 0xF0 | 1;
      v56[7] = (_BYTE *)v3;
      v78 = (_QWORD *)v66;
      v56[1] = &unk_49DF3C8;
      v56[4] = v68;
      *((_BYTE *)v56 + 52) = v69;
      sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v56, v79, (__int64)off_497B2F0);
      v8 = v78;
    }
    if ( v80 != v82 )
    {
      v77 = v8;
      _libc_free(v80, v56);
      v8 = v77;
    }
    *(_QWORD *)(a1 + 920) = v8;
  }
  return (__int64)v8;
}
