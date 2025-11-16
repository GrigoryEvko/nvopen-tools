// Function: sub_EF4820
// Address: 0xef4820
//
__int64 __fastcall sub_EF4820(__int64 a1, __int64 *a2, unsigned __int64 a3, unsigned __int64 a4)
{
  char v6; // r12
  char *v7; // rdx
  char *v8; // rax
  char *v9; // rax
  char v10; // al
  char v11; // r14
  __int64 v12; // r14
  __int64 v13; // r15
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r9
  __int64 v17; // r8
  char *v18; // rax
  unsigned __int64 *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r15
  char v22; // r14
  _QWORD *v23; // rax
  unsigned __int64 v24; // r14
  __int64 *v25; // rax
  __int64 v26; // r8
  _QWORD *v27; // r9
  unsigned __int64 v28; // rax
  _QWORD *v29; // rax
  __int64 *v30; // rax
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rax
  __int64 v33; // r9
  __int64 *v34; // r14
  char v35; // r12
  __int64 *v36; // rsi
  _QWORD *v37; // rax
  __int64 v38; // r8
  __int64 *v39; // rax
  __int64 v40; // rax
  __int64 v42; // rdx
  char *v43; // rax
  char v44; // cl
  __int64 v45; // rdi
  __int64 v46; // r9
  int v47; // edx
  char v48; // r12
  __int64 v49; // rax
  __int64 v50; // r12
  void *v51; // rax
  __int64 v52; // rax
  __int64 v53; // rsi
  __int64 v54; // rax
  char v55; // al
  _QWORD *v56; // rax
  unsigned __int64 v57; // r8
  __int64 v58; // rdx
  __int64 *v59; // rax
  unsigned __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rsi
  unsigned __int64 v63; // r15
  _BYTE *v64; // rcx
  char v65; // si
  __int64 v66; // rcx
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rsi
  unsigned __int64 v70; // r15
  int v71; // edx
  char v72; // [rsp+Fh] [rbp-101h]
  unsigned __int64 v73; // [rsp+10h] [rbp-100h]
  unsigned __int64 *v74; // [rsp+20h] [rbp-F0h]
  char v75; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v76; // [rsp+20h] [rbp-F0h]
  __int64 v77; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v78; // [rsp+28h] [rbp-E8h]
  _QWORD *v79; // [rsp+28h] [rbp-E8h]
  __int64 v80; // [rsp+28h] [rbp-E8h]
  __int64 v81; // [rsp+28h] [rbp-E8h]
  _QWORD *v82; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v83; // [rsp+28h] [rbp-E8h]
  __int64 v84; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v85; // [rsp+30h] [rbp-E0h] BYREF
  unsigned __int64 v86; // [rsp+38h] [rbp-D8h] BYREF
  __int64 *v87; // [rsp+48h] [rbp-C8h] BYREF
  _QWORD *v88; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v89; // [rsp+58h] [rbp-B8h]
  _QWORD v90[22]; // [rsp+60h] [rbp-B0h] BYREF

  v86 = a3;
  v85 = a4;
  if ( (unsigned __int8)sub_EE9010(a1, (__int64 *)&v85) )
    return 0;
  v6 = 0;
  v7 = *(char **)(a1 + 8);
  v8 = *(char **)a1;
  if ( !v86 )
  {
LABEL_52:
    if ( v8 == v7 )
      goto LABEL_9;
    if ( *v8 != 76 )
      goto LABEL_6;
LABEL_54:
    v9 = v8 + 1;
    *(_QWORD *)a1 = v9;
    if ( v9 == v7 )
      goto LABEL_9;
    goto LABEL_7;
  }
  if ( v8 == v7 )
    goto LABEL_9;
  if ( *v8 == 70 )
  {
    ++v8;
    v6 = 1;
    *(_QWORD *)a1 = v8;
    goto LABEL_52;
  }
  if ( *v8 == 76 )
    goto LABEL_54;
LABEL_6:
  v9 = *(char **)a1;
LABEL_7:
  v10 = *v9;
  if ( (unsigned __int8)(v10 - 49) <= 8u )
  {
    v24 = sub_EE6C50((__int64 *)a1);
LABEL_22:
    if ( !v24 )
      return 0;
    if ( v85 )
    {
      v75 = *(_BYTE *)(a1 + 937);
      v89 = 0x2000000000LL;
      v88 = v90;
      sub_EE40D0((__int64)&v88, 0x1Cu, v85, v24, v26, (__int64)v27);
      v29 = sub_C65B40(a1 + 904, (__int64)&v88, (__int64 *)&v87, (__int64)off_497B2F0);
      if ( v29 )
      {
        v24 = (unsigned __int64)(v29 + 1);
        if ( v88 != v90 )
          _libc_free(v88, &v88);
        v88 = (_QWORD *)v24;
        v30 = sub_EE6840(a1 + 944, (__int64 *)&v88);
        if ( v30 )
        {
          v31 = v30[1];
          if ( v31 )
            v24 = v31;
        }
        if ( *(_QWORD *)(a1 + 928) == v24 )
          *(_BYTE *)(a1 + 936) = 1;
        goto LABEL_32;
      }
      if ( !v75 )
      {
        v45 = (__int64)v88;
        if ( v88 != v90 )
        {
LABEL_49:
          _libc_free(v45, &v88);
          *(_QWORD *)(a1 + 920) = 0;
          return 0;
        }
LABEL_71:
        *(_QWORD *)(a1 + 920) = 0;
        return 0;
      }
      v61 = sub_CD1D40((__int64 *)(a1 + 808), 40, 3);
      *(_QWORD *)v61 = 0;
      v62 = v61;
      v63 = v61 + 8;
      *(_WORD *)(v61 + 16) = 16412;
      LOBYTE(v61) = *(_BYTE *)(v61 + 18);
      *(_QWORD *)(v62 + 32) = v24;
      *(_BYTE *)(v62 + 18) = v61 & 0xF0 | 5;
      *(_QWORD *)(v62 + 8) = &unk_49DF7E8;
      *(_QWORD *)(v62 + 24) = v85;
      sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v62, v87, (__int64)off_497B2F0);
      if ( v88 != v90 )
        _libc_free(v88, v62);
      *(_QWORD *)(a1 + 920) = v63;
      v24 = v63;
    }
LABEL_32:
    v32 = sub_EE9860(a1, v24);
    v17 = v32;
    if ( v32 )
    {
      if ( v6 )
      {
        v34 = (__int64 *)(a1 + 904);
        v78 = v32;
        v89 = 0x2000000000LL;
        v48 = *(_BYTE *)(a1 + 937);
        v88 = v90;
        sub_EE40D0((__int64)&v88, 0x19u, v86, v32, v32, v33);
        v36 = (__int64 *)&v88;
        v37 = sub_C65B40(a1 + 904, (__int64)&v88, (__int64 *)&v87, (__int64)off_497B2F0);
        if ( v37 )
        {
LABEL_36:
          v38 = (__int64)(v37 + 1);
          if ( v88 != v90 )
          {
            v79 = v37 + 1;
            _libc_free(v88, &v88);
            v38 = (__int64)v79;
          }
          v88 = (_QWORD *)v38;
          v80 = v38;
          v39 = sub_EE6840(a1 + 944, (__int64 *)&v88);
          v17 = v80;
          if ( v39 )
          {
            v40 = v39[1];
            if ( v40 )
              v17 = v40;
          }
          if ( *(_QWORD *)(a1 + 928) == v17 )
            *(_BYTE *)(a1 + 936) = 1;
          return v17;
        }
        if ( !v48 )
          goto LABEL_78;
        v49 = sub_CD1D40((__int64 *)(a1 + 808), 40, 3);
        *(_QWORD *)v49 = 0;
        v36 = (__int64 *)v49;
        v50 = v49 + 8;
        *(_WORD *)(v49 + 16) = 16409;
        *(_BYTE *)(v49 + 18) = *(_BYTE *)(v49 + 18) & 0xF0 | 5;
        v51 = &unk_49DF718;
      }
      else
      {
        if ( !v86 )
          return v17;
        v34 = (__int64 *)(a1 + 904);
        v78 = v32;
        v89 = 0x2000000000LL;
        v35 = *(_BYTE *)(a1 + 937);
        v88 = v90;
        sub_EE40D0((__int64)&v88, 0x18u, v86, v32, v32, v33);
        v36 = (__int64 *)&v88;
        v37 = sub_C65B40(a1 + 904, (__int64)&v88, (__int64 *)&v87, (__int64)off_497B2F0);
        if ( v37 )
          goto LABEL_36;
        if ( !v35 )
        {
LABEL_78:
          v17 = 0;
          goto LABEL_79;
        }
        v54 = sub_CD1D40((__int64 *)(a1 + 808), 40, 3);
        *(_QWORD *)v54 = 0;
        v36 = (__int64 *)v54;
        v50 = v54 + 8;
        *(_WORD *)(v54 + 16) = 16408;
        *(_BYTE *)(v54 + 18) = *(_BYTE *)(v54 + 18) & 0xF0 | 5;
        v51 = &unk_49DF6B8;
      }
      v36[1] = (__int64)v51 + 16;
      v36[3] = v86;
      v36[4] = v78;
      sub_C657C0(v34, v36, v87, (__int64)off_497B2F0);
      v17 = v50;
LABEL_79:
      if ( v88 != v90 )
      {
        v81 = v17;
        _libc_free(v88, v36);
        v17 = v81;
      }
      *(_QWORD *)(a1 + 920) = v17;
      return v17;
    }
    return 0;
  }
  if ( v10 == 85 )
  {
    v24 = sub_EEDD80((char **)a1, (__int64)a2);
    goto LABEL_22;
  }
LABEL_9:
  v11 = sub_EE3B50((const void **)a1, 2u, "DC");
  if ( v11 )
  {
    v12 = *(_QWORD *)(a1 + 24);
    v13 = *(_QWORD *)(a1 + 16);
    while ( 1 )
    {
      v88 = (_QWORD *)sub_EE6C50((__int64 *)a1);
      v17 = (__int64)v88;
      if ( !v88 )
        return v17;
      sub_E18380(a1 + 16, (__int64 *)&v88, v14, v15, (__int64)v88, v16);
      v18 = *(char **)a1;
      if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) && *v18 == 69 )
      {
        *(_QWORD *)a1 = v18 + 1;
        v19 = (unsigned __int64 *)sub_EE6060((_QWORD *)a1, (v12 - v13) >> 3);
        v21 = v20;
        v88 = v90;
        v22 = *(_BYTE *)(a1 + 937);
        v74 = v19;
        v89 = 0x2000000000LL;
        sub_EE4780((__int64)&v88, 0x35u, v74, v20, (__int64)v74, (__int64)v90);
        v23 = sub_C65B40(a1 + 904, (__int64)&v88, (__int64 *)&v87, (__int64)off_497B2F0);
        if ( v23 )
        {
          v24 = (unsigned __int64)(v23 + 1);
          if ( v88 != v90 )
            _libc_free(v88, &v88);
          v88 = (_QWORD *)v24;
          v25 = sub_EE6840(a1 + 944, (__int64 *)&v88);
          if ( v25 )
          {
            v28 = v25[1];
            if ( v28 )
              v24 = v28;
          }
          if ( *(_QWORD *)(a1 + 928) == v24 )
            *(_BYTE *)(a1 + 936) = 1;
          goto LABEL_22;
        }
        if ( v22 )
        {
          v52 = sub_CD1D40((__int64 *)(a1 + 808), 40, 3);
          *(_WORD *)(v52 + 16) = 16437;
          v53 = v52;
          v24 = v52 + 8;
          *(_QWORD *)v52 = 0;
          LOBYTE(v52) = *(_BYTE *)(v52 + 18);
          *(_QWORD *)(v53 + 24) = v74;
          *(_QWORD *)(v53 + 32) = v21;
          *(_BYTE *)(v53 + 18) = v52 & 0xF0 | 5;
          *(_QWORD *)(v53 + 8) = &unk_49E01A8;
          sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v53, v87, (__int64)off_497B2F0);
          v27 = v90;
          if ( v88 != v90 )
            _libc_free(v88, v53);
          *(_QWORD *)(a1 + 920) = v24;
          goto LABEL_22;
        }
        v45 = (__int64)v88;
        if ( v88 != v90 )
          goto LABEL_49;
        goto LABEL_71;
      }
    }
  }
  v42 = *(_QWORD *)(a1 + 8);
  v43 = *(char **)a1;
  if ( v42 == *(_QWORD *)a1 || (v44 = *v43, (unsigned __int8)(*v43 - 67) > 1u) )
  {
    v24 = sub_EF3FC0(a1, a2);
    goto LABEL_22;
  }
  v46 = v86;
  if ( !v86 )
    return 0;
  v17 = v85;
  if ( v85 )
    return 0;
  if ( *(_BYTE *)(v86 + 8) == 48 )
  {
    v55 = *(_BYTE *)(a1 + 937);
    v73 = v85;
    v90[1] = v86;
    v72 = v55;
    v88 = v90;
    v76 = v86;
    v89 = 0x2000000004LL;
    v90[0] = 47;
    v56 = sub_C65B40(a1 + 904, (__int64)&v88, (__int64 *)&v87, (__int64)off_497B2F0);
    v57 = v73;
    if ( v56 )
    {
      v58 = (__int64)(v56 + 1);
      if ( v88 != v90 )
      {
        v82 = v56 + 1;
        _libc_free(v88, &v88);
        v58 = (__int64)v82;
        v57 = v73;
      }
      v77 = v57;
      v88 = (_QWORD *)v58;
      v83 = v58;
      v59 = sub_EE6840(a1 + 944, (__int64 *)&v88);
      v17 = v77;
      if ( v59 )
      {
        v60 = v59[1];
        if ( !v60 )
          v60 = v83;
      }
      else
      {
        v60 = v83;
      }
      if ( *(_QWORD *)(a1 + 928) == v60 )
        *(_BYTE *)(a1 + 936) = 1;
      v86 = v60;
    }
    else
    {
      if ( !v72 )
      {
        v45 = (__int64)v88;
        if ( v88 != v90 )
          goto LABEL_49;
        goto LABEL_71;
      }
      v68 = sub_CD1D40((__int64 *)(a1 + 808), 24, 3);
      *(_QWORD *)v68 = 0;
      v69 = v68;
      v70 = v68 + 8;
      v71 = *(_DWORD *)(v76 + 12);
      *(_WORD *)(v68 + 16) = 16431;
      LOBYTE(v68) = *(_BYTE *)(v68 + 18);
      *(_DWORD *)(v69 + 20) = v71;
      *(_BYTE *)(v69 + 18) = v68 & 0xF0 | 5;
      *(_QWORD *)(v69 + 8) = &unk_49DFF68;
      sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v69, v87, (__int64)off_497B2F0);
      v17 = v73;
      if ( v88 != v90 )
      {
        _libc_free(v88, v69);
        v17 = v73;
      }
      *(_QWORD *)(a1 + 920) = v70;
      v86 = v70;
    }
    v43 = *(char **)a1;
    v42 = *(_QWORD *)(a1 + 8);
    if ( *(_QWORD *)a1 == v42 )
      return 0;
    v44 = *v43;
  }
  if ( v44 != 67 )
  {
    if ( v44 != 68 )
      return 0;
    if ( v42 - (_QWORD)v43 == 1 )
      return 0;
    v47 = v43[1];
    if ( (unsigned __int8)(v43[1] - 48) > 2u && (unsigned __int8)(v47 - 52) > 1u )
      return 0;
    LODWORD(v88) = v47 - 48;
    *(_QWORD *)a1 = v43 + 2;
    if ( a2 )
      *(_BYTE *)a2 = 1;
    LOBYTE(v87) = 1;
    goto LABEL_68;
  }
  v64 = v43 + 1;
  *(_QWORD *)a1 = v43 + 1;
  if ( v43 + 1 != (char *)v42 )
  {
    v65 = v43[1];
    if ( v65 == 73 )
    {
      v64 = v43 + 2;
      *(_QWORD *)a1 = v43 + 2;
      if ( v43 + 2 == (char *)v42 )
        return v17;
      v65 = v43[2];
      v11 = 1;
    }
    if ( (unsigned __int8)(v65 - 49) <= 4u )
    {
      v66 = (__int64)(v64 + 1);
      *(_QWORD *)a1 = v66;
      LODWORD(v88) = (char)(v65 - 48);
      if ( a2 )
        *(_BYTE *)a2 = 1;
      v84 = v17;
      if ( !v11 || (v67 = sub_EF1680(a1, a2, v42, v66, v17, v46), v17 = v84, v67) )
      {
        LOBYTE(v87) = 0;
LABEL_68:
        v24 = sub_EE9E40(a1 + 808, (__int64 *)&v86, (unsigned __int8 *)&v87, (int *)&v88, v17, v46);
        goto LABEL_22;
      }
    }
  }
  return v17;
}
