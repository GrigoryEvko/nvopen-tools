// Function: sub_EEE940
// Address: 0xeee940
//
_QWORD *__fastcall sub_EEE940(__int64 a1)
{
  char *v1; // rax
  __int64 v2; // rdx
  _QWORD *v3; // r13
  unsigned __int64 v5; // r15
  __int64 v6; // rdx
  __int64 v7; // r9
  signed __int64 v8; // r10
  char *v9; // rax
  char v10; // al
  _BYTE **v11; // rsi
  _QWORD *v12; // rax
  _BYTE *v13; // r10
  __int64 *v14; // rax
  _QWORD *v15; // rax
  __int64 v16; // rsi
  char *v17; // rdx
  unsigned __int8 *v18; // r15
  __int64 v19; // r8
  _BYTE *v20; // rcx
  __int64 v21; // r9
  _BYTE *v22; // r15
  __int64 v23; // r8
  _BYTE *v24; // rcx
  __int64 v25; // r9
  __int64 v26; // rax
  _BYTE *v27; // r15
  __int64 v28; // r8
  _BYTE *v29; // rcx
  __int64 v30; // r9
  __int64 v31; // rax
  char *v32; // rax
  char *v33; // rax
  char *v34; // rcx
  char v35; // dl
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r15
  char *v40; // rax
  __int64 v41; // r9
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rax
  char *v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  char *v51; // rax
  __int64 v52; // r9
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  char v60; // [rsp+7h] [rbp-E9h]
  signed __int64 v61; // [rsp+8h] [rbp-E8h]
  _BYTE *v62; // [rsp+8h] [rbp-E8h]
  __int64 v63; // [rsp+10h] [rbp-E0h]
  char v64; // [rsp+10h] [rbp-E0h]
  char v65; // [rsp+10h] [rbp-E0h]
  char v66; // [rsp+10h] [rbp-E0h]
  char v67; // [rsp+10h] [rbp-E0h]
  char v68; // [rsp+10h] [rbp-E0h]
  _BYTE *v69; // [rsp+10h] [rbp-E0h]
  __int64 *v70; // [rsp+18h] [rbp-D8h]
  __int64 *v71; // [rsp+28h] [rbp-C8h] BYREF
  _BYTE *v72; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v73; // [rsp+38h] [rbp-B8h]
  _BYTE v74[176]; // [rsp+40h] [rbp-B0h] BYREF

  v1 = *(char **)a1;
  v2 = *(_QWORD *)(a1 + 8);
  if ( v2 == *(_QWORD *)a1 || *v1 != 76 )
    return 0;
  *(_QWORD *)a1 = v1 + 1;
  if ( v1 + 1 != (char *)v2 )
  {
    switch ( v1[1] )
    {
      case 'A':
        v39 = sub_EF1F20(a1);
        if ( !v39 )
          return 0;
        v40 = *(char **)a1;
        if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) || *v40 != 69 )
          return 0;
        v41 = *(unsigned __int8 *)(a1 + 937);
        *(_QWORD *)a1 = v40 + 1;
        v67 = v41;
        v73 = 0x2000000000LL;
        v72 = v74;
        sub_D953B0((__int64)&v72, 74, v36, v37, v38, v41);
        sub_D953B0((__int64)&v72, v39, v42, v43, v44, v45);
        v11 = &v72;
        v70 = (__int64 *)(a1 + 904);
        v3 = sub_C65B40(a1 + 904, (__int64)&v72, (__int64 *)&v71, (__int64)off_497B2F0);
        if ( v3 )
          goto LABEL_12;
        if ( !v67 )
          goto LABEL_40;
        v46 = sub_CD1D40((__int64 *)(a1 + 808), 32, 3);
        *(_QWORD *)v46 = 0;
        v11 = (_BYTE **)v46;
        v3 = (_QWORD *)(v46 + 8);
        *(_WORD *)(v46 + 16) = 16458;
        *(_BYTE *)(v46 + 18) = *(_BYTE *)(v46 + 18) & 0xF0 | 5;
        v47 = (char *)&unk_49E0A38;
        goto LABEL_81;
      case 'D':
        if ( !(unsigned __int8)sub_EE3B50((const void **)a1, 2u, "Dn") )
          return 0;
        v33 = *(char **)a1;
        v34 = *(char **)(a1 + 8);
        if ( *(char **)a1 == v34 )
          return 0;
        v35 = *v33;
        if ( *v33 != 48 )
          goto LABEL_73;
        *(_QWORD *)a1 = v33 + 1;
        if ( v34 == v33 + 1 )
          return 0;
        v35 = *++v33;
LABEL_73:
        if ( v35 != 69 )
          return 0;
        *(_QWORD *)a1 = v33 + 1;
        return (_QWORD *)sub_EE68C0(a1 + 808, "nullptr");
      case 'T':
        return 0;
      case 'U':
        if ( v2 - (_QWORD)(v1 + 1) == 1 )
          return 0;
        if ( v1[2] != 108 )
          return 0;
        v39 = sub_EEDD80((char **)a1, 0);
        if ( !v39 )
          return 0;
        v51 = *(char **)a1;
        if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) || *v51 != 69 )
          return 0;
        v52 = *(unsigned __int8 *)(a1 + 937);
        *(_QWORD *)a1 = v51 + 1;
        v68 = v52;
        v73 = 0x2000000000LL;
        v72 = v74;
        sub_D953B0((__int64)&v72, 75, v48, v49, v50, v52);
        sub_D953B0((__int64)&v72, v39, v53, v54, v55, v56);
        v11 = &v72;
        v70 = (__int64 *)(a1 + 904);
        v3 = sub_C65B40(a1 + 904, (__int64)&v72, (__int64 *)&v71, (__int64)off_497B2F0);
        if ( v3 )
          goto LABEL_12;
        if ( !v68 )
          goto LABEL_40;
        v57 = sub_CD1D40((__int64 *)(a1 + 808), 32, 3);
        *(_QWORD *)v57 = 0;
        v11 = (_BYTE **)v57;
        v3 = (_QWORD *)(v57 + 8);
        *(_WORD *)(v57 + 16) = 16459;
        *(_BYTE *)(v57 + 18) = *(_BYTE *)(v57 + 18) & 0xF0 | 5;
        v47 = (char *)&unk_49E0A98;
LABEL_81:
        v11[3] = (_BYTE *)v39;
        v11[1] = v47 + 16;
        goto LABEL_82;
      case '_':
        if ( !(unsigned __int8)sub_EE3B50((const void **)a1, 2u, &unk_3C1BC40) )
          return 0;
        v3 = (_QWORD *)sub_EF05F0(a1, 1);
        if ( !v3 )
          return 0;
        v32 = *(char **)a1;
        if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) || *v32 != 69 )
          return 0;
        *(_QWORD *)a1 = v32 + 1;
        return v3;
      case 'a':
        v16 = 11;
        v17 = "signed char";
        *(_QWORD *)a1 = v1 + 2;
        return sub_EE95D0(a1, v16, (unsigned __int8 *)v17);
      case 'b':
        if ( (unsigned __int8)sub_EE3B50((const void **)a1, 3u, "b0E") )
        {
          LODWORD(v72) = 0;
          return (_QWORD *)sub_EE8700(a1 + 808, (int *)&v72);
        }
        if ( !(unsigned __int8)sub_EE3B50((const void **)a1, 3u, "b1E") )
          return 0;
        LODWORD(v72) = 1;
        return (_QWORD *)sub_EE8700(a1 + 808, (int *)&v72);
      case 'c':
        v16 = 4;
        v17 = "char";
        *(_QWORD *)a1 = v1 + 2;
        return sub_EE95D0(a1, v16, (unsigned __int8 *)v17);
      case 'd':
        v27 = v1 + 2;
        v28 = (__int64)(v1 + 18);
        *(_QWORD *)a1 = v1 + 2;
        v29 = v1 + 2;
        if ( (unsigned __int64)(v2 - (_QWORD)(v1 + 2)) <= 0x10 )
          return 0;
        while ( (unsigned __int8)(*v29 - 48) <= 9u || (unsigned __int8)(*v29 - 97) <= 5u )
        {
          if ( (_BYTE *)v28 == ++v29 )
          {
            *(_QWORD *)a1 = v28;
            if ( v2 == v28 || v1[18] != 69 )
              return 0;
            v30 = *(unsigned __int8 *)(a1 + 937);
            *(_QWORD *)a1 = v1 + 19;
            v66 = v30;
            v73 = 0x2000000000LL;
            v72 = v74;
            sub_EE3C10((__int64)&v72, 0x4Fu, 16, (unsigned __int8 *)v1 + 2, v28, v30);
            v11 = &v72;
            v70 = (__int64 *)(a1 + 904);
            v3 = sub_C65B40(a1 + 904, (__int64)&v72, (__int64 *)&v71, (__int64)off_497B2F0);
            if ( v3 )
              goto LABEL_12;
            if ( v66 )
            {
              v31 = sub_CD1D40((__int64 *)(a1 + 808), 40, 3);
              *(_QWORD *)v31 = 0;
              v11 = (_BYTE **)v31;
              v3 = (_QWORD *)(v31 + 8);
              *(_WORD *)(v31 + 16) = 16463;
              LOBYTE(v31) = *(_BYTE *)(v31 + 18);
              v11[3] = (_BYTE *)16;
              v11[4] = v27;
              *((_BYTE *)v11 + 18) = v31 & 0xF0 | 5;
              v11[1] = &unk_49E0DA8;
              goto LABEL_82;
            }
            goto LABEL_40;
          }
        }
        return 0;
      case 'e':
        v22 = v1 + 2;
        v23 = (__int64)(v1 + 22);
        *(_QWORD *)a1 = v1 + 2;
        v24 = v1 + 2;
        if ( (unsigned __int64)(v2 - (_QWORD)(v1 + 2)) <= 0x14 )
          return 0;
        while ( (unsigned __int8)(*v24 - 48) <= 9u || (unsigned __int8)(*v24 - 97) <= 5u )
        {
          if ( (_BYTE *)v23 == ++v24 )
          {
            *(_QWORD *)a1 = v23;
            if ( v2 == v23 || v1[22] != 69 )
              return 0;
            v25 = *(unsigned __int8 *)(a1 + 937);
            *(_QWORD *)a1 = v1 + 23;
            v65 = v25;
            v73 = 0x2000000000LL;
            v72 = v74;
            sub_EE3C10((__int64)&v72, 0x50u, 20, (unsigned __int8 *)v1 + 2, v23, v25);
            v11 = &v72;
            v70 = (__int64 *)(a1 + 904);
            v3 = sub_C65B40(a1 + 904, (__int64)&v72, (__int64 *)&v71, (__int64)off_497B2F0);
            if ( v3 )
              goto LABEL_12;
            if ( v65 )
            {
              v26 = sub_CD1D40((__int64 *)(a1 + 808), 40, 3);
              *(_QWORD *)v26 = 0;
              v11 = (_BYTE **)v26;
              v3 = (_QWORD *)(v26 + 8);
              *(_WORD *)(v26 + 16) = 16464;
              LOBYTE(v26) = *(_BYTE *)(v26 + 18);
              v11[3] = (_BYTE *)20;
              v11[4] = v22;
              *((_BYTE *)v11 + 18) = v26 & 0xF0 | 5;
              v11[1] = &unk_49E0E08;
              goto LABEL_82;
            }
            goto LABEL_40;
          }
        }
        return 0;
      case 'f':
        v18 = (unsigned __int8 *)(v1 + 2);
        v19 = (__int64)(v1 + 10);
        *(_QWORD *)a1 = v1 + 2;
        v20 = v1 + 2;
        if ( (unsigned __int64)(v2 - (_QWORD)(v1 + 2)) <= 8 )
          return 0;
        break;
      case 'h':
        v16 = 13;
        v17 = "unsigned char";
        *(_QWORD *)a1 = v1 + 2;
        return sub_EE95D0(a1, v16, (unsigned __int8 *)v17);
      case 'i':
        v16 = 0;
        v17 = (char *)byte_3F871B3;
        *(_QWORD *)a1 = v1 + 2;
        return sub_EE95D0(a1, v16, (unsigned __int8 *)v17);
      case 'j':
        v16 = 1;
        v17 = (char *)"u";
        *(_QWORD *)a1 = v1 + 2;
        return sub_EE95D0(a1, v16, (unsigned __int8 *)v17);
      case 'l':
        v16 = 1;
        v17 = "l";
        *(_QWORD *)a1 = v1 + 2;
        return sub_EE95D0(a1, v16, (unsigned __int8 *)v17);
      case 'm':
        v16 = 2;
        v17 = "ul";
        *(_QWORD *)a1 = v1 + 2;
        return sub_EE95D0(a1, v16, (unsigned __int8 *)v17);
      case 'n':
        v16 = 8;
        v17 = "__int128";
        *(_QWORD *)a1 = v1 + 2;
        return sub_EE95D0(a1, v16, (unsigned __int8 *)v17);
      case 'o':
        v16 = 17;
        v17 = "unsigned __int128";
        *(_QWORD *)a1 = v1 + 2;
        return sub_EE95D0(a1, v16, (unsigned __int8 *)v17);
      case 's':
        v16 = 5;
        v17 = "short";
        *(_QWORD *)a1 = v1 + 2;
        return sub_EE95D0(a1, v16, (unsigned __int8 *)v17);
      case 't':
        v16 = 14;
        v17 = "unsigned short";
        *(_QWORD *)a1 = v1 + 2;
        return sub_EE95D0(a1, v16, (unsigned __int8 *)v17);
      case 'w':
        v16 = 7;
        v17 = "wchar_t";
        *(_QWORD *)a1 = v1 + 2;
        return sub_EE95D0(a1, v16, (unsigned __int8 *)v17);
      case 'x':
        v16 = 2;
        v17 = "ll";
        *(_QWORD *)a1 = v1 + 2;
        return sub_EE95D0(a1, v16, (unsigned __int8 *)v17);
      case 'y':
        v16 = 3;
        v17 = "ull";
        *(_QWORD *)a1 = v1 + 2;
        return sub_EE95D0(a1, v16, (unsigned __int8 *)v17);
      default:
        goto LABEL_7;
    }
    while ( (unsigned __int8)(*v20 - 48) <= 9u || (unsigned __int8)(*v20 - 97) <= 5u )
    {
      if ( ++v20 == (_BYTE *)v19 )
      {
        *(_QWORD *)a1 = v20;
        if ( (_BYTE *)v2 == v20 || v1[10] != 69 )
          return 0;
        v21 = *(unsigned __int8 *)(a1 + 937);
        *(_QWORD *)a1 = v1 + 11;
        v64 = v21;
        v73 = 0x2000000000LL;
        v72 = v74;
        sub_D953B0((__int64)&v72, 78, v2, (__int64)v20, v19, v21);
        sub_C653C0((__int64)&v72, v18, 8u);
        v11 = &v72;
        v70 = (__int64 *)(a1 + 904);
        v3 = sub_C65B40(a1 + 904, (__int64)&v72, (__int64 *)&v71, (__int64)off_497B2F0);
        if ( v3 )
          goto LABEL_12;
        if ( v64 )
        {
          v59 = sub_CD1D40((__int64 *)(a1 + 808), 40, 3);
          *(_QWORD *)v59 = 0;
          v11 = (_BYTE **)v59;
          v3 = (_QWORD *)(v59 + 8);
          *(_WORD *)(v59 + 16) = 16462;
          LOBYTE(v59) = *(_BYTE *)(v59 + 18);
          v11[3] = (_BYTE *)8;
          v11[4] = v18;
          *((_BYTE *)v11 + 18) = v59 & 0xF0 | 5;
          v11[1] = &unk_49E0D48;
LABEL_82:
          sub_C657C0(v70, (__int64 *)v11, v71, (__int64)off_497B2F0);
        }
        goto LABEL_40;
      }
    }
    return 0;
  }
LABEL_7:
  v5 = sub_EF1F20(a1);
  if ( !v5 )
    return 0;
  v8 = sub_EE32C0((char **)a1, 1);
  if ( !v8 )
    return 0;
  v9 = *(char **)a1;
  if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) || *v9 != 69 )
    return 0;
  v63 = v6;
  *(_QWORD *)a1 = v9 + 1;
  v10 = *(_BYTE *)(a1 + 937);
  v61 = v8;
  v72 = v74;
  v60 = v10;
  v73 = 0x2000000000LL;
  sub_EE3E30((__int64)&v72, 0x4Cu, v5, v8, v6, v7);
  v11 = &v72;
  v12 = sub_C65B40(a1 + 904, (__int64)&v72, (__int64 *)&v71, (__int64)off_497B2F0);
  v13 = (_BYTE *)v61;
  v3 = v12;
  if ( v12 )
  {
LABEL_12:
    ++v3;
    if ( v72 != v74 )
      _libc_free(v72, &v72);
    v72 = v3;
    v14 = sub_EE6840(a1 + 944, (__int64 *)&v72);
    if ( v14 )
    {
      v15 = (_QWORD *)v14[1];
      if ( v15 )
        v3 = v15;
    }
    if ( *(_QWORD **)(a1 + 928) == v3 )
      *(_BYTE *)(a1 + 936) = 1;
  }
  else
  {
    if ( v60 )
    {
      v62 = (_BYTE *)v63;
      v69 = v13;
      v58 = sub_CD1D40((__int64 *)(a1 + 808), 48, 3);
      *(_QWORD *)v58 = 0;
      v11 = (_BYTE **)v58;
      v3 = (_QWORD *)(v58 + 8);
      *(_WORD *)(v58 + 16) = 16460;
      LOBYTE(v58) = *(_BYTE *)(v58 + 18);
      v11[3] = (_BYTE *)v5;
      v11[4] = v69;
      v11[5] = v62;
      *((_BYTE *)v11 + 18) = v58 & 0xF0 | 5;
      v11[1] = &unk_49E0B08;
      sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v11, v71, (__int64)off_497B2F0);
    }
LABEL_40:
    if ( v72 != v74 )
      _libc_free(v72, v11);
    *(_QWORD *)(a1 + 920) = v3;
  }
  return v3;
}
