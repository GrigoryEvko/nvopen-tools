// Function: sub_EF7610
// Address: 0xef7610
//
__int64 __fastcall sub_EF7610(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int8 *v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  unsigned __int8 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r15
  __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // r15
  unsigned __int8 *v25; // rax
  __int64 v26; // rcx
  __int64 v27; // r14
  __int64 v28; // r13
  char v29; // al
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  _QWORD *v38; // rax
  __int64 v39; // r13
  __int64 *v40; // rax
  __int64 v41; // rax
  __int64 *v42; // r14
  __int64 v43; // r14
  unsigned __int64 *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r13
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // rax
  char *v59; // rbx
  unsigned __int64 v60; // r15
  unsigned __int64 v61; // r15
  __int64 v62; // rax
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rax
  unsigned __int64 *v66; // rbx
  unsigned __int64 v67; // r15
  unsigned __int64 v68; // r15
  __int64 v69; // rax
  _QWORD **v70; // rsi
  _QWORD *v71; // rax
  __int64 *v72; // rax
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // r15
  char v79; // r14
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // rax
  char *v85; // rax
  __int64 v86; // r10
  __int64 v87; // r15
  __int64 v88; // rax
  char *v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // rdx
  __int64 v92; // rcx
  __int64 v93; // r8
  __int64 v94; // r9
  char v95; // r14
  __int64 v96; // rdx
  __int64 v97; // rcx
  __int64 v98; // r8
  __int64 v99; // r9
  __int64 v100; // rax
  void *v101; // rax
  void *v102; // rax
  __int64 v103; // rax
  __int64 v104; // rax
  __int64 v105; // [rsp+8h] [rbp-118h]
  char n; // [rsp+10h] [rbp-110h]
  char na; // [rsp+10h] [rbp-110h]
  size_t nb; // [rsp+10h] [rbp-110h]
  unsigned __int8 src; // [rsp+18h] [rbp-108h]
  char *srcb; // [rsp+18h] [rbp-108h]
  __int64 *srca; // [rsp+18h] [rbp-108h]
  __int64 v112; // [rsp+20h] [rbp-100h]
  char *v113; // [rsp+28h] [rbp-F8h]
  __int64 v114; // [rsp+30h] [rbp-F0h]
  __int64 *v115; // [rsp+40h] [rbp-E0h]
  unsigned __int64 *v116; // [rsp+48h] [rbp-D8h]
  __int64 *v117; // [rsp+58h] [rbp-C8h] BYREF
  _QWORD *v118; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v119; // [rsp+68h] [rbp-B8h]
  _QWORD v120[22]; // [rsp+70h] [rbp-B0h] BYREF

  v1 = 2;
  if ( (unsigned __int8)sub_EE3B50((const void **)a1, 2u, "rQ") )
  {
    v6 = (__int64)(*(_QWORD *)(a1 + 24) - *(_QWORD *)(a1 + 16)) >> 3;
    while ( 1 )
    {
      v11 = *(unsigned __int8 **)a1;
      if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) && *v11 == 95 )
        break;
      v118 = (_QWORD *)sub_EF1F20(a1, v1, v2, v3, v4, v5);
      if ( !v118 )
        return 0;
      v1 = (__int64)&v118;
      sub_E18380(a1 + 16, (__int64 *)&v118, v7, v8, v9, v10);
    }
    v12 = v6;
    *(_QWORD *)a1 = v11 + 1;
    v113 = (char *)sub_EE6060((_QWORD *)a1, v6);
    v114 = v16;
  }
  else
  {
    v12 = 2;
    v114 = 0;
    v113 = 0;
    if ( !(unsigned __int8)sub_EE3B50((const void **)a1, 2u, "rq") )
      return 0;
  }
  v112 = (__int64)(*(_QWORD *)(a1 + 24) - *(_QWORD *)(a1 + 16)) >> 3;
  v17 = *(unsigned __int8 **)a1;
  if ( *(_QWORD *)(a1 + 8) == *(_QWORD *)a1 )
    return 0;
  v18 = *v17;
  while ( (_BYTE)v18 != 88 )
  {
    if ( (_BYTE)v18 == 84 )
    {
      *(_QWORD *)a1 = v17 + 1;
      v78 = sub_EF1F20(a1, v12, v18, v13, v14, v15);
      if ( !v78 )
        return 0;
      v79 = *(_BYTE *)(a1 + 937);
      v119 = 0x2000000000LL;
      v118 = v120;
      sub_D953B0((__int64)&v118, 85, v74, v75, v76, v77);
      sub_D953B0((__int64)&v118, v78, v80, v81, v82, v83);
      v115 = (__int64 *)(a1 + 904);
      v38 = sub_C65B40(a1 + 904, (__int64)&v118, (__int64 *)&v117, (__int64)off_497B2F0);
      if ( v38 )
        goto LABEL_21;
      if ( !v79 )
        goto LABEL_78;
      v84 = sub_CD1D40((__int64 *)(a1 + 808), 32, 3);
      *(_QWORD *)v84 = 0;
      v12 = v84;
      v39 = v84 + 8;
      *(_WORD *)(v84 + 16) = 16469;
      *(_BYTE *)(v84 + 18) = *(_BYTE *)(v84 + 18) & 0xF0 | 5;
      v85 = (char *)&unk_49E0C78;
    }
    else
    {
      if ( (_BYTE)v18 != 81 )
        return 0;
      *(_QWORD *)a1 = v17 + 1;
      v78 = sub_EEA9F0(a1);
      if ( !v78 )
        return 0;
      v95 = *(_BYTE *)(a1 + 937);
      v119 = 0x2000000000LL;
      v118 = v120;
      sub_D953B0((__int64)&v118, 86, v91, v92, v93, v94);
      sub_D953B0((__int64)&v118, v78, v96, v97, v98, v99);
      v115 = (__int64 *)(a1 + 904);
      v38 = sub_C65B40(a1 + 904, (__int64)&v118, (__int64 *)&v117, (__int64)off_497B2F0);
      if ( v38 )
      {
LABEL_21:
        v39 = (__int64)(v38 + 1);
        if ( v118 != v120 )
          _libc_free(v118, &v118);
        v12 = (__int64)&v118;
        v118 = (_QWORD *)v39;
        v40 = sub_EE6840(a1 + 944, (__int64 *)&v118);
        if ( v40 )
        {
          v41 = v40[1];
          if ( v41 )
            v39 = v41;
        }
        if ( *(_QWORD *)(a1 + 928) == v39 )
          *(_BYTE *)(a1 + 936) = 1;
        goto LABEL_28;
      }
      if ( !v95 )
        goto LABEL_78;
      v100 = sub_CD1D40((__int64 *)(a1 + 808), 32, 3);
      *(_QWORD *)v100 = 0;
      v12 = v100;
      v39 = v100 + 8;
      *(_WORD *)(v100 + 16) = 16470;
      *(_BYTE *)(v100 + 18) = *(_BYTE *)(v100 + 18) & 0xF0 | 5;
      v85 = (char *)&unk_49E0CD8;
    }
    *(_QWORD *)(v12 + 24) = v78;
    *(_QWORD *)(v12 + 8) = v85 + 16;
LABEL_60:
    sub_C657C0(v115, (__int64 *)v12, v117, (__int64)off_497B2F0);
    if ( v118 != v120 )
      _libc_free(v118, v12);
    *(_QWORD *)(a1 + 920) = v39;
LABEL_28:
    if ( !v39 )
      return 0;
    v42 = *(__int64 **)(a1 + 24);
    if ( v42 == *(__int64 **)(a1 + 32) )
    {
      v86 = *(_QWORD *)(a1 + 16);
      v87 = 16 * (((__int64)v42 - v86) >> 3);
      if ( v86 == a1 + 40 )
      {
        nb = (size_t)v42 - v86;
        srca = *(__int64 **)(a1 + 16);
        v101 = (void *)malloc(
                         16 * (((__int64)v42 - v86) >> 3),
                         v12,
                         (char *)v42 - v86,
                         16 * (((__int64)v42 - v86) >> 3),
                         v14,
                         v15);
        v90 = (__int64)v101;
        if ( !v101 )
LABEL_94:
          abort();
        v89 = (char *)nb;
        if ( v42 != srca )
        {
          v12 = (__int64)srca;
          v102 = memmove(v101, srca, nb);
          v89 = (char *)nb;
          v90 = (__int64)v102;
        }
        *(_QWORD *)(a1 + 16) = v90;
      }
      else
      {
        v12 = 16 * (((__int64)v42 - v86) >> 3);
        srcb = (char *)v42 - v86;
        v88 = realloc(*(void **)(a1 + 16));
        v89 = srcb;
        *(_QWORD *)(a1 + 16) = v88;
        v90 = v88;
        if ( !v88 )
          goto LABEL_94;
      }
      v42 = (__int64 *)&v89[v90];
      v13 = v87 + v90;
      *(_QWORD *)(a1 + 32) = v13;
    }
    *(_QWORD *)(a1 + 24) = v42 + 1;
    *v42 = v39;
    v17 = *(unsigned __int8 **)a1;
    if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) )
      return 0;
    v18 = *v17;
    if ( (_BYTE)v18 == 69 )
    {
      v43 = a1;
      *(_QWORD *)a1 = v17 + 1;
      v44 = (unsigned __int64 *)sub_EE6060((_QWORD *)a1, v112);
      v118 = v120;
      v116 = v44;
      v46 = v45;
      na = *(_BYTE *)(a1 + 937);
      v119 = 0x2000000000LL;
      sub_D953B0((__int64)&v118, 83, v45, v47, v48, v49);
      sub_D953B0((__int64)&v118, v114, v50, v51, v52, v53);
      v58 = (unsigned int)v119;
      if ( &v113[8 * v114] != v113 )
      {
        v59 = v113;
        do
        {
          v60 = *(_QWORD *)v59;
          if ( v58 + 1 > (unsigned __int64)HIDWORD(v119) )
          {
            sub_C8D5F0((__int64)&v118, v120, v58 + 1, 4u, v56, v57);
            v58 = (unsigned int)v119;
          }
          *((_DWORD *)v118 + v58) = v60;
          v61 = HIDWORD(v60);
          v55 = HIDWORD(v119);
          LODWORD(v119) = v119 + 1;
          v62 = (unsigned int)v119;
          if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
          {
            sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 4u, v56, v57);
            v62 = (unsigned int)v119;
          }
          v54 = (__int64)v118;
          v59 += 8;
          *((_DWORD *)v118 + v62) = v61;
          v58 = (unsigned int)(v119 + 1);
          LODWORD(v119) = v119 + 1;
        }
        while ( &v113[8 * v114] != v59 );
        v43 = a1;
      }
      sub_D953B0((__int64)&v118, v46, v54, v55, v56, v57);
      if ( v116 != &v116[v46] )
      {
        v65 = (unsigned int)v119;
        v66 = v116;
        do
        {
          v67 = *v66;
          if ( v65 + 1 > (unsigned __int64)HIDWORD(v119) )
          {
            sub_C8D5F0((__int64)&v118, v120, v65 + 1, 4u, v63, v64);
            v65 = (unsigned int)v119;
          }
          *((_DWORD *)v118 + v65) = v67;
          v68 = HIDWORD(v67);
          LODWORD(v119) = v119 + 1;
          v69 = (unsigned int)v119;
          if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
          {
            sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 4u, v63, v64);
            v69 = (unsigned int)v119;
          }
          ++v66;
          *((_DWORD *)v118 + v69) = v68;
          v65 = (unsigned int)(v119 + 1);
          LODWORD(v119) = v119 + 1;
        }
        while ( &v116[v46] != v66 );
      }
      v70 = &v118;
      v71 = sub_C65B40((__int64)v115, (__int64)&v118, (__int64 *)&v117, (__int64)off_497B2F0);
      v19 = (__int64)v71;
      if ( v71 )
      {
        v19 = (__int64)(v71 + 1);
        if ( v118 != v120 )
          _libc_free(v118, &v118);
        v118 = (_QWORD *)v19;
        v72 = sub_EE6840(v43 + 944, (__int64 *)&v118);
        if ( v72 )
        {
          v73 = v72[1];
          if ( v73 )
            v19 = v73;
        }
        if ( *(_QWORD *)(v43 + 928) == v19 )
          *(_BYTE *)(v43 + 936) = 1;
      }
      else
      {
        if ( na )
        {
          v104 = sub_CD1D40((__int64 *)(v43 + 808), 56, 3);
          *(_QWORD *)v104 = 0;
          v70 = (_QWORD **)v104;
          v19 = v104 + 8;
          *(_WORD *)(v104 + 16) = 16467;
          LOBYTE(v104) = *(_BYTE *)(v104 + 18);
          v70[6] = (_QWORD *)v46;
          *((_BYTE *)v70 + 18) = v104 & 0xF0 | 5;
          v70[1] = &unk_49E0BC8;
          v70[3] = v113;
          v70[4] = (_QWORD *)v114;
          v70[5] = v116;
          sub_C657C0(v115, (__int64 *)v70, v117, (__int64)off_497B2F0);
        }
        if ( v118 != v120 )
          _libc_free(v118, v70);
        *(_QWORD *)(v43 + 920) = v19;
      }
      return v19;
    }
  }
  *(_QWORD *)a1 = v17 + 1;
  v24 = sub_EEA9F0(a1);
  if ( !v24 )
    return 0;
  v25 = *(unsigned __int8 **)a1;
  v26 = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)a1 == v26 )
  {
    src = 0;
    v28 = 0;
    v27 = 0;
    v105 = 0;
    goto LABEL_20;
  }
  v21 = *v25;
  src = 0;
  if ( (_BYTE)v21 == 78 )
  {
    *(_QWORD *)a1 = v25 + 1;
    if ( (unsigned __int8 *)v26 == v25 + 1 )
    {
      src = 1;
      v28 = 0;
      v27 = 1;
      v105 = 0;
      goto LABEL_20;
    }
    v21 = v25[1];
    src = 1;
    ++v25;
  }
  if ( (_BYTE)v21 == 82 )
  {
    *(_QWORD *)a1 = v25 + 1;
    v105 = sub_EF1680(a1, 0, v21, v26, v22, v23);
    v28 = v105;
    if ( !v105 )
      return 0;
    v27 = src;
  }
  else
  {
    v105 = 0;
    v27 = src;
    v28 = 0;
  }
LABEL_20:
  v29 = *(_BYTE *)(a1 + 937);
  v120[0] = 84;
  v119 = 0x2000000002LL;
  n = v29;
  v118 = v120;
  sub_D953B0((__int64)&v118, v24, v21, v26, v22, v23);
  sub_D953B0((__int64)&v118, v27, v30, v31, v32, v33);
  sub_D953B0((__int64)&v118, v28, v34, v35, v36, v37);
  v115 = (__int64 *)(a1 + 904);
  v38 = sub_C65B40(a1 + 904, (__int64)&v118, (__int64 *)&v117, (__int64)off_497B2F0);
  if ( v38 )
    goto LABEL_21;
  if ( n )
  {
    v103 = sub_CD1D40((__int64 *)(a1 + 808), 48, 3);
    *(_QWORD *)v103 = 0;
    v12 = v103;
    v39 = v103 + 8;
    *(_WORD *)(v103 + 16) = 16468;
    LOBYTE(v103) = *(_BYTE *)(v103 + 18);
    *(_QWORD *)(v12 + 24) = v24;
    *(_BYTE *)(v12 + 18) = v103 & 0xF0 | 5;
    *(_QWORD *)(v12 + 8) = &unk_49E0C28;
    *(_BYTE *)(v12 + 32) = src;
    *(_QWORD *)(v12 + 40) = v105;
    goto LABEL_60;
  }
LABEL_78:
  if ( v118 != v120 )
    _libc_free(v118, &v118);
  *(_QWORD *)(a1 + 920) = 0;
  return 0;
}
