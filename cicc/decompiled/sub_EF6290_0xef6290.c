// Function: sub_EF6290
// Address: 0xef6290
//
__int64 __fastcall sub_EF6290(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r13
  char v11; // al
  __int64 *v12; // r15
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  _QWORD **v17; // rsi
  __int64 v18; // r14
  __int64 *v19; // rax
  __int64 v20; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  char v26; // r13
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  unsigned __int64 v31; // r15
  __int64 v32; // r8
  unsigned __int64 v33; // r10
  _QWORD **v34; // rsi
  _QWORD *v35; // rax
  __int64 *v36; // rax
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  unsigned __int64 v46; // r13
  unsigned __int64 v47; // rax
  __int64 v48; // r8
  _QWORD *v49; // r15
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // r9
  __int64 v56; // rax
  char *v57; // r12
  char *v58; // r8
  __int64 v59; // rax
  _BYTE *v60; // rax
  _BYTE *v61; // rdx
  _QWORD **v62; // rsi
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 v66; // r9
  char v67; // r13
  __int64 v68; // rax
  _BYTE *v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // r8
  char v73; // al
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // rax
  char *v79; // rax
  __int64 v80; // rax
  _QWORD *v81; // r13
  __int64 v82; // rdx
  __int64 v83; // r8
  __int64 v84; // rdx
  __int64 v85; // rcx
  __int64 v86; // r8
  __int64 v87; // r9
  __int64 v88; // rcx
  __int64 v89; // r8
  __int64 v90; // rdx
  _QWORD *v91; // r9
  __int64 v92; // rax
  _QWORD *v93; // rbx
  unsigned __int64 v94; // r12
  unsigned __int64 v95; // r12
  __int64 v96; // rax
  _QWORD *v97; // rax
  __int64 *v98; // rax
  __int64 v99; // rax
  __int64 v100; // rax
  __int64 v101; // r13
  __int64 v102; // rax
  char *v103; // rcx
  __int64 v104; // rdi
  char *v105; // rax
  __int64 v106; // rax
  __int64 v107; // r12
  __int64 *v108; // rdx
  char v109; // [rsp+Fh] [rbp-191h]
  _QWORD *v110; // [rsp+10h] [rbp-190h]
  _QWORD *v111; // [rsp+20h] [rbp-180h]
  char n; // [rsp+28h] [rbp-178h]
  size_t na; // [rsp+28h] [rbp-178h]
  size_t nb; // [rsp+28h] [rbp-178h]
  char v115; // [rsp+30h] [rbp-170h]
  __int64 v116; // [rsp+30h] [rbp-170h]
  __int64 v117; // [rsp+30h] [rbp-170h]
  char v118; // [rsp+38h] [rbp-168h]
  _QWORD *v119; // [rsp+38h] [rbp-168h]
  __int64 v120; // [rsp+38h] [rbp-168h]
  char v121; // [rsp+38h] [rbp-168h]
  __int64 src; // [rsp+40h] [rbp-160h]
  char *srcb; // [rsp+40h] [rbp-160h]
  char *srca; // [rsp+40h] [rbp-160h]
  __int64 v125; // [rsp+48h] [rbp-158h] BYREF
  __int64 *v126; // [rsp+58h] [rbp-148h] BYREF
  _QWORD *v127[2]; // [rsp+60h] [rbp-140h] BYREF
  __int64 *v128; // [rsp+70h] [rbp-130h] BYREF
  __int64 v129; // [rsp+78h] [rbp-128h]
  _QWORD v130[3]; // [rsp+80h] [rbp-120h] BYREF
  _OWORD v131[4]; // [rsp+98h] [rbp-108h] BYREF
  char v132[8]; // [rsp+D8h] [rbp-C8h] BYREF
  _QWORD *v133; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v134; // [rsp+E8h] [rbp-B8h]
  _QWORD v135[22]; // [rsp+F0h] [rbp-B0h] BYREF

  v2 = a1;
  v125 = a2;
  v127[0] = (_QWORD *)a1;
  v127[1] = &v125;
  if ( (unsigned __int8)sub_EE3B50((const void **)a1, 2u, "Ty") )
  {
    v10 = sub_EE92E0(v127, 0, v3, v4, v5, v6);
    if ( v10 )
    {
      v11 = *(_BYTE *)(a1 + 937);
      v133 = v135;
      v12 = (__int64 *)(a1 + 904);
      v118 = v11;
      v134 = 0x2000000000LL;
      sub_D953B0((__int64)&v133, 35, v7, v8, v9, (__int64)v135);
      sub_D953B0((__int64)&v133, v10, v13, v14, v15, v16);
      v17 = &v133;
      v18 = (__int64)sub_C65B40(a1 + 904, (__int64)&v133, (__int64 *)&v128, (__int64)off_497B2F0);
      if ( v18 )
      {
LABEL_4:
        v18 += 8;
        if ( v133 != v135 )
          _libc_free(v133, &v133);
        v133 = (_QWORD *)v18;
        v19 = sub_EE6840(a1 + 944, (__int64 *)&v133);
        if ( v19 )
        {
          v20 = v19[1];
          if ( v20 )
            v18 = v20;
        }
        if ( *(_QWORD *)(a1 + 928) == v18 )
          *(_BYTE *)(a1 + 936) = 1;
        return v18;
      }
      if ( !v118 )
        goto LABEL_49;
      v80 = sub_CD1D40((__int64 *)(a1 + 808), 32, 3);
      *(_QWORD *)v80 = 0;
      v17 = (_QWORD **)v80;
      v18 = v80 + 8;
      *(_WORD *)(v80 + 16) = 35;
      *(_BYTE *)(v80 + 18) = *(_BYTE *)(v80 + 18) & 0xF0 | 5;
      v79 = (char *)&unk_49DFAD8;
      goto LABEL_56;
    }
    return 0;
  }
  if ( (unsigned __int8)sub_EE3B50((const void **)a1, 2u, "Tk") )
  {
    v26 = *(_BYTE *)(a1 + 778);
    *(_BYTE *)(a1 + 778) = 1;
    v31 = sub_EF1680(a1, 0, v22, v23, v24, v25);
    if ( v31 && (v33 = sub_EE92E0(v127, 0, v27, v28, v29, v30)) != 0 )
    {
      v119 = (_QWORD *)v33;
      n = *(_BYTE *)(a1 + 937);
      v133 = v135;
      v134 = 0x2000000000LL;
      sub_EE40D0((__int64)&v133, 0x24u, v31, v33, v32, (__int64)v135);
      v34 = &v133;
      v35 = sub_C65B40(a1 + 904, (__int64)&v133, (__int64 *)&v128, (__int64)off_497B2F0);
      v18 = (__int64)v35;
      if ( v35 )
      {
        v18 = (__int64)(v35 + 1);
        if ( v133 != v135 )
          _libc_free(v133, &v133);
        v133 = (_QWORD *)v18;
        v36 = sub_EE6840(a1 + 944, (__int64 *)&v133);
        if ( v36 )
        {
          v37 = v36[1];
          if ( v37 )
            v18 = v37;
        }
        if ( *(_QWORD *)(a1 + 928) == v18 )
          *(_BYTE *)(a1 + 936) = 1;
      }
      else
      {
        if ( n )
        {
          v100 = sub_CD1D40((__int64 *)(a1 + 808), 40, 3);
          *(_WORD *)(v100 + 16) = 36;
          v34 = (_QWORD **)v100;
          v18 = v100 + 8;
          *(_QWORD *)v100 = 0;
          LOBYTE(v100) = *(_BYTE *)(v100 + 18);
          v34[3] = (_QWORD *)v31;
          v34[4] = v119;
          *((_BYTE *)v34 + 18) = v100 & 0xF0 | 5;
          v34[1] = &unk_49DFB48;
          sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v34, v128, (__int64)off_497B2F0);
        }
        if ( v133 != v135 )
          _libc_free(v133, v34);
        *(_QWORD *)(a1 + 920) = v18;
      }
    }
    else
    {
      v18 = 0;
    }
    *(_BYTE *)(a1 + 778) = v26;
    return v18;
  }
  if ( (unsigned __int8)sub_EE3B50((const void **)a1, 2u, "Tn") )
  {
    v46 = sub_EE92E0(v127, 1, v38, v39, v40, v41);
    if ( v46 )
    {
      v47 = sub_EF1F20(a1, 1, v42, v43, v44, v45);
      v49 = (_QWORD *)v47;
      if ( v47 )
      {
        v115 = *(_BYTE *)(a1 + 937);
        v133 = v135;
        v134 = 0x2000000000LL;
        sub_EE40D0((__int64)&v133, 0x25u, v46, v47, v48, (__int64)v135);
        v17 = &v133;
        v18 = (__int64)sub_C65B40(a1 + 904, (__int64)&v133, (__int64 *)&v128, (__int64)off_497B2F0);
        if ( v18 )
          goto LABEL_4;
        if ( v115 )
        {
          v50 = sub_CD1D40((__int64 *)(a1 + 808), 40, 3);
          *(_WORD *)(v50 + 16) = 37;
          v17 = (_QWORD **)v50;
          v18 = v50 + 8;
          *(_QWORD *)v50 = 0;
          LOBYTE(v50) = *(_BYTE *)(v50 + 18);
          v17[3] = (_QWORD *)v46;
          v17[4] = v49;
          *((_BYTE *)v17 + 18) = v50 & 0xF0 | 5;
          v17[1] = &unk_49DFBA8;
          sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v17, v128, (__int64)off_497B2F0);
        }
        goto LABEL_49;
      }
    }
    return 0;
  }
  if ( !(unsigned __int8)sub_EE3B50((const void **)a1, 2u, "Tt") )
  {
    if ( (unsigned __int8)sub_EE3B50((const void **)a1, 2u, "Tp") )
    {
      v10 = sub_EF6290(a1, v125);
      if ( v10 )
      {
        v73 = *(_BYTE *)(a1 + 937);
        v133 = v135;
        v12 = (__int64 *)(a1 + 904);
        v121 = v73;
        v134 = 0x2000000000LL;
        sub_D953B0((__int64)&v133, 39, v70, v71, v72, (__int64)v135);
        sub_D953B0((__int64)&v133, v10, v74, v75, v76, v77);
        v17 = &v133;
        v18 = (__int64)sub_C65B40(a1 + 904, (__int64)&v133, (__int64 *)&v128, (__int64)off_497B2F0);
        if ( v18 )
          goto LABEL_4;
        if ( !v121 )
          goto LABEL_49;
        v78 = sub_CD1D40((__int64 *)(a1 + 808), 32, 3);
        *(_QWORD *)v78 = 0;
        v17 = (_QWORD **)v78;
        v18 = v78 + 8;
        *(_WORD *)(v78 + 16) = 39;
        *(_BYTE *)(v78 + 18) = *(_BYTE *)(v78 + 18) & 0xF0 | 5;
        v79 = (char *)&unk_49DFC58;
LABEL_56:
        v17[3] = (_QWORD *)v10;
        v17[1] = v79 + 16;
        sub_C657C0(v12, (__int64 *)v17, v128, (__int64)off_497B2F0);
LABEL_49:
        if ( v133 != v135 )
          _libc_free(v133, v17);
        *(_QWORD *)(a1 + 920) = v18;
        return v18;
      }
    }
    return 0;
  }
  v18 = sub_EE92E0(v127, 2, v51, v52, v53, v54);
  if ( !v18 )
    return v18;
  v56 = *(_QWORD *)(a1 + 24);
  v57 = *(char **)(a1 + 672);
  v58 = *(char **)(a1 + 664);
  v128 = (__int64 *)a1;
  v120 = v56;
  v59 = *(_QWORD *)(a1 + 16);
  v130[0] = v131;
  v116 = v59;
  v130[1] = v131;
  v129 = (v57 - v58) >> 3;
  v130[2] = v132;
  memset(v131, 0, sizeof(v131));
  if ( v57 == *(char **)(a1 + 680) )
  {
    v101 = 16 * ((v57 - v58) >> 3);
    if ( v58 == (char *)(a1 + 688) )
    {
      nb = v57 - v58;
      srca = v58;
      v104 = malloc(16 * ((v57 - v58) >> 3), 2, v132, v57 - v58, v58, v55);
      if ( v104 )
      {
        v103 = (char *)nb;
        if ( v57 != srca )
        {
          v105 = (char *)memmove((void *)v104, srca, nb);
          v103 = (char *)nb;
          v104 = (__int64)v105;
        }
        *(_QWORD *)(v2 + 664) = v104;
        goto LABEL_82;
      }
    }
    else
    {
      srcb = (char *)(v57 - v58);
      v102 = realloc(v58);
      v103 = srcb;
      *(_QWORD *)(a1 + 664) = v102;
      v104 = v102;
      if ( v102 )
      {
LABEL_82:
        v57 = &v103[v104];
        *(_QWORD *)(v2 + 680) = v104 + v101;
        goto LABEL_34;
      }
    }
    abort();
  }
LABEL_34:
  *(_QWORD *)(v2 + 672) = v57 + 8;
  *(_QWORD *)v57 = v130;
  v60 = *(_BYTE **)v2;
  v61 = *(_BYTE **)(v2 + 8);
  do
  {
    if ( v61 != v60 && *v60 == 69 )
    {
      na = 0;
      *(_QWORD *)v2 = v60 + 1;
      v110 = 0;
      goto LABEL_64;
    }
    v62 = (_QWORD **)v130;
    v133 = (_QWORD *)sub_EF6290(v2, v130);
    if ( !v133 )
      goto LABEL_43;
    v62 = &v133;
    sub_E18380(v2 + 16, (__int64 *)&v133, v63, v64, v65, v66);
    v60 = *(_BYTE **)v2;
    v61 = *(_BYTE **)(v2 + 8);
  }
  while ( *(_BYTE **)v2 == v61 || *v60 != 81 );
  v67 = *(_BYTE *)(v2 + 778);
  *(_BYTE *)(v2 + 778) = 1;
  *(_QWORD *)v2 = v60 + 1;
  v68 = sub_EEA9F0(v2);
  *(_BYTE *)(v2 + 778) = v67;
  v110 = (_QWORD *)v68;
  if ( !v68 || (v69 = *(_BYTE **)v2, *(_QWORD *)v2 == *(_QWORD *)(v2 + 8)) || *v69 != 69 )
  {
LABEL_43:
    v18 = 0;
    goto LABEL_44;
  }
  *(_QWORD *)v2 = v69 + 1;
  na = (size_t)v110;
LABEL_64:
  v81 = sub_EE6060((_QWORD *)v2, (v120 - v116) >> 3);
  src = v82;
  v109 = *(_BYTE *)(v2 + 937);
  v133 = v135;
  v134 = 0x2000000002LL;
  v135[0] = 38;
  sub_D953B0((__int64)&v133, v18, v82, 0x2000000002LL, v83, (__int64)v135);
  sub_D953B0((__int64)&v133, src, v84, v85, v86, v87);
  v90 = (unsigned int)v134;
  v91 = v135;
  if ( &v81[src] != v81 )
  {
    v111 = v81;
    v92 = (unsigned int)v134;
    v117 = v2;
    v93 = &v81[src];
    do
    {
      v94 = *v81;
      if ( v92 + 1 > (unsigned __int64)HIDWORD(v134) )
      {
        sub_C8D5F0((__int64)&v133, v135, v92 + 1, 4u, v89, (__int64)v91);
        v92 = (unsigned int)v134;
      }
      *((_DWORD *)v133 + v92) = v94;
      v95 = HIDWORD(v94);
      v88 = HIDWORD(v134);
      LODWORD(v134) = v134 + 1;
      v96 = (unsigned int)v134;
      if ( (unsigned __int64)(unsigned int)v134 + 1 > HIDWORD(v134) )
      {
        sub_C8D5F0((__int64)&v133, v135, (unsigned int)v134 + 1LL, 4u, v89, (__int64)v91);
        v96 = (unsigned int)v134;
      }
      v90 = (__int64)v133;
      ++v81;
      *((_DWORD *)v133 + v96) = v95;
      v92 = (unsigned int)(v134 + 1);
      LODWORD(v134) = v134 + 1;
    }
    while ( v93 != v81 );
    v2 = v117;
    v81 = v111;
  }
  sub_D953B0((__int64)&v133, na, v90, v88, v89, (__int64)v135);
  v62 = &v133;
  v97 = sub_C65B40(v2 + 904, (__int64)&v133, (__int64 *)&v126, (__int64)off_497B2F0);
  if ( v97 )
  {
    v18 = (__int64)(v97 + 1);
    if ( v133 != v135 )
      _libc_free(v133, &v133);
    v62 = &v133;
    v133 = (_QWORD *)v18;
    v98 = sub_EE6840(v2 + 944, (__int64 *)&v133);
    if ( v98 )
    {
      v99 = v98[1];
      if ( v99 )
        v18 = v99;
    }
    if ( *(_QWORD *)(v2 + 928) == v18 )
      *(_BYTE *)(v2 + 936) = 1;
  }
  else
  {
    if ( v109 )
    {
      v106 = sub_CD1D40((__int64 *)(v2 + 808), 56, 3);
      *(_QWORD *)v106 = 0;
      v62 = (_QWORD **)v106;
      v107 = v106 + 8;
      *(_WORD *)(v106 + 16) = 38;
      LOBYTE(v106) = *(_BYTE *)(v106 + 18);
      v62[3] = (_QWORD *)v18;
      v108 = v126;
      v18 = v107;
      v62[4] = v81;
      *((_BYTE *)v62 + 18) = v106 & 0xF0 | 5;
      v62[1] = &unk_49DFC08;
      v62[5] = (_QWORD *)src;
      v62[6] = v110;
      sub_C657C0((__int64 *)(v2 + 904), (__int64 *)v62, v108, (__int64)off_497B2F0);
    }
    else
    {
      v18 = 0;
    }
    if ( v133 != v135 )
      _libc_free(v133, v62);
    *(_QWORD *)(v2 + 920) = v18;
  }
LABEL_44:
  v128[84] = v128[83] + 8 * v129;
  if ( (_OWORD *)v130[0] != v131 )
    _libc_free(v130[0], v62);
  return v18;
}
