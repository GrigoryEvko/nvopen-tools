// Function: sub_2B32060
// Address: 0x2b32060
//
__int64 __fastcall sub_2B32060(__int64 *a1, char **a2)
{
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 *v8; // r15
  __int64 *v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rdx
  char *v12; // r9
  __int64 v13; // rdi
  __int64 v14; // rdx
  bool v15; // al
  __int64 result; // rax
  char v17; // cl
  __int64 v18; // rdi
  unsigned int v19; // edx
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 *v23; // r13
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rax
  char *v29; // r8
  __int64 v30; // rdi
  __int64 v31; // rdx
  char *v32; // r13
  __int64 v33; // r14
  __int64 *v34; // rax
  __int64 v35; // r9
  __int64 *v36; // rcx
  __int64 v37; // rax
  __int64 v38; // r12
  char v39; // si
  __int64 v40; // rdx
  __int64 v41; // r8
  int v42; // edx
  unsigned int v43; // ecx
  __int64 v44; // rdi
  __int64 v45; // rax
  __int64 *v46; // r15
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 *v49; // rcx
  __int64 v50; // rdx
  __int64 v51; // rax
  char *v52; // r8
  __int64 v53; // rdx
  __int64 v54; // rdi
  __int64 v55; // rcx
  char *v56; // r9
  __int64 v57; // rdi
  bool v58; // al
  __int64 v59; // rcx
  char *v60; // r9
  __int64 v61; // rdi
  bool v62; // al
  __int64 v63; // rcx
  char *v64; // r9
  __int64 v65; // rdi
  bool v66; // al
  __int64 v67; // rdi
  __int64 v68; // r13
  char *v69; // r12
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // r9
  char *v73; // r8
  __int64 v74; // rdi
  __int64 v75; // rcx
  __int64 v76; // r9
  char *v77; // r8
  __int64 v78; // rdi
  __int64 v79; // rcx
  __int64 v80; // r9
  char *v81; // r8
  __int64 v82; // rdi
  __int64 v83; // rdi
  char *v84; // r12
  __int64 v85; // rbx
  __int64 v86; // rcx
  __int64 v87; // r9
  char *v88; // r8
  __int64 v89; // rdi
  __int64 v90; // rcx
  __int64 v91; // r9
  char *v92; // r8
  __int64 v93; // rdi
  __int64 v94; // rcx
  __int64 v95; // r9
  char *v96; // r8
  __int64 v97; // rdi
  char *v98; // r13
  __int64 v99; // r14
  __int64 v100; // rdi
  __int64 v101; // rdi
  char *v102; // r13
  __int64 v103; // r14
  __int64 v104; // rsi
  int v105; // esi
  __int64 v106; // rdi
  int v107; // edi
  __int64 v108; // rdi
  char *v109; // r12
  __int64 v110; // rbx
  char *v111; // r14
  __int64 v112; // r15
  __int64 v113; // rdi
  __int64 v114; // rdi
  char *v115; // r14
  __int64 v116; // r15
  char *v117; // r13
  __int64 v118; // r14
  __int64 v119; // rdi
  __int64 v120; // rdi
  char *v121; // r13
  __int64 v122; // r14
  int v123; // r10d
  __int64 *v124; // [rsp+8h] [rbp-B8h]
  __int64 *v126; // [rsp+10h] [rbp-B0h]
  __int64 *v127; // [rsp+10h] [rbp-B0h]
  __int64 *v128; // [rsp+18h] [rbp-A8h]
  __int64 *v129; // [rsp+18h] [rbp-A8h]
  __int64 *v130; // [rsp+18h] [rbp-A8h]
  __int64 *v131; // [rsp+30h] [rbp-90h]
  __int64 *v132; // [rsp+30h] [rbp-90h]
  __int64 *v133; // [rsp+30h] [rbp-90h]
  unsigned __int8 **v134; // [rsp+38h] [rbp-88h]
  unsigned __int8 **v135; // [rsp+38h] [rbp-88h]
  unsigned __int8 **v136; // [rsp+38h] [rbp-88h]
  unsigned __int8 **v137; // [rsp+38h] [rbp-88h]
  unsigned __int8 **v138; // [rsp+38h] [rbp-88h]
  unsigned __int8 **v139; // [rsp+38h] [rbp-88h]
  unsigned __int8 **v140; // [rsp+38h] [rbp-88h]
  unsigned __int8 **v141; // [rsp+38h] [rbp-88h]
  __int64 v142; // [rsp+40h] [rbp-80h]
  unsigned __int8 **v143; // [rsp+40h] [rbp-80h]
  __int64 v144; // [rsp+40h] [rbp-80h]
  __int64 v145; // [rsp+40h] [rbp-80h]
  __int64 v146; // [rsp+40h] [rbp-80h]
  unsigned __int8 **v147; // [rsp+40h] [rbp-80h]
  unsigned __int8 **v148; // [rsp+40h] [rbp-80h]
  unsigned __int8 **v149; // [rsp+40h] [rbp-80h]
  __int64 v150; // [rsp+48h] [rbp-78h]
  __int64 v151; // [rsp+48h] [rbp-78h]
  __int64 v152; // [rsp+48h] [rbp-78h]
  __int64 v153; // [rsp+48h] [rbp-78h]
  __int64 v154; // [rsp+48h] [rbp-78h]
  __int64 v155; // [rsp+48h] [rbp-78h]
  __int64 v156; // [rsp+48h] [rbp-78h]
  __int64 v157; // [rsp+48h] [rbp-78h]
  __int64 v158; // [rsp+58h] [rbp-68h] BYREF
  __int64 v159; // [rsp+60h] [rbp-60h] BYREF
  __int64 v160; // [rsp+68h] [rbp-58h]
  __int64 v161; // [rsp+70h] [rbp-50h] BYREF
  __int64 *v162; // [rsp+78h] [rbp-48h]
  bool (__fastcall *v163)(_QWORD *, __int64); // [rsp+80h] [rbp-40h] BYREF
  __int64 *v164; // [rsp+88h] [rbp-38h]

  v3 = (__int64)a2[52];
  if ( v3 && a2[53] )
  {
    v4 = *a1;
    v159 = sub_2B2A0E0(*a1, v3);
    v160 = v5;
    if ( !v5 )
      goto LABEL_9;
    v8 = (__int64 *)v159;
    v9 = (__int64 *)(v159 + 8 * v5);
    v10 = (8 * v5) >> 5;
    v11 = (8 * v5) >> 3;
    v124 = v9;
    if ( v10 > 0 )
    {
      v9 = &v161;
      v128 = (__int64 *)(v159 + 32 * v10);
      while ( 1 )
      {
        v12 = *a2;
        v13 = *v8;
        v161 = v4;
        v14 = *((unsigned int *)a2 + 2);
        v162 = &v159;
        v134 = (unsigned __int8 **)v12;
        v142 = v14;
        v158 = v13;
        v163 = sub_2B200E0;
        v164 = &v161;
        v15 = sub_2B31C30(v13, v12, v14, (__int64)v9, v6, (__int64)v12);
        v7 = (__int64)v134;
        if ( v15 || &v134[v142] == sub_2B1E290(v134, (__int64)&v134[v142], &v158, (__int64)&v163) )
          break;
        v56 = *a2;
        v57 = v8[1];
        v161 = v4;
        v131 = v8 + 1;
        v136 = (unsigned __int8 **)v56;
        v144 = *((unsigned int *)a2 + 2);
        v162 = &v159;
        v158 = v57;
        v163 = sub_2B200E0;
        v164 = &v161;
        v58 = sub_2B31C30(v57, v56, v144, v55, v6, (__int64)v56);
        v7 = (__int64)v136;
        if ( v58 )
          goto LABEL_38;
        if ( &v136[v144] == sub_2B1E290(v136, (__int64)&v136[v144], &v158, (__int64)&v163) )
          goto LABEL_38;
        v60 = *a2;
        v61 = v8[2];
        v161 = v4;
        v131 = v8 + 2;
        v137 = (unsigned __int8 **)v60;
        v145 = *((unsigned int *)a2 + 2);
        v162 = &v159;
        v158 = v61;
        v163 = sub_2B200E0;
        v164 = &v161;
        v62 = sub_2B31C30(v61, v60, v145, v59, v6, (__int64)v60);
        v7 = (__int64)v137;
        if ( v62 )
          goto LABEL_38;
        if ( &v137[v145] == sub_2B1E290(v137, (__int64)&v137[v145], &v158, (__int64)&v163) )
          goto LABEL_38;
        v64 = *a2;
        v65 = v8[3];
        v161 = v4;
        v131 = v8 + 3;
        v138 = (unsigned __int8 **)v64;
        v146 = *((unsigned int *)a2 + 2);
        v162 = &v159;
        v158 = v65;
        v163 = sub_2B200E0;
        v164 = &v161;
        v66 = sub_2B31C30(v65, v64, v146, v63, v6, (__int64)v64);
        v7 = (__int64)v138;
        if ( v66 || &v138[v146] == sub_2B1E290(v138, (__int64)&v138[v146], &v158, (__int64)&v163) )
        {
LABEL_38:
          v8 = v131;
          break;
        }
        v8 += 4;
        if ( v128 == v8 )
        {
          v11 = v124 - v8;
          goto LABEL_46;
        }
      }
LABEL_7:
      result = 1;
      if ( v124 != v8 )
        return result;
      goto LABEL_8;
    }
LABEL_46:
    if ( v11 != 2 )
    {
      if ( v11 != 3 )
      {
        if ( v11 != 1 )
          goto LABEL_8;
        goto LABEL_49;
      }
      v98 = *a2;
      v99 = *((unsigned int *)a2 + 2);
      v100 = *v8;
      v162 = &v159;
      v163 = sub_2B200E0;
      v161 = v4;
      v158 = v100;
      v164 = &v161;
      if ( sub_2B31C30(v100, v98, v99, (__int64)v9, v6, v7)
        || &v98[8 * v99] == (char *)sub_2B1E290((unsigned __int8 **)v98, (__int64)&v98[8 * v99], &v158, (__int64)&v163) )
      {
        goto LABEL_7;
      }
      ++v8;
    }
    v101 = *v8;
    v161 = v4;
    v102 = *a2;
    v103 = *((unsigned int *)a2 + 2);
    v162 = &v159;
    v158 = v101;
    v163 = sub_2B200E0;
    v164 = &v161;
    if ( sub_2B31C30(v101, v102, v103, (__int64)v9, v6, v7)
      || &v102[8 * v103] == (char *)sub_2B1E290(
                                      (unsigned __int8 **)v102,
                                      (__int64)&v102[8 * v103],
                                      &v158,
                                      (__int64)&v163) )
    {
      goto LABEL_7;
    }
    ++v8;
LABEL_49:
    v67 = *v8;
    v161 = v4;
    v68 = *((unsigned int *)a2 + 2);
    v69 = *a2;
    v162 = &v159;
    v158 = v67;
    v163 = sub_2B200E0;
    v164 = &v161;
    if ( sub_2B31C30(v67, v69, v68, (__int64)v9, v6, v7)
      || &v69[8 * v68] == (char *)sub_2B1E290((unsigned __int8 **)v69, (__int64)&v69[8 * v68], &v158, (__int64)&v163) )
    {
      goto LABEL_7;
    }
LABEL_8:
    v3 = (__int64)a2[52];
    v4 = *a1;
LABEL_9:
    v17 = *(_BYTE *)(v4 + 392) & 1;
    if ( v17 )
    {
      v18 = v4 + 400;
      v6 = 3;
    }
    else
    {
      v70 = *(unsigned int *)(v4 + 408);
      v18 = *(_QWORD *)(v4 + 400);
      if ( !(_DWORD)v70 )
        goto LABEL_89;
      v6 = (unsigned int)(v70 - 1);
    }
    v19 = v6 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v20 = v18 + 72LL * v19;
    v21 = *(_QWORD *)v20;
    if ( v3 == *(_QWORD *)v20 )
      goto LABEL_12;
    v105 = 1;
    while ( v21 != -4096 )
    {
      v7 = (unsigned int)(v105 + 1);
      v19 = v6 & (v105 + v19);
      v20 = v18 + 72LL * v19;
      v21 = *(_QWORD *)v20;
      if ( *(_QWORD *)v20 == v3 )
        goto LABEL_12;
      v105 = v7;
    }
    if ( v17 )
    {
      v104 = 288;
      goto LABEL_90;
    }
    v70 = *(unsigned int *)(v4 + 408);
LABEL_89:
    v104 = 72 * v70;
LABEL_90:
    v20 = v18 + v104;
LABEL_12:
    v22 = 288;
    if ( !v17 )
      v22 = 72LL * *(unsigned int *)(v4 + 408);
    if ( v20 == v18 + v22 )
      return 0;
    v23 = *(__int64 **)(v20 + 8);
    v24 = *(unsigned int *)(v20 + 16);
    v159 = (__int64)v23;
    v160 = v24;
    if ( !v24 )
      return 0;
    v25 = 8 * v24;
    v26 = (__int64)&v23[(unsigned __int64)v25 / 8];
    v27 = v25 >> 3;
    v28 = v25 >> 5;
    v126 = (__int64 *)v26;
    if ( v28 )
    {
      v129 = &v23[4 * v28];
      while ( 1 )
      {
        v29 = *a2;
        v30 = *v23;
        v161 = v4;
        v31 = *((unsigned int *)a2 + 2);
        v163 = sub_2B1FB60;
        v135 = (unsigned __int8 **)v29;
        v150 = v31;
        v162 = &v159;
        v158 = v30;
        v164 = &v161;
        if ( sub_2B31C30(v30, v29, v31, v26, (__int64)v29, v7)
          || &v135[v150] == sub_2B1E290(v135, (__int64)&v135[v150], &v158, (__int64)&v163) )
        {
          break;
        }
        v73 = *a2;
        v74 = v23[1];
        v161 = v4;
        v132 = v23 + 1;
        v139 = (unsigned __int8 **)v73;
        v152 = *((unsigned int *)a2 + 2);
        v162 = &v159;
        v158 = v74;
        v163 = sub_2B1FB60;
        v164 = &v161;
        if ( sub_2B31C30(v74, v73, v152, v71, (__int64)v73, v72) )
          goto LABEL_57;
        if ( &v139[v152] == sub_2B1E290(v139, (__int64)&v139[v152], &v158, (__int64)&v163) )
          goto LABEL_57;
        v77 = *a2;
        v78 = v23[2];
        v161 = v4;
        v132 = v23 + 2;
        v140 = (unsigned __int8 **)v77;
        v153 = *((unsigned int *)a2 + 2);
        v162 = &v159;
        v158 = v78;
        v163 = sub_2B1FB60;
        v164 = &v161;
        if ( sub_2B31C30(v78, v77, v153, v75, (__int64)v77, v76) )
          goto LABEL_57;
        if ( &v140[v153] == sub_2B1E290(v140, (__int64)&v140[v153], &v158, (__int64)&v163) )
          goto LABEL_57;
        v81 = *a2;
        v82 = v23[3];
        v161 = v4;
        v132 = v23 + 3;
        v141 = (unsigned __int8 **)v81;
        v154 = *((unsigned int *)a2 + 2);
        v162 = &v159;
        v158 = v82;
        v163 = sub_2B1FB60;
        v164 = &v161;
        if ( sub_2B31C30(v82, v81, v154, v79, (__int64)v81, v80)
          || &v141[v154] == sub_2B1E290(v141, (__int64)&v141[v154], &v158, (__int64)&v163) )
        {
LABEL_57:
          v23 = v132;
          break;
        }
        v23 += 4;
        if ( v129 == v23 )
        {
          v27 = v126 - v23;
          goto LABEL_65;
        }
      }
LABEL_19:
      result = 1;
      if ( v126 != v23 )
        return result;
      return 0;
    }
LABEL_65:
    if ( v27 != 2 )
    {
      if ( v27 != 3 )
      {
        if ( v27 != 1 )
          return 0;
        goto LABEL_68;
      }
      v111 = *a2;
      v112 = *((unsigned int *)a2 + 2);
      v113 = *v23;
      v162 = &v159;
      v163 = sub_2B1FB60;
      v161 = v4;
      v158 = v113;
      v164 = &v161;
      if ( sub_2B31C30(v113, v111, v112, v26, v6, v7)
        || &v111[8 * v112] == (char *)sub_2B1E290(
                                        (unsigned __int8 **)v111,
                                        (__int64)&v111[8 * v112],
                                        &v158,
                                        (__int64)&v163) )
      {
        goto LABEL_19;
      }
      ++v23;
    }
    v114 = *v23;
    v161 = v4;
    v115 = *a2;
    v116 = *((unsigned int *)a2 + 2);
    v162 = &v159;
    v158 = v114;
    v163 = sub_2B1FB60;
    v164 = &v161;
    if ( sub_2B31C30(v114, v115, v116, v26, v6, v7)
      || &v115[8 * v116] == (char *)sub_2B1E290(
                                      (unsigned __int8 **)v115,
                                      (__int64)&v115[8 * v116],
                                      &v158,
                                      (__int64)&v163) )
    {
      goto LABEL_19;
    }
    ++v23;
LABEL_68:
    v83 = *v23;
    v161 = v4;
    v84 = *a2;
    v85 = *((unsigned int *)a2 + 2);
    v162 = &v159;
    v158 = v83;
    v163 = sub_2B1FB60;
    v164 = &v161;
    if ( sub_2B31C30(v83, v84, v85, v26, v6, v7)
      || &v84[8 * v85] == (char *)sub_2B1E290((unsigned __int8 **)v84, (__int64)&v84[8 * v85], &v158, (__int64)&v163) )
    {
      goto LABEL_19;
    }
    return 0;
  }
  v32 = *a2;
  v33 = *((unsigned int *)a2 + 2);
  v34 = (__int64 *)sub_2B0CB90((_BYTE **)*a2, (__int64)&(*a2)[8 * v33]);
  if ( v36 == v34 )
    return 0;
  v37 = *v34;
  v38 = *a1;
  v39 = *(_BYTE *)(*a1 + 392) & 1;
  if ( v39 )
  {
    v41 = v38 + 400;
    v42 = 3;
  }
  else
  {
    v40 = *(unsigned int *)(v38 + 408);
    v41 = *(_QWORD *)(v38 + 400);
    if ( !(_DWORD)v40 )
      goto LABEL_96;
    v42 = v40 - 1;
  }
  v43 = v42 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
  v44 = v41 + 72LL * v43;
  v35 = *(_QWORD *)v44;
  if ( v37 == *(_QWORD *)v44 )
    goto LABEL_27;
  v107 = 1;
  while ( v35 != -4096 )
  {
    v123 = v107 + 1;
    v43 = v42 & (v107 + v43);
    v44 = v41 + 72LL * v43;
    v35 = *(_QWORD *)v44;
    if ( v37 == *(_QWORD *)v44 )
      goto LABEL_27;
    v107 = v123;
  }
  if ( v39 )
  {
    v106 = 288;
    goto LABEL_97;
  }
  v40 = *(unsigned int *)(v38 + 408);
LABEL_96:
  v106 = 72 * v40;
LABEL_97:
  v44 = v41 + v106;
LABEL_27:
  v45 = 288;
  if ( !v39 )
    v45 = 72LL * *(unsigned int *)(v38 + 408);
  if ( v44 == v41 + v45 )
    return 0;
  v46 = *(__int64 **)(v44 + 8);
  v47 = *(unsigned int *)(v44 + 16);
  v159 = (__int64)v46;
  v160 = v47;
  if ( !v47 )
    return 0;
  v48 = 8 * v47;
  v49 = &v46[(unsigned __int64)v48 / 8];
  v50 = v48 >> 3;
  v51 = v48 >> 5;
  v127 = v49;
  if ( v51 )
  {
    v49 = &v161;
    v52 = v32;
    v53 = v33;
    v130 = &v46[4 * v51];
    while ( 1 )
    {
      v54 = *v46;
      v143 = (unsigned __int8 **)v52;
      v151 = v53;
      v161 = v38;
      v162 = &v159;
      v158 = v54;
      v163 = sub_2B1FE20;
      v164 = &v161;
      if ( sub_2B31C30(v54, v52, v53, (__int64)v49, (__int64)v52, v35)
        || &v143[v151] == sub_2B1E290(v143, (__int64)&v143[v151], &v158, (__int64)&v163) )
      {
        goto LABEL_34;
      }
      v88 = *a2;
      v89 = v46[1];
      v161 = v38;
      v133 = v46 + 1;
      v147 = (unsigned __int8 **)v88;
      v155 = *((unsigned int *)a2 + 2);
      v162 = &v159;
      v158 = v89;
      v163 = sub_2B1FE20;
      v164 = &v161;
      if ( sub_2B31C30(v89, v88, v155, v86, (__int64)v88, v87) )
        goto LABEL_73;
      if ( &v147[v155] == sub_2B1E290(v147, (__int64)&v147[v155], &v158, (__int64)&v163) )
        goto LABEL_73;
      v92 = *a2;
      v93 = v46[2];
      v161 = v38;
      v133 = v46 + 2;
      v148 = (unsigned __int8 **)v92;
      v156 = *((unsigned int *)a2 + 2);
      v162 = &v159;
      v158 = v93;
      v163 = sub_2B1FE20;
      v164 = &v161;
      if ( sub_2B31C30(v93, v92, v156, v90, (__int64)v92, v91) )
        goto LABEL_73;
      if ( &v148[v156] == sub_2B1E290(v148, (__int64)&v148[v156], &v158, (__int64)&v163) )
        goto LABEL_73;
      v96 = *a2;
      v97 = v46[3];
      v161 = v38;
      v133 = v46 + 3;
      v149 = (unsigned __int8 **)v96;
      v157 = *((unsigned int *)a2 + 2);
      v162 = &v159;
      v158 = v97;
      v163 = sub_2B1FE20;
      v164 = &v161;
      if ( sub_2B31C30(v97, v96, v157, v94, (__int64)v96, v95)
        || &v149[v157] == sub_2B1E290(v149, (__int64)&v149[v157], &v158, (__int64)&v163) )
      {
LABEL_73:
        v46 = v133;
        goto LABEL_34;
      }
      v46 += 4;
      if ( v46 == v130 )
      {
        v50 = v127 - v46;
        break;
      }
      v52 = *a2;
      v53 = *((unsigned int *)a2 + 2);
    }
  }
  if ( v50 == 2 )
    goto LABEL_119;
  if ( v50 == 3 )
  {
    v117 = *a2;
    v118 = *((unsigned int *)a2 + 2);
    v119 = *v46;
    v162 = &v159;
    v163 = sub_2B1FE20;
    v161 = v38;
    v158 = v119;
    v164 = &v161;
    if ( sub_2B31C30(v119, v117, v118, (__int64)v49, v41, v35)
      || &v117[8 * v118] == (char *)sub_2B1E290(
                                      (unsigned __int8 **)v117,
                                      (__int64)&v117[8 * v118],
                                      &v158,
                                      (__int64)&v163) )
    {
      goto LABEL_34;
    }
    ++v46;
LABEL_119:
    v120 = *v46;
    v161 = v38;
    v121 = *a2;
    v122 = *((unsigned int *)a2 + 2);
    v162 = &v159;
    v158 = v120;
    v163 = sub_2B1FE20;
    v164 = &v161;
    if ( sub_2B31C30(v120, v121, v122, (__int64)v49, v41, v35)
      || &v121[8 * v122] == (char *)sub_2B1E290(
                                      (unsigned __int8 **)v121,
                                      (__int64)&v121[8 * v122],
                                      &v158,
                                      (__int64)&v163) )
    {
      goto LABEL_34;
    }
    ++v46;
    goto LABEL_106;
  }
  if ( v50 != 1 )
    return 0;
LABEL_106:
  v108 = *v46;
  v161 = v38;
  v109 = *a2;
  v110 = *((unsigned int *)a2 + 2);
  v162 = &v159;
  v158 = v108;
  v163 = sub_2B1FE20;
  v164 = &v161;
  if ( !sub_2B31C30(v108, v109, v110, (__int64)v49, v41, v35)
    && &v109[8 * v110] != (char *)sub_2B1E290((unsigned __int8 **)v109, (__int64)&v109[8 * v110], &v158, (__int64)&v163) )
  {
    return 0;
  }
LABEL_34:
  result = 1;
  if ( v127 == v46 )
    return 0;
  return result;
}
