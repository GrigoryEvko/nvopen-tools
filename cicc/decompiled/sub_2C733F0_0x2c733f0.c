// Function: sub_2C733F0
// Address: 0x2c733f0
//
__int64 *__fastcall sub_2C733F0(__int64 a1, __int64 *a2, __int64 a3, unsigned __int8 a4, unsigned __int8 a5)
{
  __int64 v5; // r9
  __int16 v9; // ax
  __int64 *v10; // r12
  __int64 *v12; // rax
  __int64 v13; // r13
  __int64 v14; // rbx
  __int16 v15; // ax
  unsigned int v16; // r14d
  char v17; // al
  unsigned int v18; // ebx
  __int64 v19; // rax
  __int64 v20; // rcx
  __int16 v21; // ax
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 *v24; // rax
  __int64 v25; // r8
  __int64 v26; // rdx
  unsigned __int64 v27; // r9
  unsigned int v28; // edx
  __int64 v29; // r13
  __int16 v30; // ax
  _QWORD *v31; // rax
  __int64 v32; // r14
  __int64 v33; // r15
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // rax
  _QWORD *v38; // rax
  __int64 v39; // rcx
  __int64 v40; // rdx
  unsigned int v41; // r8d
  __int64 v42; // rsi
  __int64 *v43; // r9
  unsigned __int8 v44; // r8
  __int64 v45; // r13
  unsigned int v46; // r12d
  __int64 v47; // rax
  __int64 v48; // r14
  __int64 v49; // rax
  __int64 v50; // r8
  __int64 *v51; // r13
  __int64 v52; // rdx
  unsigned __int64 v53; // r9
  __int64 *v54; // rax
  unsigned __int8 v55; // r8
  __int64 v56; // r13
  unsigned __int64 v57; // rbx
  unsigned int v58; // r14d
  _QWORD *v59; // rax
  __int64 v60; // rax
  __int64 v61; // r9
  __int64 v62; // rdx
  unsigned __int64 v63; // r8
  __int64 *v64; // rax
  unsigned __int64 v65; // rdi
  char v66; // al
  char *v67; // rdx
  char *v68; // rcx
  unsigned int v69; // ebx
  __int64 v70; // r8
  char *v71; // rdx
  __int64 v72; // r13
  char v73; // r14
  __int64 v74; // rax
  __int64 v75; // r9
  __int64 v76; // rdx
  unsigned __int64 v77; // rdi
  bool v78; // al
  __int64 v79; // rax
  __int64 *v80; // rax
  unsigned __int8 v81; // r8
  __int64 *v82; // rbx
  __int64 v83; // r15
  __int64 v84; // rdx
  __int64 v85; // rcx
  __int64 v86; // r8
  __int64 *v87; // r14
  __int64 v88; // r9
  unsigned __int8 v89; // r8
  __int64 v90; // r13
  __int64 v91; // rbx
  _QWORD *v92; // rax
  __int64 v93; // rax
  __int64 v94; // r9
  __int64 v95; // rdx
  unsigned __int64 v96; // r8
  __int64 *v97; // rax
  __int64 *v98; // rax
  __int64 v99; // rax
  __int64 *v100; // r13
  __int64 v101; // rdx
  __int64 v102; // rcx
  __int64 v103; // r8
  __int64 *v104; // r15
  __int64 v105; // [rsp+8h] [rbp-F8h]
  unsigned int v106; // [rsp+8h] [rbp-F8h]
  __int64 v107; // [rsp+8h] [rbp-F8h]
  __int64 v108; // [rsp+8h] [rbp-F8h]
  __int64 v109; // [rsp+10h] [rbp-F0h]
  __int64 *v110; // [rsp+18h] [rbp-E8h]
  unsigned __int8 v111; // [rsp+18h] [rbp-E8h]
  char v112; // [rsp+18h] [rbp-E8h]
  __int64 *v113; // [rsp+18h] [rbp-E8h]
  __int64 v114; // [rsp+18h] [rbp-E8h]
  __int64 v115; // [rsp+20h] [rbp-E0h]
  __int64 v116; // [rsp+20h] [rbp-E0h]
  __int64 v117; // [rsp+20h] [rbp-E0h]
  __int64 v118; // [rsp+20h] [rbp-E0h]
  __int64 v119; // [rsp+20h] [rbp-E0h]
  unsigned __int8 v120; // [rsp+20h] [rbp-E0h]
  __int64 v121; // [rsp+20h] [rbp-E0h]
  unsigned __int8 v122; // [rsp+28h] [rbp-D8h]
  __int64 v123; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v124; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v125; // [rsp+30h] [rbp-D0h]
  __int64 v126; // [rsp+30h] [rbp-D0h]
  unsigned __int8 v127; // [rsp+30h] [rbp-D0h]
  __int64 v128; // [rsp+38h] [rbp-C8h]
  __int64 v129; // [rsp+38h] [rbp-C8h]
  __int64 v130; // [rsp+38h] [rbp-C8h]
  unsigned int v131; // [rsp+38h] [rbp-C8h]
  unsigned __int8 v132; // [rsp+38h] [rbp-C8h]
  unsigned __int8 v133; // [rsp+38h] [rbp-C8h]
  unsigned int v134; // [rsp+38h] [rbp-C8h]
  __int64 v135; // [rsp+38h] [rbp-C8h]
  unsigned int v136; // [rsp+38h] [rbp-C8h]
  __int64 v137; // [rsp+38h] [rbp-C8h]
  __int64 *v138; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v139; // [rsp+48h] [rbp-B8h]
  __int64 v140; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v141; // [rsp+58h] [rbp-A8h]
  char *v142; // [rsp+70h] [rbp-90h] BYREF
  __int64 v143; // [rsp+78h] [rbp-88h]
  char v144[32]; // [rsp+80h] [rbp-80h] BYREF
  __int64 *v145; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v146; // [rsp+A8h] [rbp-58h]
  __int64 v147; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v148; // [rsp+B8h] [rbp-48h]

  v5 = a1;
  v9 = *(_WORD *)(a1 + 24);
  if ( v9 != 6 )
  {
    if ( v9 == 4 )
    {
      v29 = *(_QWORD *)(a1 + 32);
      if ( a5 )
      {
        v132 = a5;
        v66 = sub_2C727F0(*(_QWORD *)(a1 + 32), a3, a4);
        a5 = v132;
        if ( !v66 )
          return (__int64 *)v5;
      }
      v30 = *(_WORD *)(v29 + 24);
      if ( v30 != 8 )
      {
        if ( v30 == 5 )
        {
          if ( !a4 && (*(_BYTE *)(v29 + 28) & 5) == 0 )
            return (__int64 *)v5;
          v122 = a5;
          v145 = &v147;
          v138 = &v140;
          v139 = 0x400000000LL;
          v142 = v144;
          v143 = 0x400000000LL;
          v146 = 0x400000000LL;
          v130 = v5;
          sub_2C728B0(
            (__int64)&v138,
            (char *)&v140,
            *(char **)(v29 + 32),
            (char *)(*(_QWORD *)(v29 + 32) + 8LL * *(_QWORD *)(v29 + 40)));
          v55 = v122;
          if ( (_DWORD)v139 )
          {
            v118 = v29;
            v123 = 8LL * (unsigned int)v139;
            v56 = v130;
            v131 = a4;
            v57 = 0;
            v58 = v55;
            do
            {
              v59 = sub_DC5000((__int64)a2, v138[v57 / 8], *(_QWORD *)(v56 + 40), 0);
              v60 = sub_2C733F0(v59, a2, a3, v131, v58);
              v62 = (unsigned int)v146;
              v63 = (unsigned int)v146 + 1LL;
              if ( v63 > HIDWORD(v146) )
              {
                v107 = v60;
                sub_C8D5F0((__int64)&v145, &v147, (unsigned int)v146 + 1LL, 8u, v63, v61);
                v62 = (unsigned int)v146;
                v60 = v107;
              }
              v57 += 8LL;
              v145[v62] = v60;
              LODWORD(v146) = v146 + 1;
            }
            while ( v123 != v57 );
            v64 = sub_DC7EB0(a2, (__int64)&v145, *(_WORD *)(v118 + 28) & 7, 0);
          }
          else
          {
            v64 = sub_DC7EB0(a2, (__int64)&v145, *(_WORD *)(v29 + 28) & 7, 0);
          }
          v10 = v64;
          if ( v145 != &v147 )
            _libc_free((unsigned __int64)v145);
          v65 = (unsigned __int64)v138;
          if ( v138 == &v140 )
            return v10;
LABEL_72:
          _libc_free(v65);
          return v10;
        }
        if ( v30 != 6 || !a4 && (*(_BYTE *)(v29 + 28) & 5) == 0 )
          return (__int64 *)v5;
        v120 = a5;
        v142 = v144;
        v143 = 0x400000000LL;
        v145 = &v147;
        v146 = 0x400000000LL;
        v135 = v5;
        sub_2C728B0(
          (__int64)&v142,
          v144,
          *(char **)(v29 + 32),
          (char *)(*(_QWORD *)(v29 + 32) + 8LL * *(_QWORD *)(v29 + 40)));
        v88 = v135;
        v89 = v120;
        if ( (_DWORD)v143 )
        {
          v114 = v29;
          v121 = 8LL * (unsigned int)v143;
          v136 = v89;
          v90 = 0;
          v91 = v88;
          do
          {
            v92 = sub_DC5000((__int64)a2, *(_QWORD *)&v142[v90], *(_QWORD *)(v91 + 40), 0);
            v93 = sub_2C733F0(v92, a2, a3, a4, v136);
            v95 = (unsigned int)v146;
            v96 = (unsigned int)v146 + 1LL;
            if ( v96 > HIDWORD(v146) )
            {
              v108 = v93;
              sub_C8D5F0((__int64)&v145, &v147, (unsigned int)v146 + 1LL, 8u, v96, v94);
              v95 = (unsigned int)v146;
              v93 = v108;
            }
            v90 += 8;
            v145[v95] = v93;
            LODWORD(v146) = v146 + 1;
          }
          while ( v121 != v90 );
          v97 = sub_DC8BD0(a2, (__int64)&v145, *(_WORD *)(v114 + 28) & 7, 0);
        }
        else
        {
          v97 = sub_DC8BD0(a2, (__int64)&v145, *(_WORD *)(v29 + 28) & 7, 0);
        }
        v77 = (unsigned __int64)v145;
        v10 = v97;
        if ( v145 == &v147 )
        {
LABEL_71:
          v65 = (unsigned __int64)v142;
          if ( v142 == v144 )
            return v10;
          goto LABEL_72;
        }
LABEL_70:
        _libc_free(v77);
        goto LABEL_71;
      }
      if ( a3 != *(_QWORD *)(v29 + 48) || !a4 && (*(_BYTE *)(v29 + 28) & 5) == 0 )
        return (__int64 *)v5;
      v129 = v5;
      v127 = a5;
      v31 = sub_DC5000((__int64)a2, **(_QWORD **)(v29 + 32), *(_QWORD *)(v5 + 40), 0);
      v32 = sub_2C733F0(v31, a2, a3, a4, v127);
      v33 = *(_QWORD *)(v129 + 40);
      v37 = sub_D33D80((_QWORD *)v29, (__int64)a2, v34, v35, v36);
      v38 = sub_DC5000((__int64)a2, v37, v33, 0);
      v39 = *(_QWORD *)(v29 + 48);
      v40 = (__int64)v38;
      v41 = *(_WORD *)(v29 + 28) & 7;
LABEL_26:
      v42 = v32;
      return sub_DC1960((__int64)a2, v42, v40, v39, v41);
    }
    v133 = a5;
    if ( v9 != 5 )
      return (__int64 *)v5;
    v67 = *(char **)(a1 + 32);
    v143 = 0x400000000LL;
    v146 = 0x400000000LL;
    v68 = &v67[8 * *(_QWORD *)(a1 + 40)];
    v142 = v144;
    v145 = &v147;
    sub_2C728B0((__int64)&v142, v144, v67, v68);
    v43 = (__int64 *)a1;
    if ( (_DWORD)v143 )
    {
      v69 = a4;
      v70 = v133;
      v71 = v142;
      v119 = 8LL * (unsigned int)v143;
      v72 = 0;
      v73 = 0;
      do
      {
        v134 = v70;
        v74 = sub_2C733F0(*(_QWORD *)&v71[v72], a2, a3, v69, v70);
        v76 = (unsigned int)v146;
        v70 = v134;
        if ( (unsigned __int64)(unsigned int)v146 + 1 > HIDWORD(v146) )
        {
          v106 = v134;
          v137 = v74;
          sub_C8D5F0((__int64)&v145, &v147, (unsigned int)v146 + 1LL, 8u, v70, v75);
          v76 = (unsigned int)v146;
          v70 = v106;
          v74 = v137;
        }
        v145[v76] = v74;
        v71 = v142;
        LODWORD(v146) = v146 + 1;
        if ( *(_QWORD *)&v142[v72] != v74 )
          v73 = 1;
        v72 += 8;
      }
      while ( v72 != v119 );
      v43 = (__int64 *)a1;
      if ( v73 )
      {
        v28 = *(_WORD *)(a1 + 28) & 7;
LABEL_18:
        v10 = sub_DC7EB0(a2, (__int64)&v145, v28, 0);
        goto LABEL_69;
      }
    }
    goto LABEL_68;
  }
  if ( *(_QWORD *)(a1 + 40) != 2 )
    return (__int64 *)v5;
  v12 = *(__int64 **)(a1 + 32);
  v13 = v12[1];
  v14 = *v12;
  v15 = *(_WORD *)(v13 + 24);
  v128 = v14;
  if ( v15 == 4 )
  {
    v16 = a4;
    if ( a5 )
    {
      v125 = a5;
      v17 = sub_2C727F0(*(_QWORD *)(v13 + 32), a3, a4);
      a5 = v125;
      if ( !v17 )
        return (__int64 *)v5;
    }
    v18 = a5;
    v126 = v5;
    v19 = sub_2C733F0(v13, a2, a3, v16, a5);
    v5 = v126;
    v105 = v19;
    v20 = v19;
    if ( v13 == v19 )
      return (__int64 *)v5;
    v21 = *(_WORD *)(v19 + 24);
    if ( v21 == 5 )
    {
      v142 = v144;
      v143 = 0x400000000LL;
      v145 = &v147;
      v146 = 0x400000000LL;
      sub_2C728B0(
        (__int64)&v142,
        v144,
        *(char **)(v20 + 32),
        (char *)(*(_QWORD *)(v20 + 32) + 8LL * *(_QWORD *)(v20 + 40)));
      if ( (_DWORD)v143 )
      {
        v22 = 0;
        v115 = 8LL * (unsigned int)v143;
        do
        {
          v23 = *(_QWORD *)&v142[v22];
          v138 = &v140;
          v140 = v128;
          v141 = v23;
          v139 = 0x200000002LL;
          v24 = sub_DC8BD0(a2, (__int64)&v138, 0, 0);
          if ( v138 != &v140 )
          {
            v110 = v24;
            _libc_free((unsigned __int64)v138);
            v24 = v110;
          }
          v26 = (unsigned int)v146;
          v27 = (unsigned int)v146 + 1LL;
          if ( v27 > HIDWORD(v146) )
          {
            v113 = v24;
            sub_C8D5F0((__int64)&v145, &v147, (unsigned int)v146 + 1LL, 8u, v25, v27);
            v26 = (unsigned int)v146;
            v24 = v113;
          }
          v22 += 8;
          v145[v26] = (__int64)v24;
          LODWORD(v146) = v146 + 1;
        }
        while ( v115 != v22 );
      }
      v28 = *(_WORD *)(v105 + 28) & 7;
      goto LABEL_18;
    }
    if ( v21 != 8 || a3 != *(_QWORD *)(v105 + 48) || !sub_DADE90((__int64)a2, v128, a3) )
    {
      v147 = v128;
      v145 = &v147;
      v148 = v105;
      v146 = 0x200000002LL;
      v98 = sub_DC8BD0(a2, (__int64)&v145, 0, 0);
      v65 = (unsigned __int64)v145;
      v10 = v98;
      if ( v145 == &v147 )
        return v10;
      goto LABEL_72;
    }
    v99 = **(_QWORD **)(v105 + 32);
    v147 = v128;
    v145 = &v147;
    v146 = 0x200000002LL;
    v148 = v99;
    v100 = sub_DC8BD0(a2, (__int64)&v145, 0, 0);
    if ( v145 != &v147 )
      _libc_free((unsigned __int64)v145);
    v32 = sub_2C733F0(v100, a2, a3, v16, v18);
    v148 = sub_D33D80((_QWORD *)v105, (__int64)a2, v101, v102, v103);
    v147 = v128;
    v145 = &v147;
    v146 = 0x200000002LL;
    v104 = sub_DC8BD0(a2, (__int64)&v145, 0, 0);
    if ( v145 != &v147 )
      _libc_free((unsigned __int64)v145);
    v40 = (__int64)v104;
    v39 = *(_QWORD *)(v105 + 48);
    v41 = *(_WORD *)(v105 + 28) & 7;
    goto LABEL_26;
  }
  if ( v15 == 5 )
  {
    v142 = v144;
    v143 = 0x400000000LL;
    v145 = &v147;
    v146 = 0x400000000LL;
    v111 = a5;
    sub_2C728B0(
      (__int64)&v142,
      v144,
      *(char **)(v13 + 32),
      (char *)(*(_QWORD *)(v13 + 32) + 8LL * *(_QWORD *)(v13 + 40)));
    v43 = (__int64 *)a1;
    v44 = v111;
    if ( (_DWORD)v143 )
    {
      v112 = 0;
      v116 = 8LL * (unsigned int)v143;
      v109 = v13;
      v45 = 0;
      v46 = v44;
      do
      {
        v47 = sub_2C733F0(*(_QWORD *)&v142[v45], a2, a3, a4, v46);
        if ( *(_QWORD *)&v142[v45] != v47 )
        {
          *(_QWORD *)&v142[v45] = v47;
          v112 = 1;
        }
        v45 += 8;
      }
      while ( v116 != v45 );
      v43 = (__int64 *)a1;
      if ( v112 )
      {
        if ( (_DWORD)v143 )
        {
          v117 = 8LL * (unsigned int)v143;
          v48 = 0;
          do
          {
            v49 = *(_QWORD *)&v142[v48];
            v138 = &v140;
            v140 = v14;
            v141 = v49;
            v139 = 0x200000002LL;
            v51 = sub_DC8BD0(a2, (__int64)&v138, 0, 0);
            if ( v138 != &v140 )
              _libc_free((unsigned __int64)v138);
            v52 = (unsigned int)v146;
            v53 = (unsigned int)v146 + 1LL;
            if ( v53 > HIDWORD(v146) )
            {
              sub_C8D5F0((__int64)&v145, &v147, (unsigned int)v146 + 1LL, 8u, v50, v53);
              v52 = (unsigned int)v146;
            }
            v48 += 8;
            v145[v52] = (__int64)v51;
            LODWORD(v146) = v146 + 1;
          }
          while ( v117 != v48 );
          v54 = sub_DC7EB0(a2, (__int64)&v145, *(_WORD *)(v109 + 28) & 7, 0);
        }
        else
        {
          v54 = sub_DC7EB0(a2, (__int64)&v145, *(_WORD *)(v109 + 28) & 7, 0);
        }
        v10 = v54;
LABEL_69:
        v77 = (unsigned __int64)v145;
        if ( v145 == &v147 )
          goto LABEL_71;
        goto LABEL_70;
      }
    }
LABEL_68:
    v10 = v43;
    goto LABEL_69;
  }
  if ( v15 != 8 )
    return (__int64 *)v5;
  if ( a3 != *(_QWORD *)(v13 + 48) )
    return (__int64 *)v5;
  v124 = a5;
  v78 = sub_DADE90((__int64)a2, v14, a3);
  v5 = a1;
  if ( !v78 )
    return (__int64 *)v5;
  v79 = **(_QWORD **)(v13 + 32);
  v147 = v14;
  v146 = 0x200000002LL;
  v145 = &v147;
  v148 = v79;
  v80 = sub_DC8BD0(a2, (__int64)&v145, 0, 0);
  v81 = v124;
  v82 = v80;
  if ( v145 != &v147 )
  {
    _libc_free((unsigned __int64)v145);
    v81 = v124;
  }
  v83 = sub_2C733F0(v82, a2, a3, a4, v81);
  v148 = sub_D33D80((_QWORD *)v13, (__int64)a2, v84, v85, v86);
  v147 = v128;
  v145 = &v147;
  v146 = 0x200000002LL;
  v87 = sub_DC8BD0(a2, (__int64)&v145, 0, 0);
  if ( v145 != &v147 )
    _libc_free((unsigned __int64)v145);
  v39 = *(_QWORD *)(v13 + 48);
  v40 = (__int64)v87;
  v42 = v83;
  v41 = *(_WORD *)(v13 + 28) & 7;
  return sub_DC1960((__int64)a2, v42, v40, v39, v41);
}
