// Function: sub_2B28940
// Address: 0x2b28940
//
unsigned __int64 __fastcall sub_2B28940(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, char a5, int a6, __int64 a7)
{
  __int64 *v7; // r15
  __int64 v10; // r13
  unsigned __int64 v11; // r12
  __int64 *v12; // rax
  __int64 v13; // rsi
  _DWORD *v14; // rax
  int v15; // esi
  int v16; // edi
  int v17; // r12d
  unsigned int v18; // r15d
  __int64 v19; // r12
  unsigned __int64 v20; // rbx
  __int64 v21; // rax
  bool v22; // of
  unsigned __int64 v23; // rbx
  __int64 v24; // rax
  unsigned __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 *v28; // rcx
  int v29; // eax
  __int64 v30; // rbx
  char i; // al
  __int64 v32; // r9
  __int64 v33; // rbx
  int v34; // edx
  int v35; // r12d
  int v36; // eax
  unsigned __int64 v37; // rax
  __int64 v38; // r10
  __int64 v39; // rbx
  __int64 v40; // r13
  __int64 **v41; // r15
  __int64 v42; // r14
  __int64 v43; // rdi
  __int64 v44; // rax
  int v45; // edi
  int v46; // ecx
  __int64 v47; // r12
  __int64 *v48; // rbx
  unsigned __int64 v49; // r13
  char v50; // al
  __int64 v51; // rax
  int v52; // edx
  bool v53; // zf
  int v54; // edx
  __int64 v55; // rbx
  __int64 v56; // r13
  __int64 v57; // rdi
  __int64 v58; // rax
  int v59; // eax
  __int64 v60; // rax
  __int64 v62; // rax
  __int64 v63; // r8
  __int64 v64; // rax
  __int64 v65; // rax
  unsigned __int64 v66; // rdx
  __int64 v67; // r12
  unsigned __int8 **v68; // rsi
  int v69; // ecx
  unsigned __int8 **v70; // r10
  unsigned __int64 v71; // rax
  __int64 v72; // r12
  int v73; // edx
  int v74; // r13d
  int v75; // eax
  __int64 v76; // rax
  __int64 v77; // r8
  __int64 v78; // r11
  __int64 v79; // r8
  __int64 v80; // r8
  unsigned __int64 v81; // rdx
  __int64 v82; // rax
  unsigned __int8 **v83; // r9
  int v84; // ecx
  unsigned __int8 **v85; // rdi
  unsigned __int64 v86; // rcx
  int v87; // edx
  __int64 v88; // r14
  int v89; // eax
  unsigned __int64 v90; // rax
  __int64 v91; // r12
  char v92; // bl
  __int64 v93; // rdx
  bool v94; // cf
  __int64 v95; // rax
  int v96; // r12d
  _QWORD *v97; // rax
  __int64 v98; // rax
  __int64 v99; // r12
  __int64 v100; // r10
  __int64 v101; // rbx
  unsigned int v102; // r12d
  __int64 v103; // rsi
  __int64 v104; // rcx
  unsigned __int64 v105; // rax
  __int64 v106; // rbx
  unsigned int v107; // r12d
  __int64 v108; // rsi
  __int64 v109; // rcx
  unsigned __int64 v110; // rax
  __int64 v111; // rax
  bool v112; // cc
  unsigned __int64 v113; // rax
  __int128 v114; // [rsp-18h] [rbp-1C8h]
  __int128 v115; // [rsp-18h] [rbp-1C8h]
  __int128 v116; // [rsp-18h] [rbp-1C8h]
  __int64 v117; // [rsp+0h] [rbp-1B0h]
  __int64 v118; // [rsp+8h] [rbp-1A8h]
  __int64 v119; // [rsp+8h] [rbp-1A8h]
  __int64 v120; // [rsp+10h] [rbp-1A0h]
  __int64 v121; // [rsp+10h] [rbp-1A0h]
  int v122; // [rsp+18h] [rbp-198h]
  __int64 v123; // [rsp+18h] [rbp-198h]
  __int64 v124; // [rsp+18h] [rbp-198h]
  int v125; // [rsp+28h] [rbp-188h]
  __int64 v126; // [rsp+30h] [rbp-180h]
  __int64 *v128; // [rsp+38h] [rbp-178h]
  int v129; // [rsp+38h] [rbp-178h]
  __int64 v130; // [rsp+40h] [rbp-170h]
  __int64 v131; // [rsp+40h] [rbp-170h]
  __int64 v132; // [rsp+40h] [rbp-170h]
  __int64 v134; // [rsp+50h] [rbp-160h]
  __int64 *v135; // [rsp+50h] [rbp-160h]
  int v136; // [rsp+58h] [rbp-158h]
  unsigned __int64 v137; // [rsp+58h] [rbp-158h]
  __int64 v138; // [rsp+60h] [rbp-150h]
  unsigned __int64 v139; // [rsp+68h] [rbp-148h]
  __int64 v142; // [rsp+78h] [rbp-138h]
  __int64 v143; // [rsp+78h] [rbp-138h]
  int v144; // [rsp+80h] [rbp-130h]
  char v145; // [rsp+80h] [rbp-130h]
  unsigned int v146; // [rsp+90h] [rbp-120h]
  int v147; // [rsp+90h] [rbp-120h]
  char v148; // [rsp+90h] [rbp-120h]
  unsigned __int64 v149; // [rsp+98h] [rbp-118h]
  __int64 v150; // [rsp+A0h] [rbp-110h]
  __int64 v151; // [rsp+A0h] [rbp-110h]
  __int64 *v152; // [rsp+B0h] [rbp-100h]
  unsigned int v153; // [rsp+B0h] [rbp-100h]
  int v155; // [rsp+B8h] [rbp-F8h]
  int v156; // [rsp+B8h] [rbp-F8h]
  _QWORD v157[2]; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v158; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 v159; // [rsp+D8h] [rbp-D8h]
  unsigned __int8 **v160; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 v161; // [rsp+E8h] [rbp-C8h]
  char v162[8]; // [rsp+F0h] [rbp-C0h] BYREF
  _BYTE *v163; // [rsp+F8h] [rbp-B8h]
  _BYTE v164[32]; // [rsp+108h] [rbp-A8h] BYREF
  _BYTE *v165; // [rsp+128h] [rbp-88h]
  _BYTE v166[120]; // [rsp+138h] [rbp-78h] BYREF

  v7 = a3;
  v10 = *(_QWORD *)(*a3 + 8);
  v11 = *(unsigned int *)(a7 + 3552);
  v12 = *(__int64 **)a7;
  if ( (_DWORD)v11
    && (v13 = *(_QWORD *)(**(_QWORD **)*v12 + 8LL), *(_BYTE *)(v13 + 8) == 12)
    && (v157[0] = sub_9208B0(*(_QWORD *)(a7 + 3344), v13),
        v157[1] = v93,
        v94 = v11 < sub_CA1930(v157),
        v12 = *(__int64 **)a7,
        v94) )
  {
    v95 = *v12;
    v96 = *(_DWORD *)(v95 + 120);
    if ( !v96 )
      v96 = *(_DWORD *)(v95 + 8);
    v153 = *(_DWORD *)(a7 + 3552);
    v97 = (_QWORD *)sub_BD5C60(**(_QWORD **)v95);
    v98 = sub_BCCE00(v97, v153);
    v130 = sub_2B08680(v98, v96);
  }
  else
  {
    v14 = (_DWORD *)*v12;
    v15 = v14[30];
    if ( !v15 )
      v15 = v14[2];
    v130 = sub_2B08680(*(_QWORD *)(**(_QWORD **)v14 + 8LL), v15);
  }
  v152 = &a3[a4];
  if ( v152 != sub_2B0BF30(a3, (__int64)v152, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B0D8B0) )
  {
    v16 = *(_DWORD *)(a1 + 1576);
    v17 = *(_DWORD *)(a1 + 1592);
    if ( v16 > 11 )
    {
      if ( (unsigned int)(v16 - 12) > 3 )
        goto LABEL_197;
    }
    else
    {
      if ( v16 > 9 )
        goto LABEL_11;
      if ( v16 <= 5 )
      {
        if ( v16 > 0 )
        {
LABEL_11:
          v18 = sub_1022EF0(v16);
          if ( *(_BYTE *)(v10 + 8) == 17 )
          {
            v146 = *(_DWORD *)(v10 + 32);
            if ( (_DWORD)a4 )
            {
              v19 = 0;
              v20 = 0;
              do
              {
                sub_9B95E0((__int64 *)&v160, v19, v146, a4);
                v21 = sub_DFBC30(a2, 7, v130, (__int64)v160, (unsigned int)v161, 0, 0, 0, 0, 0, 0);
                v22 = __OFADD__(v21, v20);
                v23 = v21 + v20;
                if ( v22 )
                {
                  v23 = 0x8000000000000000LL;
                  if ( v21 > 0 )
                    v23 = 0x7FFFFFFFFFFFFFFFLL;
                }
                if ( v160 != (unsigned __int8 **)v162 )
                  _libc_free((unsigned __int64)v160);
                BYTE4(v160) = 1;
                LODWORD(v160) = a6;
                v24 = sub_DFDC10(a2, v18, v10, (__int64)v160);
                v22 = __OFADD__(v24, v23);
                v20 = v24 + v23;
                if ( v22 )
                {
                  v20 = 0x8000000000000000LL;
                  if ( v24 > 0 )
                    v20 = 0x7FFFFFFFFFFFFFFFLL;
                }
                ++v19;
              }
              while ( (unsigned int)a4 != v19 );
            }
            else
            {
              v20 = 0;
            }
            LODWORD(v161) = v146;
            if ( v146 > 0x40 )
            {
              sub_C43690((__int64)&v160, -1, 1);
            }
            else
            {
              v25 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v146;
              if ( !v146 )
                v25 = 0;
              v160 = (unsigned __int8 **)v25;
            }
            v26 = sub_DFAAD0(a2, v10, (__int64)&v160, 1u, 0);
            v22 = __OFADD__(v26, v20);
            v27 = v26 + v20;
            if ( v22 )
            {
              v112 = v26 <= 0;
              v113 = 0x8000000000000000LL;
              if ( !v112 )
                v113 = 0x7FFFFFFFFFFFFFFFLL;
              v126 = v113;
            }
            else
            {
              v126 = v27;
            }
            if ( (unsigned int)v161 > 0x40 && v160 )
              j_j___libc_free_0_0((unsigned __int64)v160);
          }
          else
          {
            v53 = v17 == 0;
            v91 = *(_QWORD *)(v130 + 24);
            if ( v53 )
            {
              sub_2B28800((__int64)&v160, (__int64 **)a7);
              if ( !v162[0] || (v92 = v161, (unsigned __int8 **)v91 == v160) )
              {
                BYTE4(v160) = 1;
                LODWORD(v160) = a6;
                v126 = sub_DFDC10(a2, v18, v130, (__int64)v160);
              }
              else
              {
                sub_2B08680((__int64)v160, a4);
                v126 = sub_DFDCC0(a2, v18, v92 ^ 1u, v91);
              }
            }
            else
            {
              sub_2B28800((__int64)&v160, (__int64 **)a7);
              if ( v162[0] )
              {
                v143 = (__int64)v160;
                v148 = v161;
                v106 = sub_2B08680((__int64)v160, a4);
                v126 = sub_DFD800((__int64)a2, v18, v106, 0, 0, 0, 0, 0, 0, 0);
                if ( v143 != v91 )
                {
                  v107 = sub_BCB060(v91);
                  v108 = 39 - ((unsigned int)(v148 == 0) - 1);
                  if ( v107 <= (unsigned int)sub_BCB060(v143) )
                    v108 = 38;
                  v109 = sub_DFD060(a2, v108, v130, v106);
                  v110 = v109 + v126;
                  if ( __OFADD__(v109, v126) )
                  {
                    v110 = 0x8000000000000000LL;
                    if ( v109 > 0 )
                      v110 = 0x7FFFFFFFFFFFFFFFLL;
                  }
                  v126 = v110;
                }
              }
              else
              {
                v111 = sub_2B08680(v91, a4);
                v126 = sub_DFD800((__int64)a2, v18, v111, 0, 0, 0, 0, 0, 0, 0);
              }
            }
          }
          goto LABEL_60;
        }
LABEL_197:
        BUG();
      }
    }
    v136 = sub_F6F040(v16);
    if ( v17 )
    {
      v99 = *(_QWORD *)(v130 + 24);
      sub_2B28800((__int64)&v160, (__int64 **)a7);
      if ( v162[0] )
      {
        v100 = (__int64)v160;
        v145 = v161;
      }
      else
      {
        v145 = 1;
        v100 = v99;
      }
      v151 = v100;
      v101 = sub_2B08680(v100, a4);
      v158 = v101;
      v159 = v101;
      *((_QWORD *)&v116 + 1) = 1;
      *(_QWORD *)&v116 = 0;
      sub_DF8CB0((__int64)&v160, v136, v101, (char *)&v158, 2, a6, 0, v116);
      v126 = sub_DFD690((__int64)a2, (__int64)&v160);
      if ( v151 != v99 )
      {
        v102 = sub_BCB060(v99);
        v103 = 39 - ((unsigned int)(v145 == 0) - 1);
        if ( v102 <= (unsigned int)sub_BCB060(v151) )
          v103 = 38;
        v104 = sub_DFD060(a2, v103, v130, v101);
        v105 = v104 + v126;
        if ( __OFADD__(v104, v126) )
        {
          v105 = 0x8000000000000000LL;
          if ( v104 > 0 )
            v105 = 0x7FFFFFFFFFFFFFFFLL;
        }
        v126 = v105;
      }
      if ( v165 != v166 )
        _libc_free((unsigned __int64)v165);
      if ( v163 != v164 )
        _libc_free((unsigned __int64)v163);
    }
    else
    {
      v126 = sub_DFDC70((__int64)a2);
    }
    v28 = a3;
    if ( v152 == a3 )
      return v126;
    goto LABEL_32;
  }
  v45 = *(_DWORD *)(a1 + 1576);
  if ( v45 > 11 )
  {
    if ( (unsigned int)(v45 - 12) > 3 )
      goto LABEL_197;
    goto LABEL_80;
  }
  if ( v45 > 9 )
    goto LABEL_59;
  if ( v45 > 5 )
  {
LABEL_80:
    v59 = sub_F6F040(v45);
    v28 = a3;
    v136 = v59;
    v126 = 0;
    if ( v152 == a3 )
      return v126;
LABEL_32:
    v29 = a4;
    v30 = *v28;
    if ( (_DWORD)a4 == 1 )
      return v126;
    v155 = 0;
    v150 = 0;
    v144 = 3 - (a5 == 0);
    v134 = (__int64)&v28[(unsigned int)(v29 - 2) + 1];
    for ( i = sub_BD3660(v30, v144); ; i = sub_BD3660(v30, v144) )
    {
      if ( i )
      {
        v158 = v10;
        v159 = v10;
        v149 = v149 & 0xFFFFFFFF00000000LL | 1;
        *((_QWORD *)&v114 + 1) = v149;
        *(_QWORD *)&v114 = 0;
        sub_DF8CB0((__int64)&v160, v136, v10, (char *)&v158, 2, a6, 0, v114);
LABEL_35:
        v33 = sub_DFD690((__int64)a2, (__int64)&v160);
        v35 = v34;
        if ( v165 != v166 )
          _libc_free((unsigned __int64)v165);
        if ( v163 != v164 )
          _libc_free((unsigned __int64)v163);
        v36 = 1;
        if ( v35 != 1 )
          v36 = v155;
        v155 = v36;
        v37 = v33 + v150;
        if ( __OFADD__(v33, v150) )
        {
          v37 = 0x8000000000000000LL;
          if ( v33 > 0 )
            v37 = 0x7FFFFFFFFFFFFFFFLL;
        }
LABEL_42:
        v150 = v37;
        goto LABEL_43;
      }
      v38 = *(_QWORD *)(v30 + 16);
      if ( !v38 )
        goto LABEL_43;
      v131 = v10;
      v39 = 0;
      v125 = 0;
      v40 = v38;
      v128 = v7;
      v41 = (__int64 **)a2;
      do
      {
        v42 = *(_QWORD *)(v40 + 24);
        if ( !a5 )
        {
          v44 = *(_QWORD *)(v42 + 16);
          if ( !v44 )
            goto LABEL_53;
LABEL_106:
          if ( *(_QWORD *)(v44 + 8) )
            goto LABEL_53;
          goto LABEL_107;
        }
        v43 = *(_QWORD *)(v40 + 24);
        if ( *(_BYTE *)v42 == 86 )
        {
          if ( !(unsigned __int8)sub_BD3610(v43, 2) || (v44 = *(_QWORD *)(*(_QWORD *)(v42 - 96) + 16LL)) == 0 )
          {
LABEL_53:
            a2 = (__int64 *)v41;
            v10 = v131;
            v7 = v128;
LABEL_54:
            v158 = v10;
            v159 = v10;
            v139 = v139 & 0xFFFFFFFF00000000LL | 1;
            *((_QWORD *)&v115 + 1) = v139;
            *(_QWORD *)&v115 = 0;
            sub_DF8CB0((__int64)&v160, v136, v10, (char *)&v158, 2, a6, 0, v115);
            goto LABEL_35;
          }
          goto LABEL_106;
        }
        if ( !(unsigned __int8)sub_BD3610(v43, 2) )
          goto LABEL_53;
LABEL_107:
        v77 = 32LL * (*(_DWORD *)(v42 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v42 + 7) & 0x40) != 0 )
        {
          v78 = *(_QWORD *)(v42 - 8);
          v79 = v78 + v77;
        }
        else
        {
          v78 = v42 - v77;
          v79 = v42;
        }
        v80 = v79 - v78;
        v160 = (unsigned __int8 **)v162;
        v81 = v80 >> 5;
        v161 = 0x400000000LL;
        v82 = v80 >> 5;
        if ( (unsigned __int64)v80 > 0x80 )
        {
          v117 = v80 >> 5;
          v118 = v80;
          v120 = v78;
          v123 = v80 >> 5;
          sub_C8D5F0((__int64)&v160, v162, v81, 8u, v80, v32);
          v83 = v160;
          v84 = v161;
          LODWORD(v81) = v123;
          v78 = v120;
          v80 = v118;
          v82 = v117;
          v85 = &v160[(unsigned int)v161];
        }
        else
        {
          v83 = (unsigned __int8 **)v162;
          v84 = 0;
          v85 = (unsigned __int8 **)v162;
        }
        if ( v80 > 0 )
        {
          v86 = 0;
          do
          {
            v85[v86 / 8] = *(unsigned __int8 **)(v78 + 4 * v86);
            v86 += 8LL;
            --v82;
          }
          while ( v82 );
          v83 = v160;
          v84 = v161;
        }
        LODWORD(v161) = v81 + v84;
        v88 = sub_DFCEF0(v41, (unsigned __int8 *)v42, v83, (unsigned int)(v81 + v84), 0);
        if ( v160 != (unsigned __int8 **)v162 )
        {
          v122 = v87;
          _libc_free((unsigned __int64)v160);
          v87 = v122;
        }
        v89 = 1;
        if ( v87 != 1 )
          v89 = v125;
        v22 = __OFADD__(v88, v39);
        v39 += v88;
        v125 = v89;
        if ( v22 )
        {
          v39 = 0x8000000000000000LL;
          if ( v88 > 0 )
            v39 = 0x7FFFFFFFFFFFFFFFLL;
        }
        v40 = *(_QWORD *)(v40 + 8);
      }
      while ( v40 );
      a2 = (__int64 *)v41;
      v10 = v131;
      v7 = v128;
      if ( v89 )
        goto LABEL_54;
      v37 = v39 + v150;
      if ( !__OFADD__(v39, v150) )
        goto LABEL_42;
      v90 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v39 <= 0 )
        v90 = 0x8000000000000000LL;
      v150 = v90;
LABEL_43:
      if ( v152 != ++v7 )
      {
        v30 = *v7;
        if ( (__int64 *)v134 != v7 )
          continue;
      }
      goto LABEL_83;
    }
  }
  if ( v45 <= 0 )
    goto LABEL_197;
LABEL_59:
  v126 = 0;
  v18 = sub_1022EF0(v45);
LABEL_60:
  if ( v152 == a3 )
    return v126;
  v46 = a4;
  v47 = *a3;
  if ( (_DWORD)a4 == 1 )
    return v126;
  v48 = a3 + 1;
  v142 = v10;
  v156 = 0;
  v147 = 3 - (a5 == 0);
  v138 = (__int64)&a3[(unsigned int)(v46 - 2) + 2];
  v49 = 0;
  v50 = sub_BD3660(v47, v147);
  while ( 2 )
  {
    if ( v50 )
    {
LABEL_63:
      v51 = sub_DFD800((__int64)a2, v18, v142, 0, 0, 0, 0, 0, 0, 0);
      v53 = v52 == 1;
      v54 = 1;
      if ( !v53 )
        v54 = v156;
      v156 = v54;
      if ( __OFADD__(v51, v49) )
      {
        v49 = 0x8000000000000000LL;
        if ( v51 > 0 )
          v49 = 0x7FFFFFFFFFFFFFFFLL;
      }
      else
      {
        v49 += v51;
      }
      goto LABEL_67;
    }
    if ( !*(_QWORD *)(v47 + 16) )
      goto LABEL_67;
    v137 = v49;
    v129 = 0;
    v132 = 0;
    v135 = v48;
    v55 = *(_QWORD *)(v47 + 16);
    do
    {
      v56 = *(_QWORD *)(v55 + 24);
      if ( !a5 )
      {
        v58 = *(_QWORD *)(v56 + 16);
        if ( !v58 )
          goto LABEL_77;
LABEL_87:
        if ( *(_QWORD *)(v58 + 8) )
          goto LABEL_77;
        goto LABEL_88;
      }
      v57 = *(_QWORD *)(v55 + 24);
      if ( *(_BYTE *)v56 == 86 )
      {
        if ( !(unsigned __int8)sub_BD3610(v57, 2) || (v58 = *(_QWORD *)(*(_QWORD *)(v56 - 96) + 16LL)) == 0 )
        {
LABEL_77:
          v49 = v137;
          v48 = v135;
          goto LABEL_63;
        }
        goto LABEL_87;
      }
      if ( !(unsigned __int8)sub_BD3610(v57, 2) )
        goto LABEL_77;
LABEL_88:
      v62 = 32LL * (*(_DWORD *)(v56 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(v56 + 7) & 0x40) != 0 )
      {
        v63 = *(_QWORD *)(v56 - 8);
        v64 = v63 + v62;
      }
      else
      {
        v63 = v56 - v62;
        v64 = v56;
      }
      v65 = v64 - v63;
      v160 = (unsigned __int8 **)v162;
      v66 = v65 >> 5;
      v161 = 0x400000000LL;
      v67 = v65 >> 5;
      if ( (unsigned __int64)v65 > 0x80 )
      {
        v119 = v65;
        v121 = v63;
        v124 = v65 >> 5;
        sub_C8D5F0((__int64)&v160, v162, v66, 8u, v63, (__int64)v162);
        v70 = v160;
        v69 = v161;
        LODWORD(v66) = v124;
        v63 = v121;
        v65 = v119;
        v68 = &v160[(unsigned int)v161];
      }
      else
      {
        v68 = (unsigned __int8 **)v162;
        v69 = 0;
        v70 = (unsigned __int8 **)v162;
      }
      if ( v65 > 0 )
      {
        v71 = 0;
        do
        {
          v68[v71 / 8] = *(unsigned __int8 **)(v63 + 4 * v71);
          v71 += 8LL;
          --v67;
        }
        while ( v67 );
        v70 = v160;
        v69 = v161;
      }
      LODWORD(v161) = v66 + v69;
      v72 = sub_DFCEF0((__int64 **)a2, (unsigned __int8 *)v56, v70, (unsigned int)(v66 + v69), 0);
      v74 = v73;
      if ( v160 != (unsigned __int8 **)v162 )
        _libc_free((unsigned __int64)v160);
      v75 = 1;
      if ( v74 != 1 )
        v75 = v129;
      v129 = v75;
      v76 = v72 + v132;
      if ( __OFADD__(v72, v132) )
      {
        v76 = 0x8000000000000000LL;
        if ( v72 > 0 )
          v76 = 0x7FFFFFFFFFFFFFFFLL;
      }
      v55 = *(_QWORD *)(v55 + 8);
      v132 = v76;
    }
    while ( v55 );
    v49 = v137;
    v48 = v135;
    if ( v129 )
      goto LABEL_63;
    if ( __OFADD__(v76, v137) )
    {
      v49 = 0x8000000000000000LL;
      if ( v76 > 0 )
        v49 = 0x7FFFFFFFFFFFFFFFLL;
    }
    else
    {
      v49 = v76 + v137;
    }
LABEL_67:
    if ( v152 != v48 )
    {
      v47 = *v48++;
      if ( (__int64 *)v138 != v48 )
      {
        v50 = sub_BD3660(v47, v147);
        continue;
      }
    }
    break;
  }
  v150 = v49;
LABEL_83:
  v60 = v126 - v150;
  if ( __OFSUB__(v126, v150) )
  {
    v60 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v150 > 0 )
      return 0x8000000000000000LL;
  }
  return v60;
}
