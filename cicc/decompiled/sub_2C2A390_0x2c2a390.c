// Function: sub_2C2A390
// Address: 0x2c2a390
//
__int64 __fastcall sub_2C2A390(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rbx
  __int64 i; // r12
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 *v15; // rdi
  __int64 v16; // rbx
  __int64 (__fastcall *v17)(__int64); // rax
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 (*v20)(void); // rax
  __int64 v21; // rdi
  __int64 v22; // r12
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // rax
  __int64 *v28; // r15
  _QWORD *v29; // rax
  __int64 *v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 *v34; // r13
  __int64 *v35; // r14
  __int64 v36; // r15
  __int64 v37; // r12
  __int64 *v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r13
  __int64 *v41; // r9
  __int64 *v42; // r15
  unsigned __int64 v43; // rax
  __int64 *v44; // r13
  __int64 *v45; // r15
  __int64 v46; // r12
  __int64 *j; // rbx
  __int64 v48; // rdx
  __int64 *v49; // r13
  __int64 *v50; // r13
  __int64 *v51; // r14
  __int64 v52; // rsi
  __int64 v53; // r15
  int v54; // r15d
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  __int64 v57; // r13
  __int64 v58; // rax
  __int64 (__fastcall *v59)(__int64); // rax
  __int64 v60; // rax
  _QWORD *v61; // r13
  __int64 v62; // rsi
  __int64 v63; // r14
  _QWORD *v64; // rdi
  __int64 v65; // rsi
  _QWORD *v66; // rax
  __int64 v67; // r8
  _QWORD *v68; // r9
  __int64 v69; // rax
  unsigned int v70; // r14d
  _BYTE *v71; // rbx
  _BYTE *v72; // r12
  unsigned __int64 v73; // r13
  unsigned __int64 v74; // rdi
  __int64 *v76; // rcx
  __int64 *v77; // r13
  __int64 v78; // r12
  __int64 *v79; // rbx
  __int64 v80; // r15
  _QWORD *v81; // rax
  __int64 v82; // rax
  __int64 v83; // r13
  __int64 v84; // r14
  char v85; // al
  char v86; // dl
  char v87; // dl
  __int64 v88; // r8
  __int64 v89; // r9
  __int64 v90; // rax
  unsigned __int64 v91; // rdx
  __int64 v92; // rdx
  __int64 v93; // r12
  __int64 *v94; // rbx
  __int64 *v95; // r12
  __int64 v96; // rax
  __int64 v97; // r15
  __int64 v98; // r14
  char v99; // dl
  __int64 v100; // r8
  __int64 v101; // r9
  __int64 v102; // rax
  unsigned __int64 v103; // rdx
  __int64 v104; // rax
  __int64 *v105; // r9
  unsigned __int64 v106; // r13
  __int64 *v107; // r14
  unsigned __int64 v108; // rax
  __int64 *v109; // r13
  unsigned __int8 v110; // r9
  __int64 v111; // rcx
  __int64 v112; // r8
  __int64 *v113; // r15
  __int64 v114; // r12
  __int64 *v115; // rbx
  __int64 *v116; // r15
  __int64 v117; // r14
  __int64 v118; // r13
  __int64 v119; // rdx
  char v120; // al
  __int64 *v121; // r13
  __int64 *v122; // r14
  __int64 *v123; // r13
  _QWORD *v124; // rdi
  __int64 v125; // [rsp+18h] [rbp-278h]
  __int64 v126; // [rsp+20h] [rbp-270h]
  unsigned __int8 v127; // [rsp+2Bh] [rbp-265h]
  unsigned int v128; // [rsp+2Ch] [rbp-264h]
  unsigned __int8 v129; // [rsp+2Ch] [rbp-264h]
  __int64 v130; // [rsp+30h] [rbp-260h]
  __int64 v131; // [rsp+30h] [rbp-260h]
  __int64 *v132; // [rsp+38h] [rbp-258h]
  __int64 v133; // [rsp+38h] [rbp-258h]
  __int64 v134; // [rsp+38h] [rbp-258h]
  char v135; // [rsp+38h] [rbp-258h]
  __int64 *v136; // [rsp+38h] [rbp-258h]
  __int64 *v137; // [rsp+48h] [rbp-248h]
  unsigned int v139; // [rsp+70h] [rbp-220h]
  __int64 *v140; // [rsp+70h] [rbp-220h]
  __int64 v141; // [rsp+70h] [rbp-220h]
  __int64 v142; // [rsp+70h] [rbp-220h]
  __int64 *v143; // [rsp+70h] [rbp-220h]
  __int64 *v144; // [rsp+70h] [rbp-220h]
  __int64 *v145; // [rsp+78h] [rbp-218h]
  __int64 v146; // [rsp+90h] [rbp-200h] BYREF
  int v147; // [rsp+98h] [rbp-1F8h]
  __int64 *v148; // [rsp+A0h] [rbp-1F0h] BYREF
  __int64 v149; // [rsp+A8h] [rbp-1E8h]
  _BYTE v150[48]; // [rsp+B0h] [rbp-1E0h] BYREF
  __int64 v151; // [rsp+E0h] [rbp-1B0h]
  char *v152; // [rsp+E8h] [rbp-1A8h]
  __int64 v153; // [rsp+F0h] [rbp-1A0h]
  int v154; // [rsp+F8h] [rbp-198h]
  char v155; // [rsp+FCh] [rbp-194h]
  char v156; // [rsp+100h] [rbp-190h] BYREF
  __int64 *v157; // [rsp+120h] [rbp-170h] BYREF
  __int64 v158; // [rsp+128h] [rbp-168h]
  _BYTE v159[48]; // [rsp+130h] [rbp-160h] BYREF
  char *v160; // [rsp+160h] [rbp-130h] BYREF
  __int64 *v161; // [rsp+168h] [rbp-128h]
  __int64 v162; // [rsp+170h] [rbp-120h]
  int v163; // [rsp+178h] [rbp-118h]
  char v164; // [rsp+17Ch] [rbp-114h]
  _WORD v165[32]; // [rsp+180h] [rbp-110h] BYREF
  unsigned __int64 v166[2]; // [rsp+1C0h] [rbp-D0h] BYREF
  char v167; // [rsp+1D0h] [rbp-C0h] BYREF
  _BYTE *v168; // [rsp+1D8h] [rbp-B8h]
  __int64 v169; // [rsp+1E0h] [rbp-B0h]
  _BYTE v170[48]; // [rsp+1E8h] [rbp-A8h] BYREF
  __int64 v171; // [rsp+218h] [rbp-78h]
  __int64 v172; // [rsp+220h] [rbp-70h]
  __int64 v173; // [rsp+228h] [rbp-68h]
  unsigned int v174; // [rsp+230h] [rbp-60h]
  __int64 v175; // [rsp+238h] [rbp-58h]
  _QWORD *v176; // [rsp+240h] [rbp-50h]
  char v177; // [rsp+248h] [rbp-48h]
  __int64 v178; // [rsp+24Ch] [rbp-44h]

  v166[0] = (unsigned __int64)&v167;
  v166[1] = 0x100000000LL;
  v168 = v170;
  v169 = 0x600000000LL;
  v176 = a1;
  v171 = 0;
  v172 = 0;
  v173 = 0;
  v174 = 0;
  v175 = 0;
  v177 = 0;
  v178 = 0;
  sub_2C06B20((__int64)v166, (__int64)a2, a3, a4, a5, a6);
  v149 = 0x600000000LL;
  v148 = (__int64 *)v150;
  v6 = sub_2BF3F10(a1);
  v7 = sub_2BF04D0(*(_QWORD *)(v6 + 112));
  v8 = sub_2BF05A0(v7);
  v11 = *(_QWORD *)(v7 + 120);
  for ( i = v8; v11 != i; v11 = *(_QWORD *)(v11 + 8) )
  {
    if ( !v11 )
      BUG();
    if ( *(_BYTE *)(v11 - 16) == 32 )
    {
      v13 = (unsigned int)v149;
      v14 = (unsigned int)v149 + 1LL;
      if ( v14 > HIDWORD(v149) )
      {
        sub_C8D5F0((__int64)&v148, v150, v14, 8u, v9, v10);
        v13 = (unsigned int)v149;
      }
      v148[v13] = v11 - 24;
      LODWORD(v149) = v149 + 1;
    }
  }
  v15 = v148;
  v137 = &v148[(unsigned int)v149];
  if ( v137 == v148 )
  {
    v70 = 1;
    goto LABEL_79;
  }
  v145 = v148;
  while ( 2 )
  {
    v151 = 0;
    v153 = 4;
    v16 = *v145;
    v154 = 0;
    v155 = 1;
    v152 = &v156;
    v17 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v16 + 40LL);
    if ( v17 == sub_2AA7530 )
      v18 = *(_QWORD *)(*(_QWORD *)(v16 + 48) + 8LL);
    else
      v18 = v17(v16);
    v19 = sub_2BF0490(v18);
    if ( v19 )
    {
      while ( *(_BYTE *)(v19 + 8) == 32 )
      {
        v20 = *(__int64 (**)(void))(*(_QWORD *)v19 + 40LL);
        if ( (char *)v20 == (char *)sub_2AA7530 )
          v21 = *(_QWORD *)(*(_QWORD *)(v19 + 48) + 8LL);
        else
          v21 = v20();
        v19 = sub_2BF0490(v21);
        if ( !v19 )
          goto LABEL_17;
      }
      v22 = v19;
    }
    else
    {
LABEL_17:
      v22 = 0;
    }
    v160 = 0;
    v162 = 8;
    v157 = (__int64 *)v159;
    v158 = 0x600000000LL;
    v163 = 0;
    v161 = (__int64 *)v165;
    v164 = 1;
    sub_AE6EC0((__int64)&v160, v22);
    v25 = (unsigned int)v158;
    v26 = (unsigned int)v158 + 1LL;
    if ( v26 > HIDWORD(v158) )
    {
      sub_C8D5F0((__int64)&v157, v159, v26, 8u, v23, v24);
      v25 = (unsigned int)v158;
    }
    v157[v25] = v16;
    LODWORD(v158) = v158 + 1;
    if ( !(_DWORD)v158 )
      goto LABEL_50;
    v27 = 0;
    v139 = 0;
    v28 = v157;
    while ( 1 )
    {
      v29 = sub_2C2A370((_QWORD *)(v28[v27] + 16), 0);
      v33 = v29[2];
      v34 = (__int64 *)(v33 + 8LL * *((unsigned int *)v29 + 6));
      if ( (__int64 *)v33 != v34 )
        break;
LABEL_35:
      ++v139;
      v39 = (unsigned int)v158;
      v27 = v139;
      if ( (_DWORD)v158 == v139 )
      {
        v40 = v139;
        v41 = v28;
        v42 = &v28[v40];
        v140 = v42;
        if ( v41 != v42 )
        {
          v132 = v41;
          _BitScanReverse64(&v43, (v40 * 8) >> 3);
          sub_2C25AC0(v41, v42, 2LL * (int)(63 - (v43 ^ 0x3F)), (__int64)v166);
          if ( (unsigned __int64)v40 <= 16 )
          {
            sub_2C25F90(v132, v42, (__int64)v166);
          }
          else
          {
            v44 = v132 + 16;
            sub_2C25F90(v132, v132 + 16, (__int64)v166);
            if ( v132 + 16 != v42 )
            {
              v133 = v22;
              v45 = v44;
              v130 = v16;
              do
              {
                v46 = *v45;
                for ( j = v45; ; j[1] = *j )
                {
                  v48 = *(j - 1);
                  v49 = j--;
                  if ( !(unsigned __int8)sub_2BFCAA0((__int64)v166, v46, v48) )
                    break;
                }
                *v49 = v46;
                ++v45;
              }
              while ( v45 != v140 );
              v22 = v133;
              v16 = v130;
              v39 = (unsigned int)v158;
              v41 = v157;
              goto LABEL_45;
            }
          }
          v39 = (unsigned int)v158;
          v41 = v157;
        }
LABEL_45:
        v50 = &v41[v39];
        if ( v41 != v50 )
        {
          v51 = v41;
          v52 = v22;
          do
          {
            v53 = *v51;
            if ( v16 != *v51 )
            {
              sub_2C19EB0((_QWORD *)*v51, v52);
              v52 = v53;
            }
            ++v51;
          }
          while ( v50 != v51 );
        }
LABEL_50:
        v54 = 1;
        if ( v164 )
          goto LABEL_60;
LABEL_51:
        _libc_free((unsigned __int64)v161);
        goto LABEL_60;
      }
    }
    v35 = (__int64 *)v29[2];
    v36 = v22;
    while ( 1 )
    {
      v37 = *v35;
      if ( *v35 )
        v37 = *v35 - 40;
      if ( v37 == v36 )
        break;
      if ( (unsigned int)*(unsigned __int8 *)(v37 + 8) - 29 <= 7 )
        goto LABEL_33;
      if ( !v164 )
        goto LABEL_52;
      v38 = v161;
      v31 = HIDWORD(v162);
      v30 = &v161[HIDWORD(v162)];
      if ( v161 != v30 )
      {
        while ( v37 != *v38 )
        {
          if ( v30 == ++v38 )
            goto LABEL_93;
        }
        goto LABEL_33;
      }
LABEL_93:
      if ( HIDWORD(v162) < (unsigned int)v162 )
      {
        ++HIDWORD(v162);
        *v30 = v37;
        ++v160;
LABEL_53:
        if ( (unsigned __int8)sub_2BFCAA0((__int64)v166, v36, v37) )
          goto LABEL_33;
        if ( sub_2C1AB20(v37) )
          break;
        v55 = (unsigned int)v158;
        v31 = HIDWORD(v158);
        v56 = (unsigned int)v158 + 1LL;
        if ( v56 > HIDWORD(v158) )
        {
          sub_C8D5F0((__int64)&v157, v159, v56, 8u, v32, v33);
          v55 = (unsigned int)v158;
        }
        v30 = v157;
        ++v35;
        v157[v55] = v37;
        LODWORD(v158) = v158 + 1;
        if ( v34 == v35 )
        {
LABEL_34:
          v22 = v36;
          v28 = v157;
          goto LABEL_35;
        }
      }
      else
      {
LABEL_52:
        sub_C8CC70((__int64)&v160, v37, (__int64)v30, v31, v32, v33);
        if ( (_BYTE)v30 )
          goto LABEL_53;
LABEL_33:
        if ( v34 == ++v35 )
          goto LABEL_34;
      }
    }
    v22 = v36;
    v54 = 0;
    if ( !v164 )
      goto LABEL_51;
LABEL_60:
    if ( v157 != (__int64 *)v159 )
      _libc_free((unsigned __int64)v157);
    if ( (_BYTE)v54 )
    {
LABEL_63:
      v57 = *(_QWORD *)(v22 + 80);
      if ( (unsigned int)*(unsigned __int8 *)(v22 + 8) - 29 <= 7 )
        v58 = sub_2BF05A0(*(_QWORD *)(v22 + 80));
      else
        v58 = *(_QWORD *)(v22 + 32);
      *a2 = v57;
      a2[1] = v58;
      v146 = 0;
      v165[0] = 257;
      v157 = (__int64 *)(v16 + 96);
      v59 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v16 + 40LL);
      if ( v59 == sub_2AA7530 )
        v60 = *(_QWORD *)(*(_QWORD *)(v16 + 48) + 8LL);
      else
        v60 = v59(v16);
      v158 = v60;
      v61 = sub_2C27AE0(a2, 69, (__int64 *)&v157, 2, v147, 0, &v146, (void **)&v160);
      if ( v146 )
        sub_B91220((__int64)&v146, v146);
      v62 = (__int64)(v61 + 12);
      if ( !v61 )
        v62 = 0;
      sub_2BF1250(v16 + 96, v62);
      v63 = *(_QWORD *)v61[6];
      v160 = (char *)(v61 + 5);
      v64 = *(_QWORD **)(v63 + 16);
      v65 = (__int64)&v64[*(unsigned int *)(v63 + 24)];
      v66 = sub_2C25810(v64, v65, (__int64 *)&v160);
      if ( (_QWORD *)v65 != v66 )
      {
        if ( (_QWORD *)v65 != v66 + 1 )
        {
          memmove(v66, v66 + 1, v65 - (_QWORD)(v66 + 1));
          LODWORD(v67) = *(_DWORD *)(v63 + 24);
        }
        v67 = (unsigned int)(v67 - 1);
        *(_DWORD *)(v63 + 24) = v67;
        v68 = (_QWORD *)v61[6];
      }
      *v68 = v16 + 96;
      v69 = *(unsigned int *)(v16 + 120);
      if ( v69 + 1 > (unsigned __int64)*(unsigned int *)(v16 + 124) )
      {
        sub_C8D5F0(v16 + 112, (const void *)(v16 + 128), v69 + 1, 8u, v67, (__int64)v68);
        v69 = *(unsigned int *)(v16 + 120);
      }
      ++v145;
      *(_QWORD *)(*(_QWORD *)(v16 + 112) + 8 * v69) = v61 + 5;
      ++*(_DWORD *)(v16 + 120);
      if ( v137 == v145 )
      {
        v15 = v148;
        v70 = 1;
        goto LABEL_79;
      }
      continue;
    }
    break;
  }
  if ( !sub_2C1AB20(v22) && !(unsigned __int8)sub_2C1ABE0(v22) )
  {
    v160 = 0;
    v162 = 8;
    v157 = (__int64 *)v159;
    v158 = 0x600000000LL;
    v163 = 0;
    v161 = (__int64 *)v165;
    v164 = 1;
    v76 = *(__int64 **)(v16 + 112);
    v77 = &v76[*(unsigned int *)(v16 + 120)];
    if ( v76 == v77 )
    {
      v131 = 0;
    }
    else
    {
      v141 = v22;
      v134 = v16;
      v78 = 0;
      v79 = *(__int64 **)(v16 + 112);
      do
      {
        while ( 1 )
        {
          v80 = *v79;
          if ( *v79 )
            v80 = *v79 - 40;
          if ( v78 )
            break;
          ++v79;
          v78 = v80;
          if ( v77 == v79 )
            goto LABEL_112;
        }
        if ( (unsigned __int8)sub_2BFCAA0((__int64)v166, v80, v78) )
          v78 = v80;
        ++v79;
      }
      while ( v77 != v79 );
LABEL_112:
      v131 = v78;
      v16 = v134;
      v22 = v141;
      v54 = 0;
    }
    v81 = sub_2C2A370((_QWORD *)(v22 + 16), 0);
    v82 = sub_2BF0490((__int64)v81);
    v83 = v82;
    if ( v82 )
    {
      v84 = sub_2BF09E0(*(_QWORD *)(v82 + 80));
      sub_AE6EC0((__int64)&v160, v83);
      v85 = v164;
      v135 = v86 ^ 1 | (v84 == 0);
      if ( !v135 )
      {
        v135 = 1;
        if ( (unsigned int)*(unsigned __int8 *)(v83 + 8) - 29 > 7 )
        {
          v87 = sub_2BFCAA0((__int64)v166, v83, v131);
          if ( !v87 )
          {
            v90 = (unsigned int)v158;
            v91 = (unsigned int)v158 + 1LL;
            if ( v91 > HIDWORD(v158) )
            {
              sub_C8D5F0((__int64)&v157, v159, v91, 8u, v88, v89);
              v90 = (unsigned int)v158;
            }
            v157[v90] = v22;
            LODWORD(v158) = v158 + 1;
            if ( (_DWORD)v158 )
            {
              v128 = 0;
              v142 = v16 + 96;
              v92 = 0;
              v126 = v22;
              v125 = v16;
              v127 = v54;
              do
              {
                v93 = v157[v92];
                v135 = sub_2C1AB20(v93);
                if ( v135 )
                {
                  v135 = 0;
                  v22 = v126;
                  v16 = v125;
                  v54 = v127;
                  v85 = v164;
                  goto LABEL_135;
                }
                v94 = *(__int64 **)(v93 + 48);
                v95 = &v94[*(unsigned int *)(v93 + 56)];
                if ( v94 != v95 )
                {
                  while ( *v94 != v142 )
                  {
                    v96 = sub_2BF0490(*v94);
                    v97 = v96;
                    if ( v96 )
                    {
                      v98 = sub_2BF09E0(*(_QWORD *)(v96 + 80));
                      sub_AE6EC0((__int64)&v160, v97);
                      if ( v98 )
                      {
                        if ( v99 == 1
                          && (unsigned int)*(unsigned __int8 *)(v97 + 8) - 29 > 7
                          && !(unsigned __int8)sub_2BFCAA0((__int64)v166, v97, v131) )
                        {
                          v102 = (unsigned int)v158;
                          v103 = (unsigned int)v158 + 1LL;
                          if ( v103 > HIDWORD(v158) )
                          {
                            sub_C8D5F0((__int64)&v157, v159, v103, 8u, v100, v101);
                            v102 = (unsigned int)v158;
                          }
                          v157[v102] = v97;
                          LODWORD(v158) = v158 + 1;
                        }
                      }
                    }
                    if ( v95 == ++v94 )
                      goto LABEL_141;
                  }
                  v22 = v126;
                  v16 = v125;
                  v54 = v127;
                  v85 = v164;
                  goto LABEL_135;
                }
LABEL_141:
                ++v128;
                v104 = (unsigned int)v158;
                v92 = v128;
              }
              while ( (_DWORD)v158 != v128 );
              v105 = v157;
              v22 = v126;
              v106 = v128;
              v16 = v125;
              v54 = v127;
              v107 = &v157[v106];
              if ( v157 != &v157[v106] )
              {
                v143 = v157;
                _BitScanReverse64(&v108, (__int64)(v106 * 8) >> 3);
                sub_2C25D00(v157, &v157[v128], 2LL * (int)(63 - (v108 ^ 0x3F)), (__int64)v166);
                if ( v106 <= 16 )
                {
                  sub_2C26050(v143, v107, (__int64)v166);
                }
                else
                {
                  v109 = v143 + 16;
                  sub_2C26050(v143, v143 + 16, (__int64)v166);
                  if ( v107 != v143 + 16 )
                  {
                    v110 = v127;
                    v111 = v126;
                    v112 = v125;
                    v113 = v107;
                    do
                    {
                      v144 = v113;
                      v114 = *v109;
                      v115 = v109;
                      v116 = v109;
                      v117 = v112;
                      v118 = v111;
                      while ( 1 )
                      {
                        v119 = *(v115 - 1);
                        v129 = v110;
                        v136 = v115--;
                        v120 = sub_2BFCAA0((__int64)v166, v114, v119);
                        v110 = v129;
                        if ( !v120 )
                          break;
                        v115[1] = *v115;
                      }
                      v111 = v118;
                      v121 = v116;
                      v113 = v144;
                      v109 = v121 + 1;
                      *v136 = v114;
                      v112 = v117;
                    }
                    while ( v144 != v109 );
                    v22 = v111;
                    v16 = v117;
                    v54 = v129;
                  }
                }
                v104 = (unsigned int)v158;
                v105 = v157;
              }
              v122 = &v105[v104];
              if ( v105 != v122 )
              {
                v123 = v105;
                do
                {
                  v124 = (_QWORD *)*v123++;
                  sub_2C19EE0(v124, *(_QWORD *)(v131 + 80), (unsigned __int64 *)(v131 + 24));
                }
                while ( v122 != v123 );
                v135 = 1;
                v54 = (unsigned __int8)v54;
                v85 = v164;
                goto LABEL_135;
              }
            }
            goto LABEL_158;
          }
          v135 = v87;
          v85 = v164;
        }
      }
    }
    else
    {
LABEL_158:
      v135 = 1;
      v85 = v164;
    }
LABEL_135:
    if ( !v85 )
      _libc_free((unsigned __int64)v161);
    if ( v157 != (__int64 *)v159 )
      _libc_free((unsigned __int64)v157);
    if ( v135 )
      goto LABEL_63;
  }
  v15 = v148;
  v70 = v54;
LABEL_79:
  if ( v15 != (__int64 *)v150 )
    _libc_free((unsigned __int64)v15);
  sub_C7D6A0(v172, 16LL * v174, 8);
  v71 = v168;
  v72 = &v168[8 * (unsigned int)v169];
  if ( v168 != v72 )
  {
    do
    {
      v73 = *((_QWORD *)v72 - 1);
      v72 -= 8;
      if ( v73 )
      {
        v74 = *(_QWORD *)(v73 + 24);
        if ( v74 != v73 + 40 )
          _libc_free(v74);
        j_j___libc_free_0(v73);
      }
    }
    while ( v71 != v72 );
    v72 = v168;
  }
  if ( v72 != v170 )
    _libc_free((unsigned __int64)v72);
  if ( (char *)v166[0] != &v167 )
    _libc_free(v166[0]);
  return v70;
}
