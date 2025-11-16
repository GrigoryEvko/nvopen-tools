// Function: sub_1C5FDC0
// Address: 0x1c5fdc0
//
__int64 __fastcall sub_1C5FDC0(_QWORD *a1, __int64 a2, __int64 a3, char a4, __m128i a5, __m128i a6)
{
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rdi
  unsigned int v10; // edx
  __int64 *v11; // rsi
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r15
  char v16; // al
  __int64 *v17; // r13
  __int64 v18; // r14
  __int64 v19; // r15
  bool v20; // al
  __int64 v21; // rax
  __int64 *v22; // r8
  int v23; // r9d
  __int64 *v24; // rax
  _QWORD *v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdx
  _QWORD *v28; // rcx
  unsigned int v29; // esi
  __int64 *v30; // r9
  __int64 *v31; // rdi
  __int64 v32; // r14
  __int64 v33; // rdi
  unsigned int v34; // edx
  __int64 *v35; // rax
  __int64 v36; // r10
  __int64 v37; // r14
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rbx
  __int64 v41; // r12
  __int64 v42; // rdi
  int v43; // ecx
  __int64 v44; // rax
  _QWORD *v45; // rdi
  _QWORD *i; // rax
  _QWORD *v47; // rbx
  _QWORD *v48; // r12
  __int64 v50; // r14
  __int64 v51; // rax
  __int64 v52; // rax
  int v53; // edi
  _QWORD *v54; // rax
  __int64 *v55; // rsi
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // r14
  __int64 v59; // rdi
  unsigned int v60; // esi
  __int64 v61; // r10
  __int64 v62; // rdx
  unsigned int v63; // eax
  __int64 *v64; // rcx
  __int64 v65; // rdi
  int v66; // edx
  int v67; // esi
  int v68; // r10d
  _QWORD *v69; // rax
  __int64 v70; // rdx
  __int64 v71; // r12
  unsigned __int64 v72; // rdx
  const char *v73; // r14
  size_t v74; // r13
  __int64 v75; // rax
  _QWORD *v76; // rdx
  char *v77; // rsi
  size_t v78; // rdx
  unsigned __int8 v79; // dl
  int v80; // r14d
  __int64 v81; // rdi
  _BYTE *v82; // rax
  _QWORD *v83; // r12
  _QWORD *v84; // rax
  _QWORD *v85; // rbx
  unsigned __int64 *v86; // r13
  unsigned int v87; // edx
  _QWORD *v88; // rsi
  unsigned int v89; // ecx
  unsigned int v90; // edx
  int v91; // edx
  unsigned __int64 v92; // rax
  int v93; // ebx
  __int64 v94; // r12
  _QWORD *v95; // rax
  _QWORD *j; // rdx
  __int64 *v97; // rdx
  int v98; // r12d
  unsigned __int64 v99; // rbx
  unsigned int v100; // eax
  __int64 v101; // r8
  __int64 v102; // r9
  int v103; // esi
  unsigned __int64 v104; // rbx
  __int64 *v105; // r13
  __int64 v106; // rcx
  int v107; // edx
  int v108; // r14d
  __int64 v109; // rax
  unsigned __int64 *v110; // r13
  __int64 *v111; // rax
  unsigned int v112; // eax
  __int64 v113; // rbx
  __int64 v114; // r12
  __int64 v115; // rdi
  _QWORD *v116; // rax
  _QWORD *v117; // r12
  _QWORD *v118; // rbx
  void *v119; // rax
  __int64 v120; // r12
  unsigned __int64 v121; // rdx
  const char *v122; // r15
  size_t v123; // r14
  __int64 v124; // rax
  _QWORD *v125; // rdx
  size_t v126; // rdx
  char *v127; // rsi
  __int64 v128; // rax
  _QWORD *v129; // rdi
  int v130; // r15d
  int v131; // eax
  int v132; // edi
  int v133; // r11d
  __int64 v134; // rcx
  int v135; // edi
  __int64 v136; // rax
  int v137; // esi
  unsigned __int8 v138; // dl
  unsigned __int64 *v139; // r13
  __int64 *v140; // r15
  _QWORD *v141; // rdi
  __int64 v142; // [rsp+0h] [rbp-260h]
  __int64 v144; // [rsp+18h] [rbp-248h]
  __int64 v145; // [rsp+20h] [rbp-240h]
  __int64 v146; // [rsp+28h] [rbp-238h]
  unsigned __int8 v149; // [rsp+3Eh] [rbp-222h]
  __int64 v151; // [rsp+40h] [rbp-220h]
  __int64 v152; // [rsp+40h] [rbp-220h]
  __int64 v153; // [rsp+48h] [rbp-218h]
  int v154; // [rsp+5Ch] [rbp-204h] BYREF
  __int64 v155; // [rsp+60h] [rbp-200h] BYREF
  __int64 v156; // [rsp+68h] [rbp-1F8h] BYREF
  __int64 *v157; // [rsp+70h] [rbp-1F0h] BYREF
  __int64 *v158; // [rsp+78h] [rbp-1E8h]
  __int64 *v159; // [rsp+80h] [rbp-1E0h]
  __int64 v160; // [rsp+90h] [rbp-1D0h] BYREF
  _QWORD *v161; // [rsp+98h] [rbp-1C8h]
  __int64 v162; // [rsp+A0h] [rbp-1C0h]
  unsigned int v163; // [rsp+A8h] [rbp-1B8h]
  __int64 v164; // [rsp+B0h] [rbp-1B0h] BYREF
  _QWORD *v165; // [rsp+B8h] [rbp-1A8h]
  __int64 v166; // [rsp+C0h] [rbp-1A0h]
  unsigned int v167; // [rsp+C8h] [rbp-198h]
  _QWORD *v168; // [rsp+D0h] [rbp-190h] BYREF
  __int64 v169; // [rsp+D8h] [rbp-188h]
  _QWORD v170[8]; // [rsp+E0h] [rbp-180h] BYREF
  _BYTE *v171; // [rsp+120h] [rbp-140h] BYREF
  __int64 v172; // [rsp+128h] [rbp-138h]
  _BYTE v173[304]; // [rsp+130h] [rbp-130h] BYREF

  v171 = v173;
  v172 = 0x2000000000LL;
  v160 = 0;
  v161 = 0;
  v162 = 0;
  v163 = 0;
  v157 = 0;
  v158 = 0;
  v159 = 0;
  v164 = 0;
  v165 = 0;
  v166 = 0;
  v167 = 0;
  v154 = 0;
  if ( !dword_4FBCCA0 )
    goto LABEL_2;
  v119 = sub_16E8CB0();
  v120 = sub_1263B40((__int64)v119, "\n\nProcessing ");
  v122 = sub_1649960(a2);
  v123 = v121;
  if ( v122 )
  {
    v156 = v121;
    v124 = v121;
    v168 = v170;
    if ( v121 > 0xF )
    {
      v168 = (_QWORD *)sub_22409D0(&v168, &v156, 0);
      v141 = v168;
      v170[0] = v156;
    }
    else
    {
      if ( v121 == 1 )
      {
        LOBYTE(v170[0]) = *v122;
        v125 = v170;
LABEL_194:
        v169 = v124;
        *((_BYTE *)v125 + v124) = 0;
        v126 = v169;
        v127 = (char *)v168;
        goto LABEL_195;
      }
      if ( !v121 )
      {
        v125 = v170;
        goto LABEL_194;
      }
      v141 = v170;
    }
    memcpy(v141, v122, v123);
    v124 = v156;
    v125 = v168;
    goto LABEL_194;
  }
  LOBYTE(v170[0]) = 0;
  v126 = 0;
  v168 = v170;
  v127 = (char *)v170;
  v169 = 0;
LABEL_195:
  v128 = sub_16E7EE0(v120, v127, v126);
  sub_1263B40(v128, "\n\n");
  if ( v168 != v170 )
    j_j___libc_free_0(v168, v170[0] + 1LL);
LABEL_2:
  v149 = 0;
  v144 = a2 + 72;
  v146 = *(_QWORD *)(a2 + 80);
  if ( a2 + 72 == v146 )
    goto LABEL_48;
  do
  {
    v7 = v146 - 24;
    if ( !v146 )
      v7 = 0;
    v155 = v7;
    if ( !a4 )
    {
      if ( dword_4FBCCA0 <= 1 )
        goto LABEL_7;
      v69 = sub_16E8CB0();
      v70 = v69[3];
      v71 = (__int64)v69;
      if ( (unsigned __int64)(v69[2] - v70) <= 5 )
      {
        v71 = sub_16E7EE0((__int64)v69, "Block ", 6u);
      }
      else
      {
        *(_DWORD *)v70 = 1668246594;
        *(_WORD *)(v70 + 4) = 8299;
        v69[3] += 6LL;
      }
      v73 = sub_1649960(v155);
      v74 = v72;
      if ( !v73 )
      {
        LOBYTE(v170[0]) = 0;
        v78 = 0;
        v168 = v170;
        v77 = (char *)v170;
        v169 = 0;
        goto LABEL_114;
      }
      v156 = v72;
      v75 = v72;
      v168 = v170;
      if ( v72 > 0xF )
      {
        v168 = (_QWORD *)sub_22409D0(&v168, &v156, 0);
        v129 = v168;
        v170[0] = v156;
      }
      else
      {
        if ( v72 == 1 )
        {
          LOBYTE(v170[0]) = *v73;
          v76 = v170;
LABEL_102:
          v169 = v75;
          *((_BYTE *)v76 + v75) = 0;
          v77 = (char *)v168;
          v78 = v169;
LABEL_114:
          v81 = sub_16E7EE0(v71, v77, v78);
          v82 = *(_BYTE **)(v81 + 24);
          if ( *(_BYTE **)(v81 + 16) == v82 )
          {
            sub_16E7EE0(v81, "\n", 1u);
          }
          else
          {
            *v82 = 10;
            ++*(_QWORD *)(v81 + 24);
          }
          if ( v168 != v170 )
            j_j___libc_free_0(v168, v170[0] + 1LL);
          v7 = v155;
LABEL_7:
          v8 = *(_QWORD *)(v7 + 48);
          v153 = v7 + 40;
          if ( v7 + 40 != v8 )
            goto LABEL_12;
LABEL_32:
          v39 = (unsigned int)v172;
          if ( (unsigned int)v172 > 1 )
          {
            v79 = sub_1C5DFC0(a1, &v157, (__int64)&v160, a1[25], &v154, a5, a6);
            v39 = (unsigned int)v172;
            if ( v79 )
              v149 = v79;
          }
          if ( (_DWORD)v39 )
          {
            v40 = 8 * v39;
            v41 = 0;
            do
            {
              v42 = *(_QWORD *)&v171[v41];
              if ( v42 )
                j_j___libc_free_0(v42, 32);
              v41 += 8;
            }
            while ( v40 != v41 );
          }
          LODWORD(v172) = 0;
          v43 = v162;
          if ( !(_DWORD)v162 )
          {
            ++v160;
            goto LABEL_40;
          }
          v45 = v161;
          v83 = &v161[2 * v163];
          if ( v161 == v83 )
            goto LABEL_124;
          v84 = v161;
          while ( 1 )
          {
            v85 = v84;
            if ( *v84 != -16 && *v84 != -8 )
              break;
            v84 += 2;
            if ( v83 == v84 )
              goto LABEL_124;
          }
          if ( v83 == v84 )
          {
LABEL_124:
            ++v160;
          }
          else
          {
            do
            {
              v86 = (unsigned __int64 *)v85[1];
              if ( v86 )
              {
                if ( (unsigned __int64 *)*v86 != v86 + 2 )
                  _libc_free(*v86);
                j_j___libc_free_0(v86, 80);
              }
              v85 += 2;
              if ( v85 == v83 )
                break;
              while ( *v85 == -8 || *v85 == -16 )
              {
                v85 += 2;
                if ( v83 == v85 )
                  goto LABEL_134;
              }
            }
            while ( v83 != v85 );
LABEL_134:
            v43 = v162;
            ++v160;
            if ( !(_DWORD)v162 )
            {
LABEL_40:
              if ( HIDWORD(v162) )
              {
                v44 = v163;
                v45 = v161;
                if ( v163 <= 0x40 )
                {
LABEL_42:
                  for ( i = &v45[2 * v44]; i != v45; v45 += 2 )
                    *v45 = -8;
                  v162 = 0;
                  goto LABEL_45;
                }
                j___libc_free_0(v161);
                v161 = 0;
                v162 = 0;
                v163 = 0;
              }
LABEL_45:
              if ( v157 != v158 )
                v158 = v157;
              goto LABEL_47;
            }
            v45 = v161;
          }
          v87 = 4 * v43;
          v44 = v163;
          if ( (unsigned int)(4 * v43) < 0x40 )
            v87 = 64;
          if ( v87 >= v163 )
            goto LABEL_42;
          v88 = v45;
          v89 = v43 - 1;
          if ( v89 )
          {
            _BitScanReverse(&v90, v89);
            v91 = 1 << (33 - (v90 ^ 0x1F));
            if ( v91 < 64 )
              v91 = 64;
            if ( v91 == v163 )
            {
              v162 = 0;
              do
              {
                if ( v88 )
                  *v88 = -8;
                v88 += 2;
              }
              while ( &v45[2 * (unsigned int)v91] != v88 );
              goto LABEL_45;
            }
            v92 = (((4 * v91 / 3u + 1)
                  | ((unsigned __int64)(4 * v91 / 3u + 1) >> 1)
                  | (((4 * v91 / 3u + 1) | ((unsigned __int64)(4 * v91 / 3u + 1) >> 1)) >> 2)) >> 4)
                | (4 * v91 / 3u + 1)
                | ((unsigned __int64)(4 * v91 / 3u + 1) >> 1)
                | (((4 * v91 / 3u + 1) | ((unsigned __int64)(4 * v91 / 3u + 1) >> 1)) >> 2)
                | (((((4 * v91 / 3u + 1)
                    | ((unsigned __int64)(4 * v91 / 3u + 1) >> 1)
                    | (((4 * v91 / 3u + 1) | ((unsigned __int64)(4 * v91 / 3u + 1) >> 1)) >> 2)) >> 4)
                  | (4 * v91 / 3u + 1)
                  | ((unsigned __int64)(4 * v91 / 3u + 1) >> 1)
                  | (((4 * v91 / 3u + 1) | ((unsigned __int64)(4 * v91 / 3u + 1) >> 1)) >> 2)) >> 8);
            v93 = (v92 | (v92 >> 16)) + 1;
            v94 = 16 * ((v92 | (v92 >> 16)) + 1);
          }
          else
          {
            v94 = 2048;
            v93 = 128;
          }
          j___libc_free_0(v45);
          v163 = v93;
          v95 = (_QWORD *)sub_22077B0(v94);
          v162 = 0;
          v161 = v95;
          for ( j = &v95[2 * v163]; j != v95; v95 += 2 )
          {
            if ( v95 )
              *v95 = -8;
          }
          goto LABEL_45;
        }
        if ( !v72 )
        {
          v76 = v170;
          goto LABEL_102;
        }
        v129 = v170;
      }
      memcpy(v129, v73, v74);
      v75 = v156;
      v76 = v168;
      goto LABEL_102;
    }
    v8 = *(_QWORD *)(v7 + 48);
    v153 = v7 + 40;
    if ( v8 == v7 + 40 )
      goto LABEL_47;
    do
    {
      while ( 1 )
      {
LABEL_12:
        v13 = *(unsigned int *)(a3 + 24);
        v14 = v8;
        v8 = *(_QWORD *)(v8 + 8);
        v15 = v14 - 24;
        if ( (_DWORD)v13 )
        {
          v9 = *(_QWORD *)(a3 + 8);
          v10 = (v13 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
          v11 = (__int64 *)(v9 + 8LL * v10);
          v12 = *v11;
          if ( v15 == *v11 )
          {
LABEL_10:
            if ( v11 != (__int64 *)(v9 + 8 * v13) )
              goto LABEL_11;
          }
          else
          {
            v67 = 1;
            while ( v12 != -8 )
            {
              v68 = v67 + 1;
              v10 = (v13 - 1) & (v67 + v10);
              v11 = (__int64 *)(v9 + 8LL * v10);
              v12 = *v11;
              if ( v15 == *v11 )
                goto LABEL_10;
              v67 = v68;
            }
          }
        }
        v16 = *(_BYTE *)(v14 - 8);
        if ( v16 != 54 )
          break;
        v151 = *(_QWORD *)(v14 - 48);
        v17 = (__int64 *)sub_22077B0(32);
        if ( v17 )
        {
          v17[2] = v15;
          v17[3] = v151;
          v18 = a1[23];
          v19 = sub_146F1B0(v18, v151);
          v20 = sub_14560B0(v19);
          v17[1] = v19;
          if ( v20 )
          {
            v52 = sub_1456040(v19);
            *v17 = sub_145CF80(v18, v52, 0, 0);
          }
          else
          {
            v168 = v170;
            v169 = 0x800000000LL;
            v21 = sub_1456040(v19);
            *v17 = sub_145CF80(v18, v21, 0, 0);
            sub_1C54710(v19, 0, (__int64)&v168, v18, v17, a5, a6);
            if ( (_DWORD)v169 )
            {
              if ( (unsigned int)v169 == 1 )
              {
                v25 = v168;
                v17[1] = *v168;
              }
              else
              {
                v24 = sub_147DD40(v18, (__int64 *)&v168, 0, 0, a5, a6);
                v25 = v168;
                v17[1] = (__int64)v24;
              }
              if ( v25 != v170 )
                goto LABEL_20;
            }
            else
            {
              v25 = v168;
              v17[1] = 0;
              if ( v25 != v170 )
LABEL_20:
                _libc_free((unsigned __int64)v25);
            }
          }
          v26 = (unsigned int)v172;
          if ( (unsigned int)v172 < HIDWORD(v172) )
            goto LABEL_22;
LABEL_63:
          sub_16CD150((__int64)&v171, v173, 0, 8, (int)v22, v23);
          v26 = (unsigned int)v172;
          goto LABEL_22;
        }
LABEL_11:
        if ( v8 == v153 )
          goto LABEL_31;
      }
      if ( v16 != 55 )
        goto LABEL_11;
      v152 = *(_QWORD *)(v14 - 48);
      v17 = (__int64 *)sub_22077B0(32);
      if ( !v17 )
        goto LABEL_11;
      v17[2] = v15;
      v50 = a1[23];
      v17[3] = v152;
      v51 = sub_146F1B0(v50, v152);
      sub_1C54F70(v17, v51, v50, a5, a6);
      v26 = (unsigned int)v172;
      if ( (unsigned int)v172 >= HIDWORD(v172) )
        goto LABEL_63;
LABEL_22:
      *(_QWORD *)&v171[8 * v26] = v17;
      v27 = v17[1];
      LODWORD(v172) = v172 + 1;
      v156 = v27;
      if ( !v27 )
      {
        j_j___libc_free_0(v17, 32);
        goto LABEL_11;
      }
      v28 = v161;
      v29 = v163;
      if ( v163 )
      {
        LODWORD(v30) = (v163 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v31 = &v161[2 * (unsigned int)v30];
        v32 = *v31;
        if ( v27 == *v31 )
        {
LABEL_25:
          if ( &v161[2 * v163] != v31 )
          {
            if ( !a4 )
              goto LABEL_27;
            goto LABEL_74;
          }
        }
        else
        {
          v53 = 1;
          while ( v32 != -8 )
          {
            LODWORD(v22) = v53 + 1;
            LODWORD(v30) = (v163 - 1) & (v53 + (_DWORD)v30);
            v31 = &v161[2 * (unsigned int)v30];
            v32 = *v31;
            if ( v27 == *v31 )
              goto LABEL_25;
            v53 = (int)v22;
          }
        }
      }
      v54 = (_QWORD *)sub_22077B0(80);
      if ( v54 )
      {
        *v54 = v54 + 2;
        v54[1] = 0x800000000LL;
      }
      sub_1C53170((__int64)&v160, &v156)[1] = (__int64)v54;
      v55 = v158;
      if ( v158 == v159 )
      {
        sub_1C55B50((__int64)&v157, v158, &v156);
      }
      else
      {
        if ( v158 )
        {
          *v158 = v156;
          v55 = v158;
        }
        v158 = v55 + 1;
      }
      if ( !a4 )
        goto LABEL_78;
LABEL_74:
      if ( !v167 )
      {
        ++v164;
        goto LABEL_221;
      }
      v56 = v156;
      LODWORD(v57) = (v167 - 1) & (((unsigned int)v156 >> 9) ^ ((unsigned int)v156 >> 4));
      v58 = (__int64)&v165[5 * (unsigned int)v57];
      v59 = *(_QWORD *)v58;
      if ( v156 != *(_QWORD *)v58 )
      {
        v133 = 1;
        v134 = 0;
        while ( v59 != -8 )
        {
          if ( !v134 && v59 == -16 )
            v134 = v58;
          LODWORD(v22) = v133 + 1;
          v57 = (v167 - 1) & ((_DWORD)v57 + v133);
          v58 = (__int64)&v165[5 * v57];
          v59 = *(_QWORD *)v58;
          if ( v156 == *(_QWORD *)v58 )
            goto LABEL_76;
          ++v133;
        }
        if ( !v134 )
          v134 = v58;
        ++v164;
        v135 = v166 + 1;
        if ( 4 * ((int)v166 + 1) < 3 * v167 )
        {
          if ( v167 - HIDWORD(v166) - v135 > v167 >> 3 )
          {
LABEL_215:
            LODWORD(v166) = v135;
            if ( *(_QWORD *)v134 != -8 )
              --HIDWORD(v166);
            *(_QWORD *)v134 = v56;
            v30 = (__int64 *)(v134 + 8);
            v136 = 1;
            *(_QWORD *)(v134 + 8) = 0;
            *(_QWORD *)(v134 + 16) = 0;
            *(_QWORD *)(v134 + 24) = 0;
            *(_DWORD *)(v134 + 32) = 0;
            goto LABEL_218;
          }
          sub_1C5FBB0((__int64)&v164, v167);
LABEL_222:
          sub_1C57100((__int64)&v164, &v156, &v168);
          v134 = (__int64)v168;
          v56 = v156;
          v135 = v166 + 1;
          goto LABEL_215;
        }
LABEL_221:
        sub_1C5FBB0((__int64)&v164, 2 * v167);
        goto LABEL_222;
      }
LABEL_76:
      v60 = *(_DWORD *)(v58 + 32);
      v61 = *(_QWORD *)(v58 + 16);
      v30 = (__int64 *)(v58 + 8);
      if ( !v60 )
      {
        v134 = v58;
        v136 = *(_QWORD *)(v58 + 8) + 1LL;
LABEL_218:
        *(_QWORD *)(v134 + 8) = v136;
        v137 = 0;
        goto LABEL_219;
      }
      v62 = v155;
      v63 = (v60 - 1) & (((unsigned int)v155 >> 9) ^ ((unsigned int)v155 >> 4));
      v64 = (__int64 *)(v61 + 8LL * v63);
      v65 = *v64;
      if ( *v64 != v155 )
      {
        v130 = 1;
        v22 = 0;
        while ( v65 != -8 )
        {
          if ( v65 == -16 && !v22 )
            v22 = v64;
          v63 = (v60 - 1) & (v130 + v63);
          v64 = (__int64 *)(v61 + 8LL * v63);
          v65 = *v64;
          if ( v155 == *v64 )
            goto LABEL_78;
          ++v130;
        }
        v131 = *(_DWORD *)(v58 + 24);
        if ( !v22 )
          v22 = v64;
        ++*(_QWORD *)(v58 + 8);
        v132 = v131 + 1;
        if ( 4 * (v131 + 1) < 3 * v60 )
        {
          if ( v60 - *(_DWORD *)(v58 + 28) - v132 <= v60 >> 3 )
          {
            sub_13B3D40(v58 + 8, v60);
            sub_1898220(v58 + 8, &v155, &v168);
            v22 = v168;
            v62 = v155;
            v132 = *(_DWORD *)(v58 + 24) + 1;
          }
          goto LABEL_206;
        }
        v137 = 2 * v60;
        v134 = v58;
LABEL_219:
        v142 = v134;
        v145 = (__int64)v30;
        sub_13B3D40((__int64)v30, v137);
        sub_1898220(v145, &v155, &v168);
        v22 = v168;
        v62 = v155;
        v58 = v142;
        v132 = *(_DWORD *)(v142 + 24) + 1;
LABEL_206:
        *(_DWORD *)(v58 + 24) = v132;
        if ( *v22 != -8 )
          --*(_DWORD *)(v58 + 28);
        *v22 = v62;
      }
LABEL_78:
      v29 = v163;
      if ( !v163 )
      {
        ++v160;
        goto LABEL_80;
      }
      v28 = v161;
LABEL_27:
      v33 = v156;
      v34 = (v29 - 1) & (((unsigned int)v156 >> 9) ^ ((unsigned int)v156 >> 4));
      v35 = &v28[2 * v34];
      v36 = *v35;
      if ( v156 != *v35 )
      {
        v80 = 1;
        v30 = 0;
        while ( v36 != -8 )
        {
          if ( v30 || v36 != -16 )
            v35 = v30;
          LODWORD(v22) = v80 + 1;
          v34 = (v29 - 1) & (v80 + v34);
          v140 = &v28[2 * v34];
          v36 = *v140;
          if ( v156 == *v140 )
          {
            v37 = v140[1];
            goto LABEL_29;
          }
          v30 = v35;
          ++v80;
          v35 = &v28[2 * v34];
        }
        if ( !v30 )
          v30 = v35;
        ++v160;
        v66 = v162 + 1;
        if ( 4 * ((int)v162 + 1) < 3 * v29 )
        {
          if ( v29 - HIDWORD(v162) - v66 > v29 >> 3 )
            goto LABEL_82;
          goto LABEL_81;
        }
LABEL_80:
        v29 *= 2;
LABEL_81:
        sub_1C52FC0((__int64)&v160, v29);
        sub_1C50640((__int64)&v160, &v156, &v168);
        v30 = v168;
        v33 = v156;
        v66 = v162 + 1;
LABEL_82:
        LODWORD(v162) = v66;
        if ( *v30 != -8 )
          --HIDWORD(v162);
        v37 = 0;
        *v30 = v33;
        v30[1] = 0;
        v38 = MEMORY[8];
        if ( MEMORY[8] >= MEMORY[0xC] )
          goto LABEL_85;
        goto LABEL_30;
      }
      v37 = v35[1];
LABEL_29:
      v38 = *(unsigned int *)(v37 + 8);
      if ( (unsigned int)v38 >= *(_DWORD *)(v37 + 12) )
      {
LABEL_85:
        sub_16CD150(v37, (const void *)(v37 + 16), 0, 8, (int)v22, (int)v30);
        v38 = *(unsigned int *)(v37 + 8);
      }
LABEL_30:
      *(_QWORD *)(*(_QWORD *)v37 + 8 * v38) = v17;
      ++*(_DWORD *)(v37 + 8);
    }
    while ( v8 != v153 );
LABEL_31:
    if ( !a4 )
      goto LABEL_32;
LABEL_47:
    v146 = *(_QWORD *)(v146 + 8);
  }
  while ( v144 != v146 );
LABEL_48:
  if ( !a4 )
    goto LABEL_49;
  v97 = v157;
  v98 = 0;
  v99 = 0;
  if ( v158 != v157 )
  {
    while ( 1 )
    {
      v103 = v167;
      v104 = v99;
      v105 = &v97[v104];
      if ( !v167 )
        break;
      v100 = (v167 - 1) & (((unsigned int)*v105 >> 9) ^ ((unsigned int)*v105 >> 4));
      v101 = (__int64)&v165[5 * v100];
      v102 = *(_QWORD *)v101;
      if ( *v105 != *(_QWORD *)v101 )
      {
        v108 = 1;
        v106 = 0;
        while ( v102 != -8 )
        {
          if ( !v106 && v102 == -16 )
            v106 = v101;
          v100 = (v167 - 1) & (v108 + v100);
          v101 = (__int64)&v165[5 * v100];
          v102 = *(_QWORD *)v101;
          if ( *v105 == *(_QWORD *)v101 )
            goto LABEL_156;
          ++v108;
        }
        if ( !v106 )
          v106 = v101;
        ++v164;
        v107 = v166 + 1;
        if ( 4 * ((int)v166 + 1) < 3 * v167 )
        {
          if ( v167 - HIDWORD(v166) - v107 > v167 >> 3 )
            goto LABEL_170;
          goto LABEL_161;
        }
LABEL_160:
        v103 = 2 * v167;
LABEL_161:
        sub_1C5FBB0((__int64)&v164, v103);
        sub_1C57100((__int64)&v164, v105, &v168);
        v106 = (__int64)v168;
        v107 = v166 + 1;
LABEL_170:
        LODWORD(v166) = v107;
        if ( *(_QWORD *)v106 != -8 )
          --HIDWORD(v166);
        v109 = *v105;
        *(_QWORD *)(v106 + 8) = 0;
        *(_QWORD *)(v106 + 16) = 0;
        *(_QWORD *)(v106 + 24) = 0;
        *(_DWORD *)(v106 + 32) = 0;
        *(_QWORD *)v106 = v109;
        v97 = v157;
LABEL_173:
        v110 = (unsigned __int64 *)sub_1C53170((__int64)&v160, &v97[v104])[1];
        if ( v110 )
        {
          if ( (unsigned __int64 *)*v110 != v110 + 2 )
            _libc_free(*v110);
          j_j___libc_free_0(v110, 80);
        }
        v111 = sub_1C53170((__int64)&v160, &v157[v104]);
        v97 = v157;
        v111[1] = 0;
        goto LABEL_157;
      }
LABEL_156:
      if ( *(_DWORD *)(v101 + 24) <= 1u )
        goto LABEL_173;
LABEL_157:
      v99 = (unsigned int)++v98;
      if ( v98 == v158 - v97 )
        goto LABEL_178;
    }
    ++v164;
    goto LABEL_160;
  }
LABEL_178:
  v112 = v172;
  if ( (unsigned int)v172 > 1 )
  {
    v138 = sub_1C5DFC0(a1, &v157, (__int64)&v160, a1[25], &v154, a5, a6);
    v112 = v172;
    if ( v138 )
      v149 = v138;
  }
  v113 = 0;
  v114 = 8LL * v112;
  if ( v112 )
  {
    do
    {
      v115 = *(_QWORD *)&v171[v113];
      if ( v115 )
        j_j___libc_free_0(v115, 32);
      v113 += 8;
    }
    while ( v113 != v114 );
  }
  if ( (_DWORD)v162 )
  {
    v116 = v161;
    v117 = &v161[2 * v163];
    if ( v161 != v117 )
    {
      while ( 1 )
      {
        v118 = v116;
        if ( *v116 != -8 && *v116 != -16 )
          break;
        v116 += 2;
        if ( v117 == v116 )
          goto LABEL_49;
      }
      while ( v118 != v117 )
      {
        v139 = (unsigned __int64 *)v118[1];
        if ( v139 )
        {
          if ( (unsigned __int64 *)*v139 != v139 + 2 )
            _libc_free(*v139);
          j_j___libc_free_0(v139, 80);
        }
        v118 += 2;
        if ( v118 == v117 )
          break;
        while ( *v118 == -16 || *v118 == -8 )
        {
          v118 += 2;
          if ( v117 == v118 )
            goto LABEL_49;
        }
      }
    }
  }
LABEL_49:
  if ( v167 )
  {
    v47 = v165;
    v48 = &v165[5 * v167];
    do
    {
      if ( *v47 != -16 && *v47 != -8 )
        j___libc_free_0(v47[2]);
      v47 += 5;
    }
    while ( v48 != v47 );
  }
  j___libc_free_0(v165);
  if ( v157 )
    j_j___libc_free_0(v157, (char *)v159 - (char *)v157);
  j___libc_free_0(v161);
  if ( v171 != v173 )
    _libc_free((unsigned __int64)v171);
  return v149;
}
