// Function: sub_1ED2DC0
// Address: 0x1ed2dc0
//
_QWORD *__fastcall sub_1ED2DC0(_QWORD *a1, unsigned int *a2)
{
  __int64 *v2; // rax
  __int64 v3; // rdi
  unsigned int v4; // r12d
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 j; // rdi
  __int64 v9; // rcx
  __int64 v10; // rsi
  _QWORD *v11; // rdx
  float v12; // xmm0_4
  _QWORD *v13; // rax
  float v14; // xmm1_4
  bool v15; // al
  __int64 v16; // rax
  _BYTE *v17; // rsi
  unsigned int v18; // r9d
  _QWORD *v19; // r15
  __int64 v20; // rax
  unsigned int *v21; // r10
  unsigned int *v22; // r13
  __int64 v23; // rdx
  __int64 v24; // r12
  __int64 v25; // rax
  unsigned int v26; // ecx
  unsigned int v27; // ebx
  _QWORD *v28; // r14
  __int64 v29; // r8
  __int64 v30; // rsi
  __int64 v31; // rcx
  __int64 v32; // rdx
  int v33; // eax
  __int64 v34; // r11
  __int64 v35; // rdx
  unsigned int v36; // eax
  __int64 v37; // rdx
  __int64 v38; // rsi
  __int64 v39; // r8
  __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // rdi
  __int64 *v43; // rbx
  __int64 v44; // rax
  _BYTE *v45; // rsi
  unsigned int v46; // r8d
  _QWORD *v47; // rdi
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rsi
  __int64 v51; // rdx
  _DWORD *v52; // rdi
  int v53; // eax
  _QWORD *v54; // rdx
  _QWORD *v55; // rcx
  unsigned int v56; // esi
  _QWORD *v57; // rax
  _BOOL4 v58; // r11d
  __int64 v59; // rax
  int v60; // eax
  _QWORD *v61; // rdx
  _QWORD *v62; // rcx
  unsigned int v63; // esi
  _QWORD *v64; // rax
  _BOOL4 v65; // r11d
  __int64 v66; // rax
  __int64 v67; // rdi
  __int64 v68; // rax
  _BYTE *v69; // rsi
  unsigned int v70; // r9d
  _QWORD *v71; // r15
  __int64 v72; // rax
  unsigned int *v73; // r11
  unsigned int *v74; // r13
  __int64 v75; // rdx
  __int64 v76; // r12
  __int64 v77; // rax
  unsigned int v78; // ecx
  unsigned int v79; // ebx
  _QWORD *v80; // r14
  __int64 v81; // r8
  __int64 v82; // rsi
  __int64 v83; // rcx
  __int64 v84; // rdx
  int v85; // eax
  __int64 v86; // r10
  unsigned int v87; // edx
  unsigned int i; // eax
  __int64 v89; // rdx
  __int64 v90; // rsi
  __int64 v91; // r8
  __int64 v92; // rcx
  __int64 v93; // rdx
  __int64 v94; // rdi
  __int64 v95; // rsi
  __int64 v96; // rdx
  _DWORD *v97; // rdi
  int v98; // eax
  _QWORD *v99; // rdx
  _QWORD *v100; // rcx
  unsigned int v101; // esi
  _QWORD *v102; // rax
  _BOOL4 v103; // r10d
  __int64 v104; // rax
  int v105; // eax
  _QWORD *v106; // rdx
  _QWORD *v107; // rcx
  unsigned int v108; // esi
  _QWORD *v109; // rax
  _BOOL4 v110; // r10d
  __int64 v111; // rax
  __int64 v112; // rax
  _QWORD *v113; // rdi
  __int64 v114; // rax
  __int64 v115; // rax
  _QWORD *v117; // rdi
  _QWORD *v118; // rdi
  __int64 v119; // rax
  _QWORD *v120; // rdi
  unsigned int v121; // [rsp+4h] [rbp-8Ch]
  unsigned int v122; // [rsp+4h] [rbp-8Ch]
  unsigned int v123; // [rsp+4h] [rbp-8Ch]
  unsigned int v124; // [rsp+4h] [rbp-8Ch]
  unsigned int *v125; // [rsp+8h] [rbp-88h]
  unsigned int *v126; // [rsp+8h] [rbp-88h]
  unsigned int *v127; // [rsp+8h] [rbp-88h]
  unsigned int *v128; // [rsp+8h] [rbp-88h]
  _QWORD *v129; // [rsp+8h] [rbp-88h]
  _QWORD *v130; // [rsp+8h] [rbp-88h]
  _QWORD *v131; // [rsp+8h] [rbp-88h]
  unsigned int v132; // [rsp+8h] [rbp-88h]
  __int64 v133; // [rsp+10h] [rbp-80h]
  __int64 v134; // [rsp+10h] [rbp-80h]
  __int64 v135; // [rsp+10h] [rbp-80h]
  __int64 v136; // [rsp+10h] [rbp-80h]
  _QWORD *v137; // [rsp+10h] [rbp-80h]
  _QWORD *v138; // [rsp+10h] [rbp-80h]
  _QWORD *v139; // [rsp+10h] [rbp-80h]
  unsigned int *v140; // [rsp+10h] [rbp-80h]
  _BOOL4 v141; // [rsp+18h] [rbp-78h]
  _BOOL4 v142; // [rsp+18h] [rbp-78h]
  _QWORD *v143; // [rsp+18h] [rbp-78h]
  _BOOL4 v144; // [rsp+18h] [rbp-78h]
  unsigned int v145; // [rsp+18h] [rbp-78h]
  unsigned int v146; // [rsp+18h] [rbp-78h]
  unsigned int v147; // [rsp+18h] [rbp-78h]
  unsigned int v148; // [rsp+18h] [rbp-78h]
  unsigned int v149; // [rsp+18h] [rbp-78h]
  unsigned int v150; // [rsp+18h] [rbp-78h]
  unsigned int v151; // [rsp+18h] [rbp-78h]
  unsigned int v152; // [rsp+18h] [rbp-78h]
  unsigned int v153; // [rsp+18h] [rbp-78h]
  __int64 v154; // [rsp+18h] [rbp-78h]
  unsigned int v155; // [rsp+18h] [rbp-78h]
  unsigned int v156; // [rsp+18h] [rbp-78h]
  _QWORD *v157; // [rsp+20h] [rbp-70h]
  _QWORD *v158; // [rsp+20h] [rbp-70h]
  _BOOL4 v159; // [rsp+20h] [rbp-70h]
  _QWORD *v160; // [rsp+20h] [rbp-70h]
  unsigned int *v161; // [rsp+20h] [rbp-70h]
  unsigned int *v162; // [rsp+20h] [rbp-70h]
  unsigned int *v163; // [rsp+20h] [rbp-70h]
  unsigned int *v164; // [rsp+20h] [rbp-70h]
  unsigned int *v165; // [rsp+20h] [rbp-70h]
  unsigned int *v166; // [rsp+20h] [rbp-70h]
  unsigned int *v167; // [rsp+20h] [rbp-70h]
  unsigned int *v168; // [rsp+20h] [rbp-70h]
  unsigned int *v169; // [rsp+20h] [rbp-70h]
  _QWORD *v170; // [rsp+20h] [rbp-70h]
  unsigned int *v171; // [rsp+20h] [rbp-70h]
  unsigned int *v172; // [rsp+20h] [rbp-70h]
  _QWORD *v173; // [rsp+28h] [rbp-68h]
  _QWORD *v174; // [rsp+28h] [rbp-68h]
  _QWORD *v175; // [rsp+28h] [rbp-68h]
  _QWORD *v176; // [rsp+28h] [rbp-68h]
  __int64 v177; // [rsp+28h] [rbp-68h]
  __int64 v178; // [rsp+28h] [rbp-68h]
  __int64 v179; // [rsp+28h] [rbp-68h]
  __int64 v180; // [rsp+28h] [rbp-68h]
  __int64 v181; // [rsp+28h] [rbp-68h]
  __int64 v182; // [rsp+28h] [rbp-68h]
  __int64 v183; // [rsp+28h] [rbp-68h]
  _QWORD *v184; // [rsp+28h] [rbp-68h]
  __int64 v185; // [rsp+28h] [rbp-68h]
  __int64 v186; // [rsp+28h] [rbp-68h]
  __int64 v187; // [rsp+30h] [rbp-60h]
  unsigned int v190; // [rsp+58h] [rbp-38h] BYREF
  int v191[13]; // [rsp+5Ch] [rbp-34h] BYREF

  v187 = (__int64)(a2 + 28);
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
LABEL_2:
  while ( 2 )
  {
    while ( 2 )
    {
      v2 = (__int64 *)a2;
      if ( *((_QWORD *)a2 + 6) )
        goto LABEL_34;
LABEL_3:
      if ( *((_QWORD *)a2 + 12) )
      {
        v67 = *((_QWORD *)a2 + 10);
        v190 = *(_DWORD *)(v67 + 32);
        v68 = sub_220F330(v67, a2 + 16);
        j_j___libc_free_0(v68, 40);
        --*((_QWORD *)a2 + 12);
        v69 = (_BYTE *)a1[1];
        if ( v69 == (_BYTE *)a1[2] )
        {
          sub_B8BBF0((__int64)a1, v69, &v190);
          v70 = v190;
        }
        else
        {
          v70 = v190;
          if ( v69 )
          {
            *(_DWORD *)v69 = v190;
            v69 = (_BYTE *)a1[1];
            v70 = v190;
          }
          a1[1] = v69 + 4;
        }
        v71 = *(_QWORD **)a2;
        v72 = *(_QWORD *)(*(_QWORD *)a2 + 160LL) + 88LL * v70;
        v73 = *(unsigned int **)(v72 + 72);
        v74 = *(unsigned int **)(v72 + 64);
        if ( v74 == v73 )
          continue;
        while ( 1 )
        {
          v75 = v71[26];
          v76 = 48LL * *v74;
          v77 = v75 + v76;
          v78 = *(_DWORD *)(v75 + v76 + 20);
          v79 = v78;
          if ( v78 == v70 )
            v79 = *(_DWORD *)(v77 + 24);
          v80 = (_QWORD *)v71[19];
          if ( !v80 )
            goto LABEL_95;
          v81 = 88LL * v79;
          v82 = v81 + *(_QWORD *)(*v80 + 160LL);
          v83 = v76 + *(_QWORD *)(*v80 + 208LL);
          v84 = *(_QWORD *)v83;
          v85 = *(_DWORD *)(v82 + 24);
          if ( v79 == *(_DWORD *)(v83 + 24) )
          {
            *(_DWORD *)(v82 + 24) = v85 - *(_DWORD *)(v84 + 16);
            v86 = *(_QWORD *)(v84 + 32);
          }
          else
          {
            *(_DWORD *)(v82 + 24) = v85 - *(_DWORD *)(v84 + 20);
            v86 = *(_QWORD *)(v84 + 24);
          }
          v87 = *(_DWORD *)(v82 + 20);
          if ( v87 )
          {
            for ( i = 0; i < v87; ++i )
            {
              v89 = i;
              *(_DWORD *)(*(_QWORD *)(v82 + 32) + 4 * v89) -= *(unsigned __int8 *)(v86 + v89);
              v87 = *(_DWORD *)(v82 + 20);
            }
          }
          if ( *(_QWORD *)(v81 + *(_QWORD *)(*v80 + 160LL) + 72) - *(_QWORD *)(v81 + *(_QWORD *)(*v80 + 160LL) + 64) == 12 )
          {
            v191[0] = v79;
            v105 = *(_DWORD *)(*(_QWORD *)(*v80 + 160LL) + v81 + 16);
            switch ( v105 )
            {
              case 2:
                v151 = v70;
                v117 = v80 + 7;
                v167 = v73;
                v181 = 88LL * v79;
                break;
              case 3:
                v151 = v70;
                v117 = v80 + 1;
                v167 = v73;
                v181 = 88LL * v79;
                break;
              case 1:
                v150 = v70;
                v166 = v73;
                sub_1ECB700(v80 + 13, (unsigned int *)v191);
                v106 = (_QWORD *)v80[3];
                v70 = v150;
                v107 = v80 + 2;
                v73 = v166;
                v81 = 88LL * v79;
                if ( !v106 )
                  goto LABEL_164;
                while ( 1 )
                {
LABEL_129:
                  v108 = *((_DWORD *)v106 + 8);
                  v109 = (_QWORD *)v106[3];
                  if ( v79 < v108 )
                    v109 = (_QWORD *)v106[2];
                  if ( !v109 )
                    break;
                  v106 = v109;
                }
                if ( v79 >= v108 )
                {
                  if ( v79 > v108 )
                    goto LABEL_134;
LABEL_136:
                  *(_DWORD *)(*(_QWORD *)(*v80 + 160LL) + v81 + 16) = 3;
                  v75 = v71[26];
                  v77 = v75 + v76;
                  v78 = *(_DWORD *)(v75 + v76 + 20);
                  goto LABEL_95;
                }
                if ( v106 == (_QWORD *)v80[4] )
                {
LABEL_134:
                  v110 = 1;
                  if ( v107 != v106 )
                    goto LABEL_154;
                }
                else
                {
LABEL_151:
                  v148 = v70;
                  v164 = v73;
                  v179 = v81;
                  v130 = v107;
                  v138 = v106;
                  v114 = sub_220EF80(v106);
                  v81 = v179;
                  v73 = v164;
                  v70 = v148;
                  if ( v79 <= *(_DWORD *)(v114 + 32) )
                    goto LABEL_136;
                  v106 = v138;
                  v107 = v130;
                  if ( !v138 )
                    goto LABEL_136;
                  v110 = 1;
                  if ( v130 != v138 )
LABEL_154:
                    v110 = v79 < *((_DWORD *)v106 + 8);
                }
LABEL_135:
                v124 = v70;
                v128 = v73;
                v136 = v81;
                v144 = v110;
                v160 = v107;
                v176 = v106;
                v111 = sub_22077B0(40);
                *(_DWORD *)(v111 + 32) = v79;
                sub_220F040(v144, v111, v176, v160);
                ++v80[6];
                v70 = v124;
                v73 = v128;
                v81 = v136;
                goto LABEL_136;
              default:
                goto LABEL_126;
            }
            sub_1ECB700(v117, (unsigned int *)v191);
            v81 = v181;
            v73 = v167;
            v70 = v151;
LABEL_126:
            v106 = (_QWORD *)v80[3];
            v107 = v80 + 2;
            if ( v106 )
              goto LABEL_129;
LABEL_164:
            v106 = v107;
            if ( v107 != (_QWORD *)v80[4] )
              goto LABEL_151;
            v110 = 1;
            goto LABEL_135;
          }
          if ( *(_DWORD *)(v82 + 16) == 1 )
          {
            if ( v87 > *(_DWORD *)(v82 + 24) )
              break;
            v191[0] = 0;
            v97 = *(_DWORD **)(v82 + 32);
            if ( &v97[v87] != sub_1ECB090(v97, (__int64)&v97[v87], v191) )
              break;
          }
LABEL_94:
          v75 = v71[26];
          v77 = v75 + v76;
          v78 = *(_DWORD *)(v75 + v76 + 20);
LABEL_95:
          v90 = v71[20];
          if ( v79 == v78 )
          {
            v94 = *(_QWORD *)(v77 + 32);
            v95 = v90 + 88LL * v79;
            v96 = 48LL * *(unsigned int *)(*(_QWORD *)(v95 + 72) - 4LL) + v75;
            if ( v79 == *(_DWORD *)(v96 + 20) )
              *(_QWORD *)(v96 + 32) = v94;
            else
              *(_QWORD *)(v96 + 40) = v94;
            *(_DWORD *)(*(_QWORD *)(v95 + 64) + 4 * v94) = *(_DWORD *)(*(_QWORD *)(v95 + 72) - 4LL);
            *(_QWORD *)(v95 + 72) -= 4LL;
            *(_QWORD *)(v77 + 32) = -1;
          }
          else
          {
            v91 = *(_QWORD *)(v77 + 40);
            v92 = v90 + 88LL * *(unsigned int *)(v77 + 24);
            v93 = 48LL * *(unsigned int *)(*(_QWORD *)(v92 + 72) - 4LL) + v75;
            if ( *(_DWORD *)(v77 + 24) == *(_DWORD *)(v93 + 20) )
              *(_QWORD *)(v93 + 32) = v91;
            else
              *(_QWORD *)(v93 + 40) = v91;
            *(_DWORD *)(*(_QWORD *)(v92 + 64) + 4 * v91) = *(_DWORD *)(*(_QWORD *)(v92 + 72) - 4LL);
            *(_QWORD *)(v92 + 72) -= 4LL;
            *(_QWORD *)(v77 + 40) = -1;
          }
          if ( v73 == ++v74 )
            goto LABEL_2;
        }
        v191[0] = v79;
        v98 = *(_DWORD *)(*(_QWORD *)(*v80 + 160LL) + v81 + 16);
        switch ( v98 )
        {
          case 2:
            v156 = v70;
            v120 = v80 + 7;
            v172 = v73;
            v186 = v81;
            break;
          case 3:
            v156 = v70;
            v120 = v80 + 1;
            v172 = v73;
            v186 = v81;
            break;
          case 1:
            v155 = v70;
            v171 = v73;
            v185 = v81;
            sub_1ECB700(v80 + 13, (unsigned int *)v191);
            v99 = (_QWORD *)v80[9];
            v70 = v155;
            v100 = v80 + 8;
            v73 = v171;
            v81 = v185;
            if ( !v99 )
              goto LABEL_181;
            goto LABEL_114;
          default:
LABEL_111:
            v99 = (_QWORD *)v80[9];
            v100 = v80 + 8;
            if ( !v99 )
            {
LABEL_181:
              v99 = v100;
              if ( v100 == (_QWORD *)v80[10] )
              {
                v103 = 1;
                goto LABEL_120;
              }
              goto LABEL_176;
            }
            while ( 1 )
            {
LABEL_114:
              v101 = *((_DWORD *)v99 + 8);
              v102 = (_QWORD *)v99[3];
              if ( v79 < v101 )
                v102 = (_QWORD *)v99[2];
              if ( !v102 )
                break;
              v99 = v102;
            }
            if ( v79 < v101 )
            {
              if ( v99 != (_QWORD *)v80[10] )
              {
LABEL_176:
                v132 = v70;
                v140 = v73;
                v154 = v81;
                v170 = v100;
                v184 = v99;
                v119 = sub_220EF80(v99);
                v81 = v154;
                v73 = v140;
                v70 = v132;
                if ( v79 <= *(_DWORD *)(v119 + 32) )
                  goto LABEL_121;
                v99 = v184;
                v100 = v170;
                if ( !v184 )
                  goto LABEL_121;
                v103 = 1;
                if ( v170 == v184 )
                  goto LABEL_120;
                goto LABEL_179;
              }
            }
            else if ( v79 <= v101 )
            {
LABEL_121:
              *(_DWORD *)(*(_QWORD *)(*v80 + 160LL) + v81 + 16) = 2;
              goto LABEL_94;
            }
            v103 = 1;
            if ( v100 == v99 )
            {
LABEL_120:
              v123 = v70;
              v127 = v73;
              v135 = v81;
              v143 = v100;
              v159 = v103;
              v175 = v99;
              v104 = sub_22077B0(40);
              *(_DWORD *)(v104 + 32) = v79;
              sub_220F040(v159, v104, v175, v143);
              ++v80[12];
              v70 = v123;
              v73 = v127;
              v81 = v135;
              goto LABEL_121;
            }
LABEL_179:
            v103 = v79 < *((_DWORD *)v99 + 8);
            goto LABEL_120;
        }
        sub_1ECB700(v120, (unsigned int *)v191);
        v81 = v186;
        v73 = v172;
        v70 = v156;
        goto LABEL_111;
      }
      break;
    }
    if ( *((_QWORD *)a2 + 18) )
    {
      v3 = *((_QWORD *)a2 + 16);
      if ( v3 == v187 )
      {
        v5 = v187;
        v4 = a2[36];
      }
      else
      {
        v4 = *(_DWORD *)(v3 + 32);
        v5 = *((_QWORD *)a2 + 16);
        v6 = *(_QWORD *)a2;
        v7 = sub_220EF30(v3);
        for ( j = v7; v187 != v7; j = v7 )
        {
          v9 = *(_QWORD *)(v6 + 160);
          v10 = *(unsigned int *)(v7 + 32);
          v11 = (_QWORD *)(v9 + 88 * v10);
          v12 = **(float **)(*v11 + 8LL);
          v13 = (_QWORD *)(v9 + 88LL * v4);
          v14 = **(float **)(*v13 + 8LL);
          if ( v12 == v14 )
            v15 = v13[9] - v13[8] > v11[9] - v11[8];
          else
            v15 = v14 > v12;
          if ( v15 )
          {
            v5 = j;
            v4 = v10;
          }
          v7 = sub_220EF30(j);
        }
      }
      v190 = v4;
      v16 = sub_220F330(v5, v187);
      j_j___libc_free_0(v16, 40);
      --*((_QWORD *)a2 + 18);
      v17 = (_BYTE *)a1[1];
      if ( v17 == (_BYTE *)a1[2] )
      {
        sub_B8BBF0((__int64)a1, v17, &v190);
        v18 = v190;
      }
      else
      {
        v18 = v190;
        if ( v17 )
        {
          *(_DWORD *)v17 = v190;
          v17 = (_BYTE *)a1[1];
          v18 = v190;
        }
        a1[1] = v17 + 4;
      }
      v19 = *(_QWORD **)a2;
      v20 = *(_QWORD *)(*(_QWORD *)a2 + 160LL) + 88LL * v18;
      v21 = *(unsigned int **)(v20 + 72);
      v22 = *(unsigned int **)(v20 + 64);
      if ( v22 == v21 )
        continue;
      while ( 1 )
      {
        v23 = v19[26];
        v24 = 48LL * *v22;
        v25 = v23 + v24;
        v26 = *(_DWORD *)(v23 + v24 + 20);
        v27 = v26;
        if ( v26 == v18 )
          v27 = *(_DWORD *)(v25 + 24);
        v28 = (_QWORD *)v19[19];
        if ( !v28 )
          goto LABEL_28;
        v29 = 88LL * v27;
        v30 = v29 + *(_QWORD *)(*v28 + 160LL);
        v31 = v24 + *(_QWORD *)(*v28 + 208LL);
        v32 = *(_QWORD *)v31;
        v33 = *(_DWORD *)(v30 + 24);
        if ( v27 == *(_DWORD *)(v31 + 24) )
        {
          *(_DWORD *)(v30 + 24) = v33 - *(_DWORD *)(v32 + 16);
          v34 = *(_QWORD *)(v32 + 32);
        }
        else
        {
          *(_DWORD *)(v30 + 24) = v33 - *(_DWORD *)(v32 + 20);
          v34 = *(_QWORD *)(v32 + 24);
        }
        v35 = *(unsigned int *)(v30 + 20);
        if ( (_DWORD)v35 )
        {
          v36 = 0;
          do
          {
            v37 = v36++;
            *(_DWORD *)(*(_QWORD *)(v30 + 32) + 4 * v37) -= *(unsigned __int8 *)(v34 + v37);
            v35 = *(unsigned int *)(v30 + 20);
          }
          while ( (unsigned int)v35 > v36 );
        }
        if ( *(_QWORD *)(v29 + *(_QWORD *)(*v28 + 160LL) + 72) - *(_QWORD *)(v29 + *(_QWORD *)(*v28 + 160LL) + 64) == 12 )
        {
          v191[0] = v27;
          v60 = *(_DWORD *)(*(_QWORD *)(*v28 + 160LL) + v29 + 16);
          switch ( v60 )
          {
            case 2:
              v147 = v18;
              v113 = v28 + 7;
              v163 = v21;
              v178 = 88LL * v27;
              break;
            case 3:
              v147 = v18;
              v113 = v28 + 1;
              v163 = v21;
              v178 = 88LL * v27;
              break;
            case 1:
              v146 = v18;
              v162 = v21;
              sub_1ECB700(v28 + 13, (unsigned int *)v191);
              v61 = (_QWORD *)v28[3];
              v18 = v146;
              v62 = v28 + 2;
              v21 = v162;
              v29 = 88LL * v27;
              if ( !v61 )
                goto LABEL_145;
              while ( 1 )
              {
LABEL_71:
                v63 = *((_DWORD *)v61 + 8);
                v64 = (_QWORD *)v61[3];
                if ( v27 < v63 )
                  v64 = (_QWORD *)v61[2];
                if ( !v64 )
                  break;
                v61 = v64;
              }
              if ( v27 >= v63 )
              {
                if ( v27 > v63 )
                  goto LABEL_76;
LABEL_78:
                *(_DWORD *)(*(_QWORD *)(*v28 + 160LL) + v29 + 16) = 3;
                v23 = v19[26];
                v25 = v23 + v24;
                v26 = *(_DWORD *)(v23 + v24 + 20);
                goto LABEL_28;
              }
              if ( v61 == (_QWORD *)v28[4] )
              {
LABEL_76:
                v65 = 1;
                if ( v62 != v61 )
                  goto LABEL_141;
              }
              else
              {
LABEL_138:
                v145 = v18;
                v161 = v21;
                v177 = v29;
                v129 = v62;
                v137 = v61;
                v112 = sub_220EF80(v61);
                v29 = v177;
                v21 = v161;
                v18 = v145;
                if ( v27 <= *(_DWORD *)(v112 + 32) )
                  goto LABEL_78;
                v61 = v137;
                v62 = v129;
                if ( !v137 )
                  goto LABEL_78;
                v65 = 1;
                if ( v129 != v137 )
LABEL_141:
                  v65 = v27 < *((_DWORD *)v61 + 8);
              }
LABEL_77:
              v122 = v18;
              v126 = v21;
              v134 = v29;
              v142 = v65;
              v158 = v62;
              v174 = v61;
              v66 = sub_22077B0(40);
              *(_DWORD *)(v66 + 32) = v27;
              sub_220F040(v142, v66, v174, v158);
              ++v28[6];
              v18 = v122;
              v21 = v126;
              v29 = v134;
              goto LABEL_78;
            default:
              goto LABEL_68;
          }
          sub_1ECB700(v113, (unsigned int *)v191);
          v29 = v178;
          v21 = v163;
          v18 = v147;
LABEL_68:
          v61 = (_QWORD *)v28[3];
          v62 = v28 + 2;
          if ( v61 )
            goto LABEL_71;
LABEL_145:
          v61 = v62;
          if ( v62 != (_QWORD *)v28[4] )
            goto LABEL_138;
          v65 = 1;
          goto LABEL_77;
        }
        if ( *(_DWORD *)(v30 + 16) == 1 )
        {
          if ( (unsigned int)v35 > *(_DWORD *)(v30 + 24) )
            break;
          v191[0] = 0;
          v52 = *(_DWORD **)(v30 + 32);
          if ( &v52[v35] != sub_1ECB090(v52, (__int64)&v52[v35], v191) )
            break;
        }
LABEL_27:
        v23 = v19[26];
        v25 = v23 + v24;
        v26 = *(_DWORD *)(v23 + v24 + 20);
LABEL_28:
        v38 = v19[20];
        if ( v27 == v26 )
        {
          v49 = *(_QWORD *)(v25 + 32);
          v50 = v38 + 88LL * v27;
          v51 = 48LL * *(unsigned int *)(*(_QWORD *)(v50 + 72) - 4LL) + v23;
          if ( *(_DWORD *)(v51 + 20) == v27 )
            *(_QWORD *)(v51 + 32) = v49;
          else
            *(_QWORD *)(v51 + 40) = v49;
          *(_DWORD *)(*(_QWORD *)(v50 + 64) + 4 * v49) = *(_DWORD *)(*(_QWORD *)(v50 + 72) - 4LL);
          *(_QWORD *)(v50 + 72) -= 4LL;
          *(_QWORD *)(v25 + 32) = -1;
        }
        else
        {
          v39 = *(_QWORD *)(v25 + 40);
          v40 = v38 + 88LL * *(unsigned int *)(v25 + 24);
          v41 = 48LL * *(unsigned int *)(*(_QWORD *)(v40 + 72) - 4LL) + v23;
          if ( *(_DWORD *)(v25 + 24) == *(_DWORD *)(v41 + 20) )
            *(_QWORD *)(v41 + 32) = v39;
          else
            *(_QWORD *)(v41 + 40) = v39;
          *(_DWORD *)(*(_QWORD *)(v40 + 64) + 4 * v39) = *(_DWORD *)(*(_QWORD *)(v40 + 72) - 4LL);
          *(_QWORD *)(v40 + 72) -= 4LL;
          *(_QWORD *)(v25 + 40) = -1;
        }
        if ( v21 == ++v22 )
        {
          v2 = (__int64 *)a2;
          if ( *((_QWORD *)a2 + 6) )
          {
LABEL_34:
            v42 = v2[4];
            v43 = v2;
            v191[0] = *(_DWORD *)(v42 + 32);
            v44 = sub_220F330(v42, v2 + 2);
            j_j___libc_free_0(v44, 40);
            --v43[6];
            v45 = (_BYTE *)a1[1];
            if ( v45 == (_BYTE *)a1[2] )
            {
              sub_B8BBF0((__int64)a1, v45, v191);
              v46 = v191[0];
            }
            else
            {
              v46 = v191[0];
              if ( v45 )
              {
                *(_DWORD *)v45 = v191[0];
                v45 = (_BYTE *)a1[1];
                v46 = v191[0];
              }
              a1[1] = v45 + 4;
            }
            v47 = *(_QWORD **)a2;
            v48 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 160LL) + 88LL * v46 + 72)
                - *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 160LL) + 88LL * v46 + 64);
            if ( v48 == 4 )
            {
              sub_1ED1370(v47, v46);
            }
            else if ( v48 == 8 )
            {
              sub_1ED2780(v47, v46);
            }
            goto LABEL_2;
          }
          goto LABEL_3;
        }
      }
      v191[0] = v27;
      v53 = *(_DWORD *)(*(_QWORD *)(*v28 + 160LL) + v29 + 16);
      if ( v53 == 2 )
      {
        v152 = v18;
        v118 = v28 + 7;
        v168 = v21;
        v182 = v29;
      }
      else
      {
        if ( v53 != 3 )
        {
          if ( v53 == 1 )
          {
            v153 = v18;
            v169 = v21;
            v183 = v29;
            sub_1ECB700(v28 + 13, (unsigned int *)v191);
            v18 = v153;
            v21 = v169;
            v29 = v183;
          }
          v54 = (_QWORD *)v28[9];
          v55 = v28 + 8;
          if ( !v54 )
          {
LABEL_172:
            v54 = v55;
            if ( v55 == (_QWORD *)v28[10] )
            {
              v58 = 1;
              goto LABEL_62;
            }
            goto LABEL_157;
          }
          while ( 1 )
          {
LABEL_56:
            v56 = *((_DWORD *)v54 + 8);
            v57 = (_QWORD *)v54[3];
            if ( v27 < v56 )
              v57 = (_QWORD *)v54[2];
            if ( !v57 )
              break;
            v54 = v57;
          }
          if ( v27 < v56 )
          {
            if ( v54 != (_QWORD *)v28[10] )
            {
LABEL_157:
              v149 = v18;
              v165 = v21;
              v180 = v29;
              v131 = v55;
              v139 = v54;
              v115 = sub_220EF80(v54);
              v29 = v180;
              v21 = v165;
              v18 = v149;
              if ( v27 <= *(_DWORD *)(v115 + 32) )
                goto LABEL_63;
              v54 = v139;
              v55 = v131;
              if ( !v139 )
                goto LABEL_63;
              v58 = 1;
              if ( v131 == v139 )
                goto LABEL_62;
              goto LABEL_160;
            }
          }
          else if ( v27 <= v56 )
          {
LABEL_63:
            *(_DWORD *)(*(_QWORD *)(*v28 + 160LL) + v29 + 16) = 2;
            goto LABEL_27;
          }
          v58 = 1;
          if ( v55 == v54 )
          {
LABEL_62:
            v121 = v18;
            v125 = v21;
            v133 = v29;
            v141 = v58;
            v157 = v55;
            v173 = v54;
            v59 = sub_22077B0(40);
            *(_DWORD *)(v59 + 32) = v27;
            sub_220F040(v141, v59, v173, v157);
            ++v28[12];
            v18 = v121;
            v21 = v125;
            v29 = v133;
            goto LABEL_63;
          }
LABEL_160:
          v58 = v27 < *((_DWORD *)v54 + 8);
          goto LABEL_62;
        }
        v152 = v18;
        v118 = v28 + 1;
        v168 = v21;
        v182 = v29;
      }
      sub_1ECB700(v118, (unsigned int *)v191);
      v54 = (_QWORD *)v28[9];
      v29 = v182;
      v55 = v28 + 8;
      v21 = v168;
      v18 = v152;
      if ( !v54 )
        goto LABEL_172;
      goto LABEL_56;
    }
    return a1;
  }
}
