// Function: sub_A31250
// Address: 0xa31250
//
__int64 __fastcall sub_A31250(_QWORD *a1, unsigned __int64 a2, __int64 a3, char a4)
{
  __int64 v5; // rax
  int v6; // ecx
  int v7; // ebx
  __int64 v8; // rdi
  __int64 result; // rax
  __int64 v10; // r8
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 *v13; // rax
  _QWORD *v14; // r10
  __int64 v15; // rax
  _QWORD *v16; // r15
  _DWORD *v17; // r13
  _DWORD *v18; // r11
  __int64 v19; // r8
  unsigned int v20; // eax
  _DWORD *v21; // rdx
  int v22; // ecx
  _QWORD *v23; // r14
  __int64 v24; // r12
  __int64 v25; // rbx
  __int64 v26; // rax
  unsigned int v27; // esi
  int v28; // ecx
  _DWORD *v29; // rdi
  int v30; // edx
  __int64 v31; // rax
  unsigned __int64 *v32; // rdx
  _BYTE *v33; // rsi
  __int64 v34; // rdi
  unsigned int *v35; // rax
  unsigned int *v36; // r13
  unsigned __int64 v37; // rax
  int v38; // esi
  int *v39; // rcx
  int v40; // edx
  __int64 v41; // r12
  _QWORD *v42; // r8
  __int64 v43; // rbx
  __int64 v44; // r15
  __int64 v45; // r9
  __int64 v46; // r11
  int *v47; // r14
  unsigned __int64 v48; // rdx
  unsigned __int64 v49; // rax
  _DWORD *v50; // rax
  __int64 v51; // rdx
  __int64 v52; // r10
  int *v53; // r15
  _DWORD *j; // rdx
  int *v55; // rax
  _QWORD *v56; // r12
  int v57; // edx
  int v58; // ecx
  int v59; // ecx
  __int64 v60; // r9
  unsigned int v61; // edi
  _DWORD *v62; // rsi
  int v63; // r8d
  int v64; // edx
  _DWORD *v65; // rdx
  int v66; // esi
  int v67; // ecx
  int v68; // eax
  int *v69; // rdi
  unsigned int v70; // r15d
  int v71; // esi
  int v72; // r14d
  int *v73; // r10
  int v74; // ecx
  __int64 v75; // rdi
  unsigned __int64 *v76; // rax
  _BYTE *v77; // rsi
  unsigned __int64 v78; // rax
  unsigned __int64 v79; // rax
  __int64 v80; // rax
  _DWORD *v81; // rax
  int v82; // r10d
  __int64 v83; // rdx
  int *v84; // r15
  _DWORD *i; // rdx
  int *v86; // rax
  _QWORD *v87; // r12
  int v88; // edx
  int v89; // ecx
  int v90; // ecx
  __int64 v91; // r9
  unsigned int v92; // edi
  _DWORD *v93; // rsi
  int v94; // r8d
  int v95; // edx
  _QWORD *v96; // r8
  _DWORD *v97; // rdx
  int v98; // esi
  int v99; // eax
  unsigned int v100; // r10d
  unsigned int v101; // r15d
  int v102; // esi
  int v103; // r10d
  int *v104; // r14
  int v105; // r13d
  __int64 v106; // r14
  __int64 v107; // r15
  _QWORD *v108; // r12
  unsigned __int64 v109; // rbx
  _QWORD *v110; // rax
  __int64 v111; // rcx
  __int64 v112; // rdx
  __int64 v113; // rax
  _QWORD *v114; // rax
  __int64 v115; // rdx
  _BOOL8 v116; // rdi
  __int64 v117; // rdx
  __int64 v118; // rdx
  int v119; // eax
  int v120; // edi
  __int64 v121; // [rsp+10h] [rbp-B0h]
  int v122; // [rsp+1Ch] [rbp-A4h]
  int v123; // [rsp+1Ch] [rbp-A4h]
  __int64 v124; // [rsp+20h] [rbp-A0h]
  int v125; // [rsp+20h] [rbp-A0h]
  _DWORD *v126; // [rsp+20h] [rbp-A0h]
  __int64 v127; // [rsp+28h] [rbp-98h]
  __int64 v128; // [rsp+30h] [rbp-90h]
  __int64 v129; // [rsp+30h] [rbp-90h]
  __int64 v130; // [rsp+30h] [rbp-90h]
  __int64 v131; // [rsp+30h] [rbp-90h]
  _DWORD *v132; // [rsp+30h] [rbp-90h]
  __int64 v133; // [rsp+38h] [rbp-88h]
  __int64 v134; // [rsp+38h] [rbp-88h]
  __int64 v135; // [rsp+38h] [rbp-88h]
  __int64 v136; // [rsp+38h] [rbp-88h]
  unsigned int v137; // [rsp+38h] [rbp-88h]
  _QWORD *v138; // [rsp+38h] [rbp-88h]
  _DWORD *v139; // [rsp+40h] [rbp-80h]
  _QWORD *v140; // [rsp+40h] [rbp-80h]
  _QWORD *v141; // [rsp+48h] [rbp-78h]
  unsigned int v142; // [rsp+48h] [rbp-78h]
  _QWORD *v143; // [rsp+48h] [rbp-78h]
  int v144; // [rsp+48h] [rbp-78h]
  _QWORD *v145; // [rsp+48h] [rbp-78h]
  __int64 v146; // [rsp+48h] [rbp-78h]
  __int64 v147; // [rsp+50h] [rbp-70h]
  _QWORD *v148; // [rsp+50h] [rbp-70h]
  int v149; // [rsp+50h] [rbp-70h]
  _DWORD *v150; // [rsp+50h] [rbp-70h]
  _QWORD *v151; // [rsp+50h] [rbp-70h]
  __int64 v152; // [rsp+58h] [rbp-68h]
  __int64 v153; // [rsp+58h] [rbp-68h]
  __int64 v155; // [rsp+68h] [rbp-58h]
  unsigned int *v156; // [rsp+68h] [rbp-58h]
  unsigned __int64 v157; // [rsp+70h] [rbp-50h] BYREF
  __int64 v158; // [rsp+78h] [rbp-48h]
  _DWORD *v159; // [rsp+80h] [rbp-40h] BYREF
  unsigned __int64 *v160[7]; // [rsp+88h] [rbp-38h] BYREF

  v5 = *a1;
  v157 = a2;
  v6 = *(_DWORD *)(v5 + 144);
  v158 = a3;
  v7 = v6 + 1;
  *(_DWORD *)(v5 + 144) = v6 + 1;
  v8 = *a1;
  result = *(_QWORD *)(v8 + 56);
  v10 = v8 + 48;
  if ( !result )
    goto LABEL_61;
  do
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(result + 16);
      v12 = *(_QWORD *)(result + 24);
      if ( *(_QWORD *)(result + 32) >= a2 )
        break;
      result = *(_QWORD *)(result + 24);
      if ( !v12 )
        goto LABEL_6;
    }
    v10 = result;
    result = *(_QWORD *)(result + 16);
  }
  while ( v11 );
LABEL_6:
  if ( v8 + 48 == v10 || a2 < *(_QWORD *)(v10 + 32) )
  {
LABEL_61:
    v160[0] = &v157;
    result = sub_A28390((_QWORD *)(v8 + 40), (_QWORD *)v10, v160);
    *(_DWORD *)(result + 40) = v7;
    if ( a4 )
      return result;
  }
  else
  {
    *(_DWORD *)(v10 + 40) = v7;
    if ( a4 )
      return result;
  }
  result = v158;
  v133 = v158;
  if ( *(_DWORD *)(v158 + 8) == 1 )
  {
    v13 = *(__int64 **)(v158 + 96);
    if ( v13 )
    {
      v152 = v13[1];
      if ( *v13 != v152 )
      {
        v155 = *v13;
        v14 = a1;
        while ( 1 )
        {
          v15 = *(unsigned int *)(v155 + 80);
          if ( (_DWORD)v15 )
            break;
          v105 = *(_DWORD *)(*v14 + 144LL) + 1;
          *(_DWORD *)(*v14 + 144LL) = v105;
          v106 = *v14;
          v107 = *v14 + 48LL;
          v108 = (_QWORD *)v107;
          v109 = *(_QWORD *)(*(_QWORD *)v155 & 0xFFFFFFFFFFFFFFF8LL);
          v110 = *(_QWORD **)(*v14 + 56LL);
          if ( !v110 )
            goto LABEL_111;
          do
          {
            while ( 1 )
            {
              v111 = v110[2];
              v112 = v110[3];
              if ( v109 <= v110[4] )
                break;
              v110 = (_QWORD *)v110[3];
              if ( !v112 )
                goto LABEL_109;
            }
            v108 = v110;
            v110 = (_QWORD *)v110[2];
          }
          while ( v111 );
LABEL_109:
          if ( v108 == (_QWORD *)v107 || v109 < v108[4] )
          {
LABEL_111:
            v140 = v14;
            v148 = v108;
            v113 = sub_22077B0(48);
            *(_QWORD *)(v113 + 32) = v109;
            v108 = (_QWORD *)v113;
            *(_DWORD *)(v113 + 40) = 0;
            v114 = sub_A28290((_QWORD *)(v106 + 40), v148, (unsigned __int64 *)(v113 + 32));
            if ( v115 )
            {
              v116 = v107 == v115 || v114 || v109 < *(_QWORD *)(v115 + 32);
              sub_220F040(v116, v108, v115, v107);
              ++*(_QWORD *)(v106 + 80);
              v14 = v140;
            }
            else
            {
              v151 = v114;
              j_j___libc_free_0(v108, 48);
              v14 = v140;
              v108 = v151;
            }
          }
          *((_DWORD *)v108 + 10) = v105;
LABEL_28:
          v155 += 136;
          if ( v152 == v155 )
            goto LABEL_29;
        }
        v16 = v14;
        v17 = *(_DWORD **)(v155 + 72);
        v18 = &v17[v15];
        while ( 1 )
        {
          v23 = (_QWORD *)v16[1];
          v24 = (unsigned int)*v17;
          v25 = *v23;
          v26 = *(_QWORD *)(*v23 + 96LL) - *(_QWORD *)(*v23 + 88LL);
          LODWORD(v160[0]) = *v17;
          HIDWORD(v160[0]) = v26 >> 3;
          v27 = *(_DWORD *)(v25 + 136);
          if ( !v27 )
            break;
          v19 = *(_QWORD *)(v25 + 120);
          v20 = (v27 - 1) & (37 * v24);
          v21 = (_DWORD *)(v19 + 8LL * v20);
          v22 = *v21;
          if ( (_DWORD)v24 == *v21 )
          {
LABEL_16:
            if ( v18 == ++v17 )
              goto LABEL_27;
          }
          else
          {
            v149 = 1;
            v29 = 0;
            while ( v22 != -1 )
            {
              if ( v22 != -2 || v29 )
                v21 = v29;
              v20 = (v27 - 1) & (v149 + v20);
              v22 = *(_DWORD *)(v19 + 8LL * v20);
              if ( (_DWORD)v24 == v22 )
                goto LABEL_16;
              ++v149;
              v29 = v21;
              v21 = (_DWORD *)(v19 + 8LL * v20);
            }
            v119 = *(_DWORD *)(v25 + 128);
            if ( !v29 )
              v29 = v21;
            ++*(_QWORD *)(v25 + 112);
            v30 = v119 + 1;
            v159 = v29;
            if ( 4 * (v119 + 1) >= 3 * v27 )
              goto LABEL_19;
            v28 = v24;
            if ( v27 - *(_DWORD *)(v25 + 132) - v30 <= v27 >> 3 )
            {
              v139 = v18;
              goto LABEL_20;
            }
LABEL_21:
            *(_DWORD *)(v25 + 128) = v30;
            if ( *v29 != -1 )
              --*(_DWORD *)(v25 + 132);
            *v29 = v28;
            v29[1] = HIDWORD(v160[0]);
            v31 = *v23;
            v32 = *(unsigned __int64 **)(*(_QWORD *)(v23[1] + 528LL) + 8 * v24);
            v160[0] = v32;
            v33 = *(_BYTE **)(v31 + 96);
            if ( v33 == *(_BYTE **)(v31 + 104) )
            {
              v150 = v18;
              sub_A235E0(v31 + 88, v33, v160);
              v18 = v150;
              goto LABEL_16;
            }
            if ( v33 )
            {
              *(_QWORD *)v33 = v32;
              v33 = *(_BYTE **)(v31 + 96);
            }
            ++v17;
            *(_QWORD *)(v31 + 96) = v33 + 8;
            if ( v18 == v17 )
            {
LABEL_27:
              v14 = v16;
              goto LABEL_28;
            }
          }
        }
        ++*(_QWORD *)(v25 + 112);
        v159 = 0;
LABEL_19:
        v139 = v18;
        v27 *= 2;
LABEL_20:
        sub_A09770(v25 + 112, v27);
        sub_A1A0F0(v25 + 112, (int *)v160, &v159);
        v28 = (int)v160[0];
        v29 = v159;
        v18 = v139;
        v30 = *(_DWORD *)(v25 + 128) + 1;
        goto LABEL_21;
      }
    }
LABEL_29:
    result = *(_QWORD *)(v133 + 104);
    if ( result )
    {
      v34 = *(_QWORD *)result;
      result = *(_QWORD *)(result + 8);
      v127 = v34;
      v121 = result;
      if ( v34 != result )
      {
        while ( 1 )
        {
          v153 = *(_QWORD *)(v127 + 64);
          v147 = *(_QWORD *)(v127 + 72);
          if ( v147 != v153 )
            break;
LABEL_64:
          v127 += 112;
          result = v127;
          if ( v127 == v121 )
            return result;
        }
        while ( 1 )
        {
          v35 = *(unsigned int **)(v153 + 8);
          v36 = v35;
          v156 = &v35[*(unsigned int *)(v153 + 16)];
          if ( v156 != v35 )
            break;
LABEL_63:
          v153 += 72;
          if ( v147 == v153 )
            goto LABEL_64;
        }
        while ( 1 )
        {
          v41 = *v36;
          v42 = (_QWORD *)a1[1];
          v43 = *v42;
          v44 = *(unsigned int *)(*v42 + 136LL);
          v45 = *(_QWORD *)(*v42 + 96LL);
          v46 = *(_QWORD *)(*v42 + 88LL);
          v47 = *(int **)(*v42 + 120LL);
          if ( !(_DWORD)v44 )
            break;
          v37 = (unsigned int)(v44 - 1);
          v38 = v37 & (37 * v41);
          v39 = &v47[2 * v38];
          v40 = *v39;
          if ( (_DWORD)v41 != *v39 )
          {
            v144 = 1;
            v69 = 0;
            while ( v40 != -1 )
            {
              if ( v40 != -2 || v69 )
                v39 = v69;
              v38 = v37 & (v144 + v38);
              v40 = v47[2 * v38];
              if ( (_DWORD)v41 == v40 )
                goto LABEL_35;
              ++v144;
              v69 = v39;
              v39 = &v47[2 * v38];
            }
            if ( !v69 )
              v69 = v39;
            v74 = *(_DWORD *)(v43 + 128);
            ++*(_QWORD *)(v43 + 112);
            v67 = v74 + 1;
            if ( 4 * v67 < (unsigned int)(3 * v44) )
            {
              if ( (int)v44 - *(_DWORD *)(v43 + 132) - v67 <= (unsigned int)v44 >> 3 )
              {
                v130 = v46;
                v136 = v45;
                v145 = v42;
                v78 = (((v37 >> 1) | v37) >> 2) | (v37 >> 1) | v37;
                v79 = (((v78 >> 4) | v78) >> 8) | (v78 >> 4) | v78;
                v80 = ((v79 >> 16) | v79) + 1;
                if ( (unsigned int)v80 < 0x40 )
                  LODWORD(v80) = 64;
                *(_DWORD *)(v43 + 136) = v80;
                v81 = (_DWORD *)sub_C7D670(8LL * (unsigned int)v80, 4);
                v42 = v145;
                v45 = v136;
                *(_QWORD *)(v43 + 120) = v81;
                v46 = v130;
                v82 = 37 * v41;
                if ( v47 )
                {
                  v83 = *(unsigned int *)(v43 + 136);
                  *(_QWORD *)(v43 + 128) = 0;
                  v146 = 8 * v44;
                  v84 = &v47[2 * v44];
                  for ( i = &v81[2 * v83]; i != v81; v81 += 2 )
                  {
                    if ( v81 )
                      *v81 = -1;
                  }
                  v137 = v41;
                  v86 = v47;
                  v87 = v42;
                  v131 = v45;
                  do
                  {
                    while ( 1 )
                    {
                      v88 = *v86;
                      if ( (unsigned int)*v86 <= 0xFFFFFFFD )
                        break;
                      v86 += 2;
                      if ( v84 == v86 )
                        goto LABEL_92;
                    }
                    v89 = *(_DWORD *)(v43 + 136);
                    if ( !v89 )
                    {
                      MEMORY[0] = 0;
                      BUG();
                    }
                    v90 = v89 - 1;
                    v91 = *(_QWORD *)(v43 + 120);
                    v92 = v90 & (37 * v88);
                    v93 = (_DWORD *)(v91 + 8LL * v92);
                    v94 = *v93;
                    if ( v88 != *v93 )
                    {
                      v123 = 1;
                      v126 = 0;
                      while ( v94 != -1 )
                      {
                        if ( v94 != -2 || v126 )
                          v93 = v126;
                        v92 = v90 & (v123 + v92);
                        v94 = *(_DWORD *)(v91 + 8LL * v92);
                        if ( v88 == v94 )
                        {
                          v93 = (_DWORD *)(v91 + 8LL * v92);
                          goto LABEL_91;
                        }
                        v126 = v93;
                        v93 = (_DWORD *)(v91 + 8LL * v92);
                        ++v123;
                      }
                      if ( v126 )
                        v93 = v126;
                    }
LABEL_91:
                    *v93 = v88;
                    v95 = v86[1];
                    v86 += 2;
                    v93[1] = v95;
                    ++*(_DWORD *)(v43 + 128);
                  }
                  while ( v84 != v86 );
LABEL_92:
                  v96 = v87;
                  v122 = v82;
                  v41 = v137;
                  v124 = v46;
                  v138 = v96;
                  sub_C7D6A0(v47, v146, 4);
                  v97 = *(_DWORD **)(v43 + 120);
                  v98 = *(_DWORD *)(v43 + 136);
                  v42 = v138;
                  v45 = v131;
                  v46 = v124;
                  v67 = *(_DWORD *)(v43 + 128) + 1;
                  v82 = v122;
                }
                else
                {
                  v118 = *(unsigned int *)(v43 + 136);
                  *(_QWORD *)(v43 + 128) = 0;
                  v98 = v118;
                  v97 = &v81[2 * v118];
                  if ( v81 == v97 )
                  {
                    v67 = 1;
                  }
                  else
                  {
                    do
                    {
                      if ( v81 )
                        *v81 = -1;
                      v81 += 2;
                    }
                    while ( v97 != v81 );
                    v97 = *(_DWORD **)(v43 + 120);
                    v98 = *(_DWORD *)(v43 + 136);
                    v67 = *(_DWORD *)(v43 + 128) + 1;
                  }
                }
                if ( !v98 )
                {
LABEL_177:
                  ++*(_DWORD *)(v43 + 128);
                  BUG();
                }
                v99 = v98 - 1;
                v100 = (v98 - 1) & v82;
                v69 = &v97[2 * v100];
                v101 = v100;
                v102 = *v69;
                if ( (_DWORD)v41 != *v69 )
                {
                  v103 = 1;
                  v104 = 0;
                  while ( v102 != -1 )
                  {
                    if ( !v104 && v102 == -2 )
                      v104 = v69;
                    v120 = v103++;
                    v101 = v99 & (v120 + v101);
                    v69 = &v97[2 * v101];
                    v102 = *v69;
                    if ( (_DWORD)v41 == *v69 )
                      goto LABEL_72;
                  }
                  if ( v104 )
                    v69 = v104;
                }
              }
              goto LABEL_72;
            }
LABEL_38:
            v128 = v46;
            v134 = v45;
            v141 = v42;
            v48 = ((((((((unsigned int)(2 * v44 - 1) | ((unsigned __int64)(unsigned int)(2 * v44 - 1) >> 1)) >> 2)
                     | (unsigned int)(2 * v44 - 1)
                     | ((unsigned __int64)(unsigned int)(2 * v44 - 1) >> 1)) >> 4)
                   | (((unsigned int)(2 * v44 - 1) | ((unsigned __int64)(unsigned int)(2 * v44 - 1) >> 1)) >> 2)
                   | (unsigned int)(2 * v44 - 1)
                   | ((unsigned __int64)(unsigned int)(2 * v44 - 1) >> 1)) >> 8)
                 | (((((unsigned int)(2 * v44 - 1) | ((unsigned __int64)(unsigned int)(2 * v44 - 1) >> 1)) >> 2)
                   | (unsigned int)(2 * v44 - 1)
                   | ((unsigned __int64)(unsigned int)(2 * v44 - 1) >> 1)) >> 4)
                 | (((unsigned int)(2 * v44 - 1) | ((unsigned __int64)(unsigned int)(2 * v44 - 1) >> 1)) >> 2)
                 | (unsigned int)(2 * v44 - 1)
                 | ((unsigned __int64)(unsigned int)(2 * v44 - 1) >> 1)) >> 16;
            v49 = (v48
                 | (((((((unsigned int)(2 * v44 - 1) | ((unsigned __int64)(unsigned int)(2 * v44 - 1) >> 1)) >> 2)
                     | (unsigned int)(2 * v44 - 1)
                     | ((unsigned __int64)(unsigned int)(2 * v44 - 1) >> 1)) >> 4)
                   | (((unsigned int)(2 * v44 - 1) | ((unsigned __int64)(unsigned int)(2 * v44 - 1) >> 1)) >> 2)
                   | (unsigned int)(2 * v44 - 1)
                   | ((unsigned __int64)(unsigned int)(2 * v44 - 1) >> 1)) >> 8)
                 | (((((unsigned int)(2 * v44 - 1) | ((unsigned __int64)(unsigned int)(2 * v44 - 1) >> 1)) >> 2)
                   | (unsigned int)(2 * v44 - 1)
                   | ((unsigned __int64)(unsigned int)(2 * v44 - 1) >> 1)) >> 4)
                 | (((unsigned int)(2 * v44 - 1) | ((unsigned __int64)(unsigned int)(2 * v44 - 1) >> 1)) >> 2)
                 | (unsigned int)(2 * v44 - 1)
                 | ((unsigned __int64)(unsigned int)(2 * v44 - 1) >> 1))
                + 1;
            if ( (unsigned int)v49 < 0x40 )
              LODWORD(v49) = 64;
            *(_DWORD *)(v43 + 136) = v49;
            v50 = (_DWORD *)sub_C7D670(8LL * (unsigned int)v49, 4);
            v42 = v141;
            v45 = v134;
            *(_QWORD *)(v43 + 120) = v50;
            v46 = v128;
            if ( v47 )
            {
              v51 = *(unsigned int *)(v43 + 136);
              v52 = 8 * v44;
              *(_QWORD *)(v43 + 128) = 0;
              v53 = &v47[2 * v44];
              for ( j = &v50[2 * v51]; j != v50; v50 += 2 )
              {
                if ( v50 )
                  *v50 = -1;
              }
              v55 = v47;
              if ( v47 != v53 )
              {
                v142 = v41;
                v56 = v42;
                do
                {
                  while ( 1 )
                  {
                    v57 = *v55;
                    if ( (unsigned int)*v55 <= 0xFFFFFFFD )
                      break;
                    v55 += 2;
                    if ( v53 == v55 )
                      goto LABEL_52;
                  }
                  v58 = *(_DWORD *)(v43 + 136);
                  if ( !v58 )
                  {
                    MEMORY[0] = 0;
                    BUG();
                  }
                  v59 = v58 - 1;
                  v60 = *(_QWORD *)(v43 + 120);
                  v61 = v59 & (37 * v57);
                  v62 = (_DWORD *)(v60 + 8LL * v61);
                  v63 = *v62;
                  if ( *v62 != v57 )
                  {
                    v125 = 1;
                    v132 = 0;
                    while ( v63 != -1 )
                    {
                      if ( !v132 )
                      {
                        if ( v63 != -2 )
                          v62 = 0;
                        v132 = v62;
                      }
                      v61 = v59 & (v125 + v61);
                      v62 = (_DWORD *)(v60 + 8LL * v61);
                      v63 = *v62;
                      if ( v57 == *v62 )
                        goto LABEL_51;
                      ++v125;
                    }
                    if ( v132 )
                      v62 = v132;
                  }
LABEL_51:
                  *v62 = v57;
                  v64 = v55[1];
                  v55 += 2;
                  v62[1] = v64;
                  ++*(_DWORD *)(v43 + 128);
                }
                while ( v53 != v55 );
LABEL_52:
                v42 = v56;
                v45 = v134;
                v41 = v142;
              }
              v143 = v42;
              v129 = v46;
              v135 = v45;
              sub_C7D6A0(v47, v52, 4);
              v65 = *(_DWORD **)(v43 + 120);
              v66 = *(_DWORD *)(v43 + 136);
              v42 = v143;
              v45 = v135;
              v46 = v129;
              v67 = *(_DWORD *)(v43 + 128) + 1;
            }
            else
            {
              v117 = *(unsigned int *)(v43 + 136);
              *(_QWORD *)(v43 + 128) = 0;
              v66 = v117;
              v65 = &v50[2 * v117];
              if ( v50 == v65 )
              {
                v67 = 1;
              }
              else
              {
                do
                {
                  if ( v50 )
                    *v50 = -1;
                  v50 += 2;
                }
                while ( v65 != v50 );
                v65 = *(_DWORD **)(v43 + 120);
                v66 = *(_DWORD *)(v43 + 136);
                v67 = *(_DWORD *)(v43 + 128) + 1;
              }
            }
            if ( !v66 )
              goto LABEL_177;
            v68 = v66 - 1;
            v69 = &v65[2 * ((v66 - 1) & (unsigned int)(37 * v41))];
            v70 = (v66 - 1) & (37 * v41);
            v71 = *v69;
            if ( (_DWORD)v41 != *v69 )
            {
              v72 = 1;
              v73 = 0;
              while ( v71 != -1 )
              {
                if ( !v73 && v71 == -2 )
                  v73 = v69;
                v70 = v68 & (v72 + v70);
                v69 = &v65[2 * v70];
                v71 = *v69;
                if ( (_DWORD)v41 == *v69 )
                  goto LABEL_72;
                ++v72;
              }
              if ( v73 )
                v69 = v73;
            }
LABEL_72:
            *(_DWORD *)(v43 + 128) = v67;
            if ( *v69 != -1 )
              --*(_DWORD *)(v43 + 132);
            *v69 = v41;
            v69[1] = (v45 - v46) >> 3;
            v75 = *v42;
            v76 = *(unsigned __int64 **)(*(_QWORD *)(v42[1] + 528LL) + 8 * v41);
            v160[0] = v76;
            v77 = *(_BYTE **)(v75 + 96);
            if ( v77 == *(_BYTE **)(v75 + 104) )
            {
              sub_A235E0(v75 + 88, v77, v160);
            }
            else
            {
              if ( v77 )
              {
                *(_QWORD *)v77 = v76;
                v77 = *(_BYTE **)(v75 + 96);
              }
              *(_QWORD *)(v75 + 96) = v77 + 8;
            }
          }
LABEL_35:
          if ( v156 == ++v36 )
            goto LABEL_63;
        }
        ++*(_QWORD *)(v43 + 112);
        goto LABEL_38;
      }
    }
  }
  return result;
}
