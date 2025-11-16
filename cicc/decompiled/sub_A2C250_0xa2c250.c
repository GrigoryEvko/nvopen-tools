// Function: sub_A2C250
// Address: 0xa2c250
//
unsigned __int8 **__fastcall sub_A2C250(
        _BYTE **a1,
        __int64 a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        char a6,
        __int64 a7,
        __int64 a8,
        char a9,
        __int64 a10,
        __int64 *a11)
{
  _BYTE **v11; // r11
  __int64 *v13; // rax
  __int64 v14; // rdx
  _QWORD *v15; // r12
  __int64 v16; // r14
  unsigned int v17; // eax
  __int64 v18; // r13
  __int64 v19; // rcx
  _DWORD *v20; // r15
  _DWORD *v21; // r13
  bool v22; // zf
  __int64 v23; // rax
  unsigned int v24; // r13d
  unsigned __int8 **result; // rax
  unsigned __int8 **v26; // rdx
  __int64 v27; // rbx
  unsigned __int64 v28; // r12
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned __int8 *v31; // r12
  unsigned __int8 *v32; // r15
  unsigned __int8 *v33; // r11
  __int64 v34; // r9
  unsigned __int64 v35; // r14
  unsigned int v36; // r8d
  __int64 *v37; // rdx
  __int64 v38; // rdi
  __int64 v39; // rbx
  unsigned __int64 v40; // rdx
  __int64 v41; // rbx
  __int64 v42; // rbx
  int v43; // eax
  unsigned int v44; // esi
  __int64 v45; // rax
  int v46; // esi
  int v47; // esi
  __int64 v48; // r8
  unsigned int v49; // edx
  int v50; // eax
  __int64 *v51; // rcx
  __int64 v52; // rdi
  unsigned __int8 *v53; // rax
  unsigned __int8 *v54; // rcx
  unsigned __int64 v55; // rdx
  unsigned __int8 *v56; // r10
  unsigned __int8 *v57; // r15
  __int64 v58; // rax
  unsigned __int64 v59; // rdx
  __int64 v60; // rbx
  __int64 *v61; // rbx
  __int64 *v62; // r12
  __int64 v63; // r14
  __int64 v64; // rax
  __int64 v65; // r13
  unsigned __int64 v66; // rdx
  unsigned __int64 v67; // r10
  __int64 v68; // rax
  __int64 v69; // rax
  unsigned __int64 v70; // rdx
  unsigned int v71; // r12d
  unsigned int v72; // r13d
  int v73; // eax
  int v74; // edx
  int v75; // edx
  __int64 v76; // rdi
  __int64 *v77; // r8
  unsigned int v78; // r14d
  int v79; // r10d
  __int64 v80; // rsi
  __int64 v81; // r14
  unsigned __int64 v82; // r8
  unsigned int v83; // r15d
  int v84; // r13d
  unsigned int v85; // ecx
  __int64 v86; // r14
  unsigned __int64 i; // rbx
  int v88; // eax
  int v89; // eax
  _QWORD *v90; // rax
  __int64 v91; // rdx
  unsigned int v92; // r8d
  int v93; // edx
  unsigned int v94; // ecx
  int v95; // r13d
  _QWORD *v96; // rdi
  __int64 v97; // rdx
  int v98; // edx
  unsigned int v99; // r15d
  __int64 v100; // rbx
  __int64 v101; // r12
  unsigned __int64 v102; // r13
  unsigned int v103; // eax
  int v104; // r14d
  unsigned int v105; // ecx
  int v106; // eax
  int v107; // eax
  _QWORD *v108; // rax
  _BYTE *v109; // rdx
  unsigned int v110; // ecx
  int v111; // r14d
  _BYTE *v112; // rdx
  int v113; // edx
  unsigned int v114; // eax
  unsigned __int8 *v115; // r12
  unsigned __int8 *v116; // r14
  __int64 v117; // rax
  __int64 v118; // r13
  unsigned int *v119; // r13
  __int64 v120; // rax
  __int64 v121; // r12
  unsigned int *v122; // rbx
  __int64 v123; // rax
  __int64 v124; // r15
  int v125; // r14d
  __int64 *v126; // r9
  _BYTE **v127; // [rsp-10h] [rbp-120h]
  __int64 v128; // [rsp+0h] [rbp-110h]
  __int64 v130; // [rsp+10h] [rbp-100h]
  _QWORD *v131; // [rsp+10h] [rbp-100h]
  __int64 v132; // [rsp+18h] [rbp-F8h]
  unsigned __int8 **v133; // [rsp+18h] [rbp-F8h]
  unsigned __int8 *v135; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v136; // [rsp+30h] [rbp-E0h]
  __int64 v137; // [rsp+30h] [rbp-E0h]
  unsigned __int8 *v140; // [rsp+40h] [rbp-D0h]
  unsigned __int8 *v141; // [rsp+40h] [rbp-D0h]
  unsigned __int8 *v142; // [rsp+40h] [rbp-D0h]
  int v143; // [rsp+40h] [rbp-D0h]
  unsigned __int8 *v144; // [rsp+40h] [rbp-D0h]
  __int64 v145; // [rsp+40h] [rbp-D0h]
  unsigned int v146; // [rsp+40h] [rbp-D0h]
  __int64 v147; // [rsp+50h] [rbp-C0h]
  unsigned __int8 **v148; // [rsp+50h] [rbp-C0h]
  __int64 v149; // [rsp+58h] [rbp-B8h]
  __int64 v150; // [rsp+58h] [rbp-B8h]
  _QWORD *v151; // [rsp+58h] [rbp-B8h]
  _QWORD *v152; // [rsp+58h] [rbp-B8h]
  unsigned int v153; // [rsp+58h] [rbp-B8h]
  _QWORD *v154; // [rsp+58h] [rbp-B8h]
  _BYTE *v155; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v156; // [rsp+68h] [rbp-A8h]
  _BYTE v157[48]; // [rsp+70h] [rbp-A0h] BYREF
  _BYTE *v158; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v159; // [rsp+A8h] [rbp-68h]
  _BYTE v160[96]; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v161; // [rsp+128h] [rbp+18h]

  v11 = a1;
  v128 = a2;
  v155 = v157;
  v156 = 0x600000000LL;
  v13 = *(__int64 **)(a2 + 96);
  if ( v13 )
  {
    v14 = *v13;
    v132 = v13[1];
    if ( *v13 != v132 )
    {
      v147 = *v13;
      v15 = &v158;
      v16 = (__int64)a1;
      while ( 1 )
      {
        LODWORD(v156) = 0;
        if ( !*(_QWORD *)(a7 + 16) )
LABEL_163:
          sub_4263D6(a1, a2, v14);
        a1 = (_BYTE **)a7;
        a2 = v147;
        v17 = (*(__int64 (__fastcall **)(__int64, __int64))(a7 + 24))(a7, v147);
        v14 = (unsigned int)v156;
        v18 = v17;
        if ( (unsigned __int64)(unsigned int)v156 + 1 > HIDWORD(v156) )
        {
          a2 = (__int64)v157;
          a1 = &v155;
          sub_C8D5F0(&v155, v157, (unsigned int)v156 + 1LL, 8);
          v14 = (unsigned int)v156;
        }
        *(_QWORD *)&v155[8 * v14] = v18;
        v19 = (unsigned int)(v156 + 1);
        LODWORD(v156) = v156 + 1;
        if ( a6 )
        {
          v20 = *(_DWORD **)(v147 + 72);
          v21 = &v20[*(unsigned int *)(v147 + 80)];
          if ( v21 != v20 )
            goto LABEL_9;
          v24 = 26;
        }
        else
        {
          sub_A188E0((__int64)&v155, *(unsigned int *)(v147 + 80));
          a2 = *(unsigned int *)(v147 + 16);
          a1 = &v155;
          sub_A188E0((__int64)&v155, a2);
          v20 = *(_DWORD **)(v147 + 72);
          v19 = (unsigned int)v156;
          v21 = &v20[*(unsigned int *)(v147 + 80)];
          if ( v21 != v20 )
          {
            do
            {
LABEL_9:
              v22 = *(_QWORD *)(a8 + 16) == 0;
              LODWORD(v158) = *v20;
              if ( v22 )
                goto LABEL_163;
              a2 = (__int64)v15;
              a1 = (_BYTE **)a8;
              LODWORD(v23) = (*(__int64 (__fastcall **)(__int64, _QWORD *))(a8 + 24))(a8, v15);
              v14 = (unsigned int)v156;
              v23 = (unsigned int)v23;
              if ( (unsigned __int64)(unsigned int)v156 + 1 > HIDWORD(v156) )
              {
                a2 = (__int64)v157;
                a1 = &v155;
                v150 = (unsigned int)v23;
                sub_C8D5F0(&v155, v157, (unsigned int)v156 + 1LL, 8);
                v14 = (unsigned int)v156;
                v23 = v150;
              }
              ++v20;
              *(_QWORD *)&v155[8 * v14] = v23;
              v19 = (unsigned int)(v156 + 1);
              LODWORD(v156) = v156 + 1;
            }
            while ( v21 != v20 );
            v24 = 26;
            if ( a6 )
              goto LABEL_14;
          }
          v119 = *(unsigned int **)(v147 + 8);
          v120 = *(unsigned int *)(v147 + 16);
          if ( v119 == &v119[v120] )
          {
            v24 = 28;
          }
          else
          {
            v154 = v15;
            v121 = a8;
            v122 = &v119[v120];
            do
            {
              v123 = (unsigned int)v19;
              v124 = *v119;
              if ( (unsigned __int64)(unsigned int)v19 + 1 > HIDWORD(v156) )
              {
                sub_C8D5F0(&v155, v157, (unsigned int)v19 + 1LL, 8);
                v123 = (unsigned int)v156;
              }
              ++v119;
              *(_QWORD *)&v155[8 * v123] = v124;
              v19 = (unsigned int)(v156 + 1);
              LODWORD(v156) = v156 + 1;
            }
            while ( v122 != v119 );
            a8 = v121;
            v24 = 28;
            v15 = v154;
          }
        }
LABEL_14:
        if ( a3 )
        {
          a2 = a3;
          sub_A1B020(v16, a3, (__int64)v155, v19, 0, 0, v24, 1);
          a1 = v127;
        }
        else
        {
          v153 = v19;
          sub_A17B10(v16, 3u, *(_DWORD *)(v16 + 56));
          sub_A17CC0(v16, v24, 6);
          a1 = (_BYTE **)v16;
          a2 = v153;
          sub_A17CC0(v16, v153, 6);
          if ( v153 )
          {
            v161 = a8;
            v100 = 0;
            v131 = v15;
            v101 = v16;
            do
            {
              v102 = *(_QWORD *)&v155[v100];
              v103 = v102;
              if ( v102 == (unsigned int)v102 )
              {
                a2 = (unsigned int)v102;
                a1 = (_BYTE **)v101;
                sub_A17CC0(v101, v102, 6);
              }
              else
              {
                v104 = *(_DWORD *)(v101 + 52);
                v105 = *(_DWORD *)(v101 + 48);
                if ( v102 > 0x1F )
                {
                  do
                  {
                    v107 = (v102 & 0x1F | 0x20) << v105;
                    v105 += 6;
                    v104 |= v107;
                    *(_DWORD *)(v101 + 52) = v104;
                    if ( v105 > 0x1F )
                    {
                      v108 = *(_QWORD **)(v101 + 24);
                      v109 = (_BYTE *)v108[1];
                      if ( (unsigned __int64)(v109 + 4) > v108[2] )
                      {
                        a2 = (__int64)(v108 + 3);
                        a1 = *(_BYTE ***)(v101 + 24);
                        sub_C8D290(a1, v108 + 3, v109 + 4, 1);
                        v108 = a1;
                        v109 = a1[1];
                      }
                      *(_DWORD *)&v109[*v108] = v104;
                      v104 = 0;
                      v108[1] += 4LL;
                      v106 = *(_DWORD *)(v101 + 48);
                      if ( v106 )
                        v104 = (v102 & 0x1F | 0x20) >> (32 - (unsigned __int8)v106);
                      v105 = ((_BYTE)v106 + 6) & 0x1F;
                      *(_DWORD *)(v101 + 52) = v104;
                    }
                    v102 >>= 5;
                    *(_DWORD *)(v101 + 48) = v105;
                  }
                  while ( v102 > 0x1F );
                  v103 = v102;
                }
                v14 = v103 << v105;
                v110 = v105 + 6;
                v111 = v14 | v104;
                *(_DWORD *)(v101 + 52) = v111;
                if ( v110 > 0x1F )
                {
                  a1 = *(_BYTE ***)(v101 + 24);
                  v112 = a1[1];
                  if ( v112 + 4 > a1[2] )
                  {
                    a2 = (__int64)(a1 + 3);
                    v146 = v103;
                    sub_C8D290(a1, a1 + 3, v112 + 4, 1);
                    v103 = v146;
                    v112 = a1[1];
                  }
                  *(_DWORD *)&v112[(_QWORD)*a1] = v111;
                  a1[1] += 4;
                  v113 = *(_DWORD *)(v101 + 48);
                  v114 = v103 >> (32 - v113);
                  if ( !v113 )
                    v114 = 0;
                  v14 = ((_BYTE)v113 + 6) & 0x1F;
                  *(_DWORD *)(v101 + 52) = v114;
                  *(_DWORD *)(v101 + 48) = v14;
                }
                else
                {
                  *(_DWORD *)(v101 + 48) = v110;
                }
              }
              v100 += 8;
            }
            while ( 8LL * v153 != v100 );
            v16 = v101;
            a8 = v161;
            v15 = v131;
          }
        }
        v147 += 136;
        if ( v132 == v147 )
        {
          v11 = (_BYTE **)v16;
          result = *(unsigned __int8 ***)(v128 + 104);
          if ( !result )
            goto LABEL_63;
          goto LABEL_18;
        }
      }
    }
  }
  result = *(unsigned __int8 ***)(a2 + 104);
  if ( !result )
    return result;
LABEL_18:
  v26 = (unsigned __int8 **)*result;
  result = (unsigned __int8 **)result[1];
  v133 = result;
  if ( v26 == result )
    goto LABEL_63;
  v148 = v26;
  v27 = (__int64)v11;
  do
  {
    LODWORD(v156) = 0;
    v28 = 0x8E38E38E38E38E39LL * ((v148[9] - v148[8]) >> 3);
    v29 = 0;
    if ( !HIDWORD(v156) )
    {
      sub_C8D5F0(&v155, v157, 1, 8);
      v29 = 8LL * (unsigned int)v156;
    }
    *(_QWORD *)&v155[v29] = v28;
    v30 = (unsigned int)(v156 + 1);
    LODWORD(v156) = v156 + 1;
    if ( a6 )
    {
      v31 = v148[8];
      v32 = v148[9];
      if ( v31 == v32 )
      {
LABEL_39:
        if ( a9 )
        {
          v53 = v148[12];
          v54 = v148[11];
          if ( v53 != v54 )
            goto LABEL_41;
        }
        goto LABEL_78;
      }
    }
    else
    {
      sub_A188E0((__int64)&v155, (__int64)v148[1]);
      v31 = v148[8];
      v32 = v148[9];
      v30 = (unsigned int)v156;
      if ( v32 == v31 )
        goto LABEL_124;
    }
    v33 = v32;
    v149 = v27;
    while ( 2 )
    {
      v41 = *v31;
      if ( v30 + 1 > (unsigned __int64)HIDWORD(v156) )
      {
        v142 = v33;
        sub_C8D5F0(&v155, v157, v30 + 1, 8);
        v30 = (unsigned int)v156;
        v33 = v142;
      }
      *(_QWORD *)&v155[8 * v30] = v41;
      v42 = *a11;
      v43 = v156;
      ++*a11;
      v44 = *(_DWORD *)(a10 + 24);
      v45 = (unsigned int)(v43 + 1);
      LODWORD(v156) = v45;
      if ( !v44 )
      {
        ++*(_QWORD *)a10;
        goto LABEL_32;
      }
      v34 = *(_QWORD *)(a10 + 8);
      v35 = ((0xBF58476D1CE4E5B9LL * v42) >> 31) ^ (0xBF58476D1CE4E5B9LL * v42);
      v36 = v35 & (v44 - 1);
      v37 = (__int64 *)(v34 + 16LL * v36);
      v38 = *v37;
      if ( v42 == *v37 )
        goto LABEL_26;
      v143 = 1;
      v51 = 0;
      while ( 1 )
      {
        if ( v38 == -1 )
        {
          v73 = *(_DWORD *)(a10 + 16);
          if ( !v51 )
            v51 = v37;
          ++*(_QWORD *)a10;
          v50 = v73 + 1;
          if ( 4 * v50 < 3 * v44 )
          {
            if ( v44 - *(_DWORD *)(a10 + 20) - v50 > v44 >> 3 )
              goto LABEL_34;
            v144 = v33;
            sub_9E25D0(a10, v44);
            v74 = *(_DWORD *)(a10 + 24);
            if ( v74 )
            {
              v75 = v74 - 1;
              v76 = *(_QWORD *)(a10 + 8);
              v77 = 0;
              v78 = v75 & v35;
              v33 = v144;
              v79 = 1;
              v50 = *(_DWORD *)(a10 + 16) + 1;
              v51 = (__int64 *)(v76 + 16LL * v78);
              v80 = *v51;
              if ( v42 != *v51 )
              {
                while ( v80 != -1 )
                {
                  if ( v80 == -2 && !v77 )
                    v77 = v51;
                  v78 = v75 & (v79 + v78);
                  v51 = (__int64 *)(v76 + 16LL * v78);
                  v80 = *v51;
                  if ( v42 == *v51 )
                    goto LABEL_34;
                  ++v79;
                }
                if ( v77 )
                  v51 = v77;
              }
LABEL_34:
              *(_DWORD *)(a10 + 16) = v50;
              if ( *v51 != -1 )
                --*(_DWORD *)(a10 + 20);
              *v51 = v42;
              v39 = 0;
              *((_DWORD *)v51 + 2) = 0;
              v45 = (unsigned int)v156;
              v40 = (unsigned int)v156 + 1LL;
              if ( v40 <= HIDWORD(v156) )
                goto LABEL_27;
LABEL_37:
              v141 = v33;
              sub_C8D5F0(&v155, v157, v40, 8);
              v45 = (unsigned int)v156;
              v33 = v141;
              goto LABEL_27;
            }
LABEL_169:
            ++*(_DWORD *)(a10 + 16);
            BUG();
          }
LABEL_32:
          v140 = v33;
          sub_9E25D0(a10, 2 * v44);
          v46 = *(_DWORD *)(a10 + 24);
          if ( v46 )
          {
            v47 = v46 - 1;
            v48 = *(_QWORD *)(a10 + 8);
            v33 = v140;
            v49 = v47 & (((0xBF58476D1CE4E5B9LL * v42) >> 31) ^ (484763065 * v42));
            v50 = *(_DWORD *)(a10 + 16) + 1;
            v51 = (__int64 *)(v48 + 16LL * v49);
            v52 = *v51;
            if ( v42 != *v51 )
            {
              v125 = 1;
              v126 = 0;
              while ( v52 != -1 )
              {
                if ( !v126 && v52 == -2 )
                  v126 = v51;
                v49 = v47 & (v125 + v49);
                v51 = (__int64 *)(v48 + 16LL * v49);
                v52 = *v51;
                if ( v42 == *v51 )
                  goto LABEL_34;
                ++v125;
              }
              if ( v126 )
                v51 = v126;
            }
            goto LABEL_34;
          }
          goto LABEL_169;
        }
        if ( !v51 && v38 == -2 )
          v51 = v37;
        v36 = (v44 - 1) & (v143 + v36);
        v37 = (__int64 *)(v34 + 16LL * v36);
        v38 = *v37;
        if ( v42 == *v37 )
          break;
        ++v143;
      }
      v45 = (unsigned int)v45;
LABEL_26:
      v39 = *((unsigned int *)v37 + 2);
      v40 = v45 + 1;
      if ( v45 + 1 > (unsigned __int64)HIDWORD(v156) )
        goto LABEL_37;
LABEL_27:
      v31 += 72;
      *(_QWORD *)&v155[8 * v45] = v39;
      v30 = (unsigned int)(v156 + 1);
      LODWORD(v156) = v156 + 1;
      if ( v33 != v31 )
        continue;
      break;
    }
    v27 = v149;
    if ( a6 )
      goto LABEL_39;
LABEL_124:
    v115 = *v148;
    v116 = &v148[1][(_QWORD)*v148];
    if ( v116 != *v148 )
    {
      v117 = (unsigned int)v156;
      do
      {
        v118 = *v115;
        if ( v117 + 1 > (unsigned __int64)HIDWORD(v156) )
        {
          sub_C8D5F0(&v155, v157, v117 + 1, 8);
          v117 = (unsigned int)v156;
        }
        ++v115;
        *(_QWORD *)&v155[8 * v117] = v118;
        v117 = (unsigned int)(v156 + 1);
        LODWORD(v156) = v156 + 1;
      }
      while ( v116 != v115 );
    }
    if ( !a9 )
      goto LABEL_59;
    v53 = v148[12];
    v54 = v148[11];
    if ( v54 == v53 )
      goto LABEL_59;
LABEL_41:
    v158 = v160;
    v159 = 0xC00000000LL;
    v55 = 0x5555555555555556LL * ((v53 - v54) >> 3);
    if ( v55 > 0xC )
    {
      sub_C8D5F0(&v158, v160, v55, 4);
      v56 = v148[11];
      v135 = v148[12];
      if ( v135 != v56 )
        goto LABEL_43;
    }
    else
    {
      v56 = v148[11];
      v135 = v148[12];
LABEL_43:
      v130 = v27;
      v57 = v56;
      do
      {
        v58 = (unsigned int)v156;
        v59 = (unsigned int)v156 + 1LL;
        v60 = (__int64)(*((_QWORD *)v57 + 1) - *(_QWORD *)v57) >> 4;
        if ( v59 > HIDWORD(v156) )
        {
          sub_C8D5F0(&v155, v157, v59, 8);
          v58 = (unsigned int)v156;
        }
        *(_QWORD *)&v155[8 * v58] = v60;
        LODWORD(v156) = v156 + 1;
        v61 = *(__int64 **)v57;
        v62 = (__int64 *)*((_QWORD *)v57 + 1);
        if ( v62 != *(__int64 **)v57 )
        {
          do
          {
            v63 = *v61;
            v64 = (unsigned int)v159;
            v65 = v61[1];
            v66 = (unsigned int)v159 + 1LL;
            v67 = HIDWORD(*v61);
            if ( v66 > HIDWORD(v159) )
            {
              v136 = HIDWORD(*v61);
              sub_C8D5F0(&v158, v160, v66, 4);
              v64 = (unsigned int)v159;
              LODWORD(v67) = v136;
            }
            *(_DWORD *)&v158[4 * v64] = v67;
            LODWORD(v159) = v159 + 1;
            v68 = (unsigned int)v159;
            if ( (unsigned __int64)(unsigned int)v159 + 1 > HIDWORD(v159) )
            {
              sub_C8D5F0(&v158, v160, (unsigned int)v159 + 1LL, 4);
              v68 = (unsigned int)v159;
            }
            *(_DWORD *)&v158[4 * v68] = v63;
            v69 = (unsigned int)v156;
            LODWORD(v159) = v159 + 1;
            v70 = (unsigned int)v156 + 1LL;
            if ( v70 > HIDWORD(v156) )
            {
              sub_C8D5F0(&v155, v157, v70, 8);
              v69 = (unsigned int)v156;
            }
            v61 += 2;
            *(_QWORD *)&v155[8 * v69] = v65;
            LODWORD(v156) = v156 + 1;
          }
          while ( v62 != v61 );
        }
        v57 += 24;
      }
      while ( v135 != v57 );
      v27 = v130;
    }
    sub_A23520(v27, 0x1Fu, (__int64)&v158, a5);
    if ( v158 != v160 )
      _libc_free(v158, 31);
    if ( !a6 )
    {
LABEL_59:
      v71 = 29;
      goto LABEL_60;
    }
LABEL_78:
    v71 = 27;
LABEL_60:
    v72 = v156;
    if ( a4 )
    {
      a2 = a4;
      sub_A1B020(v27, a4, (__int64)v155, (unsigned int)v156, 0, 0, v71, 1);
    }
    else
    {
      sub_A17B10(v27, 3u, *(_DWORD *)(v27 + 56));
      sub_A17CC0(v27, v71, 6);
      a2 = v72;
      sub_A17CC0(v27, v72, 6);
      v137 = 8LL * v72;
      if ( v72 )
      {
        v81 = 0;
        do
        {
          v82 = *(_QWORD *)&v155[v81];
          v83 = v82;
          if ( v82 == (unsigned int)v82 )
          {
            a2 = (unsigned int)v82;
            sub_A17CC0(v27, v82, 6);
          }
          else
          {
            v84 = *(_DWORD *)(v27 + 52);
            v85 = *(_DWORD *)(v27 + 48);
            if ( v82 > 0x1F )
            {
              v145 = v81;
              v86 = v27;
              for ( i = v82; i > 0x1F; i >>= 5 )
              {
                v89 = (i & 0x1F | 0x20) << v85;
                v85 += 6;
                v84 |= v89;
                *(_DWORD *)(v86 + 52) = v84;
                if ( v85 > 0x1F )
                {
                  v90 = *(_QWORD **)(v86 + 24);
                  v91 = v90[1];
                  if ( (unsigned __int64)(v91 + 4) > v90[2] )
                  {
                    a2 = (__int64)(v90 + 3);
                    v151 = *(_QWORD **)(v86 + 24);
                    sub_C8D290(v151, v90 + 3, v91 + 4, 1);
                    v90 = v151;
                    v91 = v151[1];
                  }
                  *(_DWORD *)(*v90 + v91) = v84;
                  v84 = 0;
                  v90[1] += 4LL;
                  v88 = *(_DWORD *)(v86 + 48);
                  if ( v88 )
                    v84 = (i & 0x1F | 0x20) >> (32 - (unsigned __int8)v88);
                  v85 = ((_BYTE)v88 + 6) & 0x1F;
                  *(_DWORD *)(v86 + 52) = v84;
                }
                *(_DWORD *)(v86 + 48) = v85;
              }
              v92 = i;
              v27 = v86;
              v81 = v145;
              v83 = v92;
            }
            v93 = v83 << v85;
            v94 = v85 + 6;
            v95 = v93 | v84;
            *(_DWORD *)(v27 + 52) = v95;
            if ( v94 > 0x1F )
            {
              v96 = *(_QWORD **)(v27 + 24);
              v97 = v96[1];
              if ( (unsigned __int64)(v97 + 4) > v96[2] )
              {
                a2 = (__int64)(v96 + 3);
                v152 = *(_QWORD **)(v27 + 24);
                sub_C8D290(v96, v96 + 3, v97 + 4, 1);
                v96 = v152;
                v97 = v152[1];
              }
              *(_DWORD *)(*v96 + v97) = v95;
              v96[1] += 4LL;
              v98 = *(_DWORD *)(v27 + 48);
              v99 = v83 >> (32 - v98);
              if ( !v98 )
                v99 = 0;
              *(_DWORD *)(v27 + 52) = v99;
              *(_DWORD *)(v27 + 48) = ((_BYTE)v98 + 6) & 0x1F;
            }
            else
            {
              *(_DWORD *)(v27 + 48) = v94;
            }
          }
          v81 += 8;
        }
        while ( v81 != v137 );
      }
    }
    v148 += 14;
    result = v148;
  }
  while ( v133 != v148 );
LABEL_63:
  if ( v155 != v157 )
    return (unsigned __int8 **)_libc_free(v155, a2);
  return result;
}
