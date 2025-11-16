// Function: sub_1DADA60
// Address: 0x1dada60
//
__int64 __fastcall sub_1DADA60(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, signed __int64 a6)
{
  __int64 result; // rax
  int v7; // edx
  __int64 v8; // rcx
  int v9; // edx
  __int64 v11; // rax
  __int64 v12; // rsi
  signed __int64 v13; // r8
  __int64 v14; // rdx
  __int64 v15; // r15
  __int64 v16; // rax
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // rdx
  __int64 v21; // rax
  int v22; // r15d
  unsigned __int64 v23; // rdx
  unsigned int v24; // r13d
  unsigned int v25; // eax
  __int64 v26; // r8
  __int64 v27; // rcx
  __int64 v28; // rbx
  __int64 v29; // r12
  __int64 v30; // rdx
  unsigned int v31; // ecx
  __int64 *v32; // rdi
  __int64 v33; // rax
  __int64 v34; // r13
  _BYTE *v35; // rdx
  __int64 v36; // rdi
  _BYTE *v37; // r13
  __int64 *v38; // rcx
  __int64 v39; // rdx
  __int64 *v40; // r11
  __int64 v41; // rax
  __int64 *v42; // r12
  __int64 v43; // r14
  int v44; // ebx
  __int64 v45; // rax
  __int64 v46; // rdi
  __int64 v47; // r10
  unsigned int v48; // eax
  _DWORD *v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // rdx
  __int64 *v54; // rax
  unsigned int v55; // edx
  unsigned int i; // ecx
  __int64 v57; // r9
  __int64 v58; // rdx
  unsigned int v59; // r10d
  __int64 v60; // r15
  int v61; // edx
  int v62; // edx
  __int64 v63; // rcx
  unsigned int v64; // esi
  int v65; // r8d
  __int64 v66; // rdx
  int v67; // ebx
  void *v68; // rdi
  __int64 v69; // rax
  __int64 v70; // rcx
  __int64 v71; // rdx
  __int64 v72; // rax
  unsigned __int64 v73; // rdi
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rax
  __int64 v77; // rax
  _DWORD *v78; // rcx
  __int64 v79; // rdx
  unsigned int v80; // eax
  __int64 v81; // rax
  int v82; // edx
  unsigned int v83; // eax
  __int64 v84; // rcx
  __int64 v85; // rbx
  __int64 v86; // rdi
  _QWORD *v87; // rdx
  __int64 v88; // rdx
  __int64 v89; // r8
  __int64 v90; // rdi
  unsigned __int64 v91; // rcx
  __int64 v92; // rsi
  unsigned int v93; // eax
  int v94; // r10d
  __int64 v95; // rsi
  __int64 v96; // rdi
  __int64 v97; // rcx
  __int64 *v98; // rsi
  __int64 v99; // r14
  __int64 v100; // r8
  unsigned int v101; // edi
  unsigned __int64 v102; // r9
  __int64 v103; // rcx
  __int64 v104; // rdx
  __int64 v105; // rbx
  __int64 v106; // rdi
  __int64 *v107; // rsi
  __int64 v108; // r10
  __int64 v109; // rax
  int v110; // ecx
  __int64 v111; // r11
  __int64 v112; // rax
  __int64 v113; // r8
  __int64 v114; // rax
  bool v115; // cf
  __int64 v116; // rcx
  int v117; // r8d
  __int64 v118; // rcx
  int v119; // eax
  __int64 v120; // rsi
  __int64 v121; // rax
  int v122; // ecx
  __int64 v123; // rsi
  __int64 v124; // rsi
  __int64 v125; // rsi
  __int32 v126; // r11d
  __int64 v127; // r8
  __int64 v128; // rdi
  __int64 v129; // rax
  __m128i v130; // xmm0
  __m128i v131; // xmm1
  __m128i *v132; // rax
  __int64 v133; // rsi
  __int64 v134; // rax
  unsigned int v135; // eax
  _BYTE *v136; // rdi
  char v137; // al
  unsigned int v138; // edi
  unsigned int v139; // edi
  __int64 v140; // [rsp+0h] [rbp-160h]
  unsigned int v141; // [rsp+8h] [rbp-158h]
  __int64 v142; // [rsp+8h] [rbp-158h]
  __int64 v143; // [rsp+10h] [rbp-150h]
  int v144; // [rsp+10h] [rbp-150h]
  unsigned __int64 v145; // [rsp+18h] [rbp-148h]
  unsigned __int64 v146; // [rsp+18h] [rbp-148h]
  int v147; // [rsp+20h] [rbp-140h]
  int v148; // [rsp+24h] [rbp-13Ch]
  __int64 v149; // [rsp+28h] [rbp-138h]
  __int64 v150; // [rsp+30h] [rbp-130h]
  __int64 v151; // [rsp+38h] [rbp-128h]
  const void *v152; // [rsp+40h] [rbp-120h]
  unsigned int v153; // [rsp+50h] [rbp-110h]
  char v154; // [rsp+58h] [rbp-108h]
  unsigned int v155; // [rsp+58h] [rbp-108h]
  char v156; // [rsp+5Eh] [rbp-102h]
  char v157; // [rsp+5Fh] [rbp-101h]
  __int64 v158; // [rsp+68h] [rbp-F8h]
  unsigned int v159; // [rsp+68h] [rbp-F8h]
  unsigned int v163; // [rsp+8Ch] [rbp-D4h]
  __int64 v164; // [rsp+90h] [rbp-D0h]
  __int64 v165; // [rsp+98h] [rbp-C8h]
  __m128i v166; // [rsp+A0h] [rbp-C0h] BYREF
  __m128i v167; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v168; // [rsp+C0h] [rbp-A0h]
  __int64 v169; // [rsp+D0h] [rbp-90h] BYREF
  _BYTE *v170; // [rsp+D8h] [rbp-88h] BYREF
  __int64 v171; // [rsp+E0h] [rbp-80h]
  _BYTE v172[120]; // [rsp+E8h] [rbp-78h] BYREF

  result = *(_QWORD *)(a1 + 232);
  v149 = result;
  if ( result )
  {
    v7 = *(_DWORD *)(result + 256);
    if ( v7 )
    {
      v8 = *(_QWORD *)(result + 240);
      v9 = v7 - 1;
      v147 = 37 * a2;
      v11 = v9 & (unsigned int)(37 * a2);
      v12 = v11;
      result = v8 + 16 * v11;
      LODWORD(v13) = *(_DWORD *)result;
      if ( a2 == *(_DWORD *)result )
      {
LABEL_4:
        v14 = *(_QWORD *)(result + 8);
        if ( v14 )
        {
          v15 = *(_QWORD *)(v14 + 24);
          do
          {
            v16 = v15;
            v15 = *(_QWORD *)(v15 + 24);
          }
          while ( v16 != v15 );
          *(_QWORD *)(v14 + 24) = v15;
          v156 = 0;
          v164 = v15;
          do
          {
            v17 = *(_DWORD *)(v164 + 48);
            if ( v17 )
            {
              v18 = (unsigned int)(v17 - 1);
              v157 = 0;
              v163 = v18;
              v19 = *(_QWORD *)(v149 + 128);
              v165 = 40 * v18;
              while ( 1 )
              {
                v20 = *(_QWORD *)(v164 + 40);
                if ( !*(_BYTE *)(v20 + v165) && a2 == *(_DWORD *)(v20 + v165 + 8) )
                {
                  v170 = v172;
                  v171 = 0x400000000LL;
                  v169 = v164 + 216;
                  if ( a4 )
                  {
                    v154 = 0;
                    v21 = 0;
                    v22 = 0;
                    v152 = (const void *)(v19 + 416);
                    while ( 1 )
                    {
                      v23 = *(unsigned int *)(v19 + 408);
                      v24 = *(_DWORD *)(a3 + 4 * v21);
                      v25 = v24 & 0x7FFFFFFF;
                      v26 = 8LL * (v24 & 0x7FFFFFFF);
                      if ( (v24 & 0x7FFFFFFF) >= (unsigned int)v23 )
                        break;
                      v27 = *(_QWORD *)(v19 + 400);
                      v28 = *(_QWORD *)(v27 + 8LL * v25);
                      if ( !v28 )
                        break;
LABEL_18:
                      v13 = *(unsigned int *)(v28 + 8);
                      if ( (_DWORD)v13 )
                      {
                        v29 = v169;
                        v30 = *(unsigned int *)(v169 + 80);
                        v12 = **(_QWORD **)v28;
                        if ( (_DWORD)v30 )
                        {
                          sub_1DAAC30((__int64)&v169, v12, v30, v27, v13, a6);
                          v36 = (unsigned int)v171;
                        }
                        else
                        {
                          v31 = *(_DWORD *)(v169 + 84);
                          if ( v31 )
                          {
                            v32 = (__int64 *)(v169 + 8);
                            v12 = *(_DWORD *)((v12 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v12 >> 1) & 3;
                            do
                            {
                              v13 = *v32 & 0xFFFFFFFFFFFFFFF8LL;
                              if ( (*(_DWORD *)(v13 + 24) | (unsigned int)(*v32 >> 1) & 3) > (unsigned int)v12 )
                                break;
                              v30 = (unsigned int)(v30 + 1);
                              v32 += 2;
                            }
                            while ( v31 != (_DWORD)v30 );
                          }
                          v33 = 0;
                          LODWORD(v171) = 0;
                          v34 = (v30 << 32) | v31;
                          if ( !HIDWORD(v171) )
                          {
                            v12 = (__int64)v172;
                            sub_16CD150((__int64)&v170, v172, 0, 16, v13, a6);
                            v33 = 16LL * (unsigned int)v171;
                          }
                          v35 = v170;
                          *(_QWORD *)&v170[v33] = v29;
                          *(_QWORD *)&v35[v33 + 8] = v34;
                          v36 = (unsigned int)(v171 + 1);
                          LODWORD(v171) = v171 + 1;
                        }
                        if ( (_DWORD)v36 )
                        {
                          v37 = v170;
                          if ( *((_DWORD *)v170 + 3) < *((_DWORD *)v170 + 2) )
                          {
                            v12 = *(unsigned int *)(v28 + 8);
                            v38 = *(__int64 **)v28;
                            v39 = *(_QWORD *)(*(_QWORD *)v28 + 24 * v12 - 16);
                            a6 = *(_QWORD *)(*(_QWORD *)&v170[16 * v36 - 16]
                                           + 16LL * *(unsigned int *)&v170[16 * v36 - 4]);
                            v158 = *(_QWORD *)v28 + 24 * v12;
                            LODWORD(v13) = *(_DWORD *)((a6 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a6 >> 1) & 3;
                            LODWORD(a6) = *(_DWORD *)((v39 & 0xFFFFFFFFFFFFFFF8LL) + 24);
                            if ( (unsigned int)v13 < (unsigned __int64)((unsigned int)a6 | (v39 >> 1) & 3) )
                            {
                              v40 = *(__int64 **)v28;
                              if ( (unsigned int)v13 >= (*(_DWORD *)((v38[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                       | (unsigned int)(v38[1] >> 1) & 3) )
                              {
                                do
                                {
                                  v41 = v40[4];
                                  v40 += 3;
                                }
                                while ( (unsigned int)v13 >= (*(_DWORD *)((v41 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                            | (unsigned int)(v41 >> 1) & 3) );
                              }
                              if ( v40 != (__int64 *)v158 )
                              {
                                v148 = -1;
                                v42 = v40;
                                v151 = v19;
                                v43 = v169;
                                v150 = v28;
                                v44 = *(_DWORD *)(v169 + 80);
                                while ( 1 )
                                {
                                  v45 = (__int64)&v37[16 * v36 - 16];
                                  v46 = *(unsigned int *)(v45 + 12);
                                  v13 = *(_QWORD *)v45;
                                  v47 = 16 * v46;
                                  v48 = *(_DWORD *)((*(_QWORD *)(*(_QWORD *)v45 + v47) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                      | (*(__int64 *)(*(_QWORD *)v45 + v47) >> 1) & 3;
                                  if ( v48 < (unsigned __int64)((unsigned int)a6 | (v39 >> 1) & 3) )
                                  {
                                    while ( v48 >= (*(_DWORD *)((v42[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                  | (unsigned int)(v42[1] >> 1) & 3) )
                                      v42 += 3;
                                    if ( v42 == (__int64 *)v158 )
                                      goto LABEL_75;
                                  }
                                  else
                                  {
                                    v42 = &v38[3 * v12];
                                    if ( v42 == (__int64 *)v158 )
                                      goto LABEL_75;
                                  }
                                  if ( v44 )
                                  {
                                    v49 = (_DWORD *)(v13 + 4 * (v46 + 36));
                                    if ( (*v49 & 0x7FFFFFFF) == 0x7FFFFFFF )
                                    {
                                      v50 = (unsigned int)v171;
                                      v51 = (__int64)&v37[16 * (unsigned int)v171 - 16];
                                      v52 = *(unsigned int *)(v51 + 12);
LABEL_41:
                                      v53 = *(_QWORD *)(*(_QWORD *)v51 + 16 * v52 + 8);
                                      goto LABEL_42;
                                    }
                                  }
                                  else
                                  {
                                    v49 = (_DWORD *)(v13 + 4 * (v46 + 16));
                                    if ( (*v49 & 0x7FFFFFFF) == 0x7FFFFFFF )
                                    {
                                      v50 = (unsigned int)v171;
                                      v51 = (__int64)&v37[16 * (unsigned int)v171 - 16];
                                      v52 = *(unsigned int *)(v51 + 12);
                                      goto LABEL_108;
                                    }
                                  }
                                  if ( v163 != (*v49 & 0x7FFFFFFF)
                                    || (v88 = *v42,
                                        v89 = *(_QWORD *)(v13 + v47 + 8),
                                        v90 = v89 >> 1,
                                        v91 = *v42 & 0xFFFFFFFFFFFFFFF8LL,
                                        v13 = v89 & 0xFFFFFFFFFFFFFFF8LL,
                                        v92 = (*v42 >> 1) & 3,
                                        v93 = *(_DWORD *)(v91 + 24) | v92,
                                        v93 >= (*(_DWORD *)(v13 + 24) | (unsigned int)(v90 & 3))) )
                                  {
                                    v50 = (unsigned int)v171;
                                    v51 = (__int64)&v37[16 * (unsigned int)v171 - 16];
                                    v52 = *(unsigned int *)(v51 + 12);
                                    goto LABEL_40;
                                  }
                                  if ( v148 != -1 )
                                  {
                                    v94 = v148 & 0x7FFFFFFF;
                                    goto LABEL_114;
                                  }
                                  v166.m128i_i64[0] = 0;
                                  v94 = 0x7FFFFFFF;
                                  v167 = 0u;
                                  v126 = *(_DWORD *)(v150 + 112);
                                  v127 = *(_QWORD *)(v164 + 40);
                                  v168 = 0;
                                  v166.m128i_i32[2] = v126;
                                  v155 = *(_DWORD *)(v127 + v165);
                                  v166.m128i_i64[0] = ((v155 >> 8) & 0xFFF) << 8;
                                  if ( !v126 )
                                    goto LABEL_162;
                                  v128 = *(unsigned int *)(v164 + 48);
                                  if ( (_DWORD)v128 )
                                    break;
LABEL_163:
                                  if ( (unsigned int)v128 >= *(_DWORD *)(v164 + 52) )
                                  {
                                    sub_16CD150(v164 + 40, (const void *)(v164 + 56), 0, 40, v127, a6);
                                    v127 = *(_QWORD *)(v164 + 40);
                                    v128 = *(unsigned int *)(v164 + 48);
                                  }
                                  v130 = _mm_load_si128(&v166);
                                  v131 = _mm_load_si128(&v167);
                                  v132 = (__m128i *)(v127 + 40 * v128);
                                  v132[2].m128i_i64[0] = v168;
                                  *v132 = v130;
                                  v132[1] = v131;
                                  v133 = *(_QWORD *)(v164 + 40);
                                  v134 = (unsigned int)(*(_DWORD *)(v164 + 48) + 1);
                                  *(_DWORD *)(v164 + 48) = v134;
                                  *(_QWORD *)(v133 + 40 * v134 - 24) = 0;
                                  v135 = *(_DWORD *)(v164 + 48);
                                  v136 = (_BYTE *)(*(_QWORD *)(v164 + 40) + 40LL * v135 - 40);
                                  if ( !*v136 )
                                  {
                                    v137 = v136[3];
                                    if ( (v137 & 0x10) != 0 )
                                    {
                                      v136[3] = v137 & 0xBF;
                                      v136 = (_BYTE *)(*(_QWORD *)(v164 + 40) + 40LL * *(unsigned int *)(v164 + 48) - 40);
                                    }
                                    sub_1E31260(v136, 0);
                                    v135 = *(_DWORD *)(v164 + 48);
                                  }
                                  v88 = *v42;
                                  v154 = 1;
                                  v148 = v135 - 1;
                                  v37 = v170;
                                  v94 = (v135 - 1) & 0x7FFFFFFF;
                                  v44 = *(_DWORD *)(v169 + 80);
                                  v93 = (*v42 >> 1) & 3 | *(_DWORD *)((*v42 & 0xFFFFFFFFFFFFFFF8LL) + 24);
LABEL_114:
                                  v95 = (__int64)&v37[16 * (unsigned int)v171 - 16];
                                  v96 = *(unsigned int *)(v95 + 12);
                                  v97 = *(_QWORD *)v95;
                                  v98 = (__int64 *)(*(_QWORD *)v95 + 16 * v96);
                                  v99 = v98[1];
                                  v100 = *v98;
                                  if ( v44 )
                                    v101 = *(_DWORD *)(v97 + 4 * v96 + 144);
                                  else
                                    v101 = *(_DWORD *)(v97 + 4 * v96 + 64);
                                  v153 = v101;
                                  v102 = v100 & 0xFFFFFFFFFFFFFFF8LL;
                                  v103 = (v100 >> 1) & 3;
                                  if ( ((unsigned int)v103 | *(_DWORD *)((v100 & 0xFFFFFFFFFFFFFFF8LL) + 24)) < v93 )
                                    *v98 = v88;
                                  v104 = v42[1];
                                  v105 = (v99 >> 1) & 3;
                                  if ( ((unsigned int)v105 | *(_DWORD *)((v99 & 0xFFFFFFFFFFFFFFF8LL) + 24)) > (*(_DWORD *)((v104 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v104 >> 1) & 3) )
                                  {
                                    *(_QWORD *)(*(_QWORD *)&v170[16 * (unsigned int)v171 - 16]
                                              + 16LL * *(unsigned int *)&v170[16 * (unsigned int)v171 - 4]
                                              + 8) = v104;
                                    v124 = (unsigned int)(v171 - 1);
                                    if ( *(_DWORD *)&v170[16 * v124 + 12] == *(_DWORD *)&v170[16 * v124 + 8] - 1 )
                                    {
                                      v140 = (v100 >> 1) & 3;
                                      v142 = v100;
                                      v144 = v94;
                                      v146 = v100 & 0xFFFFFFFFFFFFFFF8LL;
                                      sub_1DA99F0((__int64)&v169, v124, v104);
                                      v103 = v140;
                                      v100 = v142;
                                      v94 = v144;
                                      v102 = v146;
                                    }
                                  }
                                  v141 = v103;
                                  v143 = v100;
                                  v145 = v102;
                                  sub_1DAB4F0((__int64)&v169, v94 | v101 & 0x80000000, v104, v103, v100);
                                  v50 = (unsigned int)v171;
                                  v37 = v170;
                                  v106 = v169;
                                  v107 = (__int64 *)(*(_QWORD *)&v170[16 * (unsigned int)v171 - 16]
                                                   + 16LL * *(unsigned int *)&v170[16 * (unsigned int)v171 - 4]);
                                  if ( (*(_DWORD *)(v145 + 24) | v141) < (*(_DWORD *)((*v107 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                        | (unsigned int)(*v107 >> 1) & 3) )
                                  {
                                    sub_1DAD0A0((__int64)&v169, v143, *v107, v153);
                                    v121 = (__int64)&v170[16 * (unsigned int)v171 - 16];
                                    v122 = *(_DWORD *)(v121 + 12) + 1;
                                    v106 = v169;
                                    *(_DWORD *)(v121 + 12) = v122;
                                    v50 = (unsigned int)v171;
                                    v37 = v170;
                                    v51 = (__int64)&v170[16 * (unsigned int)v171 - 16];
                                    if ( v122 == *(_DWORD *)(v51 + 8) )
                                    {
                                      v123 = *(unsigned int *)(v106 + 80);
                                      if ( (_DWORD)v123 )
                                      {
                                        sub_39460A0(&v170, v123);
                                        v50 = (unsigned int)v171;
                                        v37 = v170;
                                        v106 = v169;
                                        v51 = (__int64)&v170[16 * (unsigned int)v171 - 16];
                                      }
                                    }
                                  }
                                  else
                                  {
                                    v51 = (__int64)&v170[16 * (unsigned int)v171 - 16];
                                  }
                                  v52 = *(unsigned int *)(v51 + 12);
                                  LODWORD(v13) = *(_DWORD *)(v51 + 12);
                                  LODWORD(a6) = v105 | *(_DWORD *)((v99 & 0xFFFFFFFFFFFFFFF8LL) + 24);
                                  v108 = *(_QWORD *)(*(_QWORD *)v51 + 16 * v52 + 8);
                                  if ( (unsigned int)a6 > (*(_DWORD *)((v108 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                         | (unsigned int)(v108 >> 1) & 3) )
                                  {
                                    v117 = v13 + 1;
                                    *(_DWORD *)(v51 + 12) = v117;
                                    if ( v117 == *(_DWORD *)&v170[16 * (unsigned int)v171 - 8] )
                                    {
                                      v125 = *(unsigned int *)(v106 + 80);
                                      if ( (_DWORD)v125 )
                                        sub_39460A0(&v170, v125);
                                    }
                                    sub_1DAD0A0((__int64)&v169, v42[1], v99, v153);
                                    v118 = (__int64)&v170[16 * (unsigned int)v171 - 16];
                                    v119 = *(_DWORD *)(v118 + 12);
                                    if ( v119 )
                                    {
                                      v43 = v169;
                                      if ( (_DWORD)v171 && *((_DWORD *)v170 + 3) < *((_DWORD *)v170 + 2)
                                        || (v120 = *(unsigned int *)(v169 + 80), !(_DWORD)v120) )
                                      {
                                        *(_DWORD *)(v118 + 12) = v119 - 1;
LABEL_153:
                                        v50 = (unsigned int)v171;
                                        v37 = v170;
                                        v51 = (__int64)&v170[16 * (unsigned int)v171 - 16];
                                        v52 = *(unsigned int *)(v51 + 12);
                                        goto LABEL_123;
                                      }
                                    }
                                    else
                                    {
                                      v120 = *(unsigned int *)(v169 + 80);
                                    }
                                    sub_3945E40(&v170, v120);
                                    v43 = v169;
                                    goto LABEL_153;
                                  }
                                  v43 = v106;
LABEL_123:
                                  v44 = *(_DWORD *)(v43 + 80);
LABEL_40:
                                  if ( v44 )
                                    goto LABEL_41;
LABEL_108:
                                  v44 = 0;
                                  v53 = *(_QWORD *)(*(_QWORD *)v51 + 16 * v52 + 8);
LABEL_42:
                                  v12 = v53 & 0xFFFFFFFFFFFFFFF8LL;
                                  if ( (*(_DWORD *)((v42[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                      | (unsigned int)(v42[1] >> 1) & 3) >= (*(_DWORD *)((v53 & 0xFFFFFFFFFFFFFFF8LL)
                                                                                       + 24)
                                                                           | (unsigned int)(v53 >> 1) & 3) )
                                  {
                                    v109 = (__int64)&v37[16 * v50 - 16];
                                    v110 = *(_DWORD *)(v109 + 12) + 1;
                                    *(_DWORD *)(v109 + 12) = v110;
                                    v55 = v171;
                                    if ( v110 == *(_DWORD *)&v170[16 * (unsigned int)v171 - 8] )
                                    {
                                      v12 = *(unsigned int *)(v43 + 80);
                                      if ( (_DWORD)v12 )
                                      {
                                        sub_39460A0(&v170, v12);
                                        v55 = v171;
                                      }
                                    }
                                    if ( !v55
                                      || (v37 = v170,
                                          LODWORD(a6) = *((_DWORD *)v170 + 3),
                                          v59 = *((_DWORD *)v170 + 2),
                                          (unsigned int)a6 >= v59) )
                                    {
LABEL_75:
                                      v19 = v151;
                                      goto LABEL_76;
                                    }
                                    v36 = v55;
                                    v38 = *(__int64 **)v150;
                                    v111 = *(_QWORD *)(*(_QWORD *)&v170[16 * v55 - 16]
                                                     + 16LL * *(unsigned int *)&v170[16 * v55 - 4]);
                                    v12 = *(_DWORD *)((v111 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                        | (unsigned int)(v111 >> 1) & 3;
                                    v112 = 24LL * *(unsigned int *)(v150 + 8);
                                    v13 = *(_QWORD *)(*(_QWORD *)v150 + v112 - 16);
                                    v54 = (__int64 *)(*(_QWORD *)v150 + v112);
                                    LODWORD(v13) = *(_DWORD *)((v13 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v13 >> 1) & 3;
                                    if ( (unsigned int)v12 < (unsigned int)v13 )
                                    {
                                      v113 = v42[1];
                                      v114 = v113 >> 1;
                                      v13 = v113 & 0xFFFFFFFFFFFFFFF8LL;
                                      v115 = (unsigned int)v12 < (*(_DWORD *)(v13 + 24) | (unsigned int)(v114 & 3));
                                      v54 = v42;
                                      if ( v115 )
                                        goto LABEL_58;
                                      do
                                      {
                                        v116 = v54[4];
                                        v54 += 3;
                                      }
                                      while ( (unsigned int)v12 >= (*(_DWORD *)((v116 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                  | (unsigned int)(v116 >> 1) & 3) );
                                    }
                                  }
                                  else
                                  {
                                    v54 = v42 + 3;
                                    if ( v42 + 3 == (__int64 *)v158 )
                                      goto LABEL_75;
                                    v55 = v171;
                                    if ( !(_DWORD)v171 )
                                      goto LABEL_75;
                                    v12 = *((unsigned int *)v37 + 2);
                                    if ( *((_DWORD *)v37 + 3) < (unsigned int)v12 )
                                    {
                                      if ( v44 )
                                      {
                                        v12 = v42[3];
                                        sub_1DAAD40((__int64)&v169, v12);
                                        v55 = v171;
                                        v54 = v42 + 3;
                                      }
                                      else
                                      {
                                        v12 = *(unsigned int *)(v43 + 84);
                                        v13 = (signed __int64)&v37[16 * (unsigned int)v171 - 16];
                                        for ( i = *(_DWORD *)(v13 + 12); (_DWORD)v12 != i; ++i )
                                        {
                                          v57 = *(_QWORD *)(v43 + 16LL * i + 8);
                                          v58 = v57 >> 1;
                                          a6 = v57 & 0xFFFFFFFFFFFFFFF8LL;
                                          if ( (*(_DWORD *)(a6 + 24) | (unsigned int)(v58 & 3)) > (*(_DWORD *)((v42[3] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                 | (unsigned int)(v42[3] >> 1)
                                                                                                 & 3) )
                                            break;
                                        }
                                        *(_DWORD *)(v13 + 12) = i;
                                        v55 = v171;
                                      }
                                      if ( !v55 )
                                        goto LABEL_75;
                                      v37 = v170;
                                    }
                                    LODWORD(a6) = *((_DWORD *)v37 + 3);
                                    v59 = *((_DWORD *)v37 + 2);
                                  }
                                  if ( v54 == (__int64 *)v158 || (unsigned int)a6 >= v59 )
                                    goto LABEL_75;
                                  v36 = v55;
                                  v38 = *(__int64 **)v150;
LABEL_58:
                                  v43 = v169;
                                  v42 = v54;
                                  v12 = *(unsigned int *)(v150 + 8);
                                  v44 = *(_DWORD *)(v169 + 80);
                                  v39 = v38[3 * v12 - 2];
                                  LODWORD(a6) = *(_DWORD *)((v39 & 0xFFFFFFFFFFFFFFF8LL) + 24);
                                }
                                v129 = v127;
                                LODWORD(a6) = 0;
                                while ( *(_BYTE *)v129
                                     || v126 != *(_DWORD *)(v129 + 8)
                                     || ((v155 >> 8) & 0xFFF) != ((*(_DWORD *)v129 >> 8) & 0xFFF) )
                                {
                                  LODWORD(a6) = a6 + 1;
                                  v129 += 40;
                                  if ( (_DWORD)v128 == (_DWORD)a6 )
                                    goto LABEL_163;
                                }
                                v148 = a6;
                                v94 = a6 & 0x7FFFFFFF;
LABEL_162:
                                v154 = 1;
                                v93 = v92 | *(_DWORD *)(v91 + 24);
                                goto LABEL_114;
                              }
                            }
                          }
                        }
                      }
LABEL_76:
                      v21 = (unsigned int)++v22;
                      if ( v22 == a4 )
                      {
                        v157 |= v154;
                        v20 = *(_QWORD *)(v164 + 40);
                        goto LABEL_78;
                      }
                    }
                    v83 = v25 + 1;
                    if ( (unsigned int)v23 < v83 )
                    {
                      v85 = v83;
                      if ( v83 >= v23 )
                      {
                        if ( v83 > v23 )
                        {
                          if ( v83 > (unsigned __int64)*(unsigned int *)(v19 + 412) )
                          {
                            v159 = v83;
                            sub_16CD150(v19 + 400, v152, v83, 8, 8 * v24, a6);
                            v23 = *(unsigned int *)(v19 + 408);
                            v26 = 8LL * (v24 & 0x7FFFFFFF);
                            v83 = v159;
                          }
                          v84 = *(_QWORD *)(v19 + 400);
                          v86 = *(_QWORD *)(v19 + 416);
                          v12 = v84 + 8 * v85;
                          v87 = (_QWORD *)(v84 + 8 * v23);
                          if ( (_QWORD *)v12 != v87 )
                          {
                            do
                              *v87++ = v86;
                            while ( (_QWORD *)v12 != v87 );
                            v84 = *(_QWORD *)(v19 + 400);
                          }
                          *(_DWORD *)(v19 + 408) = v83;
                          goto LABEL_94;
                        }
                      }
                      else
                      {
                        *(_DWORD *)(v19 + 408) = v83;
                      }
                    }
                    v84 = *(_QWORD *)(v19 + 400);
LABEL_94:
                    *(_QWORD *)(v84 + v26) = sub_1DBA290(v24, v12);
                    v28 = *(_QWORD *)(*(_QWORD *)(v19 + 400) + 8LL * (v24 & 0x7FFFFFFF));
                    v12 = v28;
                    sub_1DBB110(v19, v28);
                    goto LABEL_18;
                  }
LABEL_78:
                  v68 = (void *)(v20 + v165);
                  v12 = v20 + v165 + 40;
                  v69 = *(unsigned int *)(v164 + 48);
                  v70 = 5 * v69;
                  v71 = v20 + 40 * v69;
                  if ( v71 != v12 )
                  {
                    memmove(v68, (const void *)v12, v71 - v12);
                    LODWORD(v69) = *(_DWORD *)(v164 + 48);
                  }
                  *(_DWORD *)(v164 + 48) = v69 - 1;
                  sub_1DA9720(&v169, v12, v71, v70, v13, a6);
                  v72 = (unsigned int)v171;
                  v73 = (unsigned __int64)v170;
                  if ( (_DWORD)v171 )
                  {
                    while ( 1 )
                    {
                      v12 = *(unsigned int *)(v73 + 8);
                      if ( *(_DWORD *)(v73 + 12) >= (unsigned int)v12 )
                        goto LABEL_90;
                      v74 = v73 + 16 * v72 - 16;
                      v75 = *(_QWORD *)v74;
                      v76 = *(unsigned int *)(v74 + 12);
                      if ( *(_DWORD *)(v169 + 80) )
                        v77 = v76 + 36;
                      else
                        v77 = v76 + 16;
                      v78 = (_DWORD *)(v75 + 4 * v77);
                      v79 = (unsigned int)*v78;
                      if ( (v79 & 0x7FFFFFFF) == 0x7FFFFFFF )
                        goto LABEL_88;
                      v80 = *v78 & 0x7FFFFFFF;
                      if ( v163 != v80 )
                        break;
                      sub_1DAB460((__int64)&v169, v12, v79, (__int64)v78, v13);
                      v72 = (unsigned int)v171;
                      v73 = (unsigned __int64)v170;
LABEL_89:
                      if ( !(_DWORD)v72 )
                        goto LABEL_90;
                    }
                    if ( v163 < v80 )
                    {
                      *v78 = v79 & 0x80000000 | (v80 - 1);
                      v73 = (unsigned __int64)v170;
                    }
LABEL_88:
                    v81 = v73 + 16LL * (unsigned int)v171 - 16;
                    v82 = *(_DWORD *)(v81 + 12) + 1;
                    *(_DWORD *)(v81 + 12) = v82;
                    v72 = (unsigned int)v171;
                    v73 = (unsigned __int64)v170;
                    if ( v82 == *(_DWORD *)&v170[16 * (unsigned int)v171 - 8] )
                    {
                      v12 = *(unsigned int *)(v169 + 80);
                      if ( (_DWORD)v12 )
                      {
                        sub_39460A0(&v170, v12);
                        v72 = (unsigned int)v171;
                        v73 = (unsigned __int64)v170;
                      }
                    }
                    goto LABEL_89;
                  }
LABEL_90:
                  if ( (_BYTE *)v73 != v172 )
                    _libc_free(v73);
                }
                v165 -= 40;
                if ( !v163 )
                  break;
                --v163;
              }
              v156 |= v157;
            }
            result = *(_QWORD *)(v164 + 32);
            v164 = result;
          }
          while ( result );
          v60 = 0;
          if ( v156 )
          {
            result = v149;
            v61 = *(_DWORD *)(v149 + 256);
            if ( v61 )
            {
              v62 = v61 - 1;
              v63 = *(_QWORD *)(v149 + 240);
              v64 = v62 & v147;
              result = v63 + 16LL * (v62 & (unsigned int)v147);
              v65 = *(_DWORD *)result;
              if ( a2 == *(_DWORD *)result )
              {
LABEL_64:
                v66 = *(_QWORD *)(result + 8);
                if ( v66 )
                {
                  v60 = *(_QWORD *)(v66 + 24);
                  do
                  {
                    result = v60;
                    v60 = *(_QWORD *)(v60 + 24);
                  }
                  while ( result != v60 );
                  *(_QWORD *)(v66 + 24) = v60;
                }
                else
                {
                  v60 = 0;
                }
              }
              else
              {
                result = 1;
                while ( v65 != -1 )
                {
                  v139 = result + 1;
                  v64 = v62 & (result + v64);
                  result = v63 + 16LL * v64;
                  v65 = *(_DWORD *)result;
                  if ( a2 == *(_DWORD *)result )
                    goto LABEL_64;
                  result = v139;
                }
              }
            }
            if ( a4 )
            {
              result = 0;
              v67 = 0;
              do
              {
                sub_1DA9020(v149, *(_DWORD *)(a3 + 4 * result), v60);
                result = (unsigned int)++v67;
              }
              while ( v67 != a4 );
            }
          }
        }
      }
      else
      {
        result = 1;
        while ( (_DWORD)v13 != -1 )
        {
          v138 = result + 1;
          v12 = v9 & (unsigned int)(result + v12);
          result = v8 + 16LL * (unsigned int)v12;
          LODWORD(v13) = *(_DWORD *)result;
          if ( a2 == *(_DWORD *)result )
            goto LABEL_4;
          result = v138;
        }
      }
    }
  }
  return result;
}
