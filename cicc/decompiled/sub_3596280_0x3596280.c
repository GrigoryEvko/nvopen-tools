// Function: sub_3596280
// Address: 0x3596280
//
__int64 __fastcall sub_3596280(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 result; // rax
  size_t v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // r12
  _QWORD *v10; // rax
  _QWORD *v11; // rax
  size_t v12; // rdx
  _QWORD *v13; // rax
  size_t v14; // rdx
  _QWORD *v15; // rax
  size_t v16; // rdx
  _QWORD *v17; // rax
  size_t v18; // rdx
  _QWORD *v19; // rax
  size_t v20; // rdx
  _QWORD *v21; // rax
  size_t v22; // rdx
  _QWORD *v23; // rax
  size_t v24; // rdx
  _QWORD *v25; // rax
  size_t v26; // rdx
  _QWORD *v27; // rax
  size_t v28; // rdx
  _QWORD *v29; // rax
  size_t v30; // rdx
  _QWORD *v31; // rax
  size_t v32; // rdx
  _QWORD *v33; // rax
  size_t v34; // rdx
  _QWORD *v35; // rax
  size_t v36; // rdx
  _QWORD *v37; // rax
  size_t v38; // rdx
  _QWORD *v39; // rax
  size_t v40; // rdx
  _QWORD *v41; // rax
  size_t v42; // rdx
  _QWORD *v43; // rax
  size_t v44; // rdx
  _QWORD *v45; // rax
  size_t v46; // rdx
  _QWORD *v47; // rax
  size_t v48; // rdx
  _QWORD *v49; // rax
  char *v50; // rax
  __int64 v51; // r10
  int v52; // r14d
  int v53; // eax
  int v54; // r9d
  unsigned __int16 **v55; // r11
  __int64 v56; // r13
  unsigned __int16 *v57; // rsi
  int v58; // r10d
  int v59; // r15d
  unsigned int v61; // ebx
  unsigned __int64 v62; // rsi
  __int64 v63; // r8
  int v64; // r11d
  __int64 v65; // rbx
  __int64 v66; // r9
  int *v67; // r10
  __int64 v68; // r9
  int v69; // r11d
  _DWORD *v70; // rax
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // rdx
  __int64 v74; // rsi
  unsigned int v75; // r13d
  __int64 v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // r9
  __int64 v80; // r12
  __int64 v81; // rax
  int v82; // edx
  size_t v83; // r15
  int v84; // ebx
  unsigned __int64 v85; // rdx
  __int64 v86; // r15
  __int64 v87; // rbx
  __int64 v88; // r12
  unsigned int v89; // ecx
  _DWORD *v90; // rax
  __int64 v91; // rdx
  __int64 v92; // rdx
  _DWORD *v93; // rax
  unsigned __int64 v94; // rdi
  unsigned int v95; // r11d
  __int64 *v96; // r9
  __int64 *v97; // rax
  __int64 v98; // rax
  __int64 v99; // rdi
  float *v100; // rax
  float *v101; // rdx
  float v102; // xmm0_4
  __int64 i; // rcx
  __int64 v104; // rax
  float *v105; // rbx
  unsigned __int64 v106; // rdx
  float *v107; // rcx
  __int64 v108; // rax
  float v109; // xmm0_4
  float v110; // xmm0_4
  __int64 v111; // rdx
  unsigned __int64 *v112; // r13
  __int64 v113; // rcx
  unsigned int v114; // r15d
  __int64 v115; // rdx
  __int64 v116; // rcx
  __int64 v117; // rbx
  __int64 v118; // r8
  __int64 v119; // r9
  __int64 v120; // rax
  __int64 v121; // rbx
  __int64 j; // r14
  __int64 v123; // rax
  _DWORD *v124; // rax
  int v125; // edx
  __int64 v126; // rdi
  __int64 v127; // rsi
  __int64 v128; // r9
  unsigned __int64 v129; // r10
  int *v130; // r8
  int v131; // eax
  _DWORD *v132; // r9
  _DWORD *v133; // [rsp+8h] [rbp-6A8h]
  float v134; // [rsp+10h] [rbp-6A0h]
  __int64 v135; // [rsp+20h] [rbp-690h]
  bool v136; // [rsp+28h] [rbp-688h]
  __int64 v137; // [rsp+28h] [rbp-688h]
  __int64 v138; // [rsp+28h] [rbp-688h]
  unsigned int v139; // [rsp+28h] [rbp-688h]
  int v140; // [rsp+30h] [rbp-680h]
  __int64 v142; // [rsp+40h] [rbp-670h]
  float v143; // [rsp+4Ch] [rbp-664h]
  __int64 v144; // [rsp+58h] [rbp-658h]
  int v145; // [rsp+60h] [rbp-650h]
  unsigned int v146; // [rsp+64h] [rbp-64Ch]
  __int64 v147; // [rsp+68h] [rbp-648h]
  __int16 *v149; // [rsp+80h] [rbp-630h]
  char v150; // [rsp+8Bh] [rbp-625h]
  unsigned int v151; // [rsp+8Ch] [rbp-624h]
  int v152; // [rsp+90h] [rbp-620h]
  __int64 v153; // [rsp+90h] [rbp-620h]
  __int64 v154; // [rsp+98h] [rbp-618h]
  __int64 v155; // [rsp+98h] [rbp-618h]
  __int64 v156; // [rsp+A0h] [rbp-610h]
  unsigned __int64 v157; // [rsp+A0h] [rbp-610h]
  __int16 *v158; // [rsp+A0h] [rbp-610h]
  unsigned int v160; // [rsp+A8h] [rbp-608h]
  __int64 v161; // [rsp+B8h] [rbp-5F8h]
  float *v162; // [rsp+C0h] [rbp-5F0h] BYREF
  __int64 v163; // [rsp+C8h] [rbp-5E8h]
  _BYTE v164[80]; // [rsp+D0h] [rbp-5E0h] BYREF
  int v165; // [rsp+120h] [rbp-590h]
  _QWORD v166[33]; // [rsp+130h] [rbp-580h] BYREF
  char v167; // [rsp+238h] [rbp-478h] BYREF
  _QWORD *v168; // [rsp+240h] [rbp-470h] BYREF
  __int64 v169; // [rsp+248h] [rbp-468h]
  _QWORD v170[34]; // [rsp+250h] [rbp-460h] BYREF
  char v171; // [rsp+360h] [rbp-350h] BYREF

  v5 = a3;
  v151 = a4;
  v161 = sub_2F50FE0((_QWORD *)a1, a2, a3, a4);
  result = 0;
  if ( BYTE4(v161) )
  {
    v7 = 8;
    v8 = (int)v161;
    v9 = *(_QWORD *)(a1 + 192);
    v10 = (_QWORD *)qword_503F7F0;
    v134 = *(float *)(a2 + 116);
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v7 *= *v10++;
      while ( (_QWORD *)qword_503F7F8 != v10 );
    }
    memset(**(void ***)(v9 + 24), 0, v7);
    v11 = (_QWORD *)qword_503F7F0;
    v12 = 8;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v12 *= *v11++;
      while ( (_QWORD *)qword_503F7F8 != v11 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 8LL), 0, v12);
    v13 = (_QWORD *)qword_503F7F0;
    v14 = 4;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v14 *= *v13++;
      while ( (_QWORD *)qword_503F7F8 != v13 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 16LL), 0, v14);
    v15 = (_QWORD *)qword_503F7F0;
    v16 = 4;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v16 *= *v15++;
      while ( (_QWORD *)qword_503F7F8 != v15 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 24LL), 0, v16);
    v17 = (_QWORD *)qword_503F7F0;
    v18 = 8;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v18 *= *v17++;
      while ( (_QWORD *)qword_503F7F8 != v17 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 32LL), 0, v18);
    v19 = (_QWORD *)qword_503F7F0;
    v20 = 8;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v20 *= *v19++;
      while ( (_QWORD *)qword_503F7F8 != v19 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 40LL), 0, v20);
    v21 = (_QWORD *)qword_503F7F0;
    v22 = 4;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v22 *= *v21++;
      while ( (_QWORD *)qword_503F7F8 != v21 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 48LL), 0, v22);
    v23 = (_QWORD *)qword_503F7F0;
    v24 = 4;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v24 *= *v23++;
      while ( (_QWORD *)qword_503F7F8 != v23 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 56LL), 0, v24);
    v25 = (_QWORD *)qword_503F7F0;
    v26 = 4;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v26 *= *v25++;
      while ( (_QWORD *)qword_503F7F8 != v25 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 64LL), 0, v26);
    v27 = (_QWORD *)qword_503F7F0;
    v28 = 4;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v28 *= *v27++;
      while ( (_QWORD *)qword_503F7F8 != v27 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 72LL), 0, v28);
    v29 = (_QWORD *)qword_503F7F0;
    v30 = 4;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v30 *= *v29++;
      while ( (_QWORD *)qword_503F7F8 != v29 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 80LL), 0, v30);
    v31 = (_QWORD *)qword_503F7F0;
    v32 = 4;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v32 *= *v31++;
      while ( (_QWORD *)qword_503F7F8 != v31 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 88LL), 0, v32);
    v33 = (_QWORD *)qword_503F7F0;
    v34 = 4;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v34 *= *v33++;
      while ( (_QWORD *)qword_503F7F8 != v33 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 96LL), 0, v34);
    v35 = (_QWORD *)qword_503F7F0;
    v36 = 4;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v36 *= *v35++;
      while ( (_QWORD *)qword_503F7F8 != v35 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 104LL), 0, v36);
    v37 = (_QWORD *)qword_503F7F0;
    v38 = 4;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v38 *= *v37++;
      while ( (_QWORD *)qword_503F7F8 != v37 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 112LL), 0, v38);
    v39 = (_QWORD *)qword_503F7F0;
    v40 = 4;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v40 *= *v39++;
      while ( (_QWORD *)qword_503F7F8 != v39 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 120LL), 0, v40);
    v41 = (_QWORD *)qword_503F7F0;
    v42 = 4;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v42 *= *v41++;
      while ( (_QWORD *)qword_503F7F8 != v41 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 128LL), 0, v42);
    v43 = (_QWORD *)qword_503F7F0;
    v44 = 4;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v44 *= *v43++;
      while ( (_QWORD *)qword_503F7F8 != v43 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 136LL), 0, v44);
    v45 = (_QWORD *)qword_503F7F0;
    v46 = 8;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v46 *= *v45++;
      while ( (_QWORD *)qword_503F7F8 != v45 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 144LL), 0, v46);
    v47 = (_QWORD *)qword_503F7F0;
    v48 = 8;
    if ( qword_503F7F0 != qword_503F7F8 )
    {
      do
        v48 *= *v47++;
      while ( (_QWORD *)qword_503F7F8 != v47 );
    }
    memset(*(void **)(*(_QWORD *)(v9 + 24) + 152LL), 0, v48);
    v49 = (_QWORD *)sub_22077B0(8u);
    *v49 = 1;
    **(_DWORD **)(*(_QWORD *)(v9 + 24) + 160LL) = 0;
    j_j___libc_free_0((unsigned __int64)v49);
    memset(v166, 0, sizeof(v166));
    v50 = (char *)v166;
    do
    {
      *(_DWORD *)v50 = 0;
      v50 += 8;
      *(v50 - 4) = 0;
    }
    while ( v50 != &v167 );
    v51 = *(_QWORD *)(v5 + 8);
    v162 = (float *)v164;
    memset(v164, 0, sizeof(v164));
    v52 = -(int)v51;
    v163 = 0x1500000015LL;
    v170[32] = &v171;
    v170[33] = 0x2100000000LL;
    v53 = *(_DWORD *)(v5 + 72);
    v152 = v53;
    v165 = 0;
    if ( (_DWORD)v161 )
    {
      v54 = v161 - 1;
      if ( (int)v161 - 1 > v53 )
      {
        v54 = v53;
      }
      else if ( v54 < v53 )
      {
        if ( (int)v161 < 0 || (int)v161 >= v53 )
        {
          v54 = v161;
        }
        else
        {
          v156 = v51;
          v55 = (unsigned __int16 **)v5;
          v56 = *(_QWORD *)(v5 + 56);
          do
          {
            v54 = v8;
            if ( (unsigned int)*(unsigned __int16 *)(v56 + 2 * v8) - 1 > 0x3FFFFFFE )
              break;
            LODWORD(v168) = *(unsigned __int16 *)(v56 + 2 * v8);
            v57 = &(*v55)[v156];
            if ( v57 == sub_3592920(*v55, (__int64)v57, (int *)&v168) )
              break;
            ++v8;
            ++v54;
          }
          while ( v58 > (int)v8 );
          v5 = (__int64)v55;
        }
      }
      v152 = v54;
    }
    if ( v52 == v152 )
      goto LABEL_69;
    v154 = 0;
    v147 = 0;
    v59 = v52;
    while ( 1 )
    {
      v157 = v59;
      v61 = v59 < 0
          ? *(unsigned __int16 *)(*(_QWORD *)v5 + 2 * (*(_QWORD *)(v5 + 8) + v59))
          : *(unsigned __int16 *)(*(_QWORD *)(v5 + 56) + 2LL * v59);
      v62 = v151;
      v150 = sub_2F510F0(a1, v151, v61);
      if ( v150 )
      {
        v62 = a2;
        if ( (int)sub_2E21680(*(_QWORD **)(a1 + 24), a2, v61) <= 1 )
          break;
      }
LABEL_61:
      v64 = *(_DWORD *)(v5 + 72);
      v59 += v59 < v64;
      if ( v64 > v59 && v59 >= 0 )
      {
        v65 = *(_QWORD *)(v5 + 56);
        v66 = v59;
        v67 = (int *)&v168;
        do
        {
          v59 = v66;
          if ( (unsigned int)*(unsigned __int16 *)(v65 + 2 * v66) - 1 > 0x3FFFFFFE )
            break;
          LODWORD(v168) = *(unsigned __int16 *)(v65 + 2 * v66);
          v62 = *(_QWORD *)v5 + 2LL * *(_QWORD *)(v5 + 8);
          if ( (unsigned __int16 *)v62 == sub_3592920(*(unsigned __int16 **)v5, v62, v67) )
            break;
          v66 = v68 + 1;
          ++v59;
        }
        while ( v69 > (int)v66 );
      }
      ++v154;
      if ( v152 == v59 )
      {
        if ( v147 )
        {
          BYTE4(v166[32]) = v134 != INFINITY || a4 != 0xFF;
          if ( v134 != INFINITY || a4 != 0xFF )
          {
            v62 = (unsigned __int64)&v168;
            v170[0] = a2;
            v168 = v170;
            v169 = 0x100000001LL;
            sub_35946F0((_QWORD *)a1, (__int64)&v168, &v162, 32, 0, 0, 0.0);
            if ( v168 != v170 )
              _libc_free((unsigned __int64)v168);
          }
          v100 = v162;
          v101 = &v162[(unsigned int)v163];
          if ( v101 != v162 )
          {
            do
            {
              v102 = *v100;
              if ( *v100 == 0.0 )
                v102 = 1.0;
              *v100++ = v102;
            }
            while ( v101 != v100 );
          }
          for ( i = 0; i != 21; ++i )
          {
            while ( 1 )
            {
              v104 = *(_QWORD *)(a1 + 216) & (1LL << i);
              if ( !v104 )
                break;
              if ( ++i == 21 )
                goto LABEL_142;
            }
            v63 = 4 * i;
            do
            {
              v62 = (unsigned __int64)v162;
              v105 = (float *)(v104 + *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 192) + 24LL) + 8 * i));
              v104 += 4;
              *v105 = *v105 / v162[i];
            }
            while ( v104 != 132 );
          }
LABEL_142:
          v106 = *(_QWORD *)(a1 + 16);
          v107 = *(float **)(*(_QWORD *)(*(_QWORD *)(a1 + 192) + 24LL) + 160LL);
          v108 = (__int64)(*(_QWORD *)(v106 + 888) - *(_QWORD *)(v106 + 880)) >> 3;
          if ( v108 < 0 )
          {
            v106 = v108 & 1 | ((unsigned __int64)v108 >> 1);
            v109 = (float)(int)v106 + (float)(int)v106;
          }
          else
          {
            v109 = (float)(int)v108;
          }
          v110 = v109 / *(float *)(a1 + 224);
          *v107 = v110;
          v153 = *(_QWORD *)(*(__int64 (__fastcall **)(_QWORD, unsigned __int64, unsigned __int64, float *, __int64, __int64, float, double))(**(_QWORD **)(a1 + 192) + 24LL))(
                              *(_QWORD *)(a1 + 192),
                              v62,
                              v106,
                              v107,
                              v63,
                              1,
                              v110,
                              0.0);
          if ( v153 != 32 )
          {
            v111 = *(_QWORD *)(a1 + 56);
            v155 = a1;
            v112 = (unsigned __int64 *)(a1 + 264);
            v113 = *(_QWORD *)(v111 + 8);
            v158 = (__int16 *)(*(_QWORD *)(v111 + 56) + 2LL
                                                      * (*(_DWORD *)(v113 + 24LL * LODWORD(v166[v153]) + 16) >> 12));
            v114 = *(_DWORD *)(v113 + 24LL * LODWORD(v166[v153]) + 16) & 0xFFF;
            do
            {
              if ( !v158 )
                break;
              v117 = sub_2E21610(*(_QWORD *)(v155 + 24), a2, v114);
              if ( *(_BYTE *)(v117 + 161) )
              {
                v120 = *(unsigned int *)(v117 + 120);
                if ( LODWORD(qword_5023488[8]) >= (unsigned int)v120 )
                  continue;
              }
              sub_2E1AC90(v117, qword_5023488[8], v115, v116, v118, v119);
              v120 = *(unsigned int *)(v117 + 120);
              v121 = *(_QWORD *)(v117 + 112);
              for ( j = v121 + 8 * v120; v121 != j; ++*v124 )
              {
                v123 = *(_QWORD *)(j - 8);
                j -= 8;
                LODWORD(v168) = *(_DWORD *)(v123 + 112);
                v124 = (_DWORD *)sub_3596020(v112, (unsigned int *)&v168);
              }
              v125 = *v158++;
              v114 += v125;
            }
            while ( (_WORD)v125 );
            result = LODWORD(v166[v153]);
            goto LABEL_70;
          }
          LODWORD(v168) = *(_DWORD *)(a2 + 112);
          v70 = (_DWORD *)sub_3596020((unsigned __int64 *)(a1 + 264), (unsigned int *)&v168);
          ++*v70;
        }
LABEL_69:
        result = 0;
LABEL_70:
        if ( v162 != (float *)v164 )
        {
          v160 = result;
          _libc_free((unsigned __int64)v162);
          return v160;
        }
        return result;
      }
    }
    v71 = sub_2E13500(*(_QWORD *)(a1 + 32), a2);
    v72 = *(_QWORD *)(a1 + 16);
    v142 = v71;
    v146 = *(_DWORD *)(*(_QWORD *)(v72 + 920) + 8LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF) + 4);
    if ( !v146 )
      v146 = *(_DWORD *)(v72 + 952);
    v73 = *(_QWORD *)(a1 + 56);
    v140 = v61;
    v168 = v170;
    v169 = 0x2000000000LL;
    v74 = *(_QWORD *)(v73 + 8);
    v135 = 0;
    v145 = v59;
    v144 = v5;
    v143 = 0.0;
    v149 = (__int16 *)(*(_QWORD *)(v73 + 56) + 2LL * (*(_DWORD *)(v74 + 24LL * v61 + 16) >> 12));
    v75 = *(_DWORD *)(v74 + 24LL * v61 + 16) & 0xFFF;
    while ( 1 )
    {
      if ( !v149 )
      {
LABEL_83:
        v62 = (unsigned __int64)&v168;
        v59 = v145;
        v5 = v144;
        sub_35946F0((_QWORD *)a1, (__int64)&v168, &v162, v154, v157 >> 63, v135, v143);
        if ( v168 != v170 )
          _libc_free((unsigned __int64)v168);
        ++v147;
        LODWORD(v166[v154]) = v140;
        BYTE4(v166[v154]) = 1;
        goto LABEL_61;
      }
      v76 = sub_2E21610(*(_QWORD *)(a1 + 24), a2, v75);
      v80 = v76;
      v62 = LODWORD(qword_5023488[8]);
      if ( !*(_BYTE *)(v76 + 161) || (v77 = *(unsigned int *)(v76 + 120), LODWORD(qword_5023488[8]) < (unsigned int)v77) )
      {
        sub_2E1AC90(v76, qword_5023488[8], v77, v78, v63, v79);
        LODWORD(v77) = *(_DWORD *)(v80 + 120);
      }
      v81 = (unsigned int)v169;
      if ( (unsigned int)v77 | (unsigned int)v169 )
      {
        if ( LODWORD(qword_5023488[8]) <= (unsigned int)v77 )
        {
LABEL_98:
          v59 = v145;
          v5 = v144;
          if ( v168 != v170 )
            _libc_free((unsigned __int64)v168);
          goto LABEL_61;
        }
        v83 = 8LL * (unsigned int)v77;
        v84 = v77;
        v85 = (unsigned int)v77 + (unsigned __int64)(unsigned int)v169;
        v63 = *(_QWORD *)(v80 + 112);
        if ( v85 > HIDWORD(v169) )
        {
          v62 = (unsigned __int64)v170;
          v137 = *(_QWORD *)(v80 + 112);
          sub_C8D5F0((__int64)&v168, v170, v85, 8u, v63, v79);
          v81 = (unsigned int)v169;
          v63 = v137;
        }
        if ( v83 )
        {
          v62 = v63;
          memcpy(&v168[v81], (const void *)v63, v83);
          LODWORD(v81) = v169;
        }
        LODWORD(v169) = v84 + v81;
        v86 = *(_QWORD *)(v80 + 112);
        v87 = v86 + 8LL * *(unsigned int *)(v80 + 120);
        if ( v86 != v87 )
          break;
      }
LABEL_82:
      v82 = *v149++;
      v75 += v82;
      if ( !(_WORD)v82 )
        goto LABEL_83;
    }
    while ( 1 )
    {
      v88 = *(_QWORD *)(v87 - 8);
      v89 = *(_DWORD *)(v88 + 112);
      if ( *(_QWORD *)(a5 + 120) )
      {
        v98 = *(_QWORD *)(a5 + 96);
        if ( v98 )
        {
          v99 = a5 + 88;
          do
          {
            v62 = *(_QWORD *)(v98 + 16);
            if ( v89 > *(_DWORD *)(v98 + 32) )
            {
              v98 = *(_QWORD *)(v98 + 24);
            }
            else
            {
              v99 = v98;
              v98 = *(_QWORD *)(v98 + 16);
            }
          }
          while ( v98 );
          if ( a5 + 88 != v99 && v89 >= *(_DWORD *)(v99 + 32) )
            goto LABEL_98;
        }
      }
      else
      {
        v90 = *(_DWORD **)a5;
        v91 = *(_QWORD *)a5 + 4LL * *(unsigned int *)(a5 + 8);
        if ( *(_QWORD *)a5 != v91 )
        {
          while ( v89 != *v90 )
          {
            if ( (_DWORD *)v91 == ++v90 )
              goto LABEL_102;
          }
          if ( (_DWORD *)v91 != v90 )
            goto LABEL_98;
        }
      }
LABEL_102:
      v92 = v89 & 0x7FFFFFFF;
      v93 = (_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 920LL) + 8 * v92);
      if ( *v93 == 6 )
        goto LABEL_98;
      if ( *(float *)(a2 + 116) == INFINITY )
      {
        if ( *(float *)(v88 + 116) == INFINITY )
        {
          v126 = *(_QWORD *)(a1 + 64);
          v127 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 56LL);
          v128 = *(_QWORD *)v126;
          v129 = *(_QWORD *)(v127 + 16LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
          v130 = (int *)(*(_QWORD *)v126 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v129 + 24LL));
          v131 = *v130;
          if ( *(_DWORD *)(v126 + 8) != *v130 )
          {
            v138 = *(_QWORD *)v126 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v129 + 24LL);
            sub_2F60630(
              v126,
              (unsigned __int16 ***)(*(_QWORD *)(v127 + 16LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF))
                                   & 0xFFFFFFFFFFFFFFF8LL));
            v126 = *(_QWORD *)(a1 + 64);
            v89 = *(_DWORD *)(v88 + 112);
            v130 = (int *)v138;
            v127 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 56LL);
            v128 = *(_QWORD *)v126;
            v131 = *(_DWORD *)(v126 + 8);
            v92 = v89 & 0x7FFFFFFF;
          }
          v63 = (unsigned int)v130[1];
          v62 = *(_QWORD *)(v127 + 16 * v92) & 0xFFFFFFFFFFFFFFF8LL;
          v132 = (_DWORD *)(v128 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v62 + 24LL));
          if ( *v132 != v131 )
          {
            v133 = v132;
            v139 = v63;
            sub_2F60630(v126, (unsigned __int16 ***)v62);
            v89 = *(_DWORD *)(v88 + 112);
            v132 = v133;
            v63 = v139;
            v92 = v89 & 0x7FFFFFFF;
          }
          v136 = (unsigned int)v63 < v132[1];
          v93 = (_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 920LL) + 8 * v92);
        }
        else
        {
          v136 = v150;
        }
      }
      else
      {
        v136 = 0;
      }
      v94 = *(_QWORD *)(a1 + 272);
      v95 = v93[1];
      v96 = *(__int64 **)(*(_QWORD *)(a1 + 264) + 8 * (v89 % v94));
      if ( v96 )
      {
        v97 = (__int64 *)*v96;
        v62 = *(unsigned int *)(*v96 + 8);
        if ( v89 == (_DWORD)v62 )
        {
LABEL_111:
          if ( *v96 && (unsigned int)qword_503F8A8 < *(_DWORD *)(*v96 + 12) )
          {
            if ( !v136 )
              goto LABEL_98;
            if ( v95 < v146 )
              goto LABEL_115;
            goto LABEL_123;
          }
        }
        else
        {
          while ( 1 )
          {
            v63 = *v97;
            if ( !*v97 )
              break;
            v62 = *(unsigned int *)(v63 + 8);
            v96 = v97;
            if ( v89 % v94 != v62 % v94 )
              break;
            v97 = (__int64 *)*v97;
            if ( v89 == (_DWORD)v62 )
              goto LABEL_111;
          }
        }
      }
      if ( v95 < v146 )
        goto LABEL_115;
      if ( !v136 )
        goto LABEL_98;
LABEL_123:
      v143 = v143 + 1.0;
LABEL_115:
      if ( v142 )
      {
        v62 = v88;
        if ( sub_2E13500(*(_QWORD *)(a1 + 32), v88) )
        {
          if ( *(_BYTE *)(a1 + 88) )
          {
            v62 = v88;
            v135 += (unsigned __int8)sub_2F50850((_QWORD *)a1, v88, v140) ^ 1u;
          }
          else
          {
            ++v135;
          }
        }
      }
      v87 -= 8;
      if ( v86 == v87 )
        goto LABEL_82;
    }
  }
  return result;
}
