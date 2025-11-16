// Function: sub_290FB60
// Address: 0x290fb60
//
__int64 __fastcall sub_290FB60(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 *a5, __int64 a6)
{
  unsigned __int64 *v8; // rsi
  __int64 v9; // r14
  unsigned int v10; // r12d
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // r8
  unsigned __int64 v17; // r9
  __int64 v18; // r15
  __int64 v19; // rcx
  __int64 v20; // r13
  __int64 v21; // r13
  __int64 v22; // r15
  __int64 v23; // rbx
  unsigned __int64 v24; // rsi
  _QWORD *v25; // rdi
  __int64 v26; // r15
  __int64 i; // r9
  __int64 v28; // rdi
  __int64 v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  _QWORD *v34; // rbx
  _QWORD *v35; // r15
  void (__fastcall *v36)(_QWORD *, _QWORD *, __int64); // rax
  _QWORD *v37; // rdi
  int v39; // eax
  __int64 v40; // rbx
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  __int64 v45; // rdx
  int v46; // eax
  __int64 *v47; // rdx
  __int64 v48; // r11
  __int64 v49; // r13
  char v50; // si
  int v51; // r10d
  __int64 v52; // rax
  __int64 v53; // rdi
  __int64 v54; // rdx
  __int64 v55; // rcx
  bool v56; // dl
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 *v59; // rax
  __int64 v60; // r11
  __int64 v61; // r9
  __int64 v62; // rsi
  int v63; // r10d
  int v64; // r10d
  __int64 v65; // r11
  __int64 v66; // r9
  __int64 *v67; // rsi
  __int64 v68; // rax
  __int64 v69; // rdi
  __int64 v70; // rax
  __int64 v71; // rax
  __int64 *v72; // rbx
  __int64 v73; // r13
  int v74; // eax
  unsigned __int8 *v75; // r12
  __int64 v76; // r12
  __int64 v77; // rax
  __int64 v78; // rsi
  __int64 v79; // r12
  unsigned int *v80; // rax
  int v81; // ecx
  unsigned int *v82; // rdx
  __int64 v83; // r11
  __int64 (__fastcall *v84)(__int64, unsigned int, char *, __int64); // rax
  __int64 v85; // rax
  _BYTE *v86; // r12
  __int64 v87; // rax
  __int64 v88; // rax
  _BYTE *v89; // r10
  unsigned __int8 *v90; // r12
  int v91; // r8d
  __int64 v92; // rdx
  unsigned int *v93; // r13
  unsigned int *v94; // rbx
  __int64 v95; // rdx
  unsigned int v96; // esi
  __int64 v97; // rax
  unsigned __int64 v98; // r8
  unsigned __int64 v99; // r9
  __int64 *v100; // [rsp+0h] [rbp-930h]
  __int64 v101; // [rsp+8h] [rbp-928h]
  __int64 v102; // [rsp+28h] [rbp-908h]
  unsigned int v103; // [rsp+34h] [rbp-8FCh]
  __int64 v104; // [rsp+38h] [rbp-8F8h]
  __int64 v105; // [rsp+48h] [rbp-8E8h]
  __int64 **v106; // [rsp+48h] [rbp-8E8h]
  __int64 v107; // [rsp+48h] [rbp-8E8h]
  unsigned __int64 v108; // [rsp+48h] [rbp-8E8h]
  int v109; // [rsp+50h] [rbp-8E0h]
  __int64 v110; // [rsp+50h] [rbp-8E0h]
  _QWORD *v111; // [rsp+50h] [rbp-8E0h]
  __int64 v112; // [rsp+58h] [rbp-8D8h]
  int v113; // [rsp+58h] [rbp-8D8h]
  __int64 v114; // [rsp+58h] [rbp-8D8h]
  __int64 v115; // [rsp+60h] [rbp-8D0h]
  __int64 v116; // [rsp+60h] [rbp-8D0h]
  char *v117; // [rsp+60h] [rbp-8D0h]
  _BYTE *v118; // [rsp+60h] [rbp-8D0h]
  int v119; // [rsp+60h] [rbp-8D0h]
  __int64 v120; // [rsp+60h] [rbp-8D0h]
  __int64 v123; // [rsp+78h] [rbp-8B8h]
  __int64 *v125; // [rsp+A0h] [rbp-890h]
  unsigned __int64 v127[2]; // [rsp+B0h] [rbp-880h] BYREF
  __int64 v128; // [rsp+C0h] [rbp-870h] BYREF
  unsigned __int64 v129[2]; // [rsp+D0h] [rbp-860h] BYREF
  __int64 v130; // [rsp+E0h] [rbp-850h] BYREF
  __int16 v131; // [rsp+F0h] [rbp-840h]
  _QWORD v132[4]; // [rsp+100h] [rbp-830h] BYREF
  __int16 v133; // [rsp+120h] [rbp-810h]
  __int64 v134; // [rsp+130h] [rbp-800h] BYREF
  __int64 v135; // [rsp+138h] [rbp-7F8h]
  __int64 v136; // [rsp+140h] [rbp-7F0h]
  unsigned int v137; // [rsp+148h] [rbp-7E8h]
  __int64 *v138; // [rsp+150h] [rbp-7E0h]
  __int64 v139; // [rsp+158h] [rbp-7D8h]
  __int64 v140; // [rsp+160h] [rbp-7D0h] BYREF
  __int64 v141; // [rsp+168h] [rbp-7C8h]
  __int64 v142; // [rsp+170h] [rbp-7C0h]
  unsigned int v143; // [rsp+178h] [rbp-7B8h]
  unsigned int **v144; // [rsp+180h] [rbp-7B0h]
  __int64 v145; // [rsp+188h] [rbp-7A8h]
  unsigned int *v146; // [rsp+190h] [rbp-7A0h] BYREF
  __int64 v147; // [rsp+198h] [rbp-798h]
  _BYTE v148[32]; // [rsp+1A0h] [rbp-790h] BYREF
  __int64 v149; // [rsp+1C0h] [rbp-770h]
  __int64 v150; // [rsp+1C8h] [rbp-768h]
  __int64 v151; // [rsp+1D0h] [rbp-760h]
  __int64 v152; // [rsp+1D8h] [rbp-758h]
  void **v153; // [rsp+1E0h] [rbp-750h]
  void **v154; // [rsp+1E8h] [rbp-748h]
  __int64 v155; // [rsp+1F0h] [rbp-740h]
  int v156; // [rsp+1F8h] [rbp-738h]
  __int16 v157; // [rsp+1FCh] [rbp-734h]
  char v158; // [rsp+1FEh] [rbp-732h]
  __int64 v159; // [rsp+200h] [rbp-730h]
  __int64 v160; // [rsp+208h] [rbp-728h]
  void *v161; // [rsp+210h] [rbp-720h] BYREF
  void *v162; // [rsp+218h] [rbp-718h] BYREF
  _BYTE *v163; // [rsp+220h] [rbp-710h] BYREF
  __int64 v164; // [rsp+228h] [rbp-708h]
  _BYTE v165[512]; // [rsp+230h] [rbp-700h] BYREF
  __int64 *v166; // [rsp+430h] [rbp-500h] BYREF
  __int64 v167; // [rsp+438h] [rbp-4F8h]
  _BYTE v168[512]; // [rsp+440h] [rbp-4F0h] BYREF
  unsigned __int64 v169[2]; // [rsp+640h] [rbp-2F0h] BYREF
  _BYTE v170[512]; // [rsp+650h] [rbp-2E0h] BYREF
  __int64 v171; // [rsp+850h] [rbp-E0h]
  __int64 v172; // [rsp+858h] [rbp-D8h]
  __int64 v173; // [rsp+860h] [rbp-D0h]
  __int64 v174; // [rsp+868h] [rbp-C8h]
  char v175; // [rsp+870h] [rbp-C0h]
  __int64 v176; // [rsp+878h] [rbp-B8h]
  char *v177; // [rsp+880h] [rbp-B0h]
  __int64 v178; // [rsp+888h] [rbp-A8h]
  int v179; // [rsp+890h] [rbp-A0h]
  char v180; // [rsp+894h] [rbp-9Ch]
  char v181; // [rsp+898h] [rbp-98h] BYREF
  __int16 v182; // [rsp+8D8h] [rbp-58h]
  _QWORD *v183; // [rsp+8E0h] [rbp-50h]
  _QWORD *v184; // [rsp+8E8h] [rbp-48h]
  __int64 v185; // [rsp+8F0h] [rbp-40h]

  v169[0] = (unsigned __int64)v170;
  v169[1] = 0x1000000000LL;
  v8 = v169;
  v173 = a3;
  v177 = &v181;
  v171 = 0;
  v172 = 0;
  v174 = 0;
  v175 = 1;
  v176 = 0;
  v178 = 8;
  v179 = 0;
  v180 = 1;
  v182 = 0;
  v183 = 0;
  v184 = 0;
  v185 = 0;
  v9 = a2 + 72;
  v10 = sub_F62E00(a2, (__int64)v169, 0, (__int64)v169, (__int64)a5, a6);
  sub_FFD350((__int64)v169, (__int64)v169, v11, v12, v13, v14);
  v18 = *(_QWORD *)(a2 + 80);
  v19 = (__int64)v168;
  v163 = v165;
  v164 = 0x4000000000LL;
  v166 = (__int64 *)v168;
  v167 = 0x4000000000LL;
  if ( a2 + 72 == v18 )
  {
    v20 = 0;
  }
  else
  {
    if ( !v18 )
      BUG();
    while ( 1 )
    {
      v20 = *(_QWORD *)(v18 + 32);
      if ( v20 != v18 + 24 )
        break;
      v18 = *(_QWORD *)(v18 + 8);
      if ( v9 == v18 )
        goto LABEL_7;
      if ( !v18 )
        BUG();
    }
  }
  while ( v9 != v18 )
  {
    if ( !v20 )
      BUG();
    v39 = *(unsigned __int8 *)(v20 - 24);
    v40 = v20 - 24;
    if ( (unsigned __int8)(v39 - 34) <= 0x33u )
    {
      v19 = 0x8000000000041LL;
      if ( _bittest64(&v19, (unsigned int)(v39 - 34)) )
      {
        v41 = *(_QWORD *)(v20 - 56);
        if ( !v41
          || *(_BYTE *)v41
          || (v19 = *(_QWORD *)(v20 + 56), *(_QWORD *)(v41 + 24) != v19)
          || *(_DWORD *)(v41 + 36) != 151 )
        {
          v8 = (unsigned __int64 *)a5;
          if ( !(unsigned __int8)sub_F57670(v20 - 24, a5) )
          {
            if ( (_BYTE)qword_5004EE8
              || (v8 = (unsigned __int64 *)(v20 - 24), sub_BDBBF0((__int64)&v146, v20 - 24, 0), v148[8]) )
            {
              v43 = (unsigned int)v164;
              v19 = HIDWORD(v164);
              v44 = (unsigned int)v164 + 1LL;
              if ( v44 > HIDWORD(v164) )
              {
                v8 = (unsigned __int64 *)v165;
                sub_C8D5F0((__int64)&v163, v165, v44, 8u, v16, v17);
                v43 = (unsigned int)v164;
              }
              *(_QWORD *)&v163[8 * v43] = v40;
              LODWORD(v164) = v164 + 1;
            }
          }
          LOBYTE(v39) = *(_BYTE *)(v20 - 24);
        }
        if ( (_BYTE)v39 == 85
          && ((unsigned int)sub_B49240(v20 - 24) == 147 || (unsigned int)sub_B49240(v20 - 24) == 148) )
        {
          v45 = (unsigned int)v167;
          v19 = HIDWORD(v167);
          v46 = v167;
          if ( (unsigned int)v167 >= (unsigned __int64)HIDWORD(v167) )
          {
            v17 = (unsigned int)v167 + 1LL;
            if ( HIDWORD(v167) < v17 )
            {
              v8 = (unsigned __int64 *)v168;
              sub_C8D5F0((__int64)&v166, v168, (unsigned int)v167 + 1LL, 8u, v16, v17);
              v45 = (unsigned int)v167;
            }
            v166[v45] = v40;
            LODWORD(v167) = v167 + 1;
          }
          else
          {
            v19 = (__int64)v166;
            v47 = &v166[(unsigned int)v167];
            if ( v47 )
            {
              *v47 = v40;
              v46 = v167;
            }
            LODWORD(v167) = v46 + 1;
          }
        }
      }
    }
    v20 = *(_QWORD *)(v20 + 8);
    v15 = 0;
    while ( 1 )
    {
      v42 = v18 - 24;
      if ( !v18 )
        v42 = 0;
      if ( v20 != v42 + 48 )
        break;
      v18 = *(_QWORD *)(v18 + 8);
      if ( v9 == v18 )
        goto LABEL_7;
      if ( !v18 )
        BUG();
      v20 = *(_QWORD *)(v18 + 32);
    }
  }
LABEL_7:
  if ( (unsigned int)v167 | (unsigned int)v164 )
  {
    v21 = *(_QWORD *)(a2 + 80);
    if ( v21 != v9 )
    {
      do
      {
        v22 = v21 - 24;
        if ( !v21 )
          v22 = 0;
        if ( sub_AA5510(v22) )
          v10 |= sub_F34590(v22, 0);
        v21 = *(_QWORD *)(v21 + 8);
      }
      while ( v21 != v9 );
      v23 = *(_QWORD *)(a2 + 80);
      if ( v23 != v9 )
      {
        do
        {
          if ( !v23 )
            BUG();
          v24 = *(_QWORD *)(v23 + 24) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v24 == v23 + 24 )
            goto LABEL_186;
          if ( !v24 )
            BUG();
          if ( (unsigned int)*(unsigned __int8 *)(v24 - 24) - 30 > 0xA )
LABEL_186:
            BUG();
          if ( *(_BYTE *)(v24 - 24) == 31 && (*(_DWORD *)(v24 - 20) & 0x7FFFFFF) == 3 )
          {
            v25 = *(_QWORD **)(v24 - 120);
            if ( *(_BYTE *)v25 == 82 )
            {
              v70 = v25[2];
              if ( v70 )
              {
                if ( !*(_QWORD *)(v70 + 8) )
                {
                  v71 = v123;
                  v10 = 1;
                  LOWORD(v71) = 0;
                  v123 = v71;
                  sub_B444E0(v25, v24, v71);
                }
              }
            }
          }
          v23 = *(_QWORD *)(v23 + 8);
        }
        while ( v23 != v9 );
        v26 = *(_QWORD *)(a2 + 80);
        if ( v9 != v26 )
        {
          if ( !v26 )
            BUG();
          while ( 1 )
          {
            i = *(_QWORD *)(v26 + 32);
            if ( i != v26 + 24 )
              break;
            v26 = *(_QWORD *)(v26 + 8);
            if ( v9 == v26 )
              goto LABEL_29;
            if ( !v26 )
              BUG();
          }
          while ( v9 != v26 )
          {
            if ( !i )
              BUG();
            if ( *(_BYTE *)(i - 24) == 63 )
            {
              v48 = i - 24;
              v49 = *(_DWORD *)(i - 20) & 0x7FFFFFF;
              v50 = *(_BYTE *)(i - 17) & 0x40;
              v51 = v49;
              if ( (_DWORD)v49 )
              {
                v52 = 0;
                v51 = 0;
                v53 = 32LL * (unsigned int)v49;
                LODWORD(v23) = v48 - v53;
                do
                {
                  v54 = v48 - v53;
                  if ( v50 )
                    v54 = *(_QWORD *)(i - 32);
                  v55 = *(_QWORD *)(*(_QWORD *)(v54 + v52) + 8LL);
                  if ( (unsigned int)*(unsigned __int8 *)(v55 + 8) - 17 <= 1 )
                    v51 = *(_DWORD *)(v55 + 32);
                  v52 += 32;
                }
                while ( v53 != v52 );
                v56 = v51 != 0;
              }
              else
              {
                v56 = 0;
              }
              if ( v50 )
                v57 = *(_QWORD *)(i - 32);
              else
                v57 = v48 - 32 * v49;
              LOBYTE(v23) = v56 && (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v57 + 8LL) + 8LL) - 17 > 1;
              if ( (_BYTE)v23 )
              {
                v109 = v51;
                v112 = i;
                v115 = i - 24;
                v152 = sub_BD5C60(i - 24);
                v157 = 512;
                LOWORD(v151) = 0;
                v146 = (unsigned int *)v148;
                v161 = &unk_49DA100;
                v147 = 0x200000000LL;
                v149 = 0;
                v150 = 0;
                v153 = &v161;
                v154 = &v162;
                v155 = 0;
                v156 = 0;
                v158 = 7;
                v159 = 0;
                v160 = 0;
                v162 = &unk_49DA0B0;
                v58 = *(_QWORD *)(v112 + 16);
                v150 = v112;
                v149 = v58;
                v59 = (__int64 *)sub_B46C60(v115);
                v60 = v115;
                v61 = v112;
                v62 = *v59;
                v63 = v109;
                v140 = *v59;
                if ( v140 )
                {
                  sub_B96E90((__int64)&v140, v62, 1);
                  v62 = v140;
                  v61 = v112;
                  v60 = v115;
                  v63 = v109;
                }
                v105 = v61;
                v110 = v60;
                v113 = v63;
                sub_F80810((__int64)&v146, 0, v62, (__int64)&v140, (__int64)&v146, v61);
                v64 = v113;
                v65 = v110;
                v66 = v105;
                if ( v140 )
                {
                  sub_B91220((__int64)&v140, v140);
                  v66 = v105;
                  v65 = v110;
                  v64 = v113;
                }
                LOWORD(v144) = 257;
                if ( (*(_BYTE *)(v66 - 17) & 0x40) != 0 )
                  v67 = *(__int64 **)(v66 - 32);
                else
                  v67 = (__int64 *)(v65 - 32LL * (*(_DWORD *)(v66 - 20) & 0x7FFFFFF));
                v114 = v66;
                v116 = v65;
                v68 = sub_B37A60(&v146, v64, *v67, &v140);
                if ( (*(_BYTE *)(v114 - 17) & 0x40) != 0 )
                  v69 = *(_QWORD *)(v114 - 32);
                else
                  v69 = v116 - 32LL * (*(_DWORD *)(v114 - 20) & 0x7FFFFFF);
                sub_AC2B30(v69, v68);
                nullsub_61();
                v161 = &unk_49DA100;
                nullsub_63();
                i = v114;
                if ( v146 != (unsigned int *)v148 )
                {
                  _libc_free((unsigned __int64)v146);
                  i = v114;
                }
                v10 = v23;
              }
            }
            for ( i = *(_QWORD *)(i + 8); i == v26 - 24 + 48; i = *(_QWORD *)(v26 + 32) )
            {
              v26 = *(_QWORD *)(v26 + 8);
              if ( v9 == v26 )
                goto LABEL_29;
              if ( !v26 )
                BUG();
            }
          }
        }
      }
    }
LABEL_29:
    v134 = 0;
    v135 = 0;
    v136 = 0;
    v137 = 0;
    v138 = &v140;
    v139 = 0;
    v140 = 0;
    v141 = 0;
    v142 = 0;
    v143 = 0;
    v144 = &v146;
    v145 = 0;
    if ( !(_DWORD)v167 )
    {
      if ( !(_DWORD)v164 )
      {
        v28 = 0;
        v29 = 0;
LABEL_32:
        sub_C7D6A0(v28, v29, 8);
        if ( v138 != &v140 )
          _libc_free((unsigned __int64)v138);
        v8 = (unsigned __int64 *)(16LL * v137);
        sub_C7D6A0(v135, (__int64)v8, 8);
        goto LABEL_35;
      }
      goto LABEL_120;
    }
    v111 = (_QWORD *)sub_B2BE50(a2);
    v102 = sub_B2BEC0(a2);
    v125 = &v166[(unsigned int)v167];
    if ( v166 == v125 )
    {
LABEL_119:
      if ( !(_DWORD)v164 )
      {
LABEL_121:
        if ( v144 != &v146 )
          _libc_free((unsigned __int64)v144);
        v28 = v141;
        v29 = 16LL * v143;
        goto LABEL_32;
      }
LABEL_120:
      v10 |= sub_290C520(a2, a3, a4, (__int64)&v163, (char *)&v134, &v140);
      goto LABEL_121;
    }
    v72 = v166;
    while ( 1 )
    {
      v73 = *v72;
      v74 = sub_B49240(*v72);
      if ( v74 != 147 )
        break;
      v75 = (unsigned __int8 *)sub_2906F40(
                                 *(_QWORD *)(v73 - 32LL * (*(_DWORD *)(v73 + 4) & 0x7FFFFFF)),
                                 (char *)&v134,
                                 &v140);
      sub_BD84D0(v73, (__int64)v75);
      if ( (v75[7] & 0x10) == 0 )
        sub_BD6B90(v75, (unsigned __int8 *)v73);
      sub_B43D60((_QWORD *)v73);
LABEL_127:
      if ( v125 == ++v72 )
      {
        v10 = 1;
        goto LABEL_119;
      }
    }
    if ( v74 != 148 )
      BUG();
    v76 = *(_QWORD *)(v73 - 32LL * (*(_DWORD *)(v73 + 4) & 0x7FFFFFF));
    v104 = v76;
    v117 = sub_2906F40(v76, (char *)&v134, &v140);
    v77 = *(_QWORD *)(v76 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v77 + 8) - 17 <= 1 )
      v77 = **(_QWORD **)(v77 + 16);
    v103 = sub_AE2980(v102, *(_DWORD *)(v77 + 8) >> 8)[1];
    v152 = sub_BD5C60(v73);
    v153 = &v161;
    v154 = &v162;
    v146 = (unsigned int *)v148;
    v161 = &unk_49DA100;
    v149 = 0;
    v150 = 0;
    v147 = 0x200000000LL;
    v155 = 0;
    v156 = 0;
    v157 = 512;
    v158 = 7;
    v159 = 0;
    v160 = 0;
    LOWORD(v151) = 0;
    v162 = &unk_49DA0B0;
    v149 = *(_QWORD *)(v73 + 40);
    v150 = v73 + 24;
    v78 = *(_QWORD *)sub_B46C60(v73);
    v132[0] = v78;
    if ( v78 && (sub_B96E90((__int64)v132, v78, 1), (v79 = v132[0]) != 0) )
    {
      v80 = v146;
      v81 = v147;
      v82 = &v146[4 * (unsigned int)v147];
      if ( v146 != v82 )
      {
        while ( *v80 )
        {
          v80 += 4;
          if ( v82 == v80 )
            goto LABEL_161;
        }
        *((_QWORD *)v80 + 1) = v132[0];
        goto LABEL_141;
      }
LABEL_161:
      if ( (unsigned int)v147 >= (unsigned __int64)HIDWORD(v147) )
      {
        v98 = (unsigned int)v147 + 1LL;
        v99 = v101 & 0xFFFFFFFF00000000LL;
        v101 &= 0xFFFFFFFF00000000LL;
        if ( HIDWORD(v147) < v98 )
        {
          v108 = v99;
          sub_C8D5F0((__int64)&v146, v148, (unsigned int)v147 + 1LL, 0x10u, v98, v99);
          v99 = v108;
          v82 = &v146[4 * (unsigned int)v147];
        }
        *(_QWORD *)v82 = v99;
        *((_QWORD *)v82 + 1) = v79;
        v79 = v132[0];
        LODWORD(v147) = v147 + 1;
      }
      else
      {
        if ( v82 )
        {
          *v82 = 0;
          *((_QWORD *)v82 + 1) = v79;
          v81 = v147;
          v79 = v132[0];
        }
        LODWORD(v147) = v81 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v146, 0);
      v79 = v132[0];
    }
    if ( !v79 )
    {
LABEL_144:
      sub_28FF1A0((__int64)v127, (__int64)v117, ".int", (void *)4, byte_3F871B3, 0);
      v131 = 260;
      v129[0] = (unsigned __int64)v127;
      v83 = sub_BCD140(v111, v103);
      if ( v83 == *((_QWORD *)v117 + 1) )
      {
        v86 = v117;
        goto LABEL_151;
      }
      v84 = (__int64 (__fastcall *)(__int64, unsigned int, char *, __int64))*((_QWORD *)*v153 + 15);
      if ( v84 == sub_920130 )
      {
        if ( (unsigned __int8)*v117 > 0x15u )
          goto LABEL_167;
        v106 = (__int64 **)v83;
        if ( (unsigned __int8)sub_AC4810(0x2Fu) )
          v85 = sub_ADAB70(47, (unsigned __int64)v117, v106, 0);
        else
          v85 = sub_AA93C0(0x2Fu, (unsigned __int64)v117, (__int64)v106);
        v83 = (__int64)v106;
        v86 = (_BYTE *)v85;
      }
      else
      {
        v107 = v83;
        v97 = v84((__int64)v153, 47u, v117, v83);
        v83 = v107;
        v86 = (_BYTE *)v97;
      }
      if ( v86 )
      {
LABEL_151:
        if ( (__int64 *)v127[0] != &v128 )
          j_j___libc_free_0(v127[0]);
        sub_28FF1A0((__int64)v129, v104, ".int", (void *)4, byte_3F871B3, 0);
        v133 = 260;
        v132[0] = v129;
        v87 = sub_BCD140(v111, v103);
        v88 = sub_10E0820((__int64 *)&v146, v104, v87, (__int64)v132);
        v89 = (_BYTE *)v88;
        if ( (__int64 *)v129[0] != &v130 )
        {
          v118 = (_BYTE *)v88;
          j_j___libc_free_0(v129[0]);
          v89 = v118;
        }
        v133 = 257;
        v90 = (unsigned __int8 *)sub_929DE0(&v146, v89, v86, (__int64)v132, 0, 0);
        sub_BD84D0(v73, (__int64)v90);
        sub_BD6B90(v90, (unsigned __int8 *)v73);
        sub_B43D60((_QWORD *)v73);
        nullsub_61();
        v161 = &unk_49DA100;
        nullsub_63();
        if ( v146 != (unsigned int *)v148 )
          _libc_free((unsigned __int64)v146);
        goto LABEL_127;
      }
LABEL_167:
      v133 = 257;
      v86 = (_BYTE *)sub_B51D30(47, (__int64)v117, v83, (__int64)v132, 0, 0);
      if ( (unsigned __int8)sub_920620((__int64)v86) )
      {
        v91 = v156;
        if ( v155 )
        {
          v119 = v156;
          sub_B99FD0((__int64)v86, 3u, v155);
          v91 = v119;
        }
        sub_B45150((__int64)v86, v91);
      }
      (*((void (__fastcall **)(void **, _BYTE *, unsigned __int64 *, __int64, __int64))*v154 + 2))(
        v154,
        v86,
        v129,
        v150,
        v151);
      v92 = 4LL * (unsigned int)v147;
      if ( v146 != &v146[v92] )
      {
        v120 = v73;
        v93 = v146;
        v100 = v72;
        v94 = &v146[v92];
        do
        {
          v95 = *((_QWORD *)v93 + 1);
          v96 = *v93;
          v93 += 4;
          sub_B99FD0((__int64)v86, v96, v95);
        }
        while ( v94 != v93 );
        v73 = v120;
        v72 = v100;
      }
      goto LABEL_151;
    }
LABEL_141:
    sub_B91220((__int64)v132, v79);
    goto LABEL_144;
  }
LABEL_35:
  if ( v166 != (__int64 *)v168 )
    _libc_free((unsigned __int64)v166);
  if ( v163 != v165 )
    _libc_free((unsigned __int64)v163);
  sub_FFCE90((__int64)v169, (__int64)v8, v15, v19, v16, v17);
  sub_FFD870((__int64)v169, (__int64)v8, v30, v31, v32, v33);
  sub_FFBC40((__int64)v169, (__int64)v8);
  v34 = v184;
  v35 = v183;
  if ( v184 != v183 )
  {
    do
    {
      v36 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v35[7];
      *v35 = &unk_49E5048;
      if ( v36 )
        v36(v35 + 5, v35 + 5, 3);
      v37 = v35 + 1;
      v35 += 9;
      *(v35 - 9) = &unk_49DB368;
      sub_D68D70(v37);
    }
    while ( v34 != v35 );
    v35 = v183;
  }
  if ( v35 )
    j_j___libc_free_0((unsigned __int64)v35);
  if ( !v180 )
    _libc_free((unsigned __int64)v177);
  if ( (_BYTE *)v169[0] != v170 )
    _libc_free(v169[0]);
  return v10;
}
