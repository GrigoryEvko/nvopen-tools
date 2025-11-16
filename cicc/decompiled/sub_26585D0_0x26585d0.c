// Function: sub_26585D0
// Address: 0x26585d0
//
__int64 __fastcall sub_26585D0(
        __int64 a1,
        __int64 **a2,
        __int64 a3,
        __m128i a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 *a10)
{
  __int64 result; // rax
  __int64 v11; // r12
  unsigned __int64 v12; // rbx
  unsigned __int64 *v13; // r13
  __int64 v14; // r14
  unsigned __int64 v15; // rax
  const __m128i *v16; // r11
  unsigned __int64 v17; // rcx
  __int8 *v18; // rax
  unsigned __int64 v19; // rdx
  __m128i *v20; // rsi
  char *v21; // r13
  char *v22; // r14
  unsigned __int64 v23; // rax
  char *v24; // r11
  char *i; // rsi
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdx
  char *v28; // rcx
  char *v29; // rax
  __int64 v30; // r8
  __int64 v31; // rdx
  char *v32; // r13
  char *v33; // r14
  unsigned __int64 v34; // rax
  char *v35; // r11
  char *j; // rsi
  unsigned __int64 v37; // rdi
  unsigned __int64 v38; // rdx
  char *v39; // rcx
  char *v40; // rax
  __int64 v41; // r8
  __int64 v42; // rdx
  char *v43; // rcx
  char *v44; // rax
  __int64 v45; // rsi
  char *v46; // rdx
  char *v47; // rax
  __int64 v48; // rsi
  char *v49; // rdx
  char *v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rdi
  char *v53; // rsi
  char *v54; // rax
  __int64 v55; // r8
  char *v56; // rdx
  _QWORD *v57; // rsi
  __int64 v58; // rax
  __int64 v59; // rcx
  unsigned __int64 *v60; // rdx
  __int64 v61; // rax
  __int64 v62; // r12
  unsigned __int8 *v63; // r15
  __int64 v64; // rax
  __int64 v65; // r9
  __int64 v66; // rbx
  __int64 v67; // rax
  bool v68; // zf
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // r13
  unsigned __int8 *v72; // rax
  __int64 v73; // r13
  __int64 v74; // rax
  unsigned __int64 *v75; // rbx
  unsigned __int64 *v76; // r13
  unsigned __int64 v77; // rdi
  __int64 v78; // rbx
  unsigned int v79; // esi
  int v80; // edx
  __int64 v81; // r15
  __int64 v82; // rax
  __int64 v83; // rdx
  __int64 v84; // rax
  _QWORD *v85; // r15
  __int64 v86; // r13
  __m128i v87; // rax
  __int64 v88; // rcx
  __int64 v89; // r8
  __int64 v90; // r9
  __int64 v91; // rdx
  __int64 v92; // r8
  __int64 v93; // rcx
  __int64 v94; // rax
  __int64 v95; // rdi
  __int64 v96; // r12
  __int64 v97; // rax
  __int64 v98; // rbx
  __int64 v99; // rdx
  __int64 v100; // rcx
  __int64 v101; // r8
  __m128i v102; // xmm3
  __m128i v103; // xmm5
  __int64 v104; // r9
  unsigned __int64 *v105; // rbx
  unsigned __int64 *v106; // r13
  unsigned __int64 v107; // rdi
  unsigned __int64 *v108; // rbx
  unsigned __int64 *v109; // r13
  unsigned __int64 v110; // rdi
  int v111; // ebx
  __int64 *v112; // r12
  __int64 v113; // r15
  __int64 v114; // r13
  __int64 v115; // rdx
  _QWORD *v116; // r15
  __int64 v117; // r13
  __int64 v118; // rax
  __int64 v119; // r12
  __int64 v120; // rbx
  __int8 *v121; // r12
  size_t v122; // rdx
  __int64 v123; // rdx
  __int64 v124; // rcx
  __int64 v125; // r9
  __m128i v126; // xmm6
  __m128i v127; // xmm6
  __int64 v128; // r8
  unsigned __int64 *v129; // rbx
  unsigned __int64 *v130; // r13
  unsigned __int64 v131; // rdi
  unsigned __int64 *v132; // rbx
  unsigned __int64 v133; // rdi
  unsigned int v134; // esi
  int v135; // eax
  int v136; // eax
  __int64 v137; // r15
  __int64 v138; // rsi
  unsigned __int64 *v139; // r8
  __int64 v140; // rsi
  int v141; // r15d
  __int64 v142; // r11
  int v143; // eax
  __int64 v144; // rax
  __int64 v145; // rax
  __int64 v146; // rax
  __int64 v147; // rax
  __int64 v149; // [rsp+8h] [rbp-518h]
  __int64 v150; // [rsp+10h] [rbp-510h]
  unsigned __int64 *v151; // [rsp+20h] [rbp-500h]
  __int64 *v153; // [rsp+30h] [rbp-4F0h]
  unsigned int v154; // [rsp+38h] [rbp-4E8h]
  int v155; // [rsp+3Ch] [rbp-4E4h]
  __int64 v156; // [rsp+40h] [rbp-4E0h]
  __int64 v158; // [rsp+B0h] [rbp-470h]
  unsigned __int64 v159; // [rsp+B8h] [rbp-468h]
  __int64 v160; // [rsp+C0h] [rbp-460h]
  __int64 v161; // [rsp+C8h] [rbp-458h]
  unsigned __int8 *v162; // [rsp+D0h] [rbp-450h]
  __int32 v163; // [rsp+D0h] [rbp-450h]
  __int64 v164; // [rsp+D0h] [rbp-450h]
  unsigned __int64 *v165; // [rsp+D8h] [rbp-448h]
  char *s; // [rsp+E8h] [rbp-438h] BYREF
  __int64 v167[2]; // [rsp+F0h] [rbp-430h] BYREF
  __int64 v168; // [rsp+100h] [rbp-420h] BYREF
  __int64 *v169; // [rsp+110h] [rbp-410h] BYREF
  __int64 v170; // [rsp+120h] [rbp-400h] BYREF
  __int64 v171[2]; // [rsp+140h] [rbp-3E0h] BYREF
  __int64 v172; // [rsp+150h] [rbp-3D0h] BYREF
  unsigned __int64 v173[2]; // [rsp+160h] [rbp-3C0h] BYREF
  __int64 v174; // [rsp+170h] [rbp-3B0h] BYREF
  __int64 *v175; // [rsp+190h] [rbp-390h] BYREF
  unsigned __int64 v176; // [rsp+198h] [rbp-388h] BYREF
  __int64 v177; // [rsp+1A0h] [rbp-380h] BYREF
  __m128i v178; // [rsp+1A8h] [rbp-378h]
  __int64 v179; // [rsp+1B8h] [rbp-368h]
  __m128i v180; // [rsp+1C0h] [rbp-360h] BYREF
  __m128i v181; // [rsp+1D0h] [rbp-350h]
  unsigned __int64 *v182; // [rsp+1E0h] [rbp-340h] BYREF
  __int64 v183; // [rsp+1E8h] [rbp-338h]
  _BYTE v184[320]; // [rsp+1F0h] [rbp-330h] BYREF
  char v185; // [rsp+330h] [rbp-1F0h]
  int v186; // [rsp+334h] [rbp-1ECh]
  __int64 v187; // [rsp+338h] [rbp-1E8h]
  __m128i v188; // [rsp+340h] [rbp-1E0h] BYREF
  __int64 v189; // [rsp+350h] [rbp-1D0h]
  __int64 v190; // [rsp+358h] [rbp-1C8h]
  __int64 v191; // [rsp+360h] [rbp-1C0h]
  unsigned __int64 *v192; // [rsp+390h] [rbp-190h]
  unsigned int v193; // [rsp+398h] [rbp-188h]
  _BYTE v194[384]; // [rsp+3A0h] [rbp-180h] BYREF

  v153 = (__int64 *)(a6 * 8);
  result = a8 + 56 * a9;
  v150 = a8;
  v149 = result;
  if ( a8 != result )
  {
LABEL_2:
    v156 = *(_QWORD *)v150;
    v159 = *(_QWORD *)(v150 + 40);
    v165 = *(unsigned __int64 **)(v150 + 8);
    v151 = *(unsigned __int64 **)(v150 + 16);
    if ( v165 == v151 )
      goto LABEL_142;
    v154 = 0;
    v158 = a3 + 136LL * *(_QWORD *)(v150 + 48);
    while ( 1 )
    {
      v155 = *(_DWORD *)(v158 + 16);
      v11 = *(_QWORD *)(a1 + 24);
      v12 = *v165;
      if ( !*(_BYTE *)(v11 + 392) )
      {
        v13 = *(unsigned __int64 **)(v11 + 80);
        v14 = *(_QWORD *)(v11 + 72);
        if ( v13 != (unsigned __int64 *)v14 )
        {
          _BitScanReverse64(&v15, 0xAAAAAAAAAAAAAAABLL * (((__int64)v13 - v14) >> 3));
          sub_ED4580(
            *(_QWORD *)(v11 + 72),
            *(__m128i **)(v11 + 80),
            2LL * (int)(63 - (v15 ^ 0x3F)),
            0xAAAAAAAAAAAAAAABLL,
            a6 * 8,
            a7);
          if ( (__int64)v13 - v14 <= 384 )
          {
            sub_263F380(v14, v13);
          }
          else
          {
            sub_263F380(v14, (unsigned __int64 *)(v14 + 384));
            for ( ;
                  v13 != (unsigned __int64 *)v16;
                  *(__m128i *)((char *)v20 + 8) = _mm_loadu_si128((const __m128i *)&v188.m128i_u64[1]) )
            {
              v188 = _mm_loadu_si128(v16);
              v189 = v16[1].m128i_i64[0];
              v17 = v16->m128i_i64[0];
              v18 = &v16[-2].m128i_i8[8];
              v19 = v16[-2].m128i_u64[1];
              if ( v16->m128i_i64[0] >= v19 )
              {
                v20 = (__m128i *)v16;
              }
              else
              {
                do
                {
                  a4 = _mm_loadu_si128((const __m128i *)(v18 + 8));
                  *((_QWORD *)v18 + 3) = v19;
                  v20 = (__m128i *)v18;
                  v18 -= 24;
                  *(__m128i *)(v18 + 56) = a4;
                  v19 = *(_QWORD *)v18;
                }
                while ( v17 < *(_QWORD *)v18 );
              }
              v20->m128i_i64[0] = v17;
              v16 = (const __m128i *)((char *)v16 + 24);
            }
          }
        }
        v21 = *(char **)(v11 + 104);
        v22 = *(char **)(v11 + 96);
        if ( v21 != v22 )
        {
          _BitScanReverse64(&v23, (v21 - v22) >> 4);
          sub_ED48E0(*(char **)(v11 + 96), *(char **)(v11 + 104), 2LL * (int)(63 - (v23 ^ 0x3F)));
          if ( v21 - v22 <= 256 )
          {
            sub_263F2D0(v22, v21);
          }
          else
          {
            sub_263F2D0(v22, v22 + 256);
            for ( i = v24; v21 != i; *((_QWORD *)v28 + 1) = v30 )
            {
              v26 = *(_QWORD *)i;
              v27 = *((_QWORD *)i - 2);
              v28 = i;
              v29 = i - 16;
              v30 = *((_QWORD *)i + 1);
              if ( *(_QWORD *)i < v27 )
              {
                do
                {
                  *((_QWORD *)v29 + 2) = v27;
                  v31 = *((_QWORD *)v29 + 1);
                  v28 = v29;
                  v29 -= 16;
                  *((_QWORD *)v29 + 5) = v31;
                  v27 = *(_QWORD *)v29;
                }
                while ( v26 < *(_QWORD *)v29 );
              }
              i += 16;
              *(_QWORD *)v28 = v26;
            }
          }
        }
        v32 = *(char **)(v11 + 160);
        v33 = *(char **)(v11 + 152);
        if ( v32 != v33 )
        {
          _BitScanReverse64(&v34, (v32 - v33) >> 4);
          sub_ED4B00(*(char **)(v11 + 152), *(char **)(v11 + 160), 2LL * (int)(63 - (v34 ^ 0x3F)));
          if ( v32 - v33 <= 256 )
          {
            sub_263F220(v33, v32);
          }
          else
          {
            sub_263F220(v33, v33 + 256);
            for ( j = v35; v32 != j; *((_QWORD *)v39 + 1) = v41 )
            {
              v37 = *(_QWORD *)j;
              v38 = *((_QWORD *)j - 2);
              v39 = j;
              v40 = j - 16;
              v41 = *((_QWORD *)j + 1);
              if ( *(_QWORD *)j < v38 )
              {
                do
                {
                  *((_QWORD *)v40 + 2) = v38;
                  v42 = *((_QWORD *)v40 + 1);
                  v39 = v40;
                  v40 -= 16;
                  *((_QWORD *)v40 + 5) = v42;
                  v38 = *(_QWORD *)v40;
                }
                while ( v37 < *(_QWORD *)v40 );
              }
              j += 16;
              *(_QWORD *)v39 = v37;
            }
          }
          v43 = *(char **)(v11 + 160);
          v44 = *(char **)(v11 + 152);
          if ( v44 != v43 )
          {
            do
            {
              v44 += 16;
              if ( v43 == v44 )
                goto LABEL_43;
              v45 = *((_QWORD *)v44 - 2);
              v46 = v44 - 16;
            }
            while ( v45 != *(_QWORD *)v44 || *((_QWORD *)v44 - 1) != *((_QWORD *)v44 + 1) );
            if ( v46 != v43 )
            {
              v47 = v44 + 16;
              if ( v43 != v47 )
              {
                while ( 1 )
                {
                  if ( *(_QWORD *)v47 == v45 && *((_QWORD *)v46 + 1) == *((_QWORD *)v47 + 1) )
                  {
                    v47 += 16;
                    if ( v43 == v47 )
                      break;
                  }
                  else
                  {
                    *((_QWORD *)v46 + 2) = *(_QWORD *)v47;
                    v48 = *((_QWORD *)v47 + 1);
                    v47 += 16;
                    v46 += 16;
                    *((_QWORD *)v46 + 1) = v48;
                    if ( v43 == v47 )
                      break;
                  }
                  v45 = *(_QWORD *)v46;
                }
              }
              v49 = v46 + 16;
              if ( v49 != v43 )
              {
                v50 = *(char **)(v11 + 160);
                v51 = v50 - v43;
                if ( v50 == v43 )
                {
                  v56 = &v49[v51];
LABEL_42:
                  *(_QWORD *)(v11 + 160) = v56;
                  goto LABEL_43;
                }
                v52 = v51 >> 4;
                if ( v51 > 0 )
                {
                  v53 = v49;
                  v54 = v43;
                  do
                  {
                    v55 = *(_QWORD *)v54;
                    v53 += 16;
                    v54 += 16;
                    *((_QWORD *)v53 - 2) = v55;
                    *((_QWORD *)v53 - 1) = *((_QWORD *)v54 - 1);
                    --v52;
                  }
                  while ( v52 );
                  v50 = *(char **)(v11 + 160);
                  v51 = v50 - v43;
                }
                v56 = &v49[v51];
                if ( v56 != v50 )
                  goto LABEL_42;
              }
            }
          }
        }
LABEL_43:
        *(_BYTE *)(v11 + 392) = 1;
      }
      v57 = *(_QWORD **)(v11 + 96);
      v58 = (__int64)(*(_QWORD *)(v11 + 104) - (_QWORD)v57) >> 4;
      if ( (__int64)(*(_QWORD *)(v11 + 104) - (_QWORD)v57) > 0 )
      {
        do
        {
          while ( 1 )
          {
            v59 = v58 >> 1;
            v60 = &v57[2 * (v58 >> 1)];
            if ( v12 <= *v60 )
              break;
            v57 = v60 + 2;
            v58 = v58 - v59 - 1;
            if ( v58 <= 0 )
              goto LABEL_49;
          }
          v58 >>= 1;
        }
        while ( v59 > 0 );
      }
LABEL_49:
      if ( *(_QWORD **)(v11 + 104) == v57
        || v12 != *v57
        || (v61 = v57[1], (v161 = v61) == 0)
        || (_BYTE)qword_4FF32A8 && sub_B2FC80(v61) )
      {
        v96 = *a10;
        v97 = sub_B2BE50(*a10);
        if ( sub_B6EA50(v97)
          || (v146 = sub_B2BE50(v96),
              v147 = sub_B6F970(v146),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v147 + 48LL))(v147)) )
        {
          sub_B176B0((__int64)&v188, (__int64)"memprof-context-disambiguation", (__int64)"UnableToFindTarget", 18, v156);
          sub_B18290((__int64)&v188, "Memprof cannot promote indirect call: target with md5sum ", 0x39u);
          sub_B16B10(v171, "target md5sum", 13, *v165);
          v98 = sub_2445430((__int64)&v188, (__int64)v171);
          sub_B18290(v98, " not found", 0xAu);
          LODWORD(v176) = *(_DWORD *)(v98 + 8);
          BYTE4(v176) = *(_BYTE *)(v98 + 12);
          v177 = *(_QWORD *)(v98 + 16);
          v102 = _mm_loadu_si128((const __m128i *)(v98 + 24));
          v175 = (__int64 *)&unk_49D9D40;
          v178 = v102;
          v179 = *(_QWORD *)(v98 + 40);
          v180 = _mm_loadu_si128((const __m128i *)(v98 + 48));
          v103 = _mm_loadu_si128((const __m128i *)(v98 + 64));
          v182 = (unsigned __int64 *)v184;
          v183 = 0x400000000LL;
          v181 = v103;
          v104 = *(unsigned int *)(v98 + 88);
          if ( (_DWORD)v104 )
            sub_264ACC0((__int64)&v182, v98 + 80, v99, v100, v101, v104);
          v185 = *(_BYTE *)(v98 + 416);
          v186 = *(_DWORD *)(v98 + 420);
          v187 = *(_QWORD *)(v98 + 424);
          v175 = (__int64 *)&unk_49D9DB0;
          sub_2240A30(v173);
          sub_2240A30((unsigned __int64 *)v171);
          v105 = v192;
          v188.m128i_i64[0] = (__int64)&unk_49D9D40;
          v106 = &v192[10 * v193];
          if ( v192 != v106 )
          {
            do
            {
              v106 -= 10;
              v107 = v106[4];
              if ( (unsigned __int64 *)v107 != v106 + 6 )
                j_j___libc_free_0(v107);
              if ( (unsigned __int64 *)*v106 != v106 + 2 )
                j_j___libc_free_0(*v106);
            }
            while ( v105 != v106 );
            v106 = v192;
          }
          if ( v106 != (unsigned __int64 *)v194 )
            _libc_free((unsigned __int64)v106);
          sub_1049740(a10, (__int64)&v175);
          v108 = v182;
          v175 = (__int64 *)&unk_49D9D40;
          v109 = &v182[10 * (unsigned int)v183];
          if ( v182 != v109 )
          {
            do
            {
              v109 -= 10;
              v110 = v109[4];
              if ( (unsigned __int64 *)v110 != v109 + 6 )
                j_j___libc_free_0(v110);
              if ( (unsigned __int64 *)*v109 != v109 + 2 )
                j_j___libc_free_0(*v109);
            }
            while ( v108 != v109 );
LABEL_133:
            v109 = v182;
          }
LABEL_134:
          if ( v109 != (unsigned __int64 *)v184 )
            _libc_free((unsigned __int64)v109);
        }
      }
      else
      {
        s = 0;
        if ( (unsigned __int8)sub_29A3A40(v156, v161, &s) )
        {
          if ( v155 )
          {
            v62 = 0;
            v63 = (unsigned __int8 *)v156;
            v160 = v156;
            while ( 1 )
            {
              v64 = sub_2445EC0((__int64)v63, (unsigned __int8 *)v161, v165[1], v159, *(_BYTE *)(a1 + 16), a10);
              v65 = v161;
              v66 = v64;
              v67 = *(_QWORD *)(v158 + 8);
              if ( *(_DWORD *)(v67 + 4 * v62) )
              {
                v163 = *(_DWORD *)(v67 + 4 * v62);
                v86 = *(_QWORD *)(v161 + 24);
                v87.m128i_i64[0] = (__int64)sub_BD5D20(v161);
                LOWORD(v191) = 261;
                v188 = v87;
                sub_2644DA0((__int64 *)&v175, v163, v87.m128i_i64[1], v88, v89, v90, a4);
                sub_BA8CA0((__int64)a2, (__int64)v175, v176, v86);
                v65 = v91;
                if ( v175 != &v177 )
                {
                  v164 = v91;
                  j_j___libc_free_0((unsigned __int64)v175);
                  v65 = v164;
                }
              }
              v68 = *(_QWORD *)(v66 - 32) == 0;
              *(_QWORD *)(v66 + 80) = *(_QWORD *)(v65 + 24);
              if ( !v68 )
              {
                v69 = *(_QWORD *)(v66 - 24);
                **(_QWORD **)(v66 - 16) = v69;
                if ( v69 )
                  *(_QWORD *)(v69 + 16) = *(_QWORD *)(v66 - 16);
              }
              *(_QWORD *)(v66 - 32) = v65;
              v70 = *(_QWORD *)(v65 + 16);
              *(_QWORD *)(v66 - 24) = v70;
              if ( v70 )
                *(_QWORD *)(v70 + 16) = v66 - 24;
              *(_QWORD *)(v66 - 16) = v65 + 16;
              *(_QWORD *)(v65 + 16) = v66 - 32;
              v162 = (unsigned __int8 *)v65;
              sub_B174A0(
                (__int64)&v188,
                (__int64)"memprof-context-disambiguation",
                (__int64)"MemprofCall",
                11,
                (__int64)v63);
              sub_B16080((__int64)v167, "Call", 4, v63);
              v71 = sub_2647050((__int64)&v188, (__int64)v167);
              sub_B18290(v71, " in clone ", 0xAu);
              v72 = (unsigned __int8 *)sub_B43CB0(v160);
              sub_B16080((__int64)v171, "Caller", 6, v72);
              v73 = sub_23FD640(v71, (__int64)v171);
              sub_B18290(v73, " promoted and assigned to call function clone ", 0x2Eu);
              sub_B16080((__int64)&v175, "Callee", 6, v162);
              v74 = sub_23FD640(v73, (__int64)&v175);
              sub_1049740(a10, v74);
              if ( (__m128i *)v178.m128i_i64[1] != &v180 )
                j_j___libc_free_0(v178.m128i_u64[1]);
              if ( v175 != &v177 )
                j_j___libc_free_0((unsigned __int64)v175);
              if ( (__int64 *)v173[0] != &v174 )
                j_j___libc_free_0(v173[0]);
              if ( (__int64 *)v171[0] != &v172 )
                j_j___libc_free_0(v171[0]);
              if ( v169 != &v170 )
                j_j___libc_free_0((unsigned __int64)v169);
              if ( (__int64 *)v167[0] != &v168 )
                j_j___libc_free_0(v167[0]);
              v75 = v192;
              v188.m128i_i64[0] = (__int64)&unk_49D9D40;
              a6 = 10LL * v193;
              v76 = &v192[a6];
              if ( v192 != &v192[a6] )
              {
                do
                {
                  v76 -= 10;
                  v77 = v76[4];
                  if ( (unsigned __int64 *)v77 != v76 + 6 )
                    j_j___libc_free_0(v77);
                  if ( (unsigned __int64 *)*v76 != v76 + 2 )
                    j_j___libc_free_0(*v76);
                }
                while ( v75 != v76 );
                v76 = v192;
              }
              if ( v76 != (unsigned __int64 *)v194 )
                _libc_free((unsigned __int64)v76);
              if ( v155 - 1 == v62 )
                goto LABEL_153;
              if ( (_DWORD)v62 != -1 )
                break;
LABEL_110:
              ++v62;
            }
            v176 = 2;
            v177 = 0;
            v78 = v153[v62];
            v178.m128i_i64[0] = v156;
            if ( v156 != -4096 && v156 != 0 && v156 != -8192 )
              sub_BD73F0((__int64)&v176);
            v178.m128i_i64[1] = v78;
            v175 = (__int64 *)&unk_49DD7B0;
            v79 = *(_DWORD *)(v78 + 24);
            if ( v79 )
            {
              v83 = v178.m128i_i64[0];
              v92 = *(_QWORD *)(v78 + 8);
              LODWORD(v93) = (v79 - 1)
                           & (((unsigned __int32)v178.m128i_i32[0] >> 9)
                            ^ ((unsigned __int32)v178.m128i_i32[0] >> 4));
              v94 = v92 + ((unsigned __int64)(unsigned int)v93 << 6);
              v95 = *(_QWORD *)(v94 + 24);
              if ( v178.m128i_i64[0] == v95 )
              {
LABEL_114:
                v85 = (_QWORD *)(v94 + 40);
LABEL_106:
                v175 = (__int64 *)&unk_49DB368;
                if ( v83 != -4096 && v83 != 0 && v83 != -8192 )
                  sub_BD60C0(&v176);
                v63 = (unsigned __int8 *)v85[2];
                v160 = (__int64)v63;
                goto LABEL_110;
              }
              v141 = 1;
              v142 = 0;
              while ( v95 != -4096 )
              {
                if ( v95 == -8192 && !v142 )
                  v142 = v94;
                v93 = (v79 - 1) & ((_DWORD)v93 + v141);
                v94 = v92 + (v93 << 6);
                v95 = *(_QWORD *)(v94 + 24);
                if ( v178.m128i_i64[0] == v95 )
                  goto LABEL_114;
                ++v141;
              }
              if ( !v142 )
                v142 = v94;
              v171[0] = v142;
              v143 = *(_DWORD *)(v78 + 16);
              ++*(_QWORD *)v78;
              v80 = v143 + 1;
              if ( 4 * (v143 + 1) < 3 * v79 )
              {
                if ( v79 - *(_DWORD *)(v78 + 20) - v80 > v79 >> 3 )
                {
LABEL_93:
                  *(_DWORD *)(v78 + 16) = v80;
                  v81 = v171[0];
                  v188.m128i_i64[1] = 2;
                  v189 = 0;
                  v190 = -4096;
                  v191 = 0;
                  if ( *(_QWORD *)(v171[0] + 24) != -4096 )
                  {
                    --*(_DWORD *)(v78 + 20);
                    v188.m128i_i64[0] = (__int64)&unk_49DB368;
                    if ( v190 != -8192 && v190 != -4096 )
                    {
                      if ( v190 )
                        sub_BD60C0(&v188.m128i_i64[1]);
                    }
                  }
                  v82 = *(_QWORD *)(v81 + 24);
                  v83 = v178.m128i_i64[0];
                  if ( v82 != v178.m128i_i64[0] )
                  {
                    if ( v82 != 0 && v82 != -4096 && v82 != -8192 )
                    {
                      sub_BD60C0((_QWORD *)(v81 + 8));
                      v83 = v178.m128i_i64[0];
                    }
                    *(_QWORD *)(v81 + 24) = v83;
                    if ( v83 != 0 && v83 != -4096 && v83 != -8192 )
                      sub_BD6050((unsigned __int64 *)(v81 + 8), v176 & 0xFFFFFFFFFFFFFFF8LL);
                    v83 = v178.m128i_i64[0];
                  }
                  v84 = v178.m128i_i64[1];
                  v85 = (_QWORD *)(v81 + 40);
                  *v85 = 6;
                  v85[1] = 0;
                  *(v85 - 1) = v84;
                  v85[2] = 0;
                  goto LABEL_106;
                }
LABEL_92:
                sub_CF32C0(v78, v79);
                sub_F9E960(v78, (__int64)&v175, v171);
                v80 = *(_DWORD *)(v78 + 16) + 1;
                goto LABEL_93;
              }
            }
            else
            {
              v171[0] = 0;
              ++*(_QWORD *)v78;
            }
            v79 *= 2;
            goto LABEL_92;
          }
LABEL_153:
          ++v154;
          v159 -= v165[1];
          goto LABEL_136;
        }
        v117 = *a10;
        v118 = sub_B2BE50(*a10);
        if ( sub_B6EA50(v118)
          || (v144 = sub_B2BE50(v117),
              v145 = sub_B6F970(v144),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v145 + 48LL))(v145)) )
        {
          sub_B176B0((__int64)&v188, (__int64)"memprof-context-disambiguation", (__int64)"UnableToPromote", 15, v156);
          sub_B18290((__int64)&v188, "Memprof cannot promote indirect call to ", 0x28u);
          sub_B16080((__int64)v171, "TargetFunction", 14, (unsigned __int8 *)v161);
          v119 = sub_2445430((__int64)&v188, (__int64)v171);
          sub_B18290(v119, " with count of ", 0xFu);
          sub_B16B10(v167, "TotalCount", 10, v159);
          v120 = sub_2445430(v119, (__int64)v167);
          sub_B18290(v120, ": ", 2u);
          v121 = s;
          v122 = 0;
          if ( s )
            v122 = strlen(s);
          sub_B18290(v120, v121, v122);
          LODWORD(v176) = *(_DWORD *)(v120 + 8);
          BYTE4(v176) = *(_BYTE *)(v120 + 12);
          v177 = *(_QWORD *)(v120 + 16);
          v126 = _mm_loadu_si128((const __m128i *)(v120 + 24));
          v175 = (__int64 *)&unk_49D9D40;
          v178 = v126;
          v179 = *(_QWORD *)(v120 + 40);
          v180 = _mm_loadu_si128((const __m128i *)(v120 + 48));
          v127 = _mm_loadu_si128((const __m128i *)(v120 + 64));
          v182 = (unsigned __int64 *)v184;
          v183 = 0x400000000LL;
          v181 = v127;
          v128 = *(unsigned int *)(v120 + 88);
          if ( (_DWORD)v128 )
            sub_264ACC0((__int64)&v182, v120 + 80, v123, v124, v128, v125);
          v185 = *(_BYTE *)(v120 + 416);
          v186 = *(_DWORD *)(v120 + 420);
          v187 = *(_QWORD *)(v120 + 424);
          v175 = (__int64 *)&unk_49D9DB0;
          sub_2240A30((unsigned __int64 *)&v169);
          sub_2240A30((unsigned __int64 *)v167);
          sub_2240A30(v173);
          sub_2240A30((unsigned __int64 *)v171);
          v129 = v192;
          v188.m128i_i64[0] = (__int64)&unk_49D9D40;
          v130 = &v192[10 * v193];
          if ( v192 != v130 )
          {
            do
            {
              v130 -= 10;
              v131 = v130[4];
              if ( (unsigned __int64 *)v131 != v130 + 6 )
                j_j___libc_free_0(v131);
              if ( (unsigned __int64 *)*v130 != v130 + 2 )
                j_j___libc_free_0(*v130);
            }
            while ( v129 != v130 );
            v130 = v192;
          }
          if ( v130 != (unsigned __int64 *)v194 )
            _libc_free((unsigned __int64)v130);
          sub_1049740(a10, (__int64)&v175);
          v132 = v182;
          v175 = (__int64 *)&unk_49D9D40;
          v109 = &v182[10 * (unsigned int)v183];
          if ( v182 != v109 )
          {
            do
            {
              v109 -= 10;
              v133 = v109[4];
              if ( (unsigned __int64 *)v133 != v109 + 6 )
                j_j___libc_free_0(v133);
              if ( (unsigned __int64 *)*v109 != v109 + 2 )
                j_j___libc_free_0(*v109);
            }
            while ( v132 != v109 );
            goto LABEL_133;
          }
          goto LABEL_134;
        }
      }
LABEL_136:
      v165 += 2;
      v158 += 136;
      if ( v151 == v165 )
      {
        if ( v155 )
        {
          v111 = 0;
          v112 = v153;
          v113 = v156;
          while ( 1 )
          {
            sub_B99FD0(v113, 2u, 0);
            if ( v159 )
              sub_ED2230(
                a2,
                v113,
                (__int64 *)(16LL * v154 + *(_QWORD *)(v150 + 8)),
                ((__int64)(*(_QWORD *)(v150 + 16) - *(_QWORD *)(v150 + 8)) >> 4) - v154,
                v159,
                0,
                *(_DWORD *)(v150 + 32));
            if ( v155 == ++v111 )
              break;
            v178.m128i_i64[0] = v156;
            v114 = *v112;
            v176 = 2;
            v177 = 0;
            if ( v156 != 0 && v156 != -4096 && v156 != -8192 )
              sub_BD73F0((__int64)&v176);
            v178.m128i_i64[1] = v114;
            v175 = (__int64 *)&unk_49DD7B0;
            if ( (unsigned __int8)sub_F9E960(v114, (__int64)&v175, v167) )
            {
              v115 = v178.m128i_i64[0];
              v116 = (_QWORD *)(v167[0] + 40);
              goto LABEL_149;
            }
            v171[0] = v167[0];
            v134 = *(_DWORD *)(v114 + 24);
            v135 = *(_DWORD *)(v114 + 16);
            ++*(_QWORD *)v114;
            v136 = v135 + 1;
            if ( 4 * v136 >= 3 * v134 )
            {
              v134 *= 2;
            }
            else if ( v134 - *(_DWORD *)(v114 + 20) - v136 > v134 >> 3 )
            {
              goto LABEL_178;
            }
            sub_CF32C0(v114, v134);
            sub_F9E960(v114, (__int64)&v175, v171);
            v136 = *(_DWORD *)(v114 + 16) + 1;
LABEL_178:
            *(_DWORD *)(v114 + 16) = v136;
            v137 = v171[0];
            v188.m128i_i64[1] = 2;
            v189 = 0;
            v190 = -4096;
            v191 = 0;
            if ( *(_QWORD *)(v171[0] + 24) != -4096 )
            {
              --*(_DWORD *)(v114 + 20);
              v188.m128i_i64[0] = (__int64)&unk_49DB368;
              if ( v190 != -4096 && v190 != 0 && v190 != -8192 )
                sub_BD60C0(&v188.m128i_i64[1]);
            }
            v138 = *(_QWORD *)(v137 + 24);
            v115 = v178.m128i_i64[0];
            if ( v138 != v178.m128i_i64[0] )
            {
              v139 = (unsigned __int64 *)(v137 + 8);
              if ( v138 != -4096 && v138 != 0 && v138 != -8192 )
              {
                sub_BD60C0((_QWORD *)(v137 + 8));
                v115 = v178.m128i_i64[0];
                v139 = (unsigned __int64 *)(v137 + 8);
              }
              *(_QWORD *)(v137 + 24) = v115;
              if ( v115 == 0 || v115 == -4096 || v115 == -8192 )
              {
                v115 = v178.m128i_i64[0];
              }
              else
              {
                sub_BD6050(v139, v176 & 0xFFFFFFFFFFFFFFF8LL);
                v115 = v178.m128i_i64[0];
              }
            }
            v140 = v178.m128i_i64[1];
            v116 = (_QWORD *)(v137 + 40);
            *v116 = 6;
            v116[1] = 0;
            *(v116 - 1) = v140;
            v116[2] = 0;
LABEL_149:
            v175 = (__int64 *)&unk_49DB368;
            if ( v115 != 0 && v115 != -4096 && v115 != -8192 )
              sub_BD60C0(&v176);
            v113 = v116[2];
            ++v112;
          }
        }
LABEL_142:
        v150 += 56;
        result = v150;
        if ( v149 == v150 )
          return result;
        goto LABEL_2;
      }
    }
  }
  return result;
}
