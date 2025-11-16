// Function: sub_266C6F0
// Address: 0x266c6f0
//
__int64 __fastcall sub_266C6F0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  int v8; // eax
  int v9; // esi
  __int128 v10; // xmm2
  int v11; // ecx
  __int128 v12; // xmm3
  __int128 v13; // xmm4
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // rdi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rsi
  int v21; // edi
  unsigned int i; // eax
  __int64 v23; // rcx
  unsigned int v24; // eax
  void (*v25)(); // rax
  __int64 *v26; // rax
  __int64 v27; // rax
  __int64 *v28; // rbx
  __int64 *v29; // r14
  __int64 *v30; // r12
  __int64 *v31; // r15
  __int64 *j; // rbx
  __int64 v33; // rcx
  __int64 v34; // r13
  __int64 *v35; // rax
  __int64 **v36; // r12
  __int64 **v37; // rbx
  __int64 v38; // rax
  void (__fastcall *v39)(__int64, __int128 *); // rdx
  __int64 v40; // r15
  unsigned int v41; // edx
  unsigned int v42; // ebx
  __int64 v43; // r12
  __int64 v44; // rax
  __int64 *v45; // rax
  _QWORD *v46; // r13
  __int64 v47; // rax
  void (*v48)(); // rdx
  void (__fastcall *v49)(_QWORD *); // rax
  __int64 v50; // rsi
  void (__fastcall *v51)(__int64, __int128 *); // rax
  __int64 v52; // rax
  unsigned __int8 *v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r9
  __m128i v57; // xmm7
  __m128i v58; // xmm2
  __m128i v59; // xmm3
  int v60; // r8d
  __int64 v61; // r8
  __int64 *v62; // r14
  unsigned __int64 *v63; // r12
  __int64 *v64; // r13
  unsigned __int64 *v65; // rbx
  unsigned __int64 v66; // rdi
  unsigned __int64 *v67; // r13
  unsigned __int64 *v68; // r14
  unsigned __int64 v69; // rdi
  __int64 v70; // r13
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  int v74; // eax
  unsigned __int64 v75; // rsi
  __int64 v76; // rax
  __int64 v77; // r8
  __int64 v78; // r9
  unsigned __int64 v79; // rdx
  __int64 v80; // rcx
  unsigned __int64 v81; // rdi
  __int64 v82; // rax
  void (*v83)(); // rax
  _QWORD *v84; // rbx
  _QWORD *v85; // r12
  __int64 v86; // rax
  __int64 v87; // rax
  _QWORD *v88; // rbx
  _QWORD *v89; // r15
  __int64 v90; // rax
  unsigned __int64 v91; // r13
  void (__fastcall *v92)(_QWORD *); // rax
  __int64 v93; // rsi
  __int64 *v94; // r8
  __int64 *v95; // r12
  __int64 *v96; // rbx
  _QWORD *v97; // r13
  const char *v98; // rax
  __int64 v99; // rdx
  unsigned __int64 *v100; // rcx
  unsigned __int64 v101; // rdx
  __int64 v102; // rsi
  __int64 v103; // rdx
  void (*v104)(); // rax
  __int64 v105; // rax
  __int64 v106; // rax
  __m128i *v107; // r14
  __int64 v108; // rsi
  unsigned __int64 *v109; // rbx
  __int64 *v110; // r13
  unsigned __int64 *v111; // r12
  __int64 v112; // rcx
  __int64 v113; // rax
  unsigned __int64 v114; // rbx
  __int64 v115; // rax
  __int64 *v116; // rax
  unsigned __int64 v117; // rbx
  _BYTE *v118; // r15
  __int64 *v119; // r12
  __int64 v120; // rdi
  void (__fastcall *v121)(__int64, __int64 **); // rax
  __int64 v122; // rbx
  size_t v123; // rdx
  void (__fastcall *v124)(__int64, bool (__fastcall *)(_QWORD *, __int64), __int64 **); // rax
  __int64 v125; // r8
  __int64 v126; // r9
  __int64 v127; // rax
  unsigned __int64 v128; // rdx
  unsigned __int64 v129; // [rsp+50h] [rbp-530h]
  __int64 v130; // [rsp+58h] [rbp-528h]
  int v131; // [rsp+6Ch] [rbp-514h]
  unsigned __int64 v132; // [rsp+70h] [rbp-510h]
  __int64 v133; // [rsp+80h] [rbp-500h]
  __int64 v134; // [rsp+90h] [rbp-4F0h]
  __int64 *v135; // [rsp+98h] [rbp-4E8h]
  int v136; // [rsp+98h] [rbp-4E8h]
  __int64 *v137; // [rsp+98h] [rbp-4E8h]
  __int64 v139; // [rsp+A8h] [rbp-4D8h]
  __int64 v140; // [rsp+A8h] [rbp-4D8h]
  __int64 v141; // [rsp+B0h] [rbp-4D0h]
  __int64 *v142; // [rsp+B8h] [rbp-4C8h]
  __int64 v143; // [rsp+C8h] [rbp-4B8h]
  __int64 v144; // [rsp+D0h] [rbp-4B0h]
  char v145; // [rsp+D0h] [rbp-4B0h]
  int v146; // [rsp+D0h] [rbp-4B0h]
  __int64 v147; // [rsp+D8h] [rbp-4A8h]
  __int64 v148; // [rsp+E8h] [rbp-498h]
  __int64 *v149; // [rsp+F8h] [rbp-488h]
  char v150; // [rsp+F8h] [rbp-488h]
  __int64 v151; // [rsp+100h] [rbp-480h] BYREF
  __int64 v152; // [rsp+108h] [rbp-478h] BYREF
  unsigned __int64 v153; // [rsp+110h] [rbp-470h] BYREF
  __int64 v154; // [rsp+118h] [rbp-468h] BYREF
  __int64 v155; // [rsp+120h] [rbp-460h] BYREF
  __int64 v156; // [rsp+128h] [rbp-458h]
  __int64 v157; // [rsp+130h] [rbp-450h]
  __int64 v158; // [rsp+138h] [rbp-448h]
  __int64 **v159; // [rsp+140h] [rbp-440h]
  __int64 v160; // [rsp+148h] [rbp-438h]
  __int64 *v161; // [rsp+150h] [rbp-430h] BYREF
  __int64 v162; // [rsp+158h] [rbp-428h]
  __int64 v163; // [rsp+160h] [rbp-420h] BYREF
  __int64 *v164; // [rsp+170h] [rbp-410h]
  __int64 v165; // [rsp+178h] [rbp-408h]
  __int64 v166; // [rsp+180h] [rbp-400h] BYREF
  __m128i v167; // [rsp+190h] [rbp-3F0h] BYREF
  __int64 *v168; // [rsp+1A0h] [rbp-3E0h] BYREF
  __int64 v169; // [rsp+1A8h] [rbp-3D8h]
  _QWORD v170[2]; // [rsp+1B0h] [rbp-3D0h] BYREF
  __int64 *v171; // [rsp+1C0h] [rbp-3C0h]
  __int64 v172; // [rsp+1C8h] [rbp-3B8h]
  __int64 v173; // [rsp+1D0h] [rbp-3B0h] BYREF
  __m128i v174; // [rsp+1E0h] [rbp-3A0h] BYREF
  __int64 *v175; // [rsp+1F0h] [rbp-390h] BYREF
  __int64 v176; // [rsp+1F8h] [rbp-388h]
  __int64 v177; // [rsp+200h] [rbp-380h] BYREF
  __m128i v178; // [rsp+208h] [rbp-378h] BYREF
  __m128i *v179; // [rsp+218h] [rbp-368h]
  __m128i v180; // [rsp+220h] [rbp-360h] BYREF
  __m128i v181; // [rsp+230h] [rbp-350h]
  __m128i *v182; // [rsp+240h] [rbp-340h] BYREF
  __int64 v183; // [rsp+248h] [rbp-338h]
  _BYTE v184[324]; // [rsp+250h] [rbp-330h] BYREF
  int v185; // [rsp+394h] [rbp-1ECh]
  __int64 v186; // [rsp+398h] [rbp-1E8h]
  __int128 v187; // [rsp+3A0h] [rbp-1E0h] BYREF
  __int128 v188; // [rsp+3B0h] [rbp-1D0h] BYREF
  __int64 v189; // [rsp+3C0h] [rbp-1C0h]
  __m128i *v190; // [rsp+3C8h] [rbp-1B8h]
  __m128i v191; // [rsp+3D0h] [rbp-1B0h] BYREF
  __m128i v192; // [rsp+3E0h] [rbp-1A0h] BYREF
  unsigned __int64 *v193; // [rsp+3F0h] [rbp-190h]
  _BYTE *v194; // [rsp+3F8h] [rbp-188h]
  __int64 v195; // [rsp+400h] [rbp-180h] BYREF
  _BYTE v196[192]; // [rsp+408h] [rbp-178h] BYREF
  _BYTE *v197; // [rsp+4C8h] [rbp-B8h]
  __int64 v198; // [rsp+4D0h] [rbp-B0h]
  _BYTE v199[108]; // [rsp+4D8h] [rbp-A8h] BYREF
  int v200; // [rsp+544h] [rbp-3Ch]
  __int64 v201; // [rsp+548h] [rbp-38h]

  v8 = sub_BC0510(a4, &unk_502F110, (__int64)a3);
  v9 = *(_DWORD *)(a2 + 92);
  v10 = (__int128)_mm_loadu_si128((const __m128i *)(a2 + 8));
  v11 = *(_DWORD *)(a2 + 96);
  v187 = 0;
  v12 = (__int128)_mm_loadu_si128((const __m128i *)(a2 + 24));
  v13 = (__int128)_mm_loadu_si128((const __m128i *)(a2 + 40));
  v188 = 0;
  if ( !(unsigned __int8)sub_30CC0B0(
                           v8 + 8,
                           v9,
                           (unsigned int)&v187,
                           v11,
                           v14,
                           v15,
                           v10,
                           v12,
                           v13,
                           *(_OWORD *)&_mm_loadu_si128((const __m128i *)(a2 + 56)),
                           *(_OWORD *)&_mm_loadu_si128((const __m128i *)(a2 + 72)),
                           *(_DWORD *)(a2 + 88)) )
  {
    v16 = *a3;
    *(_QWORD *)&v187 = "Could not setup Inlining Advisor for the requested mode and/or options";
    LOWORD(v189) = 259;
    sub_B6ECE0(v16, (__int64)&v187);
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v18 = sub_BC0510(a4, &unk_502E1A8, (__int64)a3);
  v19 = *(unsigned int *)(a4 + 88);
  v20 = *(_QWORD *)(a4 + 72);
  v143 = v18;
  v134 = v18 + 8;
  if ( !(_DWORD)v19 )
    goto LABEL_10;
  v21 = 1;
  for ( i = (v19 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F87C68 >> 9) ^ ((unsigned int)&unk_4F87C68 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v19 - 1) & v24 )
  {
    v23 = v20 + 24LL * i;
    if ( *(_UNKNOWN **)v23 == &unk_4F87C68 && a3 == *(__int64 **)(v23 + 8) )
      break;
    if ( *(_QWORD *)v23 == -4096 && *(_QWORD *)(v23 + 8) == -4096 )
      goto LABEL_10;
    v24 = v21 + i;
    ++v21;
  }
  if ( v23 == v20 + 24 * v19 )
  {
LABEL_10:
    v133 = 0;
  }
  else
  {
    v112 = *(_QWORD *)(*(_QWORD *)(v23 + 16) + 24LL);
    v113 = v112 + 8;
    if ( !v112 )
      v113 = 0;
    v133 = v113;
  }
  v148 = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, (__int64)a3) + 8);
  v141 = sub_266C480(a2, a4, v148, (__int64)a3);
  v25 = *(void (**)())(*(_QWORD *)v141 + 16LL);
  if ( v25 != nullsub_1536 )
    ((void (__fastcall *)(__int64, _QWORD))v25)(v141, 0);
  sub_30E2CB0(&v151, v148, a2 + 8, a4, a3);
  v155 = 0;
  v159 = &v161;
  v26 = (__int64 *)a3[4];
  v156 = 0;
  v157 = 0;
  v158 = 0;
  v160 = 0;
  v149 = v26;
  v147 = (__int64)(a3 + 3);
  if ( a3 + 3 != v26 )
  {
    do
    {
      if ( !v149 )
      {
        sub_BC1CD0(v148, &unk_4F8FAE8, 0);
        BUG();
      }
      v27 = sub_BC1CD0(v148, &unk_4F8FAE8, (__int64)(v149 - 7));
      v28 = (__int64 *)v149[3];
      v144 = v27;
      v142 = (__int64 *)(v27 + 8);
      if ( v149 + 2 == v28 )
      {
        v29 = 0;
      }
      else
      {
        if ( !v28 )
          BUG();
        while ( 1 )
        {
          v29 = (__int64 *)v28[4];
          if ( v29 != v28 + 3 )
            break;
          v28 = (__int64 *)v28[1];
          if ( v149 + 2 == v28 )
            break;
          if ( !v28 )
            BUG();
        }
      }
      v30 = v149 + 2;
      v31 = v28;
      j = v29;
LABEL_22:
      while ( v31 != v30 )
      {
        if ( !j )
          BUG();
        if ( (unsigned __int8)(*((_BYTE *)j - 24) - 34) <= 0x33u )
        {
          v33 = 0x8000000000041LL;
          if ( _bittest64(&v33, (unsigned int)*((unsigned __int8 *)j - 24) - 34) )
          {
            v34 = *(j - 7);
            if ( v34 && !*(_BYTE *)v34 && j[7] == *(_QWORD *)(v34 + 24) )
            {
              if ( sub_B2FC80(*(j - 7)) )
              {
                if ( *((_BYTE *)j - 24) != 85
                  || (v87 = *(j - 7)) == 0
                  || *(_BYTE *)v87
                  || *(_QWORD *)(v87 + 24) != j[7]
                  || (*(_BYTE *)(v87 + 33) & 0x20) == 0 )
                {
                  sub_30CB170(j - 3, "unavailable definition", 22);
                  v139 = *(_QWORD *)(v144 + 8);
                  v52 = sub_B2BE50(v139);
                  if ( sub_B6EA50(v52)
                    || (v105 = sub_B2BE50(v139),
                        v106 = sub_B6F970(v105),
                        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v106 + 48LL))(v106)) )
                  {
                    sub_B176B0((__int64)&v187, (__int64)"module-inline", (__int64)"NoDefinition", 12, (__int64)(j - 3));
                    sub_B16080((__int64)&v168, "Callee", 6, (unsigned __int8 *)v34);
                    v175 = &v177;
                    sub_266C3D0((__int64 *)&v175, v168, (__int64)v168 + v169);
                    v178.m128i_i64[1] = (__int64)&v180;
                    sub_266C3D0(&v178.m128i_i64[1], v171, (__int64)v171 + v172);
                    v181 = _mm_load_si128(&v174);
                    sub_B180C0((__int64)&v187, (unsigned __int64)&v175);
                    if ( (__m128i *)v178.m128i_i64[1] != &v180 )
                      j_j___libc_free_0(v178.m128i_u64[1]);
                    if ( v175 != &v177 )
                      j_j___libc_free_0((unsigned __int64)v175);
                    sub_B18290((__int64)&v187, " will not be inlined into ", 0x1Au);
                    v53 = (unsigned __int8 *)sub_B491C0((__int64)(j - 3));
                    sub_B16080((__int64)&v161, "Caller", 6, v53);
                    v175 = &v177;
                    sub_266C3D0((__int64 *)&v175, v161, (__int64)v161 + v162);
                    v178.m128i_i64[1] = (__int64)&v180;
                    sub_266C3D0(&v178.m128i_i64[1], v164, (__int64)v164 + v165);
                    v181 = _mm_load_si128(&v167);
                    sub_B180C0((__int64)&v187, (unsigned __int64)&v175);
                    if ( (__m128i *)v178.m128i_i64[1] != &v180 )
                      j_j___libc_free_0(v178.m128i_u64[1]);
                    if ( v175 != &v177 )
                      j_j___libc_free_0((unsigned __int64)v175);
                    sub_B18290((__int64)&v187, " because its definition is unavailable", 0x26u);
                    sub_B17B40((__int64)&v187);
                    v57 = _mm_loadu_si128((const __m128i *)((char *)&v188 + 8));
                    v58 = _mm_load_si128(&v191);
                    v59 = _mm_load_si128(&v192);
                    LODWORD(v176) = DWORD2(v187);
                    v60 = (int)v194;
                    v178 = v57;
                    BYTE4(v176) = BYTE12(v187);
                    v180 = v58;
                    v177 = v188;
                    v181 = v59;
                    v175 = (__int64 *)&unk_49D9D40;
                    v179 = v190;
                    v182 = (__m128i *)v184;
                    v183 = 0x400000000LL;
                    if ( (_DWORD)v194 )
                    {
                      v107 = (__m128i *)v184;
                      v108 = (unsigned int)v194;
                      if ( (unsigned int)v194 > 4 )
                      {
                        v136 = (int)v194;
                        sub_11F02D0((__int64)&v182, (unsigned int)v194, v54, v55, (unsigned int)v194, v56);
                        v107 = v182;
                        v108 = (unsigned int)v194;
                        v60 = v136;
                      }
                      if ( v193 != &v193[10 * v108] )
                      {
                        v137 = j;
                        v109 = v193;
                        v110 = v30;
                        v111 = &v193[10 * v108];
                        do
                        {
                          if ( v107 )
                          {
                            v131 = v60;
                            v107->m128i_i64[0] = (__int64)v107[1].m128i_i64;
                            sub_266C3D0(v107->m128i_i64, (_BYTE *)*v109, *v109 + v109[1]);
                            v107[2].m128i_i64[0] = (__int64)v107[3].m128i_i64;
                            sub_266C3D0(v107[2].m128i_i64, (_BYTE *)v109[4], v109[4] + v109[5]);
                            v60 = v131;
                            v107[4] = _mm_loadu_si128((const __m128i *)v109 + 4);
                          }
                          v109 += 10;
                          v107 += 5;
                        }
                        while ( v111 != v109 );
                        v30 = v110;
                        j = v137;
                      }
                      LODWORD(v183) = v60;
                    }
                    v184[320] = v199[104];
                    v185 = v200;
                    v186 = v201;
                    v175 = (__int64 *)&unk_49D9DB0;
                    if ( v164 != &v166 )
                      j_j___libc_free_0((unsigned __int64)v164);
                    if ( v161 != &v163 )
                      j_j___libc_free_0((unsigned __int64)v161);
                    if ( v171 != &v173 )
                      j_j___libc_free_0((unsigned __int64)v171);
                    if ( v168 != v170 )
                      j_j___libc_free_0((unsigned __int64)v168);
                    *(_QWORD *)&v187 = &unk_49D9D40;
                    v61 = 10LL * (unsigned int)v194;
                    v62 = (__int64 *)&v193[v61];
                    if ( v193 != &v193[v61] )
                    {
                      v135 = v30;
                      v63 = &v193[v61];
                      v64 = j;
                      v65 = v193;
                      do
                      {
                        v63 -= 10;
                        v66 = v63[4];
                        if ( (unsigned __int64 *)v66 != v63 + 6 )
                          j_j___libc_free_0(v66);
                        if ( (unsigned __int64 *)*v63 != v63 + 2 )
                          j_j___libc_free_0(*v63);
                      }
                      while ( v65 != v63 );
                      j = v64;
                      v30 = v135;
                      v62 = (__int64 *)v193;
                    }
                    if ( v62 != &v195 )
                      _libc_free((unsigned __int64)v62);
                    sub_1049740(v142, (__int64)&v175);
                    v67 = (unsigned __int64 *)v182;
                    v175 = (__int64 *)&unk_49D9D40;
                    v68 = (unsigned __int64 *)&v182[5 * (unsigned int)v183];
                    if ( v182 != (__m128i *)v68 )
                    {
                      do
                      {
                        v68 -= 10;
                        v69 = v68[4];
                        if ( (unsigned __int64 *)v69 != v68 + 6 )
                          j_j___libc_free_0(v69);
                        if ( (unsigned __int64 *)*v68 != v68 + 2 )
                          j_j___libc_free_0(*v68);
                      }
                      while ( v67 != v68 );
                      v68 = (unsigned __int64 *)v182;
                    }
                    if ( v68 != (unsigned __int64 *)v184 )
                      _libc_free((unsigned __int64)v68);
                  }
                }
              }
              else
              {
                v51 = *(void (__fastcall **)(__int64, __int128 *))(*(_QWORD *)v151 + 24LL);
                *(_QWORD *)&v187 = j - 3;
                DWORD2(v187) = -1;
                v51(v151, &v187);
              }
            }
            else if ( (_BYTE)qword_4FF4508 && *(_QWORD *)(v143 + 48) && sub_B491E0((__int64)(j - 3)) )
            {
              sub_30A85F0(j - 3, v134, &v155);
            }
          }
        }
        for ( j = (__int64 *)j[1]; ; j = (__int64 *)v31[4] )
        {
          v35 = v31 - 3;
          if ( !v31 )
            v35 = 0;
          if ( j != v35 + 6 )
            break;
          v31 = (__int64 *)v31[1];
          if ( v30 == v31 )
            goto LABEL_22;
          if ( !v31 )
            BUG();
        }
      }
      v149 = (__int64 *)v149[1];
    }
    while ( (__int64 *)v147 != v149 );
    v36 = v159;
    v37 = &v159[2 * (unsigned int)v160];
    if ( v37 != v159 )
    {
      do
      {
        v38 = sub_29A5A80(*v36, v36[1], v134);
        if ( v38 )
        {
          v39 = *(void (__fastcall **)(__int64, __int128 *))(*(_QWORD *)v151 + 24LL);
          *(_QWORD *)&v187 = v38;
          DWORD2(v187) = -1;
          v39(v151, &v187);
        }
        v36 += 2;
      }
      while ( v37 != v36 );
    }
  }
  if ( !(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v151 + 16LL))(v151) )
  {
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_161;
  }
  v145 = 0;
  v175 = &v177;
  v176 = 0x1000000000LL;
  v168 = v170;
  v169 = 0x400000000LL;
  while ( (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v151 + 16LL))(v151) )
  {
    v40 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v151 + 32LL))(v151);
    v42 = v41;
    sub_B491C0(v40);
    v43 = *(_QWORD *)(v40 - 32);
    if ( v43 )
    {
      if ( *(_BYTE *)v43 )
      {
        v43 = 0;
      }
      else if ( *(_QWORD *)(v43 + 24) != *(_QWORD *)(v40 + 80) )
      {
        v43 = 0;
      }
    }
    v152 = v148;
    if ( v42 == -1 )
    {
LABEL_55:
      sub_30CC6B0(&v153, v141, v40, 0);
      v46 = (_QWORD *)v153;
      v150 = *(_BYTE *)(v153 + 56);
      if ( v150 )
      {
        v70 = sub_BC1CD0(v148, &unk_4F8D9A8, v43);
        v71 = sub_B491C0(v40);
        v72 = sub_BC1CD0(v148, &unk_4F8D9A8, v71);
        v199[64] = 1;
        *(_QWORD *)&v187 = sub_266C3B0;
        *((_QWORD *)&v188 + 1) = v72 + 8;
        *(_QWORD *)&v188 = v133;
        v189 = v70 + 8;
        v191.m128i_i64[0] = 0x400000000LL;
        v197 = v199;
        *((_QWORD *)&v187 + 1) = &v152;
        v190 = (__m128i *)&v191.m128i_u64[1];
        v194 = v196;
        v195 = 0x800000000LL;
        v198 = 0x800000000LL;
        v73 = sub_B491C0(v40);
        v74 = sub_BC1CD0(v148, &unk_4F86540, v73);
        v75 = (unsigned __int64)&v187;
        v76 = sub_29F2720(v40, (unsigned int)&v187, v134, 1, v74 + 8, 1, 0);
        v79 = v129;
        v80 = v130;
        v154 = v76;
        if ( v76 )
        {
          v81 = v153;
          v82 = *(_QWORD *)v153;
          *(_BYTE *)(v153 + 57) = 1;
          v83 = *(void (**)())(v82 + 32);
          if ( v83 != nullsub_1534 )
            ((void (__fastcall *)(unsigned __int64, __int64 *, unsigned __int64))v83)(v81, &v154, v129);
          if ( v197 != v199 )
            _libc_free((unsigned __int64)v197);
          v84 = v194;
          v85 = &v194[24 * (unsigned int)v195];
          if ( v194 != (_BYTE *)v85 )
          {
            do
            {
              v86 = *(v85 - 1);
              v85 -= 3;
              if ( v86 != 0 && v86 != -4096 && v86 != -8192 )
                sub_BD60C0(v85);
            }
            while ( v84 != v85 );
            v85 = v194;
          }
          if ( v85 != (_QWORD *)v196 )
            _libc_free((unsigned __int64)v85);
          if ( v190 != (__m128i *)&v191.m128i_u64[1] )
            _libc_free((unsigned __int64)v190);
          goto LABEL_118;
        }
        if ( (_DWORD)v198 )
        {
          v75 = (unsigned int)v176;
          v80 = HIDWORD(v176);
          v114 = v132 & 0xFFFFFFFF00000000LL | v42;
          v115 = (unsigned int)v176;
          v146 = v176;
          v79 = (unsigned int)v176 + 1LL;
          v132 = v114;
          if ( v79 > HIDWORD(v176) )
          {
            v75 = (unsigned __int64)&v177;
            sub_C8D5F0((__int64)&v175, &v177, v79, 0x10u, v77, v78);
            v115 = (unsigned int)v176;
          }
          v116 = &v175[2 * v115];
          *v116 = v43;
          v116[1] = v114;
          v117 = (unsigned __int64)v197;
          LODWORD(v176) = v176 + 1;
          v118 = &v197[8 * (unsigned int)v198];
          if ( v197 != v118 )
          {
            v140 = v43;
            do
            {
              v119 = (__int64 *)*((_QWORD *)v118 - 1);
              v120 = *(v119 - 4);
              if ( (v120 && !*(_BYTE *)v120 && *(_QWORD *)(v120 + 24) == v119[10]
                 || !*(_QWORD *)(v143 + 48)
                 && (unsigned __int8)sub_29A4580(*((_QWORD *)v118 - 1))
                 && (v120 = *(v119 - 4)) != 0
                 && !*(_BYTE *)v120
                 && *(_QWORD *)(v120 + 24) == v119[10])
                && !sub_B2FC80(v120) )
              {
                v75 = (unsigned __int64)&v161;
                v121 = *(void (__fastcall **)(__int64, __int64 **))(*(_QWORD *)v151 + 24LL);
                v161 = v119;
                LODWORD(v162) = v146;
                v121(v151, &v161);
              }
              v118 -= 8;
            }
            while ( (_BYTE *)v117 != v118 );
            v43 = v140;
          }
        }
        if ( (*(_BYTE *)(v43 + 32) & 0xFu) - 7 > 1
          || (sub_AD0030(v43), *(_QWORD *)(v43 + 16))
          || (v75 = v43,
              v122 = sub_BC1CD0(v148, &unk_4F6D3F8, v43),
              sub_981210(*(_QWORD *)(v122 + 8), v43, (unsigned int *)&v161))
          || (v75 = (unsigned __int64)sub_BD5D20(v43), sub_97F890(*(_QWORD *)(v122 + 8), (_BYTE *)v75, v123)) )
        {
          sub_30CACB0(v153, v75, v79, v80);
        }
        else
        {
          v124 = *(void (__fastcall **)(__int64, bool (__fastcall *)(_QWORD *, __int64), __int64 **))(*(_QWORD *)v151 + 40LL);
          v161 = (__int64 *)v43;
          v124(v151, sub_266C390, &v161);
          sub_B2CA40(v43, 1);
          v127 = (unsigned int)v169;
          v128 = (unsigned int)v169 + 1LL;
          if ( v128 > HIDWORD(v169) )
          {
            sub_C8D5F0((__int64)&v168, v170, v128, 8u, v125, v126);
            v127 = (unsigned int)v169;
          }
          v168[v127] = v43;
          LODWORD(v169) = v169 + 1;
          sub_30CACE0(v153);
        }
        if ( v197 != v199 )
          _libc_free((unsigned __int64)v197);
        v88 = v194;
        v89 = &v194[24 * (unsigned int)v195];
        if ( v194 != (_BYTE *)v89 )
        {
          do
          {
            v90 = *(v89 - 1);
            v89 -= 3;
            if ( v90 != 0 && v90 != -4096 && v90 != -8192 )
              sub_BD60C0(v89);
          }
          while ( v88 != v89 );
          v89 = v194;
        }
        if ( v89 != (_QWORD *)v196 )
          _libc_free((unsigned __int64)v89);
        if ( v190 != (__m128i *)&v191.m128i_u64[1] )
          _libc_free((unsigned __int64)v190);
        v91 = v153;
        if ( v153 )
        {
          v92 = *(void (__fastcall **)(_QWORD *))(*(_QWORD *)v153 + 8LL);
          if ( v92 == sub_2610030 )
          {
            v93 = *(_QWORD *)(v153 + 32);
            *(_QWORD *)v153 = &unk_4A1F3E0;
            if ( v93 )
              sub_B91220(v91 + 32, v93);
            j_j___libc_free_0(v91);
          }
          else
          {
            v92((_QWORD *)v153);
          }
        }
        v145 = v150;
      }
      else
      {
        v47 = *(_QWORD *)v153;
        *(_BYTE *)(v153 + 57) = 1;
        v48 = *(void (**)())(v47 + 40);
        if ( v48 == nullsub_1535 )
        {
          v49 = *(void (__fastcall **)(_QWORD *))(v47 + 8);
          if ( v49 == sub_2610030 )
            goto LABEL_58;
LABEL_120:
          v49(v46);
        }
        else
        {
          ((void (__fastcall *)(_QWORD *))v48)(v46);
LABEL_118:
          v46 = (_QWORD *)v153;
          if ( v153 )
          {
            v49 = *(void (__fastcall **)(_QWORD *))(*(_QWORD *)v153 + 8LL);
            if ( v49 != sub_2610030 )
              goto LABEL_120;
LABEL_58:
            v50 = v46[4];
            *v46 = &unk_4A1F3E0;
            if ( v50 )
              sub_B91220((__int64)(v46 + 4), v50);
            j_j___libc_free_0((unsigned __int64)v46);
          }
        }
      }
    }
    else
    {
      v44 = (int)v42;
      while ( 1 )
      {
        v45 = &v175[2 * v44];
        if ( v43 == *v45 )
          break;
        v44 = *((int *)v45 + 2);
        if ( (_DWORD)v44 == -1 )
          goto LABEL_55;
      }
      sub_30CB170(v40, "recursive", 9);
    }
  }
  v94 = v168;
  v95 = &v168[(unsigned int)v169];
  if ( v95 != v168 )
  {
    v96 = v168;
    do
    {
      v97 = (_QWORD *)*v96;
      v98 = sub_BD5D20(*v96);
      sub_BBB260(v148, (__int64)v97, (__int64)v98, v99);
      if ( !v97 )
      {
        sub_BA8570(v147, -56);
        BUG();
      }
      ++v96;
      sub_BA8570(v147, (__int64)v97);
      v100 = (unsigned __int64 *)v97[8];
      v101 = v97[7] & 0xFFFFFFFFFFFFFFF8LL;
      *v100 = v101 | *v100 & 7;
      *(_QWORD *)(v101 + 8) = v100;
      v97[7] &= 7uLL;
      v97[8] = 0;
      sub_B2E780(v97);
      sub_BD2DD0((__int64)v97);
    }
    while ( v95 != v96 );
    v94 = v168;
  }
  v102 = a1 + 32;
  v103 = a1 + 80;
  if ( v145 )
  {
    memset((void *)a1, 0, 0x60u);
    *(_QWORD *)(a1 + 8) = v102;
    *(_DWORD *)(a1 + 16) = 2;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 56) = v103;
    *(_DWORD *)(a1 + 64) = 2;
    *(_BYTE *)(a1 + 76) = 1;
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v102;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v103;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)a1 = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
  }
  if ( v94 != v170 )
    _libc_free((unsigned __int64)v94);
  if ( v175 != &v177 )
    _libc_free((unsigned __int64)v175);
LABEL_161:
  if ( v159 != &v161 )
    _libc_free((unsigned __int64)v159);
  sub_C7D6A0(v156, 16LL * (unsigned int)v158, 8);
  if ( v151 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v151 + 8LL))(v151);
  v104 = *(void (**)())(*(_QWORD *)v141 + 24LL);
  if ( v104 != nullsub_1537 )
    ((void (__fastcall *)(__int64, _QWORD))v104)(v141, 0);
  return a1;
}
