// Function: sub_226C400
// Address: 0x226c400
//
__int64 __fastcall sub_226C400(__int64 a1, const char **a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r12
  unsigned __int8 *v8; // r15
  size_t v9; // rdx
  const char *v10; // r14
  __int64 *v11; // r13
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  bool v15; // zf
  unsigned __int64 v16; // rax
  void *v17; // r12
  unsigned __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rdi
  _BYTE *v21; // rax
  unsigned int v22; // r12d
  unsigned __int64 v23; // r13
  _BYTE *v24; // rbx
  unsigned __int64 v25; // r13
  unsigned __int64 v26; // rdi
  __int64 v27; // rsi
  _QWORD *v28; // r15
  unsigned __int8 **v29; // r13
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rax
  _BYTE *v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // rax
  int v35; // edi
  __int64 v36; // rax
  __int64 v37; // rax
  _QWORD *v38; // rcx
  unsigned __int8 *v39; // rdi
  size_t v40; // rdx
  unsigned __int8 *v41; // rdi
  _BYTE *v42; // rsi
  char *v43; // rsi
  unsigned __int64 v44; // rax
  __int64 v45; // r8
  char *v46; // rsi
  size_t v47; // rax
  size_t v48; // rdx
  __m128i *v49; // rdi
  unsigned __int64 v50; // rax
  __m128i *v51; // rdx
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // r8
  __int64 v56; // rax
  __int64 v57; // rdi
  _BYTE *v58; // rax
  void *v59; // rax
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rdx
  __int64 v64; // r12
  char *v65; // r13
  size_t v66; // rax
  __m128i *v67; // rdi
  size_t v68; // r14
  __m128i v69; // xmm0
  __int64 v70; // rax
  size_t v71; // rdx
  void *v72; // rdi
  size_t v73; // r13
  unsigned __int64 v74; // rax
  unsigned __int8 *v75; // rdi
  _BYTE *v76; // rdx
  size_t v77; // rcx
  _BYTE *v78; // rsi
  unsigned __int8 *v79; // rdi
  __int64 v80; // rbx
  int v81; // r15d
  __int64 v82; // r12
  __int64 v83; // r12
  char *v84; // r14
  size_t v85; // rax
  __m128i *v86; // rdi
  size_t v87; // r13
  unsigned __int64 v88; // rax
  __int64 v89; // rax
  __m128i si128; // xmm0
  __int64 v91; // rdi
  _BYTE *v92; // rax
  unsigned __int8 v93; // cl
  unsigned __int8 v94; // si
  int v95; // ebx
  int v96; // edx
  int v97; // eax
  __int64 v98; // r8
  __int64 v99; // rax
  unsigned __int8 v100; // cl
  unsigned __int8 v101; // dl
  unsigned __int8 v102; // r11
  unsigned __int8 v103; // r10
  unsigned __int8 v104; // r9
  unsigned __int8 v105; // r14
  unsigned __int8 v106; // di
  unsigned __int8 v107; // si
  int v108; // edx
  int v109; // r11d
  int v110; // esi
  int v111; // r10d
  int v112; // r9d
  unsigned int v113; // ebx
  int v114; // ecx
  int v115; // r15d
  int v116; // ebx
  const char *v117; // r14
  const char *v118; // rdi
  int v119; // eax
  _QWORD *v120; // rax
  __m128i *v121; // rcx
  __m128i v122; // xmm0
  __int64 v123; // rax
  __int64 v124; // rbx
  __int64 v125; // r13
  unsigned __int64 v126; // rdi
  __int64 v127; // r12
  char *v128; // r14
  size_t v129; // rax
  _WORD *v130; // rdi
  size_t v131; // r13
  unsigned __int64 v132; // rax
  __int64 v133; // rax
  __int64 v134; // rdx
  __m128i v135; // xmm0
  size_t v136; // rdx
  size_t v137; // rdx
  _QWORD *v138; // rax
  __m128i *v139; // rdx
  __m128i v140; // xmm0
  void *v141; // rax
  __m128i *v142; // rdx
  char *v143; // rax
  char v144; // al
  __int64 v145; // rax
  __int64 v146; // rax
  _BYTE *v147; // rdi
  int v148; // eax
  __int64 v149; // rax
  __int64 v150; // [rsp+8h] [rbp-3C8h]
  int v151; // [rsp+14h] [rbp-3BCh]
  int v152; // [rsp+18h] [rbp-3B8h]
  int v153; // [rsp+1Ch] [rbp-3B4h]
  int v154; // [rsp+20h] [rbp-3B0h]
  char v155; // [rsp+24h] [rbp-3ACh]
  int v156; // [rsp+24h] [rbp-3ACh]
  __int64 v157; // [rsp+28h] [rbp-3A8h]
  unsigned __int8 v158; // [rsp+28h] [rbp-3A8h]
  int v159; // [rsp+28h] [rbp-3A8h]
  __int64 v160; // [rsp+28h] [rbp-3A8h]
  __int64 v161; // [rsp+28h] [rbp-3A8h]
  unsigned __int64 v162; // [rsp+50h] [rbp-380h]
  char *srcb; // [rsp+58h] [rbp-378h]
  void *srcc; // [rsp+58h] [rbp-378h]
  _BYTE *srca; // [rsp+58h] [rbp-378h]
  unsigned __int8 *v167; // [rsp+60h] [rbp-370h]
  const char *v168; // [rsp+60h] [rbp-370h]
  unsigned __int8 **v170; // [rsp+90h] [rbp-340h]
  int v172; // [rsp+98h] [rbp-338h]
  unsigned __int64 v173; // [rsp+A0h] [rbp-330h] BYREF
  __int64 v174; // [rsp+A8h] [rbp-328h] BYREF
  _BYTE *v175; // [rsp+B0h] [rbp-320h] BYREF
  unsigned __int8 v176; // [rsp+B8h] [rbp-318h]
  unsigned __int64 v177; // [rsp+C0h] [rbp-310h] BYREF
  char v178; // [rsp+C8h] [rbp-308h]
  void *dest; // [rsp+D0h] [rbp-300h] BYREF
  size_t v180; // [rsp+D8h] [rbp-2F8h]
  _QWORD v181[2]; // [rsp+E0h] [rbp-2F0h] BYREF
  void *v182; // [rsp+F0h] [rbp-2E0h] BYREF
  size_t v183; // [rsp+F8h] [rbp-2D8h]
  _QWORD v184[2]; // [rsp+100h] [rbp-2D0h] BYREF
  unsigned __int64 v185; // [rsp+110h] [rbp-2C0h] BYREF
  size_t v186; // [rsp+118h] [rbp-2B8h]
  _QWORD v187[2]; // [rsp+120h] [rbp-2B0h] BYREF
  __int64 v188; // [rsp+130h] [rbp-2A0h]
  __int64 v189; // [rsp+138h] [rbp-298h]
  __int64 v190; // [rsp+140h] [rbp-290h]
  __int64 v191; // [rsp+150h] [rbp-280h] BYREF
  size_t n; // [rsp+158h] [rbp-278h]
  _BYTE *v193; // [rsp+160h] [rbp-270h] BYREF
  __int64 v194; // [rsp+168h] [rbp-268h]
  __int64 v195; // [rsp+170h] [rbp-260h]
  __int64 v196; // [rsp+178h] [rbp-258h]
  __int64 v197; // [rsp+180h] [rbp-250h]
  __int64 v198; // [rsp+1E0h] [rbp-1F0h]
  unsigned int v199; // [rsp+1F0h] [rbp-1E0h]
  unsigned __int64 v200; // [rsp+200h] [rbp-1D0h]
  unsigned __int64 v201; // [rsp+218h] [rbp-1B8h]
  unsigned __int64 v202[2]; // [rsp+230h] [rbp-1A0h] BYREF
  _QWORD *v203; // [rsp+240h] [rbp-190h]
  __int64 v204; // [rsp+248h] [rbp-188h]
  _QWORD v205[3]; // [rsp+250h] [rbp-180h] BYREF
  int v206; // [rsp+268h] [rbp-168h]
  _QWORD *v207; // [rsp+270h] [rbp-160h]
  __int64 v208; // [rsp+278h] [rbp-158h]
  _BYTE v209[16]; // [rsp+280h] [rbp-150h] BYREF
  _QWORD *v210; // [rsp+290h] [rbp-140h]
  __int64 v211; // [rsp+298h] [rbp-138h]
  _BYTE v212[16]; // [rsp+2A0h] [rbp-130h] BYREF
  unsigned __int64 v213; // [rsp+2B0h] [rbp-120h]
  __int64 v214; // [rsp+2B8h] [rbp-118h]
  __int64 v215; // [rsp+2C0h] [rbp-110h]
  _BYTE *v216; // [rsp+2C8h] [rbp-108h]
  __int64 v217; // [rsp+2D0h] [rbp-100h]
  _BYTE v218[248]; // [rsp+2D8h] [rbp-F8h] BYREF

  if ( *(_BYTE *)(a4 + 1400) )
  {
    v59 = sub_CB7210(a1, (__int64)a2, a3, a4, a5);
    v22 = 0;
    sub_2273520(v59);
    return v22;
  }
  v7 = a3;
  v8 = (unsigned __int8 *)a4;
  v9 = 0;
  v10 = *a2;
  if ( *a2 )
    v9 = strlen(*a2);
  v11 = *(__int64 **)v7;
  if ( byte_4FD7A48 )
    sub_C98D50(qword_4FD7968, (__int64)v10, v9, 0);
  v202[0] = 0;
  v203 = v205;
  v207 = v209;
  v210 = v212;
  v216 = v218;
  v202[1] = 0;
  v204 = 0;
  LOBYTE(v205[0]) = 0;
  v205[2] = 0;
  v206 = 0;
  v208 = 0;
  v209[0] = 0;
  v211 = 0;
  v212[0] = 0;
  v213 = 0;
  v214 = 0;
  v215 = 0;
  v217 = 0x400000000LL;
  sub_B6F950(v11, qword_4FD7B28);
  if ( !v8[240] )
    sub_B6F900(v11);
  sub_B7DFC0(
    (__int64)&v175,
    v11,
    qword_4FD7468,
    qword_4FD7470,
    qword_4FD7368,
    qword_4FD7370,
    qword_4FD7268,
    qword_4FD7270,
    qword_4FD7788,
    qword_4FD7568,
    qword_4FD7570);
  v14 = v176 & 0xFD;
  v15 = (v176 & 1) == 0;
  v16 = (unsigned __int64)v175;
  v176 &= ~2u;
  if ( v15 )
  {
    v162 = (unsigned __int64)v175;
  }
  else
  {
    v175 = 0;
    v14 = v16 | 1;
    v182 = (void *)(v16 | 1);
    if ( (v16 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v17 = sub_CB72A0();
      v18 = (unsigned __int64)v182;
      v182 = 0;
      v185 = v18 | 1;
      sub_C64870((__int64)&v191, (__int64 *)&v185);
      v19 = v191;
      v20 = sub_CB6200((__int64)v17, (unsigned __int8 *)v191, n);
      v21 = *(_BYTE **)(v20 + 32);
      if ( (unsigned __int64)v21 >= *(_QWORD *)(v20 + 24) )
      {
        v19 = 10;
        sub_CB5D20(v20, 10);
      }
      else
      {
        *(_QWORD *)(v20 + 32) = v21 + 1;
        *v21 = 10;
      }
      if ( (_BYTE **)v191 != &v193 )
      {
        v19 = (__int64)(v193 + 1);
        j_j___libc_free_0(v191);
      }
      if ( (v185 & 1) != 0 || (v185 & 0xFFFFFFFFFFFFFFFELL) != 0 )
LABEL_234:
        sub_C63C30(&v185, v19);
      if ( ((unsigned __int8)v182 & 1) != 0 || (v22 = 1, ((unsigned __int64)v182 & 0xFFFFFFFFFFFFFFFELL) != 0) )
        sub_C63C30(&v182, v19);
      goto LABEL_18;
    }
    v162 = 0;
  }
  v15 = v8[1680] == 0;
  v175 = 0;
  if ( !v15 )
    sub_AEB840((_QWORD *)v7, (__int64)v11, v14);
  if ( v8[1720] && v7 + 72 != (*(_QWORD *)(v7 + 72) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    do
      sub_BA9050(v7, *(_QWORD **)(v7 + 80));
    while ( v7 + 72 != (*(_QWORD *)(v7 + 72) & 0xFFFFFFFFFFFFFFF8LL) );
  }
  if ( qword_4FD8510 )
  {
    sub_CCABA0((__int64)&v182, (char *)qword_4FD8508, (char *)qword_4FD8510, 0, v12, v13);
    v185 = (unsigned __int64)&v182;
    LOWORD(v188) = 260;
    sub_CC9F70((__int64)&v191, (void **)&v185);
    v75 = *(unsigned __int8 **)(v7 + 232);
    if ( (_BYTE **)v191 == &v193 )
    {
      v137 = n;
      if ( n )
      {
        if ( n == 1 )
          *v75 = (unsigned __int8)v193;
        else
          memcpy(v75, &v193, n);
        v137 = n;
        v75 = *(unsigned __int8 **)(v7 + 232);
      }
      *(_QWORD *)(v7 + 240) = v137;
      v75[v137] = 0;
      v75 = (unsigned __int8 *)v191;
      goto LABEL_129;
    }
    v76 = v193;
    v77 = n;
    if ( v75 == (unsigned __int8 *)(v7 + 248) )
    {
      *(_QWORD *)(v7 + 232) = v191;
      *(_QWORD *)(v7 + 240) = v77;
      *(_QWORD *)(v7 + 248) = v76;
    }
    else
    {
      v78 = *(_BYTE **)(v7 + 248);
      *(_QWORD *)(v7 + 232) = v191;
      *(_QWORD *)(v7 + 240) = v77;
      *(_QWORD *)(v7 + 248) = v76;
      if ( v75 )
      {
        v191 = (__int64)v75;
        v193 = v78;
LABEL_129:
        n = 0;
        *v75 = 0;
        v79 = (unsigned __int8 *)v191;
        *(_QWORD *)(v7 + 264) = v195;
        *(_QWORD *)(v7 + 272) = v196;
        *(_QWORD *)(v7 + 280) = v197;
        if ( v79 != (unsigned __int8 *)&v193 )
          j_j___libc_free_0((unsigned __int64)v79);
        if ( v182 != v184 )
          j_j___libc_free_0((unsigned __int64)v182);
        goto LABEL_62;
      }
    }
    v191 = (__int64)&v193;
    v75 = (unsigned __int8 *)&v193;
    goto LABEL_129;
  }
LABEL_62:
  if ( v8[440] || (v19 = (__int64)sub_CB72A0(), !(unsigned __int8)sub_C09360((__int64 *)v7, v19, 0)) )
  {
    v191 = 0;
    n = 0;
    v193 = 0;
    v194 = 0;
    sub_26FAD40(v7, 0, &v191, 0, sub_226AA60, &v185);
    sub_C7D6A0(n, 8LL * (unsigned int)v194, 8);
    v32 = *(_BYTE **)(v7 + 232);
    v33 = (__int64)&v32[*(_QWORD *)(v7 + 240)];
    v185 = (unsigned __int64)v187;
    sub_226AE30((__int64 *)&v185, v32, v33);
    v34 = *(_QWORD *)(v7 + 264);
    v35 = *(_DWORD *)(v7 + 264);
    v180 = 0;
    LOBYTE(v181[0]) = 0;
    v188 = v34;
    v36 = *(_QWORD *)(v7 + 272);
    v183 = 0;
    v189 = v36;
    v37 = *(_QWORD *)(v7 + 280);
    LOBYTE(v184[0]) = 0;
    v190 = v37;
    dest = v181;
    v182 = v184;
    if ( v35 )
    {
      sub_2D92D10(&v191);
      v38 = &v193;
      v39 = (unsigned __int8 *)dest;
      if ( (_BYTE **)v191 == &v193 )
      {
        v40 = n;
        if ( n )
        {
          if ( n == 1 )
          {
            *(_BYTE *)dest = (_BYTE)v193;
          }
          else
          {
            v32 = &v193;
            memcpy(dest, &v193, n);
          }
          v40 = n;
          v39 = (unsigned __int8 *)dest;
        }
        v180 = v40;
        v39[v40] = 0;
        v39 = (unsigned __int8 *)v191;
      }
      else
      {
        v38 = (_QWORD *)n;
        v40 = (size_t)v193;
        if ( dest == v181 )
        {
          dest = (void *)v191;
          v180 = n;
          v181[0] = v193;
        }
        else
        {
          v32 = (_BYTE *)v181[0];
          dest = (void *)v191;
          v180 = n;
          v181[0] = v193;
          if ( v39 )
          {
            v191 = (__int64)v39;
            v193 = v32;
            goto LABEL_69;
          }
        }
        v191 = (__int64)&v193;
        v39 = (unsigned __int8 *)&v193;
      }
LABEL_69:
      n = 0;
      *v39 = 0;
      if ( (_BYTE **)v191 != &v193 )
      {
        v32 = v193 + 1;
        j_j___libc_free_0(v191);
      }
      if ( !v180 )
      {
        v32 = (_BYTE *)(a5 + 16);
        sub_2240AE0((unsigned __int64 *)&dest, (unsigned __int64 *)(a5 + 16));
      }
      sub_2D92DA0(&v191, v32, v40, v38);
      v41 = (unsigned __int8 *)v182;
      if ( (_BYTE **)v191 == &v193 )
      {
        v136 = n;
        if ( n )
        {
          if ( n == 1 )
            *(_BYTE *)v182 = (_BYTE)v193;
          else
            memcpy(v182, &v193, n);
          v136 = n;
          v41 = (unsigned __int8 *)v182;
        }
        v183 = v136;
        v41[v136] = 0;
        v41 = (unsigned __int8 *)v191;
      }
      else
      {
        if ( v182 == v184 )
        {
          v182 = (void *)v191;
          v183 = n;
          v184[0] = v193;
        }
        else
        {
          v42 = (_BYTE *)v184[0];
          v182 = (void *)v191;
          v183 = n;
          v184[0] = v193;
          if ( v41 )
          {
            v191 = (__int64)v41;
            v193 = v42;
            goto LABEL_77;
          }
        }
        v191 = (__int64)&v193;
        v41 = (unsigned __int8 *)&v193;
      }
LABEL_77:
      n = 0;
      *v41 = 0;
      if ( (_BYTE **)v191 != &v193 )
        j_j___libc_free_0(v191);
      v43 = (char *)v185;
      sub_2D940D0(&v177, v185, v186, (unsigned int)qword_4FD8608, &dest);
      v15 = (v178 & 1) == 0;
      v44 = v177;
      v178 &= ~2u;
      if ( v15 )
      {
        srca = (_BYTE *)v177;
      }
      else
      {
        v177 = 0;
        v173 = v44 | 1;
        if ( (v44 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          v45 = (__int64)sub_CB72A0();
          v46 = (char *)*a2;
          if ( *a2 )
          {
            v157 = v45;
            v47 = strlen(*a2);
            v45 = v157;
            v48 = v47;
            v49 = *(__m128i **)(v157 + 32);
            v50 = *(_QWORD *)(v157 + 24) - (_QWORD)v49;
            if ( v48 <= v50 )
            {
              if ( v48 )
              {
                srcb = (char *)v48;
                memcpy(v49, v46, v48);
                v45 = v157;
                v51 = (__m128i *)&srcb[*(_QWORD *)(v157 + 32)];
                v52 = *(_QWORD *)(v157 + 24);
                *(_QWORD *)(v157 + 32) = v51;
                v49 = v51;
                v50 = v52 - (_QWORD)v51;
              }
              goto LABEL_85;
            }
            v45 = sub_CB6200(v157, (unsigned __int8 *)v46, v48);
          }
          v49 = *(__m128i **)(v45 + 32);
          v50 = *(_QWORD *)(v45 + 24) - (_QWORD)v49;
LABEL_85:
          if ( v50 <= 0x2F )
          {
            v45 = sub_CB6200(v45, ": WARNING: failed to create target machine for '", 0x30u);
          }
          else
          {
            *v49 = _mm_load_si128((const __m128i *)&xmmword_43645E0);
            v49[1] = _mm_load_si128((const __m128i *)&xmmword_43645F0);
            v49[2] = _mm_load_si128((const __m128i *)&xmmword_4364600);
            *(_QWORD *)(v45 + 32) += 48LL;
          }
          v53 = sub_CB6200(v45, (unsigned __int8 *)v185, v186);
          v54 = *(_QWORD *)(v53 + 32);
          v55 = v53;
          if ( (unsigned __int64)(*(_QWORD *)(v53 + 24) - v54) <= 2 )
          {
            v55 = sub_CB6200(v53, "': ", 3u);
          }
          else
          {
            *(_BYTE *)(v54 + 2) = 32;
            *(_WORD *)v54 = 14887;
            *(_QWORD *)(v53 + 32) += 3LL;
          }
          v56 = v173;
          srcc = (void *)v55;
          v173 = 0;
          v174 = v56 | 1;
          sub_C64870((__int64)&v191, &v174);
          v43 = (char *)v191;
          v57 = sub_CB6200((__int64)srcc, (unsigned __int8 *)v191, n);
          v58 = *(_BYTE **)(v57 + 32);
          if ( *(_BYTE **)(v57 + 24) == v58 )
          {
            v43 = "\n";
            sub_CB6200(v57, (unsigned __int8 *)"\n", 1u);
          }
          else
          {
            *v58 = 10;
            ++*(_QWORD *)(v57 + 32);
          }
          if ( (_BYTE **)v191 != &v193 )
          {
            v43 = v193 + 1;
            j_j___libc_free_0(v191);
          }
          if ( (v174 & 1) != 0 || (v174 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v174, (__int64)v43);
          if ( (v173 & 1) != 0 || (v173 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v173, (__int64)v43);
          srca = 0;
          goto LABEL_136;
        }
        srca = 0;
      }
      v177 = 0;
LABEL_136:
      if ( (v178 & 2) != 0 )
        sub_226C390(&v177, (__int64)v43);
      if ( v177 )
        (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v177 + 8LL))(v177);
LABEL_139:
      sub_2D94060(dest, v180, v182, v183, v7);
      if ( byte_4FD89A8 )
      {
        sub_BA93D0((__int64 **)v7, 1u, "EnableSplitLTOUnit", 0x12u, (unsigned __int8)qword_4FD88C8);
        if ( (_BYTE)qword_4FD87E8 )
          sub_BA93D0((__int64 **)v7, 1u, "UnifiedLTO", 0xAu, 1u);
      }
      sub_982C80((__int64)&v191, &v185);
      if ( byte_4FD8268 )
      {
        sub_97F7E0(&v191);
      }
      else if ( qword_4FD8170 != qword_4FD8168 )
      {
        v167 = v8;
        v80 = qword_4FD8168;
        v81 = v7;
        v82 = qword_4FD8170;
        while ( (unsigned __int8)sub_980AF0((__int64)&v191, *(_BYTE **)v80, *(_QWORD *)(v80 + 8), &v177) )
        {
          v80 += 32;
          *((_BYTE *)&v191 + ((unsigned int)v177 >> 2)) &= ~(3 << (2 * (v177 & 3)));
          if ( v82 == v80 )
          {
            LODWORD(v7) = v81;
            v8 = v167;
            goto LABEL_154;
          }
        }
        v83 = (__int64)sub_CB72A0();
        v84 = (char *)*a2;
        if ( *a2 )
        {
          v85 = strlen(*a2);
          v86 = *(__m128i **)(v83 + 32);
          v87 = v85;
          v88 = *(_QWORD *)(v83 + 24) - (_QWORD)v86;
          if ( v87 <= v88 )
          {
            if ( v87 )
            {
              memcpy(v86, v84, v87);
              v89 = *(_QWORD *)(v83 + 24);
              v86 = (__m128i *)(v87 + *(_QWORD *)(v83 + 32));
              *(_QWORD *)(v83 + 32) = v86;
              v88 = v89 - (_QWORD)v86;
            }
            goto LABEL_149;
          }
          v83 = sub_CB6200(v83, (unsigned __int8 *)v84, v87);
        }
        v86 = *(__m128i **)(v83 + 32);
        v88 = *(_QWORD *)(v83 + 24) - (_QWORD)v86;
LABEL_149:
        if ( v88 <= 0x2D )
        {
          v83 = sub_CB6200(v83, ": cannot disable nonexistent builtin function ", 0x2Eu);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_4364620);
          qmemcpy(&v86[2], "ltin function ", 14);
          *v86 = si128;
          v86[1] = _mm_load_si128((const __m128i *)&xmmword_4364630);
          *(_QWORD *)(v83 + 32) += 46LL;
        }
        v91 = sub_CB6200(v83, *(unsigned __int8 **)v80, *(_QWORD *)(v80 + 8));
        v92 = *(_BYTE **)(v91 + 32);
        if ( (unsigned __int64)v92 >= *(_QWORD *)(v91 + 24) )
        {
          sub_CB5D20(v91, 10);
        }
        else
        {
          *(_QWORD *)(v91 + 32) = v92 + 1;
          *v92 = 10;
        }
        goto LABEL_173;
      }
LABEL_154:
      if ( sub_B808A0() )
      {
        v141 = sub_CB72A0();
        v142 = (__m128i *)*((_QWORD *)v141 + 4);
        if ( *((_QWORD *)v141 + 3) - (_QWORD)v142 <= 0x5Fu )
        {
          sub_CB6200(
            (__int64)v141,
            "-debug-pass does not work with the new PM, either use -debug-pass-manager, or use the legacy PM\n",
            0x60u);
        }
        else
        {
          *v142 = _mm_load_si128((const __m128i *)&xmmword_4364640);
          v142[1] = _mm_load_si128((const __m128i *)&xmmword_4364650);
          v142[2] = _mm_load_si128((const __m128i *)&xmmword_4364660);
          v142[3] = _mm_load_si128((const __m128i *)&xmmword_4364670);
          v142[4] = _mm_load_si128((const __m128i *)&xmmword_4364680);
          v142[5] = _mm_load_si128((const __m128i *)&xmmword_4364690);
          *((_QWORD *)v141 + 4) += 96LL;
        }
        goto LABEL_173;
      }
      v93 = v8[968];
      v94 = v8[1008];
      v95 = v8[888];
      v96 = v8[928];
      v97 = v94 + v93 + v96 + v95;
      if ( v97 > 1 )
      {
        v138 = sub_CB72A0();
        v139 = (__m128i *)v138[4];
        if ( v138[3] - (_QWORD)v139 <= 0x1Bu )
        {
          sub_CB6200((__int64)v138, "Cannot specify multiple -O#\n", 0x1Cu);
        }
        else
        {
          v140 = _mm_load_si128((const __m128i *)&xmmword_43646A0);
          qmemcpy(&v139[1], "ultiple -O#\n", 12);
          *v139 = v140;
          v138[4] += 28LL;
        }
        goto LABEL_173;
      }
      v98 = *((_QWORD *)v8 + 169);
      if ( v97 == 1
        || *((_QWORD *)v8 + 132) == 3
        && ((v99 = *((_QWORD *)v8 + 131), *(_WORD *)v99 == 24941) && *(_BYTE *)(v99 + 2) == 120
         || *(_WORD *)v99 == 26989 && *(_BYTE *)(v99 + 2) == 100
         || *(_WORD *)v99 == 26989 && *(_BYTE *)(v99 + 2) == 110) )
      {
        if ( v98 )
        {
          v120 = sub_CB72A0();
          v121 = (__m128i *)v120[4];
          if ( v120[3] - (_QWORD)v121 <= 0x95u )
          {
            sub_CB6200(
              (__int64)v120,
              "Cannot specify -O#/-Ofast-compile=<min,mid,max> and --passes=/--foo-pass, use -passes='default<O#>,other-p"
              "ass' or -passes='default<Ofcmax>,other-pass\n",
              0x96u);
          }
          else
          {
            v122 = _mm_load_si128((const __m128i *)&xmmword_43646B0);
            v121[9].m128i_i32[0] = 1935765549;
            v121[9].m128i_i16[2] = 2675;
            *v121 = v122;
            v121[1] = _mm_load_si128((const __m128i *)&xmmword_43646C0);
            v121[2] = _mm_load_si128((const __m128i *)&xmmword_43646D0);
            v121[3] = _mm_load_si128((const __m128i *)&xmmword_43646E0);
            v121[4] = _mm_load_si128((const __m128i *)&xmmword_43646F0);
            v121[5] = _mm_load_si128((const __m128i *)&xmmword_4364700);
            v121[6] = _mm_load_si128((const __m128i *)&xmmword_4364710);
            v121[7] = _mm_load_si128((const __m128i *)&xmmword_4364720);
            v121[8] = _mm_load_si128((const __m128i *)&xmmword_4364730);
            v120[4] += 150LL;
          }
LABEL_173:
          v22 = 1;
LABEL_174:
          if ( v201 )
            j_j___libc_free_0(v201);
          if ( v200 )
            j_j___libc_free_0(v200);
          v123 = v199;
          if ( v199 )
          {
            v124 = v198;
            v125 = v198 + 40LL * v199;
            do
            {
              while ( 1 )
              {
                if ( *(_DWORD *)v124 <= 0xFFFFFFFD )
                {
                  v126 = *(_QWORD *)(v124 + 8);
                  if ( v126 != v124 + 24 )
                    break;
                }
                v124 += 40;
                if ( v125 == v124 )
                  goto LABEL_184;
              }
              v124 += 40;
              j_j___libc_free_0(v126);
            }
            while ( v125 != v124 );
LABEL_184:
            v123 = v199;
          }
          v19 = 40 * v123;
          sub_C7D6A0(v198, 40 * v123, 8);
          if ( srca )
            (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)srca + 8LL))(srca);
          goto LABEL_115;
        }
      }
      v168 = (const char *)*((_QWORD *)v8 + 168);
      if ( (_BYTE)v96 )
      {
        v98 = 9;
        v168 = "nvopt<O1>";
      }
      else if ( v93 )
      {
        v98 = 9;
        v168 = "nvopt<O2>";
      }
      else if ( v94 )
      {
        v98 = 9;
        v168 = "nvopt<O3>";
      }
      else
      {
        if ( *((_QWORD *)v8 + 132) != 3 )
          goto LABEL_269;
        v147 = (_BYTE *)*((_QWORD *)v8 + 131);
        if ( *(_WORD *)v147 == 24941 && v147[2] == 120 )
        {
          v98 = 13;
          v168 = "nvopt<Ofcmax>";
          goto LABEL_165;
        }
        if ( *(_WORD *)v147 == 26989 && v147[2] == 100 )
        {
          v98 = 13;
          v168 = "nvopt<Ofcmid>";
          goto LABEL_165;
        }
        v161 = *((_QWORD *)v8 + 169);
        v148 = memcmp(v147, "min", 3u);
        v98 = v161;
        if ( v148 )
        {
LABEL_269:
          if ( !v98 || (_BYTE)v95 )
          {
            v98 = 9;
            v168 = "nvopt<O0>";
          }
          goto LABEL_165;
        }
        v98 = 13;
        v168 = "nvopt<Ofcmin>";
      }
LABEL_165:
      v100 = v8[320];
      v101 = v8[400];
      v158 = v8[360];
      v102 = v8[1104];
      v103 = v8[480];
      v104 = v8[520];
      v105 = v8[600];
      v106 = v8[848];
      v107 = v8[1144];
      v155 = (unsigned __int8)**((_DWORD **)v8 + 222) >> 7;
      srca[539449] = v8[280];
      srca[539456] = v100;
      srca[539448] = v101;
      v108 = 0;
      srca[539450] = v102;
      srca[539451] = v103;
      srca[539452] = v104;
      srca[539453] = v105;
      srca[539454] = v106;
      srca[539455] = v107;
      srca[539457] = v158;
      srca[539458] = v155;
      if ( !v8[440] )
      {
        v160 = v98;
        v143 = (char *)sub_C94E20((__int64)qword_4F86270);
        v98 = v160;
        if ( v143 )
          v144 = *v143;
        else
          v144 = qword_4F86270[2];
        v108 = 1 - ((v144 == 0) - 1);
      }
      v109 = (unsigned __int8)qword_4FD87E8;
      v110 = 0;
      v111 = (unsigned __int8)byte_4FD7FA8;
      v112 = (unsigned __int8)qword_4FD8088;
      v113 = **((_DWORD **)v8 + 222);
      v114 = (unsigned __int8)qword_4FD8428;
      v115 = (unsigned __int8)byte_4FD7CE8;
      v159 = (unsigned __int8)qword_4FD8348;
      v116 = (v113 >> 2) & 1;
      v156 = (unsigned __int8)byte_4FD7DC8;
      v117 = *a2;
      if ( *a2 )
      {
        v118 = *a2;
        v150 = v98;
        v151 = v108;
        v152 = (unsigned __int8)qword_4FD8428;
        v153 = (unsigned __int8)qword_4FD8088;
        v154 = (unsigned __int8)byte_4FD7FA8;
        v172 = (unsigned __int8)qword_4FD87E8;
        v119 = strlen(v118);
        v98 = v150;
        v108 = v151;
        v114 = v152;
        v112 = v153;
        v110 = v119;
        v111 = v154;
        v109 = v172;
      }
      v22 = (unsigned __int8)sub_2277440(
                               (_DWORD)v117,
                               v110,
                               v7,
                               (_DWORD)srca,
                               (unsigned int)&v191,
                               0,
                               0,
                               v162,
                               (__int64)v168,
                               v98,
                               0,
                               0,
                               0,
                               0,
                               0,
                               v108,
                               v115,
                               v156,
                               v114,
                               v159,
                               v112,
                               v111,
                               a6,
                               a7,
                               v109,
                               v116)
          ^ 1;
      goto LABEL_174;
    }
    v61 = sub_CC7280((__int64 *)&v185);
    if ( v62 == 7 && *(_DWORD *)v61 == 1852534389 && *(_WORD *)(v61 + 4) == 30575 && *(_BYTE *)(v61 + 6) == 110
      || (sub_CC7280((__int64 *)&v185), !v63) )
    {
      srca = 0;
      goto LABEL_139;
    }
    v64 = (__int64)sub_CB72A0();
    v65 = (char *)*a2;
    if ( *a2 )
    {
      v66 = strlen(*a2);
      v67 = *(__m128i **)(v64 + 32);
      v68 = v66;
      if ( v66 <= *(_QWORD *)(v64 + 24) - (_QWORD)v67 )
      {
        if ( v66 )
        {
          memcpy(v67, v65, v66);
          v67 = (__m128i *)(v68 + *(_QWORD *)(v64 + 32));
          *(_QWORD *)(v64 + 32) = v67;
        }
        goto LABEL_107;
      }
      v64 = sub_CB6200(v64, (unsigned __int8 *)v65, v66);
    }
    v67 = *(__m128i **)(v64 + 32);
LABEL_107:
    if ( *(_QWORD *)(v64 + 24) - (_QWORD)v67 <= 0x1Cu )
    {
      v64 = sub_CB6200(v64, ": unrecognized architecture '", 0x1Du);
    }
    else
    {
      v69 = _mm_load_si128((const __m128i *)&xmmword_4364610);
      qmemcpy(&v67[1], "rchitecture '", 13);
      *v67 = v69;
      *(_QWORD *)(v64 + 32) += 29LL;
    }
    v70 = sub_CC7280((__int64 *)&v185);
    v72 = *(void **)(v64 + 32);
    v19 = v70;
    v73 = v71;
    v74 = *(_QWORD *)(v64 + 24) - (_QWORD)v72;
    if ( v71 > v74 )
    {
      v146 = sub_CB6200(v64, (unsigned __int8 *)v19, v71);
      v72 = *(void **)(v146 + 32);
      v64 = v146;
      v74 = *(_QWORD *)(v146 + 24) - (_QWORD)v72;
    }
    else if ( v71 )
    {
      memcpy(v72, (const void *)v19, v71);
      v149 = *(_QWORD *)(v64 + 24);
      v72 = (void *)(v73 + *(_QWORD *)(v64 + 32));
      *(_QWORD *)(v64 + 32) = v72;
      v74 = v149 - (_QWORD)v72;
    }
    if ( v74 <= 0xB )
    {
      v19 = (__int64)"' provided.\n";
      sub_CB6200(v64, "' provided.\n", 0xCu);
    }
    else
    {
      qmemcpy(v72, "' provided.\n", 12);
      *(_QWORD *)(v64 + 32) += 12LL;
    }
    v22 = 1;
LABEL_115:
    if ( v182 != v184 )
    {
      v19 = v184[0] + 1LL;
      j_j___libc_free_0((unsigned __int64)v182);
    }
    if ( dest != v181 )
    {
      v19 = v181[0] + 1LL;
      j_j___libc_free_0((unsigned __int64)dest);
    }
    if ( (_QWORD *)v185 != v187 )
    {
      v19 = v187[0] + 1LL;
      j_j___libc_free_0(v185);
    }
    goto LABEL_200;
  }
  v127 = (__int64)sub_CB72A0();
  v128 = (char *)*a2;
  if ( *a2 )
  {
    v129 = strlen(*a2);
    v130 = *(_WORD **)(v127 + 32);
    v131 = v129;
    v132 = *(_QWORD *)(v127 + 24) - (_QWORD)v130;
    if ( v131 <= v132 )
    {
      if ( v131 )
      {
        v19 = (__int64)v128;
        memcpy(v130, v128, v131);
        v133 = *(_QWORD *)(v127 + 24);
        v130 = (_WORD *)(v131 + *(_QWORD *)(v127 + 32));
        *(_QWORD *)(v127 + 32) = v130;
        v132 = v133 - (_QWORD)v130;
      }
      goto LABEL_195;
    }
    v19 = (__int64)v128;
    v127 = sub_CB6200(v127, (unsigned __int8 *)v128, v131);
  }
  v130 = *(_WORD **)(v127 + 32);
  v132 = *(_QWORD *)(v127 + 24) - (_QWORD)v130;
LABEL_195:
  if ( v132 <= 1 )
  {
    v19 = (__int64)": ";
    v145 = sub_CB6200(v127, (unsigned __int8 *)": ", 2u);
    v134 = *(_QWORD *)(v145 + 32);
    v127 = v145;
  }
  else
  {
    *v130 = 8250;
    v134 = *(_QWORD *)(v127 + 32) + 2LL;
    *(_QWORD *)(v127 + 32) = v134;
  }
  if ( (unsigned __int64)(*(_QWORD *)(v127 + 24) - v134) <= 0x20 )
  {
    v19 = (__int64)": error: input module is broken!\n";
    sub_CB6200(v127, ": error: input module is broken!\n", 0x21u);
  }
  else
  {
    v135 = _mm_load_si128((const __m128i *)&xmmword_43645C0);
    *(_BYTE *)(v134 + 32) = 10;
    *(__m128i *)v134 = v135;
    *(__m128i *)(v134 + 16) = _mm_load_si128((const __m128i *)&xmmword_43645D0);
    *(_QWORD *)(v127 + 32) += 33LL;
  }
  v22 = 1;
LABEL_200:
  if ( v162 )
  {
    if ( *(_BYTE *)(v162 + 136) )
    {
      *(_BYTE *)(v162 + 136) = 0;
      sub_CB5B00((int *)(v162 + 40), v19);
    }
    sub_CA0D30(v162);
    v19 = 152;
    j_j___libc_free_0(v162);
  }
LABEL_18:
  if ( (v176 & 2) != 0 )
    sub_226C320(&v175, v19);
  v23 = (unsigned __int64)v175;
  if ( (v176 & 1) != 0 )
  {
    if ( v175 )
      (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v175 + 8LL))(v175);
  }
  else if ( v175 )
  {
    if ( v175[136] )
    {
      v175[136] = 0;
      sub_CB5B00((int *)(v23 + 40), v19);
    }
    sub_CA0D30(v23);
    j_j___libc_free_0(v23);
  }
  v24 = v216;
  v25 = (unsigned __int64)&v216[48 * (unsigned int)v217];
  if ( v216 != (_BYTE *)v25 )
  {
    do
    {
      v25 -= 48LL;
      v26 = *(_QWORD *)(v25 + 16);
      if ( v26 != v25 + 32 )
        j_j___libc_free_0(v26);
    }
    while ( v24 != (_BYTE *)v25 );
    v25 = (unsigned __int64)v216;
  }
  if ( (_BYTE *)v25 != v218 )
    _libc_free(v25);
  if ( v213 )
    j_j___libc_free_0(v213);
  if ( v210 != (_QWORD *)v212 )
    j_j___libc_free_0((unsigned __int64)v210);
  if ( v207 != (_QWORD *)v209 )
    j_j___libc_free_0((unsigned __int64)v207);
  if ( v203 != v205 )
    j_j___libc_free_0((unsigned __int64)v203);
  if ( byte_4FD7A48 )
  {
    v27 = qword_4FD7868;
    sub_C9C600((__int64 *)&v173, (char *)qword_4FD7868, qword_4FD7870, (__int64)"-", 1);
    v28 = (_QWORD *)(v173 & 0xFFFFFFFFFFFFFFFELL);
    if ( (v173 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v173 = 0;
      v19 = (__int64)&unk_4F84052;
      v174 = 0;
      v177 = 0;
      if ( (*(unsigned __int8 (__fastcall **)(_QWORD *, void *))(*v28 + 48LL))(v28, &unk_4F84052) )
      {
        dest = (void *)1;
        v170 = (unsigned __int8 **)v28[2];
        if ( (unsigned __int8 **)v28[1] == v170 )
        {
          v31 = 1;
        }
        else
        {
          v29 = (unsigned __int8 **)v28[1];
          do
          {
            v191 = (__int64)*v29;
            *v29 = 0;
            sub_226B280((__int64 *)&v185, &v191);
            v30 = (unsigned __int64)dest;
            v19 = (__int64)v202;
            dest = 0;
            v202[0] = v30 | 1;
            sub_9CDB40((unsigned __int64 *)&v182, v202, &v185);
            if ( ((unsigned __int8)dest & 1) != 0 || ((unsigned __int64)dest & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_C63C30(&dest, (__int64)v202);
            dest = (void *)((unsigned __int64)v182 | (unsigned __int64)dest | 1);
            if ( (v202[0] & 1) != 0 || (v202[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_C63C30(v202, (__int64)v202);
            if ( (v185 & 1) != 0 || (v185 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              goto LABEL_234;
            if ( v191 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v191 + 8LL))(v191);
            ++v29;
          }
          while ( v170 != v29 );
          v31 = (unsigned __int64)dest | 1;
        }
        v191 = v31;
        (*(void (__fastcall **)(_QWORD *))(*v28 + 8LL))(v28);
      }
      else
      {
        v19 = (__int64)v202;
        v202[0] = (unsigned __int64)v28;
        sub_226B280(&v191, v202);
        if ( v202[0] )
          (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v202[0] + 8LL))(v202[0]);
      }
      if ( (v191 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        BUG();
      if ( (v177 & 1) != 0 || (v177 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v177, v19);
      if ( (v174 & 1) != 0 || (v174 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v174, v19);
      if ( (v173 & 1) != 0 || (v173 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v173, v19);
    }
    else
    {
      sub_C99310((__int64)&v173, v27);
    }
  }
  return v22;
}
