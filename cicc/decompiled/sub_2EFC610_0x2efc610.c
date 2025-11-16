// Function: sub_2EFC610
// Address: 0x2efc610
//
__int64 __fastcall sub_2EFC610(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // rbx
  int v8; // eax
  _BYTE *v9; // rcx
  _BYTE **v10; // rsi
  __int64 v11; // r8
  __int64 v12; // r9
  _BYTE *v13; // rdi
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned __int64 v19; // rbx
  __int64 v20; // rcx
  _QWORD *v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rcx
  __int64 *v24; // r15
  _QWORD *v25; // r13
  __int64 *v26; // r14
  __int64 v27; // rdx
  char v28; // r10
  __int64 v29; // rsi
  _QWORD *v30; // rax
  __int64 *v31; // r15
  __int64 *v32; // r14
  char v33; // r10
  __int64 v34; // rsi
  _QWORD *v35; // rax
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 result; // rax
  __int64 v39; // rdx
  _BYTE *v40; // rdx
  __int64 v41; // rbx
  _BYTE *v42; // rax
  _BYTE *i; // rdx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r8
  __int64 v47; // r9
  const __m128i *v48; // rsi
  unsigned __int64 v49; // rdi
  unsigned __int64 v50; // rbx
  __int64 v51; // rax
  __m128i *v52; // rcx
  const __m128i *v53; // rdx
  unsigned __int64 v54; // rdx
  unsigned __int64 v55; // rdi
  unsigned __int64 v56; // r13
  unsigned __int64 v57; // rax
  _BYTE *v58; // rax
  __int64 v59; // rdi
  __m128i *v60; // rax
  __m128i si128; // xmm0
  __int64 v62; // rdi
  __m128i *v63; // rax
  __int64 v64; // rdi
  _BYTE *v65; // rax
  int v66; // r12d
  __int64 v67; // rax
  __int64 v68; // r15
  char v69; // dl
  __int64 v70; // r14
  int v71; // eax
  _BYTE *v72; // rax
  __int64 *v73; // rbx
  __int64 v74; // r15
  __int64 *v75; // rax
  __int64 *v76; // rdx
  _BYTE *v77; // rax
  __int64 v78; // r13
  void *v79; // rdx
  unsigned __int64 v80; // rsi
  __m128i *v81; // rdi
  __int64 v82; // rdx
  __m128i *v83; // rdx
  __m128i v84; // xmm0
  __int64 v85; // rax
  _WORD *v86; // rdx
  __int64 v87; // rdi
  __int64 v88; // rax
  __int64 v89; // rdx
  __int64 v90; // r15
  __m128i *v91; // rdx
  __m128i v92; // xmm0
  __int64 v93; // rax
  _WORD *v94; // rdx
  __int64 v95; // rdi
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 *v98; // rbx
  __int64 v99; // r15
  __int64 *v100; // rax
  __int64 *v101; // rdx
  _BYTE *v102; // rax
  __int64 v103; // r13
  void *v104; // rdx
  __m128i *v105; // rdx
  __m128i v106; // xmm0
  __int64 v107; // rax
  _WORD *v108; // rdx
  __int64 v109; // rdi
  __int64 v110; // rax
  __int64 v111; // rdx
  __int64 v112; // r15
  __m128i *v113; // rdx
  __m128i v114; // xmm0
  __int64 v115; // rax
  _WORD *v116; // rdx
  __int64 v117; // rdi
  __int64 v118; // rax
  __int64 v119; // rdx
  unsigned __int64 v120; // rdx
  unsigned __int64 v121; // rdi
  __int64 v122; // rax
  int v123; // edx
  __int64 j; // rdi
  __int64 v125; // rax
  __int64 v126; // r12
  __int64 v127; // r13
  __int64 *v128; // rax
  __int64 v129; // rcx
  __int64 *v130; // rdx
  __int64 v131; // rbx
  __int64 *v132; // rax
  __int64 v133; // r13
  __int64 (*v134)(); // rax
  __int64 v135; // rax
  __int64 v136; // r9
  int v137; // edi
  int v138; // r11d
  int v139; // r8d
  __int64 v140; // r10
  __int64 v141; // r12
  int v142; // esi
  __int64 *v143; // rdx
  __int64 v144; // rcx
  __int64 v145; // rdx
  __int64 v146; // rsi
  __int64 v147; // rax
  int v148; // r13d
  unsigned int v149; // r10d
  __int64 v150; // rdi
  unsigned int v151; // r10d
  void *v152; // rdx
  __int64 v153; // rax
  unsigned int v154; // r10d
  __int64 v155; // rdi
  __m128i *v156; // rax
  __m128i v157; // xmm0
  __int64 v158; // rax
  __int64 v159; // rdx
  __int64 v160; // rsi
  __int64 v161; // rdx
  char v162; // dl
  __int64 v163; // rax
  __int64 v164; // rax
  __int64 *v165; // rdx
  char *v166; // rsi
  int v167; // [rsp+14h] [rbp-1FCh]
  __int64 v168; // [rsp+28h] [rbp-1E8h]
  unsigned __int8 v169; // [rsp+3Fh] [rbp-1D1h]
  int v170; // [rsp+40h] [rbp-1D0h]
  __int64 v171; // [rsp+40h] [rbp-1D0h]
  __int64 v172; // [rsp+48h] [rbp-1C8h]
  unsigned __int8 v173; // [rsp+48h] [rbp-1C8h]
  unsigned __int64 v174; // [rsp+48h] [rbp-1C8h]
  __int64 v175; // [rsp+50h] [rbp-1C0h]
  int v176; // [rsp+50h] [rbp-1C0h]
  __int64 v177; // [rsp+58h] [rbp-1B8h]
  unsigned __int64 v178; // [rsp+58h] [rbp-1B8h]
  unsigned int v179; // [rsp+58h] [rbp-1B8h]
  unsigned int v180; // [rsp+58h] [rbp-1B8h]
  unsigned __int64 v181; // [rsp+60h] [rbp-1B0h]
  __int64 *v182; // [rsp+60h] [rbp-1B0h]
  __int64 *v183; // [rsp+60h] [rbp-1B0h]
  __int64 v184; // [rsp+68h] [rbp-1A8h]
  unsigned __int64 v185; // [rsp+68h] [rbp-1A8h]
  __int64 v186; // [rsp+70h] [rbp-1A0h]
  __int64 v187; // [rsp+78h] [rbp-198h] BYREF
  __int64 v188; // [rsp+80h] [rbp-190h]
  unsigned __int64 v189; // [rsp+88h] [rbp-188h]
  _QWORD v190[2]; // [rsp+90h] [rbp-180h] BYREF
  void (__fastcall *v191)(_QWORD *, _QWORD *, __int64); // [rsp+A0h] [rbp-170h]
  void (__fastcall *v192)(_QWORD *, __int64); // [rsp+A8h] [rbp-168h]
  _QWORD v193[2]; // [rsp+B0h] [rbp-160h] BYREF
  void (__fastcall *v194)(_QWORD *, _QWORD *, __int64); // [rsp+C0h] [rbp-150h]
  void (__fastcall *v195)(_QWORD *, __int64); // [rsp+C8h] [rbp-148h]
  __m128i v196; // [rsp+D0h] [rbp-140h] BYREF
  void (__fastcall *v197)(__m128i *, __m128i *, __int64); // [rsp+E0h] [rbp-130h]
  void (__fastcall *v198)(__m128i *, __int64); // [rsp+E8h] [rbp-128h]
  __m128i v199; // [rsp+F0h] [rbp-120h] BYREF
  const __m128i *v200; // [rsp+100h] [rbp-110h]
  void (__fastcall *v201)(__m128i *, __int64); // [rsp+108h] [rbp-108h]
  __int64 v202; // [rsp+110h] [rbp-100h] BYREF
  __int64 *v203; // [rsp+118h] [rbp-F8h]
  __int64 v204; // [rsp+120h] [rbp-F0h]
  int v205; // [rsp+128h] [rbp-E8h]
  char v206; // [rsp+12Ch] [rbp-E4h]
  __int64 v207; // [rsp+130h] [rbp-E0h] BYREF
  _BYTE *v208; // [rsp+170h] [rbp-A0h] BYREF
  __int64 v209; // [rsp+178h] [rbp-98h]
  _BYTE v210[48]; // [rsp+180h] [rbp-90h] BYREF
  int v211; // [rsp+1B0h] [rbp-60h]

  v6 = a1;
  v7 = *(_QWORD *)(a1 + 64);
  *(_QWORD *)(a1 + 592) = 0;
  v8 = *(_DWORD *)(v7 + 448);
  if ( v8 )
  {
    v9 = v210;
    v208 = v210;
    v209 = 0x600000000LL;
    if ( *(_DWORD *)(v7 + 392) )
    {
      sub_2EEE630((__int64)&v208, v7 + 384, a3, (__int64)v210, a5, a6);
      v8 = *(_DWORD *)(v7 + 448);
    }
    v211 = v8;
  }
  else
  {
    (*(void (__fastcall **)(_BYTE **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 56) + 136LL))(
      &v208,
      *(_QWORD *)(a1 + 56),
      *(_QWORD *)(a1 + 32));
  }
  v10 = &v208;
  sub_2EEE630(a1 + 200, (__int64)&v208, a3, (__int64)v9, a5, a6);
  v13 = v208;
  *(_DWORD *)(v6 + 264) = v211;
  if ( v13 != v210 )
    _libc_free((unsigned __int64)v13);
  v14 = *(_QWORD *)(v6 + 32);
  v15 = *(_QWORD *)(v14 + 320) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v15 != v14 + 320 )
  {
    v10 = *(_BYTE ***)(v14 + 328);
    sub_2EEFF00(v6, (__int64)v10);
  }
  ++*(_QWORD *)(v6 + 104);
  v172 = v6 + 104;
  if ( !*(_BYTE *)(v6 + 132) )
  {
    v16 = 4 * (*(_DWORD *)(v6 + 124) - *(_DWORD *)(v6 + 128));
    v17 = *(unsigned int *)(v6 + 120);
    if ( v16 < 0x20 )
      v16 = 32;
    if ( (unsigned int)v17 > v16 )
    {
      sub_C8C990(v172, (__int64)v10);
      goto LABEL_13;
    }
    memset(*(void **)(v6 + 112), -1, 8 * v17);
  }
  *(_QWORD *)(v6 + 124) = 0;
LABEL_13:
  v18 = *(_QWORD *)(v6 + 32);
  v175 = v6 + 600;
  v19 = *(_QWORD *)(v18 + 328);
  v177 = v18 + 320;
  if ( v18 + 320 == v19 )
    goto LABEL_44;
  v184 = v6;
  do
  {
    v20 = v184;
    if ( !*(_BYTE *)(v184 + 132) )
      goto LABEL_203;
    v21 = *(_QWORD **)(v184 + 112);
    v20 = *(unsigned int *)(v184 + 124);
    v15 = (unsigned __int64)&v21[v20];
    if ( v21 != (_QWORD *)v15 )
    {
      while ( *v21 != v19 )
      {
        if ( (_QWORD *)v15 == ++v21 )
          goto LABEL_204;
      }
      goto LABEL_20;
    }
LABEL_204:
    if ( (unsigned int)v20 < *(_DWORD *)(v184 + 120) )
    {
      *(_DWORD *)(v184 + 124) = v20 + 1;
      *(_QWORD *)v15 = v19;
      ++*(_QWORD *)(v184 + 104);
    }
    else
    {
LABEL_203:
      sub_C8CC70(v172, v19, v15, v20, v11, v12);
    }
LABEL_20:
    v208 = (_BYTE *)v19;
    v22 = sub_2EEFC50(v175, &v208);
    v24 = *(__int64 **)(v19 + 64);
    v25 = v22;
    v26 = &v24[*(unsigned int *)(v19 + 72)];
    v27 = *(unsigned int *)(v19 + 72);
    if ( v24 == v26 )
      goto LABEL_29;
    v28 = *((_BYTE *)v22 + 196);
    do
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v29 = *v24;
          if ( v28 )
            break;
LABEL_195:
          ++v24;
          sub_C8CC70((__int64)(v25 + 21), v29, v27, v23, v11, v12);
          v28 = *((_BYTE *)v25 + 196);
          if ( v26 == v24 )
            goto LABEL_28;
        }
        v30 = (_QWORD *)v25[22];
        v23 = *((unsigned int *)v25 + 47);
        v27 = (__int64)&v30[v23];
        if ( v30 != (_QWORD *)v27 )
          break;
LABEL_200:
        if ( (unsigned int)v23 >= *((_DWORD *)v25 + 46) )
          goto LABEL_195;
        v23 = (unsigned int)(v23 + 1);
        ++v24;
        *((_DWORD *)v25 + 47) = v23;
        *(_QWORD *)v27 = v29;
        v28 = *((_BYTE *)v25 + 196);
        ++v25[21];
        if ( v26 == v24 )
          goto LABEL_28;
      }
      while ( v29 != *v30 )
      {
        if ( (_QWORD *)v27 == ++v30 )
          goto LABEL_200;
      }
      ++v24;
    }
    while ( v26 != v24 );
LABEL_28:
    LODWORD(v27) = *(_DWORD *)(v19 + 72);
LABEL_29:
    if ( *((_DWORD *)v25 + 47) - *((_DWORD *)v25 + 48) != (_DWORD)v27 )
      sub_2EF03A0(v184, "MBB has duplicate entries in its predecessor list.", v19);
    v31 = *(__int64 **)(v19 + 112);
    v32 = &v31[*(unsigned int *)(v19 + 120)];
    v15 = *(unsigned int *)(v19 + 120);
    if ( v31 != v32 )
    {
      v33 = *((_BYTE *)v25 + 292);
      while ( 1 )
      {
        while ( 1 )
        {
          v34 = *v31;
          if ( v33 )
            break;
LABEL_193:
          ++v31;
          sub_C8CC70((__int64)(v25 + 33), v34, v15, v23, v11, v12);
          v33 = *((_BYTE *)v25 + 292);
          if ( v32 == v31 )
            goto LABEL_39;
        }
        v35 = (_QWORD *)v25[34];
        v23 = *((unsigned int *)v25 + 71);
        v15 = (unsigned __int64)&v35[v23];
        if ( v35 == (_QWORD *)v15 )
        {
LABEL_197:
          if ( (unsigned int)v23 >= *((_DWORD *)v25 + 70) )
            goto LABEL_193;
          v23 = (unsigned int)(v23 + 1);
          ++v31;
          *((_DWORD *)v25 + 71) = v23;
          *(_QWORD *)v15 = v34;
          v33 = *((_BYTE *)v25 + 292);
          ++v25[33];
          if ( v32 == v31 )
            goto LABEL_39;
        }
        else
        {
          while ( v34 != *v35 )
          {
            if ( (_QWORD *)v15 == ++v35 )
              goto LABEL_197;
          }
          if ( v32 == ++v31 )
          {
LABEL_39:
            v15 = *(unsigned int *)(v19 + 120);
            break;
          }
        }
      }
    }
    if ( *((_DWORD *)v25 + 71) - *((_DWORD *)v25 + 72) != (_DWORD)v15 )
      sub_2EF03A0(v184, "MBB has duplicate entries in its successor list.", v19);
    v19 = *(_QWORD *)(v19 + 8);
  }
  while ( v177 != v19 );
  v6 = v184;
LABEL_44:
  nullsub_1608();
  result = *(_QWORD *)(v6 + 32);
  if ( result + 320 == (*(_QWORD *)(result + 320) & 0xFFFFFFFFFFFFFFF8LL) )
    return result;
  v39 = *(_QWORD *)(v6 + 48);
  v176 = *(_DWORD *)(v39 + 64);
  v167 = *(_DWORD *)(v39 + 68);
  if ( (v167 & v176) != 0xFFFFFFFF )
  {
    v40 = v210;
    v209 = 0x800000000LL;
    v208 = v210;
    v41 = (__int64)(*(_QWORD *)(result + 104) - *(_QWORD *)(result + 96)) >> 3;
    if ( (_DWORD)v41 )
    {
      v42 = v210;
      if ( (unsigned int)v41 > 8uLL )
      {
        sub_C8D5F0((__int64)&v208, v210, (unsigned int)v41, 0xCu, v36, v37);
        v40 = v208;
        v42 = &v208[12 * (unsigned int)v209];
      }
      for ( i = &v40[12 * (unsigned int)v41]; i != v42; v42 += 12 )
      {
        if ( v42 )
        {
          *(_DWORD *)v42 = 0;
          *((_DWORD *)v42 + 1) = 0;
          v42[8] = 0;
          v42[9] = 0;
        }
      }
      LODWORD(v209) = v41;
      result = *(_QWORD *)(v6 + 32);
    }
    v44 = *(_QWORD *)(result + 328);
    v204 = 0x100000008LL;
    v207 = v44;
    v196.m128i_i64[0] = v44;
    v203 = &v207;
    v199.m128i_i64[0] = (__int64)&v202;
    v199.m128i_i64[1] = 0;
    v200 = 0;
    v201 = 0;
    v205 = 0;
    v206 = 1;
    v202 = 1;
    LOBYTE(v197) = 0;
    sub_2EFC5D0(&v199.m128i_u64[1], &v196);
    v48 = v200;
    v187 = 0;
    v49 = v199.m128i_u64[1];
    v188 = 0;
    v186 = v199.m128i_i64[0];
    v189 = 0;
    v50 = (unsigned __int64)v200 - v199.m128i_i64[1];
    if ( v200 == (const __m128i *)v199.m128i_i64[1] )
    {
      v50 = 0;
      v51 = 0;
    }
    else
    {
      if ( v50 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4261EA(v199.m128i_i64[1], v200, v45);
      v51 = sub_22077B0((unsigned __int64)v200 - v199.m128i_i64[1]);
      v48 = v200;
      v49 = v199.m128i_u64[1];
    }
    v187 = v51;
    v188 = v51;
    v189 = v51 + v50;
    if ( v48 != (const __m128i *)v49 )
    {
      v52 = (__m128i *)v51;
      v53 = (const __m128i *)v49;
      do
      {
        if ( v52 )
        {
          *v52 = _mm_loadu_si128(v53);
          v46 = v53[1].m128i_i64[0];
          v52[1].m128i_i64[0] = v46;
        }
        v53 = (const __m128i *)((char *)v53 + 24);
        v52 = (__m128i *)((char *)v52 + 24);
      }
      while ( v53 != v48 );
      v54 = (unsigned __int64)&v53[-2].m128i_u64[1] - v49;
      v49 = v199.m128i_u64[1];
      v51 += 8 * (v54 >> 3) + 24;
    }
    v188 = v51;
    if ( v49 )
    {
      j_j___libc_free_0(v49);
      v51 = v188;
    }
    v55 = v187;
LABEL_66:
    if ( v51 != v55 )
    {
      v56 = *(_QWORD *)(v51 - 24);
      v57 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v51 - v55) >> 3);
      if ( (unsigned int)v57 <= 1 )
      {
        v173 = 0;
        v170 = 0;
      }
      else
      {
        v58 = &v208[12 * *(int *)(*(_QWORD *)(v55 + 24LL * (unsigned int)(v57 - 2)) + 24LL)];
        v170 = *((_DWORD *)v58 + 1);
        v173 = v58[9];
      }
      if ( *(_DWORD *)(v56 + 28) != -v170 )
      {
        sub_2EF03A0(v6, "Call frame size on entry does not match value computed from predecessor", v56);
        v59 = *(_QWORD *)(v6 + 16);
        v60 = *(__m128i **)(v59 + 32);
        if ( *(_QWORD *)(v59 + 24) - (_QWORD)v60 <= 0x18u )
        {
          v59 = sub_CB6200(v59, "Call frame size on entry ", 0x19u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_4453E20);
          v60[1].m128i_i8[8] = 32;
          v60[1].m128i_i64[0] = 0x7972746E65206E6FLL;
          *v60 = si128;
          *(_QWORD *)(v59 + 32) += 25LL;
        }
        v62 = sub_CB59D0(v59, *(unsigned int *)(v56 + 28));
        v63 = *(__m128i **)(v62 + 32);
        if ( *(_QWORD *)(v62 + 24) - (_QWORD)v63 <= 0x2Fu )
        {
          v62 = sub_CB6200(v62, " does not match value computed from predecessor ", 0x30u);
        }
        else
        {
          *v63 = _mm_load_si128((const __m128i *)&xmmword_4453E30);
          v63[1] = _mm_load_si128((const __m128i *)&xmmword_4453E40);
          v63[2] = _mm_load_si128((const __m128i *)&xmmword_4453E50);
          *(_QWORD *)(v62 + 32) += 48LL;
        }
        v64 = sub_CB59F0(v62, -v170);
        v65 = *(_BYTE **)(v64 + 32);
        if ( (unsigned __int64)v65 >= *(_QWORD *)(v64 + 24) )
        {
          sub_CB5D20(v64, 10);
        }
        else
        {
          *(_QWORD *)(v64 + 32) = v65 + 1;
          *v65 = 10;
        }
      }
      v66 = v170;
      v185 = v56 + 48;
      if ( *(_QWORD *)(v56 + 56) == v56 + 48 )
      {
        v169 = v173;
      }
      else
      {
        v67 = v6;
        v181 = v56;
        v68 = *(_QWORD *)(v56 + 56);
        v69 = v173;
        v70 = v67;
        do
        {
          while ( 1 )
          {
            v71 = *(unsigned __int16 *)(v68 + 68);
            if ( v176 == v71 )
            {
              if ( v69 )
                sub_2EF06E0(v70, "FrameSetup is after another FrameSetup", v68);
              if ( (*(_BYTE *)(**(_QWORD **)(v70 + 64) + 344LL) & 1) == 0
                && !*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v70 + 32) + 48LL) + 65LL) )
              {
                sub_2EF06E0(v70, "AdjustsStack not set in presence of a frame pseudo instruction.", v68);
              }
              v160 = *(_QWORD *)(v68 + 32);
              v71 = *(unsigned __int16 *)(v68 + 68);
              v161 = *(_QWORD *)(v160 + 24);
              if ( *(_DWORD *)(*(_QWORD *)(v70 + 48) + 64LL) == v71 )
                v161 += *(_QWORD *)(v160 + 64);
              v66 -= v161;
              v69 = 1;
            }
            if ( v167 == v71 )
            {
              v146 = *(_QWORD *)(v68 + 32);
              v147 = *(_QWORD *)(v146 + 24);
              if ( *(_DWORD *)(*(_QWORD *)(v70 + 48) + 64LL) == v167 )
                v147 += *(_QWORD *)(v146 + 64);
              v148 = v147;
              if ( v69 )
              {
                v149 = abs32(v66);
                if ( (_DWORD)v147 != v149 )
                {
                  v179 = v149;
                  sub_2EF06E0(v70, "FrameDestroy <n> is after FrameSetup <m>", v68);
                  v150 = *(_QWORD *)(v70 + 16);
                  v151 = v179;
                  v152 = *(void **)(v150 + 32);
                  if ( *(_QWORD *)(v150 + 24) - (_QWORD)v152 <= 0xDu )
                  {
                    v164 = sub_CB6200(v150, "FrameDestroy <", 0xEu);
                    v151 = v179;
                    v150 = v164;
                  }
                  else
                  {
                    qmemcpy(v152, "FrameDestroy <", 14);
                    *(_QWORD *)(v150 + 32) += 14LL;
                  }
                  v180 = v151;
                  v153 = sub_CB59F0(v150, v148);
                  v154 = v180;
                  v155 = v153;
                  v156 = *(__m128i **)(v153 + 32);
                  if ( *(_QWORD *)(v155 + 24) - (_QWORD)v156 <= 0x16u )
                  {
                    v163 = sub_CB6200(v155, "> is after FrameSetup <", 0x17u);
                    v154 = v180;
                    v155 = v163;
                  }
                  else
                  {
                    v157 = _mm_load_si128((const __m128i *)&xmmword_4453E60);
                    v156[1].m128i_i32[0] = 1970562387;
                    v156[1].m128i_i16[2] = 8304;
                    v156[1].m128i_i8[6] = 60;
                    *v156 = v157;
                    *(_QWORD *)(v155 + 32) += 23LL;
                  }
                  v158 = sub_CB59F0(v155, (int)v154);
                  v159 = *(_QWORD *)(v158 + 32);
                  if ( (unsigned __int64)(*(_QWORD *)(v158 + 24) - v159) <= 2 )
                  {
                    sub_CB6200(v158, ">.\n", 3u);
                  }
                  else
                  {
                    v46 = 11838;
                    *(_BYTE *)(v159 + 2) = 10;
                    *(_WORD *)v159 = 11838;
                    *(_QWORD *)(v158 + 32) += 3LL;
                  }
                }
              }
              else
              {
                sub_2EF06E0(v70, "FrameDestroy is not after a FrameSetup", v68);
              }
              if ( (*(_BYTE *)(**(_QWORD **)(v70 + 64) + 344LL) & 1) == 0
                && !*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v70 + 32) + 48LL) + 65LL) )
              {
                sub_2EF06E0(v70, "AdjustsStack not set in presence of a frame pseudo instruction.", v68);
              }
              v66 += v148;
              v69 = 0;
            }
            if ( (*(_BYTE *)v68 & 4) == 0 )
              break;
            v68 = *(_QWORD *)(v68 + 8);
            if ( v185 == v68 )
              goto LABEL_83;
          }
          while ( (*(_BYTE *)(v68 + 44) & 8) != 0 )
            v68 = *(_QWORD *)(v68 + 8);
          v68 = *(_QWORD *)(v68 + 8);
        }
        while ( v185 != v68 );
LABEL_83:
        v169 = v69;
        v56 = v181;
        v6 = v70;
      }
      v72 = &v208[12 * *(int *)(v56 + 24)];
      v72[8] = v173;
      *(_DWORD *)v72 = v170;
      v72[9] = v169;
      *((_DWORD *)v72 + 1) = v66;
      v73 = *(__int64 **)(v56 + 64);
      v182 = &v73[*(unsigned int *)(v56 + 72)];
      if ( v73 == v182 )
        goto LABEL_115;
      v178 = v56;
      v168 = v6;
      while ( 2 )
      {
        v74 = *v73;
        if ( v206 )
        {
          v75 = v203;
          v76 = &v203[HIDWORD(v204)];
          if ( v203 == v76 )
            goto LABEL_113;
          while ( v74 != *v75 )
          {
            if ( v76 == ++v75 )
              goto LABEL_113;
          }
LABEL_91:
          v77 = &v208[12 * *(int *)(v74 + 24)];
          if ( v170 != *((_DWORD *)v77 + 1) || v77[9] != v173 )
          {
            sub_2EF03A0(v168, "The exit stack state of a predecessor is inconsistent.", v178);
            v78 = *(_QWORD *)(v168 + 16);
            v79 = *(void **)(v78 + 32);
            if ( *(_QWORD *)(v78 + 24) - (_QWORD)v79 <= 0xBu )
            {
              v78 = sub_CB6200(v78, "Predecessor ", 0xCu);
            }
            else
            {
              qmemcpy(v79, "Predecessor ", 12);
              *(_QWORD *)(v78 + 32) += 12LL;
            }
            v80 = v74;
            v81 = &v199;
            sub_2E31000(&v199, v74);
            if ( !v200 )
              goto LABEL_276;
            v201(&v199, v78);
            v83 = *(__m128i **)(v78 + 32);
            if ( *(_QWORD *)(v78 + 24) - (_QWORD)v83 <= 0x10u )
            {
              v78 = sub_CB6200(v78, " has exit state (", 0x11u);
            }
            else
            {
              v84 = _mm_load_si128((const __m128i *)&xmmword_4453E70);
              v83[1].m128i_i8[0] = 40;
              *v83 = v84;
              *(_QWORD *)(v78 + 32) += 17LL;
            }
            v85 = sub_CB59F0(v78, *(int *)&v208[12 * *(int *)(v74 + 24) + 4]);
            v86 = *(_WORD **)(v85 + 32);
            v87 = v85;
            if ( *(_QWORD *)(v85 + 24) - (_QWORD)v86 <= 1u )
            {
              v87 = sub_CB6200(v85, (unsigned __int8 *)", ", 2u);
            }
            else
            {
              *v86 = 8236;
              *(_QWORD *)(v85 + 32) += 2LL;
            }
            v88 = sub_CB59F0(v87, (unsigned __int8)v208[12 * *(int *)(v74 + 24) + 9]);
            v89 = *(_QWORD *)(v88 + 32);
            v90 = v88;
            if ( (unsigned __int64)(*(_QWORD *)(v88 + 24) - v89) <= 8 )
            {
              v90 = sub_CB6200(v88, "), while ", 9u);
            }
            else
            {
              *(_BYTE *)(v89 + 8) = 32;
              *(_QWORD *)v89 = 0x656C696877202C29LL;
              *(_QWORD *)(v88 + 32) += 9LL;
            }
            v80 = v178;
            v81 = &v196;
            sub_2E31000(&v196, v178);
            if ( !v197 )
LABEL_276:
              sub_4263D6(v81, v80, v82);
            v198(&v196, v90);
            v91 = *(__m128i **)(v90 + 32);
            if ( *(_QWORD *)(v90 + 24) - (_QWORD)v91 <= 0x11u )
            {
              v90 = sub_CB6200(v90, " has entry state (", 0x12u);
            }
            else
            {
              v92 = _mm_load_si128((const __m128i *)&xmmword_4453E80);
              v91[1].m128i_i16[0] = 10272;
              *v91 = v92;
              *(_QWORD *)(v90 + 32) += 18LL;
            }
            v93 = sub_CB59F0(v90, v170);
            v94 = *(_WORD **)(v93 + 32);
            v95 = v93;
            if ( *(_QWORD *)(v93 + 24) - (_QWORD)v94 <= 1u )
            {
              v95 = sub_CB6200(v93, (unsigned __int8 *)", ", 2u);
            }
            else
            {
              *v94 = 8236;
              *(_QWORD *)(v93 + 32) += 2LL;
            }
            v96 = sub_CB59F0(v95, v173);
            v97 = *(_QWORD *)(v96 + 32);
            if ( (unsigned __int64)(*(_QWORD *)(v96 + 24) - v97) <= 2 )
            {
              sub_CB6200(v96, ").\n", 3u);
            }
            else
            {
              *(_BYTE *)(v97 + 2) = 10;
              *(_WORD *)v97 = 11817;
              *(_QWORD *)(v96 + 32) += 3LL;
            }
            if ( v197 )
              v197(&v196, &v196, 3);
            if ( v200 )
              ((void (__fastcall *)(__m128i *, __m128i *, __int64))v200)(&v199, &v199, 3);
          }
        }
        else if ( sub_C8CA60((__int64)&v202, v74) )
        {
          goto LABEL_91;
        }
LABEL_113:
        if ( v182 == ++v73 )
        {
          v56 = v178;
          v6 = v168;
LABEL_115:
          v98 = *(__int64 **)(v56 + 112);
          v183 = &v98[*(unsigned int *)(v56 + 120)];
          if ( v98 == v183 )
            goto LABEL_146;
          v174 = v56;
          v171 = v6;
          while ( 2 )
          {
            v99 = *v98;
            if ( v206 )
            {
              v100 = v203;
              v101 = &v203[HIDWORD(v204)];
              if ( v203 == v101 )
                goto LABEL_144;
              while ( v99 != *v100 )
              {
                if ( v101 == ++v100 )
                  goto LABEL_144;
              }
LABEL_122:
              v102 = &v208[12 * *(int *)(v99 + 24)];
              if ( *(_DWORD *)v102 != v66 || v102[8] != v169 )
              {
                sub_2EF03A0(v171, "The entry stack state of a successor is inconsistent.", v174);
                v103 = *(_QWORD *)(v171 + 16);
                v104 = *(void **)(v103 + 32);
                if ( *(_QWORD *)(v103 + 24) - (_QWORD)v104 <= 9u )
                {
                  v103 = sub_CB6200(v103, "Successor ", 0xAu);
                }
                else
                {
                  qmemcpy(v104, "Successor ", 10);
                  *(_QWORD *)(v103 + 32) += 10LL;
                }
                v80 = v99;
                v81 = (__m128i *)v193;
                sub_2E31000(v193, v99);
                if ( !v194 )
                  goto LABEL_276;
                v195(v193, v103);
                v105 = *(__m128i **)(v103 + 32);
                if ( *(_QWORD *)(v103 + 24) - (_QWORD)v105 <= 0x11u )
                {
                  v103 = sub_CB6200(v103, " has entry state (", 0x12u);
                }
                else
                {
                  v106 = _mm_load_si128((const __m128i *)&xmmword_4453E80);
                  v105[1].m128i_i16[0] = 10272;
                  *v105 = v106;
                  *(_QWORD *)(v103 + 32) += 18LL;
                }
                v107 = sub_CB59F0(v103, *(int *)&v208[12 * *(int *)(v99 + 24)]);
                v108 = *(_WORD **)(v107 + 32);
                v109 = v107;
                if ( *(_QWORD *)(v107 + 24) - (_QWORD)v108 <= 1u )
                {
                  v109 = sub_CB6200(v107, (unsigned __int8 *)", ", 2u);
                }
                else
                {
                  *v108 = 8236;
                  *(_QWORD *)(v107 + 32) += 2LL;
                }
                v110 = sub_CB59F0(v109, (unsigned __int8)v208[12 * *(int *)(v99 + 24) + 8]);
                v111 = *(_QWORD *)(v110 + 32);
                v112 = v110;
                if ( (unsigned __int64)(*(_QWORD *)(v110 + 24) - v111) <= 8 )
                {
                  v112 = sub_CB6200(v110, "), while ", 9u);
                }
                else
                {
                  *(_BYTE *)(v111 + 8) = 32;
                  *(_QWORD *)v111 = 0x656C696877202C29LL;
                  *(_QWORD *)(v110 + 32) += 9LL;
                }
                v80 = v174;
                v81 = (__m128i *)v190;
                sub_2E31000(v190, v174);
                if ( !v191 )
                  goto LABEL_276;
                v192(v190, v112);
                v113 = *(__m128i **)(v112 + 32);
                if ( *(_QWORD *)(v112 + 24) - (_QWORD)v113 <= 0x10u )
                {
                  v112 = sub_CB6200(v112, " has exit state (", 0x11u);
                }
                else
                {
                  v114 = _mm_load_si128((const __m128i *)&xmmword_4453E70);
                  v113[1].m128i_i8[0] = 40;
                  *v113 = v114;
                  *(_QWORD *)(v112 + 32) += 17LL;
                }
                v115 = sub_CB59F0(v112, v66);
                v116 = *(_WORD **)(v115 + 32);
                v117 = v115;
                if ( *(_QWORD *)(v115 + 24) - (_QWORD)v116 <= 1u )
                {
                  v117 = sub_CB6200(v115, (unsigned __int8 *)", ", 2u);
                }
                else
                {
                  *v116 = 8236;
                  *(_QWORD *)(v115 + 32) += 2LL;
                }
                v118 = sub_CB59F0(v117, v169);
                v119 = *(_QWORD *)(v118 + 32);
                if ( (unsigned __int64)(*(_QWORD *)(v118 + 24) - v119) <= 2 )
                {
                  sub_CB6200(v118, ").\n", 3u);
                }
                else
                {
                  *(_BYTE *)(v119 + 2) = 10;
                  *(_WORD *)v119 = 11817;
                  *(_QWORD *)(v118 + 32) += 3LL;
                }
                if ( v191 )
                  v191(v190, v190, 3);
                if ( v194 )
                  v194(v193, v193, 3);
              }
            }
            else if ( sub_C8CA60((__int64)&v202, v99) )
            {
              goto LABEL_122;
            }
LABEL_144:
            if ( v183 != ++v98 )
              continue;
            break;
          }
          v56 = v174;
          v6 = v171;
LABEL_146:
          v120 = *(_QWORD *)(v56 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          v121 = v120;
          if ( v185 != v120 )
          {
            if ( !v120 )
              BUG();
            v122 = *(_QWORD *)v120;
            v123 = *(_DWORD *)(v120 + 44);
            if ( (v122 & 4) != 0 )
            {
              if ( (v123 & 4) == 0 )
                goto LABEL_153;
            }
            else
            {
              if ( (v123 & 4) != 0 )
              {
                for ( j = v122; ; j = *(_QWORD *)v121 )
                {
                  v121 = j & 0xFFFFFFFFFFFFFFF8LL;
                  v123 = *(_DWORD *)(v121 + 44) & 0xFFFFFF;
                  if ( (*(_DWORD *)(v121 + 44) & 4) == 0 )
                    break;
                }
              }
LABEL_153:
              if ( (v123 & 8) != 0 )
              {
                LOBYTE(v125) = sub_2E88A90(v121, 32, 1);
                goto LABEL_155;
              }
            }
            v125 = (*(_QWORD *)(*(_QWORD *)(v121 + 16) + 24LL) >> 5) & 1LL;
LABEL_155:
            if ( (_BYTE)v125 )
            {
              if ( v169 )
              {
                sub_2EF03A0(v6, "A return block ends with a FrameSetup.", v56);
                if ( v66 )
                  goto LABEL_271;
              }
              else
              {
                if ( !v66 )
                  goto LABEL_158;
LABEL_271:
                sub_2EF03A0(v6, "A return block ends with a nonzero stack adjustment.", v56);
              }
            }
          }
LABEL_158:
          v126 = v188;
LABEL_159:
          v127 = *(_QWORD *)(v126 - 24);
          if ( !*(_BYTE *)(v126 - 8) )
          {
            v128 = *(__int64 **)(v127 + 112);
            *(_BYTE *)(v126 - 8) = 1;
            *(_QWORD *)(v126 - 16) = v128;
            goto LABEL_161;
          }
          while ( 1 )
          {
            v128 = *(__int64 **)(v126 - 16);
LABEL_161:
            v129 = *(unsigned int *)(v127 + 120);
            if ( v128 == (__int64 *)(*(_QWORD *)(v127 + 112) + 8 * v129) )
            {
              v188 -= 24;
              v55 = v187;
              v126 = v188;
              if ( v188 == v187 )
              {
                v51 = v187;
                goto LABEL_66;
              }
              goto LABEL_159;
            }
            v130 = v128 + 1;
            *(_QWORD *)(v126 - 16) = v128 + 1;
            v131 = *v128;
            if ( *(_BYTE *)(v186 + 28) )
            {
              v132 = *(__int64 **)(v186 + 8);
              v129 = *(unsigned int *)(v186 + 20);
              v130 = &v132[v129];
              if ( v132 != v130 )
              {
                while ( v131 != *v132 )
                {
                  if ( v130 == ++v132 )
                    goto LABEL_239;
                }
                continue;
              }
LABEL_239:
              if ( (unsigned int)v129 < *(_DWORD *)(v186 + 16) )
              {
                *(_DWORD *)(v186 + 20) = v129 + 1;
                *v130 = v131;
                ++*(_QWORD *)v186;
LABEL_238:
                v199.m128i_i64[0] = v131;
                LOBYTE(v200) = 0;
                sub_2EFC5D0((unsigned __int64 *)&v187, &v199);
                v55 = v187;
                v51 = v188;
                goto LABEL_66;
              }
            }
            sub_C8CC70(v186, v131, (__int64)v130, v129, v46, v47);
            if ( v162 )
              goto LABEL_238;
          }
        }
        continue;
      }
    }
    if ( v55 )
      j_j___libc_free_0(v55);
    if ( !v206 )
      _libc_free((unsigned __int64)v203);
    if ( v208 != v210 )
      _libc_free((unsigned __int64)v208);
    result = *(_QWORD *)(v6 + 32);
  }
  v133 = *(_QWORD *)(result + 48);
  if ( *(_DWORD *)(v133 + 68) == -1 || !*(_QWORD *)(v133 + 48) )
    return result;
  v134 = *(__int64 (**)())(**(_QWORD **)(result + 16) + 136LL);
  if ( v134 == sub_2DD19D0 )
    BUG();
  v135 = v134();
  v136 = *(_QWORD *)(v133 + 8);
  v137 = *(_DWORD *)(v133 + 68);
  v138 = *(_DWORD *)(v135 + 8);
  v139 = *(_DWORD *)(v133 + 32);
  result = v136 + 40LL * (unsigned int)(v137 + v139);
  v140 = *(_QWORD *)result;
  v141 = *(_QWORD *)result + *(_QWORD *)(result + 8);
  v142 = -858993459 * ((*(_QWORD *)(v133 + 16) - v136) >> 3) - v139;
  if ( !v142 )
    return result;
  LODWORD(result) = 0;
  while ( 2 )
  {
    if ( v137 == (_DWORD)result
      || (v143 = (__int64 *)(v136 + 40LL * (unsigned int)(v139 + result)), v144 = v143[1], v144 == -1)
      || *((_BYTE *)v143 + 20)
      || !v144
      || *((_BYTE *)v143 + 18) )
    {
LABEL_191:
      result = (unsigned int)(result + 1);
      if ( v142 == (_DWORD)result )
        return result;
      continue;
    }
    break;
  }
  v145 = *v143;
  if ( v140 >= v145 + v144 || v141 <= v145 )
  {
    if ( v140 <= v145 && v138 == 1 || v138 != 1 && v140 >= v145 )
    {
      v165 = *(__int64 **)(v6 + 32);
      v166 = "Stack protector is not the top-most object on the stack";
      return sub_2EEFF60(v6, v166, v165);
    }
    goto LABEL_191;
  }
  v165 = *(__int64 **)(v6 + 32);
  v166 = "Stack protector overlaps with another stack object";
  return sub_2EEFF60(v6, v166, v165);
}
