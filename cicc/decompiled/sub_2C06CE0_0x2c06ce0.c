// Function: sub_2C06CE0
// Address: 0x2c06ce0
//
void __fastcall sub_2C06CE0(__int64 a1, __int64 a2, _QWORD *a3, char a4, char a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // r12
  int v10; // r8d
  __int64 *v11; // rdi
  __int64 v12; // rsi
  _QWORD *v13; // rax
  int v14; // r8d
  _QWORD *v15; // rdi
  __int64 v16; // rsi
  _QWORD *v17; // rax
  int v18; // r8d
  int v19; // r8d
  __int64 v20; // r14
  _QWORD *v21; // rax
  int v22; // r8d
  char *v23; // r9
  _QWORD *v24; // rdi
  __int64 v25; // rsi
  _QWORD *v26; // rax
  int v27; // r8d
  __int64 v28; // r8
  __int64 v29; // r9
  _BYTE *v30; // rsi
  unsigned __int64 v31; // rdx
  __int64 *v32; // rdi
  __int64 v33; // rax
  __int64 v34; // r12
  __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 *v38; // r12
  __int64 v39; // rax
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // r12
  _BYTE *v43; // rsi
  unsigned __int64 v44; // rdx
  __int64 *v45; // rdi
  __int64 v46; // rax
  unsigned __int64 v47; // rcx
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // r15
  _BYTE *v53; // rsi
  __int64 v54; // rdx
  __int64 v55; // rax
  char *v56; // rdi
  __int64 v57; // rax
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rax
  unsigned __int64 v60; // rax
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rax
  unsigned __int64 v63; // rax
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 v66; // r9
  unsigned __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  const __m128i *v70; // rsi
  __int64 v71; // rdx
  __int64 v72; // r8
  __int64 v73; // r9
  __int64 v74; // rcx
  __int64 *v75; // rdi
  unsigned __int64 v76; // rbx
  __int64 v77; // rax
  unsigned __int64 v78; // rsi
  __m128i *v79; // rdx
  const __m128i *v80; // rax
  __int64 v81; // r8
  __int64 v82; // r9
  const __m128i *v83; // rcx
  unsigned __int64 v84; // rbx
  __int64 v85; // rax
  unsigned __int64 v86; // rdi
  __m128i *v87; // rdx
  const __m128i *v88; // rax
  __int64 v89; // rcx
  unsigned __int64 v90; // rax
  __int64 v91; // rdx
  unsigned __int64 v92; // rbx
  __int64 v93; // r14
  __int64 *v94; // rax
  __int64 v95; // rcx
  __int64 *v96; // rdx
  __int64 v97; // r13
  __int64 *v98; // rax
  char v99; // dl
  char v100; // si
  __int64 v101; // rbx
  __int64 v102; // r8
  __int64 v103; // r9
  _BYTE *v104; // rsi
  unsigned __int64 v105; // rdx
  __int64 *v106; // rdi
  __int64 v107; // rax
  unsigned __int64 v108; // rcx
  __int64 v109; // rdx
  __int64 v110; // r8
  __int64 v111; // r9
  __int64 v112; // r14
  _BYTE *v113; // rsi
  unsigned __int64 v114; // rdx
  __int64 *v115; // rdi
  __int64 v116; // rax
  __int64 v117; // rax
  unsigned __int64 v118; // rcx
  __int64 v119; // r15
  __int64 v120; // rax
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // rdx
  __int64 v124; // rcx
  __int64 v125; // r8
  __int64 v126; // r9
  __int64 v127; // r8
  __int64 v128; // r9
  __int64 v129; // rax
  __int64 v130; // rax
  unsigned __int64 v131; // rcx
  _QWORD *v132; // rdx
  unsigned __int64 v133; // rax
  int v134; // edx
  unsigned __int64 v135; // r14
  unsigned __int64 v136; // rax
  __int64 v137; // rsi
  int v138; // r15d
  __int64 v139; // r13
  _QWORD *v140; // rax
  __int64 v141; // rsi
  __int64 v142; // rdx
  __int64 v143; // r13
  __int64 v144; // rsi
  __int64 v145; // rsi
  __int64 v146; // r8
  __int64 v147; // r14
  __int64 v148; // rdx
  __int64 v149; // r9
  __int64 v150; // rsi
  __int64 v151; // rsi
  __int64 v152; // rsi
  __int64 v153; // rcx
  __int64 v154; // rsi
  __int64 v155; // rax
  __int64 v156; // rax
  unsigned __int64 v157; // rcx
  _QWORD **v158; // rax
  __int64 v159; // rax
  __int64 v160; // rax
  _QWORD *v163; // [rsp+10h] [rbp-380h]
  __int64 v164; // [rsp+18h] [rbp-378h]
  __int64 v166; // [rsp+30h] [rbp-360h]
  __int64 v168; // [rsp+40h] [rbp-350h]
  _QWORD *v169; // [rsp+40h] [rbp-350h]
  __int64 v170; // [rsp+48h] [rbp-348h]
  __int64 v171; // [rsp+48h] [rbp-348h]
  __int64 v172; // [rsp+60h] [rbp-330h] BYREF
  __int64 v173; // [rsp+68h] [rbp-328h] BYREF
  __int64 v174[16]; // [rsp+70h] [rbp-320h] BYREF
  __m128i v175; // [rsp+F0h] [rbp-2A0h] BYREF
  __int64 v176; // [rsp+100h] [rbp-290h]
  int v177; // [rsp+108h] [rbp-288h]
  char v178; // [rsp+10Ch] [rbp-284h]
  _QWORD v179[8]; // [rsp+110h] [rbp-280h] BYREF
  unsigned __int64 v180; // [rsp+150h] [rbp-240h] BYREF
  unsigned __int64 v181; // [rsp+158h] [rbp-238h]
  unsigned __int64 v182; // [rsp+160h] [rbp-230h]
  _BYTE *v183; // [rsp+170h] [rbp-220h] BYREF
  __int64 *v184; // [rsp+178h] [rbp-218h]
  unsigned int v185; // [rsp+180h] [rbp-210h]
  unsigned int v186; // [rsp+184h] [rbp-20Ch]
  char v187; // [rsp+18Ch] [rbp-204h]
  _BYTE v188[64]; // [rsp+190h] [rbp-200h] BYREF
  unsigned __int64 v189; // [rsp+1D0h] [rbp-1C0h] BYREF
  unsigned __int64 v190; // [rsp+1D8h] [rbp-1B8h]
  unsigned __int64 v191; // [rsp+1E0h] [rbp-1B0h]
  _BYTE *v192; // [rsp+1F0h] [rbp-1A0h] BYREF
  unsigned __int64 v193; // [rsp+1F8h] [rbp-198h]
  __int64 v194; // [rsp+200h] [rbp-190h] BYREF
  char v195; // [rsp+20Ch] [rbp-184h]
  _BYTE v196[64]; // [rsp+210h] [rbp-180h] BYREF
  unsigned __int64 v197; // [rsp+250h] [rbp-140h]
  __int64 v198; // [rsp+258h] [rbp-138h]
  __int64 v199; // [rsp+260h] [rbp-130h]
  __m128i v200; // [rsp+270h] [rbp-120h] BYREF
  char v201[16]; // [rsp+280h] [rbp-110h] BYREF
  __int16 v202; // [rsp+290h] [rbp-100h] BYREF
  __int64 *v203; // [rsp+2D0h] [rbp-C0h]
  __int64 v204; // [rsp+2D8h] [rbp-B8h]
  unsigned __int64 v205; // [rsp+2E0h] [rbp-B0h]
  char v206[8]; // [rsp+2E8h] [rbp-A8h] BYREF
  unsigned __int64 v207; // [rsp+2F0h] [rbp-A0h]
  char v208; // [rsp+304h] [rbp-8Ch]
  char v209[64]; // [rsp+308h] [rbp-88h] BYREF
  const __m128i *v210; // [rsp+348h] [rbp-48h]
  const __m128i *v211; // [rsp+350h] [rbp-40h]
  __int64 v212; // [rsp+358h] [rbp-38h]

  v8 = 0;
  v9 = *(_QWORD *)a1;
  v10 = *(_DWORD *)(*(_QWORD *)a1 + 88LL);
  v11 = *(__int64 **)(*(_QWORD *)a1 + 80LL);
  if ( v10 == 1 )
    v8 = *v11;
  v200.m128i_i64[0] = v8;
  v12 = (__int64)&v11[v10];
  v13 = sub_2C06B30(v11, v12, v200.m128i_i64);
  if ( v13 + 1 != (_QWORD *)v12 )
  {
    memmove(v13, v13 + 1, v12 - (_QWORD)(v13 + 1));
    v14 = *(_DWORD *)(v9 + 88);
  }
  *(_DWORD *)(v9 + 88) = v14 - 1;
  v200.m128i_i64[0] = v9;
  v15 = *(_QWORD **)(v8 + 56);
  v16 = (__int64)&v15[*(unsigned int *)(v8 + 64)];
  v17 = sub_2C06B30(v15, v16, v200.m128i_i64);
  if ( v17 + 1 != (_QWORD *)v16 )
  {
    memmove(v17, v17 + 1, v16 - (_QWORD)(v17 + 1));
    v18 = *(_DWORD *)(v8 + 64);
  }
  v19 = v18 - 1;
  *(_DWORD *)(v8 + 64) = v19;
  if ( v19 != 1 )
    BUG();
  v20 = **(_QWORD **)(v8 + 56);
  v200.m128i_i64[0] = v8;
  v21 = sub_2C06B30(*(_QWORD **)(v20 + 80), *(_QWORD *)(v20 + 80) + 8LL * *(unsigned int *)(v20 + 88), v200.m128i_i64);
  if ( v21 + 1 != (_QWORD *)v23 )
  {
    memmove(v21, v21 + 1, v23 - (char *)(v21 + 1));
    v22 = *(_DWORD *)(v20 + 88);
  }
  *(_DWORD *)(v20 + 88) = v22 - 1;
  v200.m128i_i64[0] = v20;
  v24 = *(_QWORD **)(v8 + 56);
  v25 = (__int64)&v24[*(unsigned int *)(v8 + 64)];
  v26 = sub_2C06B30(v24, v25, v200.m128i_i64);
  if ( v26 + 1 != (_QWORD *)v25 )
  {
    memmove(v26, v26 + 1, v25 - (_QWORD)(v26 + 1));
    v27 = *(_DWORD *)(v8 + 64);
  }
  *(_DWORD *)(v8 + 64) = v27 - 1;
  v200.m128i_i64[0] = (__int64)"vector.ph";
  v202 = 259;
  v170 = sub_22077B0(0x80u);
  if ( v170 )
  {
    sub_CA0F50((__int64 *)&v192, (void **)&v200);
    v30 = v192;
    v31 = v193;
    *(_BYTE *)(v170 + 8) = 1;
    *(_QWORD *)v170 = &unk_4A23970;
    *(_QWORD *)(v170 + 16) = v170 + 32;
    sub_2C06BF0((__int64 *)(v170 + 16), v30, (__int64)&v30[v31]);
    v32 = (__int64 *)v192;
    *(_QWORD *)(v170 + 56) = v170 + 72;
    *(_QWORD *)(v170 + 64) = 0x100000000LL;
    *(_QWORD *)(v170 + 88) = 0x100000000LL;
    *(_QWORD *)(v170 + 48) = 0;
    *(_QWORD *)(v170 + 80) = v170 + 96;
    *(_QWORD *)(v170 + 104) = 0;
    if ( v32 != &v194 )
      j_j___libc_free_0((unsigned __int64)v32);
    *(_QWORD *)v170 = &unk_4A23A00;
    *(_QWORD *)(v170 + 120) = v170 + 112;
    *(_QWORD *)(v170 + 112) = (v170 + 112) | 4;
  }
  v166 = a1 + 592;
  v33 = *(unsigned int *)(a1 + 600);
  if ( v33 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 604) )
  {
    sub_C8D5F0(v166, (const void *)(a1 + 608), v33 + 1, 8u, v28, v29);
    v33 = *(unsigned int *)(a1 + 600);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 592) + 8 * v33) = v170;
  v34 = *(_QWORD *)a1;
  ++*(_DWORD *)(a1 + 600);
  v35 = *(unsigned int *)(v34 + 88);
  if ( v35 + 1 > (unsigned __int64)*(unsigned int *)(v34 + 92) )
  {
    sub_C8D5F0(v34 + 80, (const void *)(v34 + 96), v35 + 1, 8u, v35 + 1, v29);
    v35 = *(unsigned int *)(v34 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(v34 + 80) + 8 * v35) = v170;
  ++*(_DWORD *)(v34 + 88);
  v36 = *(unsigned int *)(v170 + 64);
  if ( v36 + 1 > (unsigned __int64)*(unsigned int *)(v170 + 68) )
  {
    sub_C8D5F0(v170 + 56, (const void *)(v170 + 72), v36 + 1, 8u, v36 + 1, v29);
    v36 = *(unsigned int *)(v170 + 64);
  }
  *(_QWORD *)(*(_QWORD *)(v170 + 56) + 8 * v36) = v34;
  ++*(_DWORD *)(v170 + 64);
  v37 = sub_DEF9D0(a3);
  v38 = (__int64 *)a3[14];
  v163 = sub_DE5A20(v38, v37, a2, a6);
  v39 = sub_2C47690(a1, v163, v38);
  HIBYTE(v202) = 1;
  *(_QWORD *)(a1 + 200) = v39;
  v200.m128i_i64[0] = (__int64)"vector.latch";
  LOBYTE(v202) = 3;
  v42 = sub_22077B0(0x80u);
  if ( v42 )
  {
    sub_CA0F50((__int64 *)&v192, (void **)&v200);
    *(_BYTE *)(v42 + 8) = 1;
    v43 = v192;
    v44 = v193;
    *(_QWORD *)v42 = &unk_4A23970;
    *(_QWORD *)(v42 + 16) = v42 + 32;
    sub_2C06BF0((__int64 *)(v42 + 16), v43, (__int64)&v43[v44]);
    v45 = (__int64 *)v192;
    *(_QWORD *)(v42 + 56) = v42 + 72;
    *(_QWORD *)(v42 + 64) = 0x100000000LL;
    *(_QWORD *)(v42 + 88) = 0x100000000LL;
    *(_QWORD *)(v42 + 48) = 0;
    *(_QWORD *)(v42 + 80) = v42 + 96;
    *(_QWORD *)(v42 + 104) = 0;
    if ( v45 != &v194 )
      j_j___libc_free_0((unsigned __int64)v45);
    *(_QWORD *)v42 = &unk_4A23A00;
    *(_QWORD *)(v42 + 120) = v42 + 112;
    *(_QWORD *)(v42 + 112) = (v42 + 112) | 4;
  }
  v46 = *(unsigned int *)(a1 + 600);
  v47 = *(unsigned int *)(a1 + 604);
  if ( v46 + 1 > v47 )
  {
    sub_C8D5F0(v166, (const void *)(a1 + 608), v46 + 1, 8u, v40, v41);
    v46 = *(unsigned int *)(a1 + 600);
  }
  v48 = *(_QWORD *)(a1 + 592);
  *(_QWORD *)(v48 + 8 * v46) = v42;
  ++*(_DWORD *)(a1 + 600);
  sub_2BEFE80(v42, v20, v48, v47, v40, v41);
  v200.m128i_i64[0] = (__int64)v201;
  strcpy(v201, "vector loop");
  v200.m128i_i64[1] = 11;
  v49 = sub_22077B0(0x88u);
  v168 = v49;
  v52 = v49;
  if ( v49 )
  {
    v53 = (_BYTE *)v200.m128i_i64[0];
    *(_BYTE *)(v49 + 8) = 0;
    v54 = v200.m128i_i64[1];
    *(_QWORD *)v49 = &unk_4A23970;
    *(_QWORD *)(v49 + 16) = v49 + 32;
    sub_2C06BF0((__int64 *)(v49 + 16), v53, (__int64)&v53[v54]);
    *(_QWORD *)(v52 + 48) = 0;
    *(_QWORD *)(v52 + 56) = v52 + 72;
    *(_QWORD *)(v52 + 64) = 0x100000000LL;
    *(_QWORD *)(v52 + 88) = 0x100000000LL;
    *(_QWORD *)(v52 + 80) = v52 + 96;
    *(_QWORD *)(v52 + 104) = 0;
    *(_QWORD *)v52 = &unk_4A23A38;
    *(_QWORD *)(v8 + 48) = v52;
    *(_QWORD *)(v52 + 112) = v8;
    *(_QWORD *)(v52 + 120) = v42;
    *(_BYTE *)(v52 + 128) = 0;
    *(_QWORD *)(v42 + 48) = v52;
  }
  v55 = *(unsigned int *)(a1 + 600);
  if ( v55 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 604) )
  {
    sub_C8D5F0(v166, (const void *)(a1 + 608), v55 + 1, 8u, v50, v51);
    v55 = *(unsigned int *)(a1 + 600);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 592) + 8 * v55) = v168;
  v56 = (char *)v200.m128i_i64[0];
  ++*(_DWORD *)(a1 + 600);
  if ( v56 != v201 )
    j_j___libc_free_0((unsigned __int64)v56);
  memset(v174, 0, 0x78u);
  v176 = 0x100000008LL;
  v175.m128i_i64[1] = (__int64)v179;
  v174[1] = (__int64)&v174[4];
  v179[0] = v8;
  v200.m128i_i64[0] = v8;
  LODWORD(v174[2]) = 8;
  BYTE4(v174[3]) = 1;
  v180 = 0;
  v181 = 0;
  v182 = 0;
  v177 = 0;
  v178 = 1;
  v175.m128i_i64[0] = 1;
  v201[0] = 0;
  sub_2C06CA0(&v180, &v200);
  sub_C8CF70((__int64)&v192, v196, 8, (__int64)&v174[4], (__int64)v174);
  v57 = v174[12];
  memset(&v174[12], 0, 24);
  v197 = v57;
  v198 = v174[13];
  v199 = v174[14];
  sub_C8CF70((__int64)&v183, v188, 8, (__int64)v179, (__int64)&v175);
  v58 = v180;
  v180 = 0;
  v189 = v58;
  v59 = v181;
  v181 = 0;
  v190 = v59;
  v60 = v182;
  v182 = 0;
  v191 = v60;
  sub_C8CF70((__int64)&v200, &v202, 8, (__int64)v188, (__int64)&v183);
  v61 = v189;
  v189 = 0;
  v203 = (__int64 *)v61;
  v62 = v190;
  v190 = 0;
  v204 = v62;
  v63 = v191;
  v191 = 0;
  v205 = v63;
  sub_C8CF70((__int64)v206, v209, 8, (__int64)v196, (__int64)&v192);
  v67 = v197;
  v197 = 0;
  v210 = (const __m128i *)v67;
  v68 = v198;
  v198 = 0;
  v211 = (const __m128i *)v68;
  v69 = v199;
  v199 = 0;
  v212 = v69;
  if ( v189 )
    j_j___libc_free_0(v189);
  if ( !v187 )
    _libc_free((unsigned __int64)v184);
  if ( v197 )
    j_j___libc_free_0(v197);
  if ( !v195 )
    _libc_free(v193);
  if ( v180 )
    j_j___libc_free_0(v180);
  if ( !v178 )
    _libc_free(v175.m128i_u64[1]);
  if ( v174[12] )
    j_j___libc_free_0(v174[12]);
  if ( !BYTE4(v174[3]) )
    _libc_free(v174[1]);
  v70 = (const __m128i *)v188;
  sub_C8CD80((__int64)&v183, (__int64)v188, (__int64)&v200, v64, v65, v66);
  v74 = v204;
  v75 = v203;
  v189 = 0;
  v190 = 0;
  v191 = 0;
  v76 = v204 - (_QWORD)v203;
  if ( (__int64 *)v204 == v203 )
  {
    v76 = 0;
    v78 = 0;
  }
  else
  {
    if ( v76 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_184;
    v77 = sub_22077B0(v204 - (_QWORD)v203);
    v74 = v204;
    v75 = v203;
    v78 = v77;
  }
  v189 = v78;
  v190 = v78;
  v191 = v78 + v76;
  if ( v75 != (__int64 *)v74 )
  {
    v79 = (__m128i *)v78;
    v80 = (const __m128i *)v75;
    do
    {
      if ( v79 )
      {
        *v79 = _mm_loadu_si128(v80);
        v72 = v80[1].m128i_i64[0];
        v79[1].m128i_i64[0] = v72;
      }
      v80 = (const __m128i *)((char *)v80 + 24);
      v79 = (__m128i *)((char *)v79 + 24);
    }
    while ( (const __m128i *)v74 != v80 );
    v78 += 8 * ((unsigned __int64)(v74 - 24 - (_QWORD)v75) >> 3) + 24;
  }
  v75 = (__int64 *)&v192;
  v190 = v78;
  sub_C8CD80((__int64)&v192, (__int64)v196, (__int64)v206, v74, v72, v73);
  v83 = v211;
  v70 = v210;
  v197 = 0;
  v198 = 0;
  v199 = 0;
  v84 = (char *)v211 - (char *)v210;
  if ( v211 != v210 )
  {
    if ( v84 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v85 = sub_22077B0((char *)v211 - (char *)v210);
      v83 = v211;
      v70 = v210;
      v86 = v85;
      goto LABEL_62;
    }
LABEL_184:
    sub_4261EA(v75, v70, v71);
  }
  v84 = 0;
  v86 = 0;
LABEL_62:
  v197 = v86;
  v87 = (__m128i *)v86;
  v198 = v86;
  v199 = v86 + v84;
  if ( v70 != v83 )
  {
    v88 = v70;
    do
    {
      if ( v87 )
      {
        *v87 = _mm_loadu_si128(v88);
        v81 = v88[1].m128i_i64[0];
        v87[1].m128i_i64[0] = v81;
      }
      v88 = (const __m128i *)((char *)v88 + 24);
      v87 = (__m128i *)((char *)v87 + 24);
    }
    while ( v83 != v88 );
    v87 = (__m128i *)(v86 + 8 * ((unsigned __int64)((char *)&v83[-2].m128i_u64[1] - (char *)v70) >> 3) + 24);
  }
  v89 = v190;
  v90 = v189;
  v198 = (__int64)v87;
  v91 = (__int64)v87->m128i_i64 - v86;
  v164 = a1;
  if ( v190 - v189 == v91 )
    goto LABEL_82;
  do
  {
LABEL_69:
    *(_QWORD *)(*(_QWORD *)(v89 - 24) + 48LL) = v168;
    v92 = v190;
    do
    {
      v93 = *(_QWORD *)(v92 - 24);
      if ( !*(_BYTE *)(v92 - 8) )
      {
        v94 = *(__int64 **)(v93 + 80);
        *(_BYTE *)(v92 - 8) = 1;
        *(_QWORD *)(v92 - 16) = v94;
        goto LABEL_72;
      }
      while ( 1 )
      {
        v94 = *(__int64 **)(v92 - 16);
LABEL_72:
        v95 = *(unsigned int *)(v93 + 88);
        if ( v94 == (__int64 *)(*(_QWORD *)(v93 + 80) + 8 * v95) )
          break;
        v96 = v94 + 1;
        *(_QWORD *)(v92 - 16) = v94 + 1;
        v97 = *v94;
        if ( !v187 )
          goto LABEL_79;
        v98 = v184;
        v96 = &v184[v186];
        if ( v184 == v96 )
        {
LABEL_166:
          if ( v186 < v185 )
          {
            ++v186;
            *v96 = v97;
            ++v183;
LABEL_80:
            v175.m128i_i64[0] = v97;
            LOBYTE(v176) = 0;
            sub_2C06CA0(&v189, &v175);
            v90 = v189;
            v89 = v190;
            goto LABEL_81;
          }
LABEL_79:
          sub_C8CC70((__int64)&v183, v97, (__int64)v96, v95, v81, v82);
          if ( v99 )
            goto LABEL_80;
        }
        else
        {
          while ( v97 != *v98 )
          {
            if ( v96 == ++v98 )
              goto LABEL_166;
          }
        }
      }
      v190 -= 24LL;
      v90 = v189;
      v92 = v190;
    }
    while ( v190 != v189 );
    v89 = v189;
LABEL_81:
    v86 = v197;
    v91 = v198 - v197;
  }
  while ( v89 - v90 != v198 - v197 );
LABEL_82:
  if ( v90 != v89 )
  {
    v91 = v86;
    while ( *(_QWORD *)v90 == *(_QWORD *)v91 )
    {
      v100 = *(_BYTE *)(v90 + 16);
      if ( v100 != *(_BYTE *)(v91 + 16) || v100 && *(_QWORD *)(v90 + 8) != *(_QWORD *)(v91 + 8) )
        break;
      v90 += 24LL;
      v91 += 24;
      if ( v90 == v89 )
        goto LABEL_89;
    }
    goto LABEL_69;
  }
LABEL_89:
  if ( v86 )
    j_j___libc_free_0(v86);
  if ( !v195 )
    _libc_free(v193);
  if ( v189 )
    j_j___libc_free_0(v189);
  if ( !v187 )
    _libc_free((unsigned __int64)v184);
  if ( v210 )
    j_j___libc_free_0((unsigned __int64)v210);
  if ( !v208 )
    _libc_free(v207);
  if ( v203 )
    j_j___libc_free_0((unsigned __int64)v203);
  if ( !v201[12] )
    _libc_free(v200.m128i_u64[1]);
  sub_2BEFE80(v168, v170, v91, v89, v81, v82);
  v200.m128i_i64[0] = (__int64)"middle.block";
  v202 = 259;
  v101 = sub_22077B0(0x80u);
  if ( v101 )
  {
    sub_CA0F50((__int64 *)&v192, (void **)&v200);
    v104 = v192;
    *(_BYTE *)(v101 + 8) = 1;
    v105 = v193;
    *(_QWORD *)v101 = &unk_4A23970;
    *(_QWORD *)(v101 + 16) = v101 + 32;
    sub_2C06BF0((__int64 *)(v101 + 16), v104, (__int64)&v104[v105]);
    v106 = (__int64 *)v192;
    *(_QWORD *)(v101 + 56) = v101 + 72;
    *(_QWORD *)(v101 + 64) = 0x100000000LL;
    *(_QWORD *)(v101 + 88) = 0x100000000LL;
    *(_QWORD *)(v101 + 48) = 0;
    *(_QWORD *)(v101 + 80) = v101 + 96;
    *(_QWORD *)(v101 + 104) = 0;
    if ( v106 != &v194 )
      j_j___libc_free_0((unsigned __int64)v106);
    *(_QWORD *)v101 = &unk_4A23A00;
    *(_QWORD *)(v101 + 120) = v101 + 112;
    *(_QWORD *)(v101 + 112) = (v101 + 112) | 4;
  }
  v107 = *(unsigned int *)(v164 + 600);
  v108 = *(unsigned int *)(v164 + 604);
  if ( v107 + 1 > v108 )
  {
    sub_C8D5F0(v166, (const void *)(v164 + 608), v107 + 1, 8u, v102, v103);
    v107 = *(unsigned int *)(v164 + 600);
  }
  v109 = *(_QWORD *)(v164 + 592);
  *(_QWORD *)(v109 + 8 * v107) = v101;
  ++*(_DWORD *)(v164 + 600);
  sub_2BEFE80(v101, v168, v109, v108, v102, v103);
  v200.m128i_i64[0] = (__int64)"scalar.ph";
  v202 = 259;
  v112 = sub_22077B0(0x80u);
  if ( v112 )
  {
    sub_CA0F50((__int64 *)&v192, (void **)&v200);
    *(_BYTE *)(v112 + 8) = 1;
    v113 = v192;
    v114 = v193;
    *(_QWORD *)v112 = &unk_4A23970;
    *(_QWORD *)(v112 + 16) = v112 + 32;
    sub_2C06BF0((__int64 *)(v112 + 16), v113, (__int64)&v113[v114]);
    v115 = (__int64 *)v192;
    *(_QWORD *)(v112 + 56) = v112 + 72;
    *(_QWORD *)(v112 + 64) = 0x100000000LL;
    *(_QWORD *)(v112 + 88) = 0x100000000LL;
    *(_QWORD *)(v112 + 48) = 0;
    *(_QWORD *)(v112 + 80) = v112 + 96;
    *(_QWORD *)(v112 + 104) = 0;
    if ( v115 != &v194 )
      j_j___libc_free_0((unsigned __int64)v115);
    *(_QWORD *)v112 = &unk_4A23A00;
    *(_QWORD *)(v112 + 120) = v112 + 112;
    *(_QWORD *)(v112 + 112) = (v112 + 112) | 4;
  }
  v116 = *(unsigned int *)(v164 + 600);
  if ( v116 + 1 > (unsigned __int64)*(unsigned int *)(v164 + 604) )
  {
    sub_C8D5F0(v166, (const void *)(v164 + 608), v116 + 1, 8u, v110, v111);
    v116 = *(unsigned int *)(v164 + 600);
  }
  *(_QWORD *)(*(_QWORD *)(v164 + 592) + 8 * v116) = v112;
  v117 = *(unsigned int *)(v112 + 88);
  v118 = *(unsigned int *)(v112 + 92);
  ++*(_DWORD *)(v164 + 600);
  v119 = *(_QWORD *)(v164 + 8);
  if ( v117 + 1 > v118 )
  {
    sub_C8D5F0(v112 + 80, (const void *)(v112 + 96), v117 + 1, 8u, v110, v111);
    v117 = *(unsigned int *)(v112 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(v112 + 80) + 8 * v117) = v119;
  ++*(_DWORD *)(v112 + 88);
  v120 = *(unsigned int *)(v119 + 64);
  if ( v120 + 1 > (unsigned __int64)*(unsigned int *)(v119 + 68) )
  {
    sub_C8D5F0(v119 + 56, (const void *)(v119 + 72), v120 + 1, 8u, v110, v111);
    v120 = *(unsigned int *)(v119 + 64);
  }
  *(_QWORD *)(*(_QWORD *)(v119 + 56) + 8 * v120) = v112;
  ++*(_DWORD *)(v119 + 64);
  if ( !a4 )
  {
    v155 = *(unsigned int *)(v101 + 88);
    if ( v155 + 1 > (unsigned __int64)*(unsigned int *)(v101 + 92) )
    {
      sub_C8D5F0(v101 + 80, (const void *)(v101 + 96), v155 + 1, 8u, v110, v111);
      v155 = *(unsigned int *)(v101 + 88);
    }
    *(_QWORD *)(*(_QWORD *)(v101 + 80) + 8 * v155) = v112;
    v156 = *(unsigned int *)(v112 + 64);
    v157 = *(unsigned int *)(v112 + 68);
    ++*(_DWORD *)(v101 + 88);
    if ( v156 + 1 > v157 )
    {
      sub_C8D5F0(v112 + 56, (const void *)(v112 + 72), v156 + 1, 8u, v110, v111);
      v156 = *(unsigned int *)(v112 + 64);
    }
    *(_QWORD *)(*(_QWORD *)(v112 + 56) + 8 * v156) = v101;
    ++*(_DWORD *)(v112 + 64);
    return;
  }
  v121 = sub_D4B030(a6);
  v122 = sub_2BF0B50(v164, v121);
  sub_2BEFE80(v122, v101, v123, v124, v125, v126);
  v129 = *(unsigned int *)(v101 + 88);
  if ( v129 + 1 > (unsigned __int64)*(unsigned int *)(v101 + 92) )
  {
    sub_C8D5F0(v101 + 80, (const void *)(v101 + 96), v129 + 1, 8u, v127, v128);
    v129 = *(unsigned int *)(v101 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(v101 + 80) + 8 * v129) = v112;
  v130 = *(unsigned int *)(v112 + 64);
  v131 = *(unsigned int *)(v112 + 68);
  ++*(_DWORD *)(v101 + 88);
  if ( v130 + 1 > v131 )
  {
    sub_C8D5F0(v112 + 56, (const void *)(v112 + 72), v130 + 1, 8u, v127, v128);
    v130 = *(unsigned int *)(v112 + 64);
  }
  *(_QWORD *)(*(_QWORD *)(v112 + 56) + 8 * v130) = v101;
  ++*(_DWORD *)(v112 + 64);
  v132 = (_QWORD *)(sub_D47930(a6) + 48);
  v133 = *v132 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v133 == v132 )
  {
    v135 = 0;
  }
  else
  {
    if ( !v133 )
      BUG();
    v134 = *(unsigned __int8 *)(v133 - 24);
    v135 = 0;
    v136 = v133 - 24;
    if ( (unsigned int)(v134 - 30) < 0xB )
      v135 = v136;
  }
  v171 = v101 + 112;
  if ( a5 )
  {
    v158 = (_QWORD **)sub_D95540((__int64)v163);
    v159 = sub_BCB2A0(*v158);
    v160 = sub_AD6400(v159);
    v143 = sub_2AC42A0(v164, v160);
  }
  else
  {
    v200.m128i_i64[0] = (__int64)"cmp.n";
    v202 = 259;
    v137 = *(_QWORD *)(v135 + 48);
    v183 = (_BYTE *)v137;
    if ( v137 )
    {
      v138 = v164 + 216;
      sub_B96E90((__int64)&v183, v137, 1);
      v139 = *(_QWORD *)(v164 + 200);
      v192 = v183;
      if ( v183 )
        sub_B96E90((__int64)&v192, (__int64)v183, 1);
    }
    else
    {
      v138 = v164 + 216;
      v192 = 0;
      v139 = *(_QWORD *)(v164 + 200);
    }
    v140 = (_QWORD *)sub_22077B0(0xC8u);
    if ( v140 )
    {
      v169 = v140;
      sub_2C1A5F0((_DWORD)v140, 53, 32, v139, v138, (unsigned int)&v192, (__int64)&v200);
      v140 = v169;
    }
    v141 = *(_QWORD *)(v101 + 112);
    v142 = v140[3];
    v140[10] = v101;
    v143 = (__int64)(v140 + 12);
    v141 &= 0xFFFFFFFFFFFFFFF8LL;
    v140[4] = v171;
    v140[3] = v141 | v142 & 7;
    *(_QWORD *)(v141 + 8) = v140 + 3;
    v144 = (__int64)v192;
    *(_QWORD *)(v101 + 112) = *(_QWORD *)(v101 + 112) & 7LL | (unsigned __int64)(v140 + 3);
    if ( v144 )
      sub_B91220((__int64)&v192, v144);
    if ( v183 )
      sub_B91220((__int64)&v183, (__int64)v183);
  }
  v202 = 257;
  v145 = *(_QWORD *)(v135 + 48);
  v172 = v145;
  if ( !v145 )
  {
    v173 = 0;
    goto LABEL_173;
  }
  sub_B96E90((__int64)&v172, v145, 1);
  v173 = v172;
  if ( !v172 )
  {
LABEL_173:
    v174[0] = 0;
    goto LABEL_144;
  }
  sub_B96E90((__int64)&v173, v172, 1);
  v174[0] = v173;
  if ( v173 )
    sub_B96E90((__int64)v174, v173, 1);
LABEL_144:
  v147 = sub_22077B0(0xC8u);
  if ( !v147 )
    goto LABEL_159;
  v175.m128i_i64[0] = v174[0];
  if ( !v174[0] )
  {
    v183 = 0;
    goto LABEL_175;
  }
  sub_B96E90((__int64)&v175, v174[0], 1);
  v183 = (_BYTE *)v175.m128i_i64[0];
  if ( !v175.m128i_i64[0] )
  {
LABEL_175:
    v192 = 0;
    goto LABEL_149;
  }
  sub_B96E90((__int64)&v183, v175.m128i_i64[0], 1);
  v192 = v183;
  if ( v183 )
    sub_B96E90((__int64)&v192, (__int64)v183, 1);
LABEL_149:
  *(_BYTE *)(v147 + 8) = 4;
  *(_QWORD *)(v147 + 24) = 0;
  *(_QWORD *)(v147 + 32) = 0;
  *(_QWORD *)v147 = &unk_4A231A8;
  *(_QWORD *)(v147 + 16) = 0;
  *(_QWORD *)(v147 + 64) = v143;
  *(_QWORD *)(v147 + 40) = &unk_4A23170;
  *(_QWORD *)(v147 + 48) = v147 + 64;
  *(_QWORD *)(v147 + 56) = 0x200000001LL;
  v148 = *(unsigned int *)(v143 + 24);
  v149 = v148 + 1;
  if ( v148 + 1 > (unsigned __int64)*(unsigned int *)(v143 + 28) )
  {
    sub_C8D5F0(v143 + 16, (const void *)(v143 + 32), v148 + 1, 8u, v146, v149);
    v148 = *(unsigned int *)(v143 + 24);
  }
  *(_QWORD *)(*(_QWORD *)(v143 + 16) + 8 * v148) = v147 + 40;
  ++*(_DWORD *)(v143 + 24);
  v150 = (__int64)v192;
  *(_QWORD *)(v147 + 80) = 0;
  *(_QWORD *)v147 = &unk_4A23A70;
  *(_QWORD *)(v147 + 40) = &unk_4A23AA8;
  *(_QWORD *)(v147 + 88) = v150;
  if ( v150 )
  {
    sub_B96E90(v147 + 88, v150, 1);
    if ( v192 )
      sub_B91220((__int64)&v192, (__int64)v192);
  }
  sub_2BF0340(v147 + 96, 1, 0, v147, v146, v149);
  v151 = (__int64)v183;
  *(_QWORD *)v147 = &unk_4A231C8;
  *(_QWORD *)(v147 + 40) = &unk_4A23200;
  *(_QWORD *)(v147 + 96) = &unk_4A23238;
  if ( v151 )
    sub_B91220((__int64)&v183, v151);
  v152 = v175.m128i_i64[0];
  *(_BYTE *)(v147 + 152) = 7;
  *(_DWORD *)(v147 + 156) = 0;
  *(_QWORD *)v147 = &unk_4A23258;
  *(_QWORD *)(v147 + 40) = &unk_4A23290;
  *(_QWORD *)(v147 + 96) = &unk_4A232C8;
  if ( v152 )
    sub_B91220((__int64)&v175, v152);
  *(_BYTE *)(v147 + 160) = 79;
  *(_QWORD *)v147 = &unk_4A23B70;
  *(_QWORD *)(v147 + 40) = &unk_4A23BB8;
  *(_QWORD *)(v147 + 96) = &unk_4A23BF0;
  sub_CA0F50((__int64 *)(v147 + 168), (void **)&v200);
LABEL_159:
  v153 = *(_QWORD *)(v101 + 112);
  *(_QWORD *)(v147 + 80) = v101;
  *(_QWORD *)(v147 + 32) = v171;
  v153 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(v147 + 24) = v153 | *(_QWORD *)(v147 + 24) & 7LL;
  *(_QWORD *)(v153 + 8) = v147 + 24;
  v154 = v174[0];
  *(_QWORD *)(v101 + 112) = *(_QWORD *)(v101 + 112) & 7LL | (v147 + 24);
  if ( v154 )
    sub_B91220((__int64)v174, v154);
  if ( v173 )
    sub_B91220((__int64)&v173, v173);
  if ( v172 )
    sub_B91220((__int64)&v172, v172);
}
