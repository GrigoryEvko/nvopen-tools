// Function: sub_F77B70
// Address: 0xf77b70
//
__int64 __fastcall sub_F77B70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r12
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  int v10; // edx
  _QWORD *v11; // r13
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // r15
  _QWORD *v18; // rax
  __int64 v19; // r9
  __int64 v20; // r13
  unsigned int *v21; // r14
  unsigned int *v22; // rbx
  __int64 v23; // rdx
  unsigned int v24; // esi
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rbx
  __int64 v28; // r13
  __int64 v29; // rax
  unsigned __int64 v30; // rax
  int v31; // edx
  __int64 v32; // rsi
  __int64 v33; // rax
  _QWORD *v34; // rax
  __int64 v35; // r14
  __int64 v36; // rsi
  unsigned int *v37; // r13
  unsigned int *v38; // rbx
  __int64 v39; // rdx
  unsigned __int64 v40; // rax
  int v41; // edx
  _QWORD *v42; // rdi
  _QWORD *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // r9
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __m128i *v51; // rsi
  __m128i *v52; // rax
  __int64 *v53; // rcx
  _QWORD *v54; // r12
  __int64 v55; // rax
  __int64 v56; // r13
  __int64 v57; // r14
  __int64 v58; // r15
  __int64 v59; // rbx
  __int64 v60; // rax
  __int64 v61; // rsi
  _QWORD *v62; // rax
  _QWORD *v63; // rcx
  __int64 *v64; // rsi
  __int64 v65; // rax
  unsigned __int8 v66; // dl
  _QWORD **v67; // r12
  unsigned __int64 *v68; // r13
  char v69; // dh
  char v70; // al
  __int64 v71; // rcx
  unsigned __int8 v72; // r14
  unsigned __int64 *v73; // r14
  _QWORD **v74; // r13
  __int64 v75; // r15
  _QWORD *v76; // rdi
  _BYTE *v77; // r12
  unsigned __int8 v78; // r15
  __int64 v79; // r14
  _BYTE *v80; // r13
  __int64 v81; // rax
  __int64 v82; // r13
  __int64 v83; // rbx
  __int64 v84; // r14
  __int64 v85; // r12
  __int64 v86; // rsi
  char *v87; // rbx
  char *v88; // r12
  __int64 v89; // rbx
  __int64 v90; // r12
  __int64 v91; // rdi
  char *v92; // rbx
  char *v93; // r12
  char *v94; // rbx
  char *v95; // r12
  char *v96; // rbx
  char *v97; // r12
  __int64 *v98; // rbx
  __int64 *k; // r12
  __int64 v100; // rdi
  _QWORD *v101; // rdi
  _QWORD **v102; // r12
  _QWORD **v103; // rbx
  _QWORD *v104; // rdi
  __int64 *v105; // rbx
  __int64 *v106; // r12
  __int64 v107; // r8
  __int64 v108; // rsi
  _QWORD *v109; // rax
  __int64 *v110; // r14
  __int64 *v111; // rax
  __int64 v112; // r12
  _BYTE *v113; // rax
  __int64 v114; // r8
  _QWORD *v115; // rbx
  unsigned __int64 v116; // r12
  __int64 *v117; // r13
  __int64 *v118; // rbx
  __int64 v119; // rdi
  __int64 v120; // rax
  __int64 v121; // rax
  __int64 v122; // rdi
  __int64 v123; // rdi
  _QWORD *v124; // rbx
  _QWORD *v125; // r14
  void (__fastcall *v126)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v127; // rax
  __int64 result; // rax
  __int64 v129; // rbx
  __int64 v130; // r12
  __int64 v131; // rax
  _QWORD *v132; // rbx
  _QWORD *v133; // r12
  __int64 v134; // rax
  _QWORD *v135; // rbx
  _QWORD *v136; // r12
  __int64 v137; // rax
  __int64 v138; // rax
  __int64 v139; // rax
  __int64 v140; // rax
  unsigned int v141; // eax
  __int64 v142; // r8
  __int64 v143; // r9
  __m128i *v144; // rdx
  unsigned int v145; // eax
  unsigned int v146; // esi
  unsigned __int64 v147; // rcx
  __int64 v148; // rax
  unsigned __int64 v149; // rdx
  __int64 v150; // rdi
  _QWORD *v151; // rdx
  _QWORD *v152; // rbx
  _QWORD *i; // r13
  __int64 j; // r14
  __int64 v155; // rsi
  __int64 v156; // r12
  __m128i *v157; // rcx
  unsigned int v158; // eax
  unsigned int v159; // esi
  __int64 v160; // r8
  __int64 v161; // r9
  __int64 v162; // rax
  unsigned __int64 v163; // rdx
  unsigned int v164; // eax
  __int64 v165; // rdx
  char v166; // al
  __int64 *v167; // r15
  __int64 v168; // rsi
  __int64 v169; // rax
  unsigned int v170; // edx
  __int64 *v171; // rbx
  __int64 v172; // rdi
  __int64 *m; // r13
  char *v174; // rdx
  char *v175; // rdi
  __int64 v176; // rax
  __int64 v177; // rcx
  char *v178; // rax
  bool v179; // zf
  _QWORD *v180; // rsi
  _QWORD *v181; // rdx
  _QWORD *v182; // rax
  __int64 v183; // rcx
  __int64 *v184; // rax
  __int64 *v185; // rax
  int v186; // r9d
  _QWORD *v187; // rax
  __int64 v188; // r14
  unsigned int *v189; // rbx
  unsigned int *v190; // r13
  __int64 v191; // rdx
  __int64 *v192; // [rsp+30h] [rbp-6E0h]
  _QWORD *v193; // [rsp+38h] [rbp-6D8h]
  __int64 v196; // [rsp+58h] [rbp-6B8h]
  __int64 *v197; // [rsp+60h] [rbp-6B0h]
  __int64 v198; // [rsp+68h] [rbp-6A8h]
  __int64 v199; // [rsp+78h] [rbp-698h]
  __int64 v200; // [rsp+90h] [rbp-680h]
  __int64 v201; // [rsp+98h] [rbp-678h]
  __int64 v202; // [rsp+A0h] [rbp-670h]
  _QWORD *v204; // [rsp+B0h] [rbp-660h]
  __int64 v205; // [rsp+B0h] [rbp-660h]
  unsigned __int64 v206; // [rsp+B8h] [rbp-658h] BYREF
  __m128i *v207; // [rsp+C0h] [rbp-650h] BYREF
  __m128i *v208; // [rsp+C8h] [rbp-648h] BYREF
  _BYTE *v209; // [rsp+D0h] [rbp-640h] BYREF
  __int64 v210; // [rsp+D8h] [rbp-638h]
  _BYTE v211[32]; // [rsp+E0h] [rbp-630h] BYREF
  _BYTE *v212; // [rsp+100h] [rbp-610h] BYREF
  __int64 v213; // [rsp+108h] [rbp-608h]
  _BYTE v214[32]; // [rsp+110h] [rbp-600h] BYREF
  unsigned int *v215; // [rsp+130h] [rbp-5E0h] BYREF
  __int64 v216; // [rsp+138h] [rbp-5D8h]
  _BYTE v217[32]; // [rsp+140h] [rbp-5D0h] BYREF
  __int64 v218; // [rsp+160h] [rbp-5B0h]
  __int64 v219; // [rsp+168h] [rbp-5A8h]
  __int64 v220; // [rsp+170h] [rbp-5A0h]
  __int64 *v221; // [rsp+178h] [rbp-598h]
  void **v222; // [rsp+180h] [rbp-590h]
  void **v223; // [rsp+188h] [rbp-588h]
  __int64 v224; // [rsp+190h] [rbp-580h]
  int v225; // [rsp+198h] [rbp-578h]
  __int16 v226; // [rsp+19Ch] [rbp-574h]
  char v227; // [rsp+19Eh] [rbp-572h]
  __int64 v228; // [rsp+1A0h] [rbp-570h]
  __int64 v229; // [rsp+1A8h] [rbp-568h]
  void *v230; // [rsp+1B0h] [rbp-560h] BYREF
  void *v231; // [rsp+1B8h] [rbp-558h] BYREF
  __int64 v232; // [rsp+1C0h] [rbp-550h] BYREF
  __int64 v233; // [rsp+1C8h] [rbp-548h]
  __int64 v234; // [rsp+1D0h] [rbp-540h] BYREF
  unsigned int v235; // [rsp+1D8h] [rbp-538h]
  __m128i v236; // [rsp+270h] [rbp-4A0h] BYREF
  __m128i v237; // [rsp+280h] [rbp-490h] BYREF
  char *v238; // [rsp+290h] [rbp-480h] BYREF
  char v239; // [rsp+2A0h] [rbp-470h] BYREF
  char *v240; // [rsp+2A8h] [rbp-468h]
  int v241; // [rsp+2B0h] [rbp-460h]
  char v242; // [rsp+2B8h] [rbp-458h] BYREF
  char *v243; // [rsp+2D8h] [rbp-438h]
  int v244; // [rsp+2E0h] [rbp-430h]
  char v245; // [rsp+2E8h] [rbp-428h] BYREF
  char *v246; // [rsp+308h] [rbp-408h]
  char v247; // [rsp+318h] [rbp-3F8h] BYREF
  char *v248; // [rsp+338h] [rbp-3D8h]
  char v249; // [rsp+348h] [rbp-3C8h] BYREF
  char *v250; // [rsp+368h] [rbp-3A8h]
  int v251; // [rsp+370h] [rbp-3A0h]
  char v252; // [rsp+378h] [rbp-398h] BYREF
  __int64 v253; // [rsp+3A0h] [rbp-370h]
  unsigned int v254; // [rsp+3B0h] [rbp-360h]
  __int64 v255; // [rsp+3B8h] [rbp-358h]
  unsigned int v256; // [rsp+3C0h] [rbp-350h]
  char *v257; // [rsp+3C8h] [rbp-348h] BYREF
  int v258; // [rsp+3D0h] [rbp-340h]
  char v259; // [rsp+3D8h] [rbp-338h] BYREF
  __int64 v260; // [rsp+408h] [rbp-308h]
  unsigned int v261; // [rsp+418h] [rbp-2F8h]
  _QWORD v262[2]; // [rsp+420h] [rbp-2F0h] BYREF
  _BYTE v263[512]; // [rsp+430h] [rbp-2E0h] BYREF
  __int64 v264; // [rsp+630h] [rbp-E0h]
  __int64 v265; // [rsp+638h] [rbp-D8h]
  __int64 v266; // [rsp+640h] [rbp-D0h]
  __int64 v267; // [rsp+648h] [rbp-C8h]
  char v268; // [rsp+650h] [rbp-C0h]
  __int64 v269; // [rsp+658h] [rbp-B8h]
  char *v270; // [rsp+660h] [rbp-B0h]
  __int64 v271; // [rsp+668h] [rbp-A8h]
  int v272; // [rsp+670h] [rbp-A0h]
  char v273; // [rsp+674h] [rbp-9Ch]
  char v274; // [rsp+678h] [rbp-98h] BYREF
  __int16 v275; // [rsp+6B8h] [rbp-58h]
  _QWORD *v276; // [rsp+6C0h] [rbp-50h]
  _QWORD *v277; // [rsp+6C8h] [rbp-48h]
  __int64 v278; // [rsp+6D0h] [rbp-40h]

  v206 = a1;
  v198 = 0;
  v7 = sub_D4B130(a1);
  if ( a5 )
  {
    v8 = sub_22077B0(760);
    v198 = v8;
    if ( v8 )
    {
      *(_QWORD *)v8 = a5;
      *(_QWORD *)(v8 + 8) = v8 + 24;
      *(_QWORD *)(v8 + 16) = 0x1000000000LL;
      *(_QWORD *)(v8 + 416) = v8 + 440;
      *(_QWORD *)(v8 + 504) = v8 + 520;
      *(_QWORD *)(v8 + 512) = 0x800000000LL;
      *(_QWORD *)(v8 + 408) = 0;
      *(_QWORD *)(v8 + 424) = 8;
      *(_DWORD *)(v8 + 432) = 0;
      *(_BYTE *)(v8 + 436) = 1;
      *(_DWORD *)(v8 + 720) = 0;
      *(_QWORD *)(v8 + 728) = 0;
      *(_QWORD *)(v8 + 736) = v8 + 720;
      *(_QWORD *)(v8 + 744) = v8 + 720;
      *(_QWORD *)(v8 + 752) = 0;
    }
  }
  if ( a3 )
  {
    sub_DAC210(a3, v206);
    sub_D9D700(a3, 0);
  }
  v202 = v7 + 48;
  v9 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v7 + 48 == v9 )
  {
    v204 = 0;
  }
  else
  {
    if ( !v9 )
      BUG();
    v10 = *(unsigned __int8 *)(v9 - 24);
    v11 = 0;
    v12 = v9 - 24;
    if ( (unsigned int)(v10 - 30) < 0xB )
      v11 = (_QWORD *)v12;
    v204 = v11;
  }
  v221 = (__int64 *)sub_BD5C60((__int64)v204);
  v222 = &v230;
  v223 = &v231;
  v215 = (unsigned int *)v217;
  v230 = &unk_49DA100;
  v216 = 0x200000000LL;
  v226 = 512;
  LOWORD(v220) = 0;
  v231 = &unk_49DA0B0;
  v224 = 0;
  v225 = 0;
  v227 = 7;
  v228 = 0;
  v229 = 0;
  v218 = 0;
  v219 = 0;
  sub_D5F1F0((__int64)&v215, (__int64)v204);
  v13 = sub_D47600(v206);
  v264 = 0;
  v196 = v13;
  v14 = v13;
  v262[0] = v263;
  v262[1] = 0x1000000000LL;
  v265 = 0;
  v266 = a2;
  v267 = 0;
  v268 = 0;
  v269 = 0;
  v270 = &v274;
  v271 = 8;
  v272 = 0;
  v273 = 1;
  v275 = 0;
  v276 = 0;
  v277 = 0;
  v278 = 0;
  if ( v13 )
  {
    v15 = **(_QWORD **)(v206 + 32);
    v16 = sub_ACD720(v221);
    LOWORD(v238) = 257;
    v17 = v16;
    v18 = sub_BD2C40(72, 3u);
    v20 = (__int64)v18;
    if ( v18 )
      sub_B4C9A0((__int64)v18, v15, v14, v17, 3u, v19, 0, 0);
    (*((void (__fastcall **)(void **, __int64, __m128i *, __int64, __int64))*v223 + 2))(v223, v20, &v236, v219, v220);
    v21 = v215;
    v22 = &v215[4 * (unsigned int)v216];
    if ( v215 != v22 )
    {
      do
      {
        v23 = *((_QWORD *)v21 + 1);
        v24 = *v21;
        v21 += 4;
        sub_B99FD0(v20, v24, v23);
      }
      while ( v22 != v21 );
    }
    sub_B43D60(v204);
    v25 = sub_AA5930(v196);
    v27 = v26;
    v28 = v25;
    while ( v27 != v28 )
    {
      *(_QWORD *)(*(_QWORD *)(v28 - 8) + 32LL * *(unsigned int *)(v28 + 72)) = v7;
      sub_B57920(v28, (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_F6B860, (__int64)&v236, 0);
      v29 = *(_QWORD *)(v28 + 32);
      if ( !v29 )
        BUG();
      v28 = 0;
      if ( *(_BYTE *)(v29 - 24) == 84 )
        v28 = v29 - 24;
    }
    if ( a2 )
    {
      v236.m128i_i64[0] = v7;
      v236.m128i_i64[1] = v196 & 0xFFFFFFFFFFFFFFFBLL;
      sub_FFB3D0(v262, &v236, 1);
      if ( a5 )
      {
        v236.m128i_i64[0] = v7;
        v236.m128i_i64[1] = v196 & 0xFFFFFFFFFFFFFFFBLL;
        sub_D75690((__int64 *)v198, (unsigned __int64 *)&v236, 1, a2, 0);
        if ( byte_4F8F8E8[0] )
          nullsub_390(a5, 0);
      }
    }
    v30 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v202 == v30 )
    {
      v32 = 0;
    }
    else
    {
      if ( !v30 )
        BUG();
      v31 = *(unsigned __int8 *)(v30 - 24);
      v32 = 0;
      v33 = v30 - 24;
      if ( (unsigned int)(v31 - 30) < 0xB )
        v32 = v33;
    }
    sub_D5F1F0((__int64)&v215, v32);
    LOWORD(v238) = 257;
    v34 = sub_BD2C40(72, 1u);
    v35 = (__int64)v34;
    if ( v34 )
      sub_B4C8F0((__int64)v34, v196, 1u, 0, 0);
    v36 = v35;
    (*((void (__fastcall **)(void **, __int64, __m128i *, __int64, __int64))*v223 + 2))(v223, v35, &v236, v219, v220);
    v37 = v215;
    v38 = &v215[4 * (unsigned int)v216];
    if ( v215 != v38 )
    {
      do
      {
        v39 = *((_QWORD *)v37 + 1);
        v36 = *v37;
        v37 += 4;
        sub_B99FD0(v35, v36, v39);
      }
      while ( v38 != v37 );
    }
    v40 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v202 != v40 )
    {
      if ( !v40 )
        BUG();
LABEL_35:
      v41 = *(unsigned __int8 *)(v40 - 24);
      v42 = 0;
      v43 = (_QWORD *)(v40 - 24);
      if ( (unsigned int)(v41 - 30) < 0xB )
        v42 = v43;
      goto LABEL_37;
    }
  }
  else
  {
    sub_D5F1F0((__int64)&v215, (__int64)v204);
    LOWORD(v238) = 257;
    v187 = sub_BD2C40(72, unk_3F148B8);
    v188 = (__int64)v187;
    if ( v187 )
      sub_B4C8A0((__int64)v187, (__int64)v221, 0, 0);
    v36 = v188;
    (*((void (__fastcall **)(void **, __int64, __m128i *, __int64, __int64))*v223 + 2))(v223, v188, &v236, v219, v220);
    v189 = v215;
    v190 = &v215[4 * (unsigned int)v216];
    if ( v215 != v190 )
    {
      do
      {
        v191 = *((_QWORD *)v189 + 1);
        v36 = *v189;
        v189 += 4;
        sub_B99FD0(v188, v36, v191);
      }
      while ( v190 != v189 );
    }
    v40 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v202 != v40 )
    {
      if ( !v40 )
        BUG();
      goto LABEL_35;
    }
  }
  v42 = 0;
LABEL_37:
  sub_B43D60(v42);
  if ( a2 )
  {
    v36 = (__int64)&v236;
    v46 = **(_QWORD **)(v206 + 32);
    v236.m128i_i64[0] = v7;
    v236.m128i_i64[1] = v46 | 4;
    sub_FFB3D0(v262, &v236, 1);
    if ( a5 )
    {
      v47 = **(_QWORD **)(v206 + 32);
      v236.m128i_i64[0] = v7;
      v236.m128i_i64[1] = v47 | 4;
      sub_D75690((__int64 *)v198, (unsigned __int64 *)&v236, 1, a2, 0);
      sub_F774E0((__int64)&v236, *(__int64 **)(v206 + 32), *(__int64 **)(v206 + 40), v48, v49, v50);
      v51 = &v236;
      sub_D6F970((_QWORD *)v198, (__int64)&v236);
      if ( byte_4F8F8E8[0] )
      {
        v51 = 0;
        nullsub_390(a5, 0);
      }
      if ( v238 != &v239 )
        _libc_free(v238, v51);
      v36 = 8LL * v237.m128i_u32[2];
      sub_C7D6A0(v236.m128i_i64[1], v36, 8);
    }
  }
  v232 = 0;
  v52 = (__m128i *)&v234;
  v233 = 1;
  do
  {
    v52->m128i_i64[0] = 0;
    v52 = (__m128i *)((char *)v52 + 40);
    v52[-1].m128i_i8[0] = 0;
    v52[-1].m128i_i64[1] = 0;
  }
  while ( v52 != &v236 );
  v53 = (__int64 *)v214;
  v209 = v211;
  v210 = 0x400000000LL;
  v212 = v214;
  v213 = 0x400000000LL;
  if ( v196 )
  {
    v192 = *(__int64 **)(v206 + 40);
    v197 = *(__int64 **)(v206 + 32);
    if ( v197 != v192 )
    {
      while ( 1 )
      {
        v54 = *(_QWORD **)(*v197 + 56);
        v200 = *v197;
        v199 = *v197 + 48;
        if ( (_QWORD *)v199 != v54 )
          break;
LABEL_61:
        if ( v192 == ++v197 )
          goto LABEL_62;
      }
      while ( 1 )
      {
        if ( !v54 )
          BUG();
        v201 = (__int64)(v54 - 3);
        v55 = sub_ACADE0((__int64 **)*(v54 - 2));
        v56 = *(v54 - 1);
        v57 = v55;
        v58 = v55 + 16;
        if ( v56 )
        {
          while ( 1 )
          {
            v59 = v56;
            v56 = *(_QWORD *)(v56 + 8);
            v60 = *(_QWORD *)(v59 + 24);
            if ( *(_BYTE *)v60 <= 0x1Cu )
              goto LABEL_212;
            v61 = *(_QWORD *)(v60 + 40);
            if ( *(_BYTE *)(v206 + 84) )
            {
              v62 = *(_QWORD **)(v206 + 64);
              v63 = &v62[*(unsigned int *)(v206 + 76)];
              if ( v62 != v63 )
              {
                while ( v61 != *v62 )
                {
                  if ( v63 == ++v62 )
                    goto LABEL_212;
                }
                goto LABEL_57;
              }
LABEL_212:
              if ( *(_QWORD *)v59 )
              {
                v138 = *(_QWORD *)(v59 + 8);
                **(_QWORD **)(v59 + 16) = v138;
                if ( v138 )
                  *(_QWORD *)(v138 + 16) = *(_QWORD *)(v59 + 16);
              }
              *(_QWORD *)v59 = v57;
              if ( !v57 )
                goto LABEL_57;
              v139 = *(_QWORD *)(v57 + 16);
              *(_QWORD *)(v59 + 8) = v139;
              if ( v139 )
                *(_QWORD *)(v139 + 16) = v59 + 8;
              *(_QWORD *)(v59 + 16) = v58;
              *(_QWORD *)(v57 + 16) = v59;
              if ( !v56 )
                break;
            }
            else
            {
              if ( !sub_C8CA60(v206 + 56, v61) )
                goto LABEL_212;
LABEL_57:
              if ( !v56 )
                break;
            }
          }
        }
        if ( *(_BYTE *)(v200 + 40) )
        {
          v150 = v54[5];
          if ( v150 )
          {
            v152 = (_QWORD *)sub_B14240(v150);
            for ( i = v151; v152 != v151; v152 = (_QWORD *)v152[1] )
            {
              if ( !*((_BYTE *)v152 + 32) )
                break;
            }
          }
          else
          {
            i = &qword_4F81430[1];
            v152 = &qword_4F81430[1];
          }
          if ( i != v152 )
          {
            v193 = v54;
            while ( 1 )
            {
              for ( j = v152[1]; (_QWORD *)j != i; j = *(_QWORD *)(j + 8) )
              {
                if ( !*(_BYTE *)(j + 32) )
                  break;
              }
              v155 = v152[3];
              v208 = (__m128i *)v155;
              if ( v155 )
                sub_B96E90((__int64)&v208, v155, 1);
              v156 = sub_B10CD0((__int64)&v208);
              v205 = sub_B11F60((__int64)(v152 + 10));
              v236.m128i_i64[0] = sub_B12000((__int64)(v152 + 9));
              if ( v205 )
                sub_AF47B0(
                  (__int64)&v236.m128i_i64[1],
                  *(unsigned __int64 **)(v205 + 16),
                  *(unsigned __int64 **)(v205 + 24));
              else
                v237.m128i_i8[8] = 0;
              v238 = (char *)v156;
              if ( v208 )
                sub_B91220((__int64)&v208, (__int64)v208);
              if ( (unsigned __int8)sub_F38D60((__int64)&v232, (__int64)&v236, (__int64 *)&v207) )
              {
                if ( (_QWORD *)j == i )
                  goto LABEL_266;
                goto LABEL_256;
              }
              v157 = v207;
              ++v232;
              v208 = v207;
              v158 = ((unsigned int)v233 >> 1) + 1;
              if ( (v233 & 1) != 0 )
              {
                v159 = 4;
                if ( 4 * v158 >= 0xC )
                {
LABEL_268:
                  v159 *= 2;
LABEL_269:
                  sub_F3E3C0((__int64)&v232, v159);
                  sub_F38D60((__int64)&v232, (__int64)&v236, (__int64 *)&v208);
                  v157 = v208;
                  v158 = ((unsigned int)v233 >> 1) + 1;
                  goto LABEL_261;
                }
              }
              else
              {
                v159 = v235;
                if ( 4 * v158 >= 3 * v235 )
                  goto LABEL_268;
              }
              if ( v159 - (v158 + HIDWORD(v233)) <= v159 >> 3 )
                goto LABEL_269;
LABEL_261:
              LODWORD(v233) = v233 & 1 | (2 * v158);
              if ( v157->m128i_i64[0] || v157[1].m128i_i8[8] || v157[2].m128i_i64[0] )
                --HIDWORD(v233);
              *v157 = _mm_loadu_si128(&v236);
              v157[1] = _mm_loadu_si128(&v237);
              v157[2].m128i_i64[0] = (__int64)v238;
              sub_B14260(v152);
              v162 = (unsigned int)v213;
              v163 = (unsigned int)v213 + 1LL;
              if ( v163 > HIDWORD(v213) )
              {
                sub_C8D5F0((__int64)&v212, v214, v163, 8u, v160, v161);
                v162 = (unsigned int)v213;
              }
              *(_QWORD *)&v212[8 * v162] = v152;
              LODWORD(v213) = v213 + 1;
              if ( (_QWORD *)j == i )
              {
LABEL_266:
                v54 = v193;
                break;
              }
LABEL_256:
              v152 = (_QWORD *)j;
            }
          }
        }
        if ( *((_BYTE *)v54 - 24) == 85 )
        {
          v140 = *(v54 - 7);
          if ( v140 )
          {
            if ( !*(_BYTE *)v140 && *(_QWORD *)(v140 + 24) == v54[7] && (*(_BYTE *)(v140 + 33) & 0x20) != 0 )
            {
              v141 = *(_DWORD *)(v140 + 36);
              if ( v141 > 0x45 )
              {
                if ( v141 != 71 )
                  goto LABEL_60;
              }
              else if ( v141 <= 0x43 )
              {
                goto LABEL_60;
              }
              sub_AF4850((__int64)&v236, v201);
              if ( !(unsigned __int8)sub_F38D60((__int64)&v232, (__int64)&v236, (__int64 *)&v207) )
              {
                v144 = v207;
                ++v232;
                v208 = v207;
                v145 = ((unsigned int)v233 >> 1) + 1;
                if ( (v233 & 1) == 0 )
                {
                  v146 = v235;
                  if ( 3 * v235 > 4 * v145 )
                    goto LABEL_229;
LABEL_271:
                  v146 *= 2;
                  goto LABEL_272;
                }
                v146 = 4;
                if ( 4 * v145 >= 0xC )
                  goto LABEL_271;
LABEL_229:
                if ( v146 - (v145 + HIDWORD(v233)) <= v146 >> 3 )
                {
LABEL_272:
                  sub_F3E3C0((__int64)&v232, v146);
                  sub_F38D60((__int64)&v232, (__int64)&v236, (__int64 *)&v208);
                  v144 = v208;
                  v145 = ((unsigned int)v233 >> 1) + 1;
                }
                LODWORD(v233) = v233 & 1 | (2 * v145);
                if ( v144->m128i_i64[0] || v144[1].m128i_i8[8] || v144[2].m128i_i64[0] )
                  --HIDWORD(v233);
                *v144 = _mm_loadu_si128(&v236);
                v147 = HIDWORD(v210);
                v144[1] = _mm_loadu_si128(&v237);
                v144[2].m128i_i64[0] = (__int64)v238;
                v148 = (unsigned int)v210;
                v149 = (unsigned int)v210 + 1LL;
                if ( v149 > v147 )
                {
                  sub_C8D5F0((__int64)&v209, v211, v149, 8u, v142, v143);
                  v148 = (unsigned int)v210;
                }
                *(_QWORD *)&v209[8 * v148] = v201;
                LODWORD(v210) = v210 + 1;
              }
            }
          }
        }
LABEL_60:
        v54 = (_QWORD *)v54[1];
        if ( (_QWORD *)v199 == v54 )
          goto LABEL_61;
      }
    }
LABEL_62:
    v64 = (__int64 *)sub_AA4B30(v196);
    sub_AE0470((__int64)&v236, v64, 1, 0);
    v65 = sub_AA5190(v196);
    v67 = (_QWORD **)v209;
    v68 = (unsigned __int64 *)v65;
    v70 = v69;
    if ( !v68 )
    {
      v66 = 0;
      v70 = 0;
    }
    v71 = v66;
    BYTE1(v71) = v70;
    v72 = v66;
    if ( &v209[8 * (unsigned int)v210] != v209 )
    {
      v73 = v68;
      v74 = (_QWORD **)&v209[8 * (unsigned int)v210];
      v75 = v71;
      do
      {
        v76 = *v67;
        v64 = (__int64 *)v196;
        ++v67;
        sub_B44550(v76, v196, v73, v75);
      }
      while ( v74 != v67 );
      v68 = v73;
      v72 = v75;
    }
    v77 = &v212[8 * (unsigned int)v213];
    if ( v212 != v77 )
    {
      v78 = v72;
      v79 = (__int64)v68;
      v80 = v212;
      do
      {
        v64 = (__int64 *)*((_QWORD *)v77 - 1);
        v77 -= 8;
        sub_AA8770(v196, (__int64)v64, v79, v78);
      }
      while ( v80 != v77 );
    }
    v81 = v261;
    if ( v261 )
    {
      v82 = v260;
      v83 = v260 + 56LL * v261;
      do
      {
        if ( *(_QWORD *)v82 != -8192 && *(_QWORD *)v82 != -4096 )
        {
          v84 = *(_QWORD *)(v82 + 8);
          v85 = v84 + 8LL * *(unsigned int *)(v82 + 16);
          if ( v84 != v85 )
          {
            do
            {
              v64 = *(__int64 **)(v85 - 8);
              v85 -= 8;
              if ( v64 )
                sub_B91220(v85, (__int64)v64);
            }
            while ( v84 != v85 );
            v85 = *(_QWORD *)(v82 + 8);
          }
          if ( v85 != v82 + 24 )
            _libc_free(v85, v64);
        }
        v82 += 56;
      }
      while ( v83 != v82 );
      v81 = v261;
    }
    v86 = 56 * v81;
    sub_C7D6A0(v260, 56 * v81, 8);
    v87 = v257;
    v88 = &v257[8 * v258];
    if ( v257 != v88 )
    {
      do
      {
        v86 = *((_QWORD *)v88 - 1);
        v88 -= 8;
        if ( v86 )
          sub_B91220((__int64)v88, v86);
      }
      while ( v87 != v88 );
      v88 = v257;
    }
    if ( v88 != &v259 )
      _libc_free(v88, v86);
    v89 = v255;
    v90 = v255 + 56LL * v256;
    if ( v255 != v90 )
    {
      do
      {
        v90 -= 56;
        v91 = *(_QWORD *)(v90 + 40);
        if ( v91 != v90 + 56 )
          _libc_free(v91, v86);
        v86 = 8LL * *(unsigned int *)(v90 + 32);
        sub_C7D6A0(*(_QWORD *)(v90 + 16), v86, 8);
      }
      while ( v89 != v90 );
      v90 = v255;
    }
    if ( (char **)v90 != &v257 )
      _libc_free(v90, v86);
    v36 = 16LL * v254;
    sub_C7D6A0(v253, v36, 8);
    v92 = v250;
    v93 = &v250[8 * v251];
    if ( v250 != v93 )
    {
      do
      {
        v36 = *((_QWORD *)v93 - 1);
        v93 -= 8;
        if ( v36 )
          sub_B91220((__int64)v93, v36);
      }
      while ( v92 != v93 );
      v93 = v250;
    }
    if ( v93 != &v252 )
      _libc_free(v93, v36);
    if ( v248 != &v249 )
      _libc_free(v248, v36);
    if ( v246 != &v247 )
      _libc_free(v246, v36);
    v94 = v243;
    v95 = &v243[8 * v244];
    if ( v243 != v95 )
    {
      do
      {
        v36 = *((_QWORD *)v95 - 1);
        v95 -= 8;
        if ( v36 )
          sub_B91220((__int64)v95, v36);
      }
      while ( v94 != v95 );
      v95 = v243;
    }
    if ( v95 != &v245 )
      _libc_free(v95, v36);
    v96 = v240;
    v97 = &v240[8 * v241];
    if ( v240 != v97 )
    {
      do
      {
        v36 = *((_QWORD *)v97 - 1);
        v97 -= 8;
        if ( v36 )
          sub_B91220((__int64)v97, v36);
      }
      while ( v96 != v97 );
      v97 = v240;
    }
    if ( v97 != &v242 )
      _libc_free(v97, v36);
  }
  v98 = *(__int64 **)(v206 + 40);
  for ( k = *(__int64 **)(v206 + 32); v98 != k; ++k )
  {
    v100 = *k;
    sub_AA5200(v100);
  }
  if ( a5 && byte_4F8F8E8[0] )
  {
    v36 = 0;
    nullsub_390(a5, 0);
  }
  if ( !a4 )
    goto LABEL_163;
  v101 = (_QWORD *)v206;
  v102 = *(_QWORD ***)(v206 + 32);
  v103 = *(_QWORD ***)(v206 + 40);
  if ( v102 != v103 )
  {
    do
    {
      v104 = *v102++;
      sub_AA5450(v104);
    }
    while ( v103 != v102 );
    v101 = (_QWORD *)v206;
  }
  v236.m128i_i64[0] = 0;
  v236.m128i_i64[1] = (__int64)&v238;
  v237.m128i_i64[0] = 8;
  v237.m128i_i32[2] = 0;
  v237.m128i_i8[12] = 1;
  v105 = (__int64 *)v101[4];
  v106 = (__int64 *)v101[5];
  if ( v105 == v106 )
    goto LABEL_147;
  v107 = 1;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v108 = *v105;
        if ( (_BYTE)v107 )
          break;
LABEL_277:
        ++v105;
        sub_C8CC70((__int64)&v236, v108, v44, (__int64)v53, v107, v45);
        v107 = v237.m128i_u8[12];
        v44 = v236.m128i_i64[1];
        if ( v106 == v105 )
          goto LABEL_140;
      }
      v44 = v236.m128i_i64[1];
      v53 = (__int64 *)(v236.m128i_i64[1] + 8LL * v237.m128i_u32[1]);
      if ( (__int64 *)v236.m128i_i64[1] != v53 )
        break;
LABEL_279:
      if ( v237.m128i_i32[1] >= (unsigned __int32)v237.m128i_i32[0] )
        goto LABEL_277;
      ++v105;
      ++v237.m128i_i32[1];
      *v53 = v108;
      v44 = v236.m128i_i64[1];
      ++v236.m128i_i64[0];
      v107 = v237.m128i_u8[12];
      if ( v106 == v105 )
        goto LABEL_140;
    }
    v109 = (_QWORD *)v236.m128i_i64[1];
    while ( v108 != *v109 )
    {
      if ( v53 == ++v109 )
        goto LABEL_279;
    }
    ++v105;
  }
  while ( v106 != v105 );
LABEL_140:
  if ( (_BYTE)v107 )
    v110 = (__int64 *)(v44 + 8LL * v237.m128i_u32[1]);
  else
    v110 = (__int64 *)(v44 + 8LL * v237.m128i_u32[0]);
  if ( (__int64 *)v44 != v110 )
  {
    v111 = (__int64 *)v44;
    while ( 1 )
    {
      v112 = *v111;
      if ( (unsigned __int64)*v111 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v110 == ++v111 )
        goto LABEL_146;
    }
    if ( v111 != v110 )
    {
      v167 = v111;
      do
      {
        v168 = *(_QWORD *)(a4 + 8);
        v169 = *(unsigned int *)(a4 + 24);
        if ( (_DWORD)v169 )
        {
          v170 = (v169 - 1) & (((unsigned int)v112 >> 9) ^ ((unsigned int)v112 >> 4));
          v171 = (__int64 *)(v168 + 16LL * v170);
          v172 = *v171;
          if ( v112 == *v171 )
          {
LABEL_296:
            if ( v171 != (__int64 *)(v168 + 16 * v169) )
            {
              for ( m = (__int64 *)v171[1]; m; m = (__int64 *)*m )
              {
                v174 = (char *)m[5];
                v175 = (char *)m[4];
                v176 = (v174 - v175) >> 5;
                v177 = (v174 - v175) >> 3;
                if ( v176 > 0 )
                {
                  v178 = &v175[32 * v176];
                  while ( v112 != *(_QWORD *)v175 )
                  {
                    if ( v112 == *((_QWORD *)v175 + 1) )
                    {
                      v175 += 8;
                      goto LABEL_305;
                    }
                    if ( v112 == *((_QWORD *)v175 + 2) )
                    {
                      v175 += 16;
                      goto LABEL_305;
                    }
                    if ( v112 == *((_QWORD *)v175 + 3) )
                    {
                      v175 += 24;
                      goto LABEL_305;
                    }
                    v175 += 32;
                    if ( v175 == v178 )
                    {
                      v177 = (v174 - v175) >> 3;
                      goto LABEL_324;
                    }
                  }
                  goto LABEL_305;
                }
LABEL_324:
                if ( v177 != 2 )
                {
                  if ( v177 != 3 )
                  {
                    if ( v177 == 1 )
                      goto LABEL_335;
                    v175 = (char *)m[5];
                    goto LABEL_305;
                  }
                  if ( v112 == *(_QWORD *)v175 )
                    goto LABEL_305;
                  v175 += 8;
                }
                if ( v112 != *(_QWORD *)v175 )
                {
                  v175 += 8;
LABEL_335:
                  if ( v112 != *(_QWORD *)v175 )
                    v175 = (char *)m[5];
                }
LABEL_305:
                if ( v175 + 8 != v174 )
                {
                  memmove(v175, v175 + 8, v174 - (v175 + 8));
                  v174 = (char *)m[5];
                }
                v179 = *((_BYTE *)m + 84) == 0;
                m[5] = (__int64)(v174 - 8);
                if ( v179 )
                {
                  v185 = sub_C8CA60((__int64)(m + 7), v112);
                  if ( v185 )
                  {
                    *v185 = -2;
                    ++*((_DWORD *)m + 20);
                    ++m[7];
                  }
                }
                else
                {
                  v180 = (_QWORD *)m[8];
                  v181 = &v180[*((unsigned int *)m + 19)];
                  v182 = v180;
                  if ( v180 != v181 )
                  {
                    while ( v112 != *v182 )
                    {
                      if ( v181 == ++v182 )
                        goto LABEL_313;
                    }
                    v183 = (unsigned int)(*((_DWORD *)m + 19) - 1);
                    *((_DWORD *)m + 19) = v183;
                    *v182 = v180[v183];
                    ++m[7];
                  }
                }
LABEL_313:
                ;
              }
              *v171 = -8192;
              --*(_DWORD *)(a4 + 16);
              ++*(_DWORD *)(a4 + 20);
            }
          }
          else
          {
            v186 = 1;
            while ( v172 != -4096 )
            {
              v170 = (v169 - 1) & (v186 + v170);
              v171 = (__int64 *)(v168 + 16LL * v170);
              v172 = *v171;
              if ( v112 == *v171 )
                goto LABEL_296;
              ++v186;
            }
          }
        }
        v184 = v167 + 1;
        if ( v167 + 1 == v110 )
          break;
        v112 = *v184;
        for ( ++v167; (unsigned __int64)*v184 >= 0xFFFFFFFFFFFFFFFELL; v167 = v184 )
        {
          if ( v110 == ++v184 )
            goto LABEL_146;
          v112 = *v184;
        }
      }
      while ( v184 != v110 );
    }
  }
LABEL_146:
  v101 = (_QWORD *)v206;
LABEL_147:
  if ( *v101 )
  {
    v113 = sub_F6B930(*(_QWORD **)(*v101 + 8LL), *(_QWORD *)(*v101 + 16LL), (__int64 *)&v206);
    v115 = *(_QWORD **)v113;
    v36 = (__int64)v113;
    sub_D4C9B0(v114 + 8, v113);
    *v115 = 0;
  }
  else
  {
    v36 = (__int64)sub_F6B930(*(_QWORD **)(a4 + 32), *(_QWORD *)(a4 + 40), (__int64 *)&v206);
    sub_D4C9B0(a4 + 32, (_BYTE *)v36);
  }
  v116 = v206;
  v117 = *(__int64 **)(v206 + 8);
  v118 = *(__int64 **)(v206 + 16);
  if ( v117 == v118 )
  {
    *(_BYTE *)(v206 + 152) = 1;
  }
  else
  {
    do
    {
      v119 = *v117++;
      sub_D47BB0(v119, v36);
    }
    while ( v118 != v117 );
    *(_BYTE *)(v116 + 152) = 1;
    v120 = *(_QWORD *)(v116 + 8);
    if ( *(_QWORD *)(v116 + 16) != v120 )
      *(_QWORD *)(v116 + 16) = v120;
  }
  v121 = *(_QWORD *)(v116 + 32);
  if ( v121 != *(_QWORD *)(v116 + 40) )
    *(_QWORD *)(v116 + 40) = v121;
  ++*(_QWORD *)(v116 + 56);
  if ( *(_BYTE *)(v116 + 84) )
  {
    *(_QWORD *)v116 = 0;
  }
  else
  {
    v164 = 4 * (*(_DWORD *)(v116 + 76) - *(_DWORD *)(v116 + 80));
    v165 = *(unsigned int *)(v116 + 72);
    if ( v164 < 0x20 )
      v164 = 32;
    if ( (unsigned int)v165 > v164 )
    {
      sub_C8C990(v116 + 56, v36);
    }
    else
    {
      v36 = 0xFFFFFFFFLL;
      memset(*(void **)(v116 + 64), -1, 8 * v165);
    }
    v166 = *(_BYTE *)(v116 + 84);
    *(_QWORD *)v116 = 0;
    if ( !v166 )
      _libc_free(*(_QWORD *)(v116 + 64), v36);
  }
  v122 = *(_QWORD *)(v116 + 32);
  if ( v122 )
  {
    v36 = *(_QWORD *)(v116 + 48) - v122;
    j_j___libc_free_0(v122, v36);
  }
  v123 = *(_QWORD *)(v116 + 8);
  if ( v123 )
  {
    v36 = *(_QWORD *)(v116 + 24) - v123;
    j_j___libc_free_0(v123, v36);
  }
  if ( !v237.m128i_i8[12] )
    _libc_free(v236.m128i_i64[1], v36);
LABEL_163:
  if ( v212 != v214 )
    _libc_free(v212, v36);
  if ( v209 != v211 )
    _libc_free(v209, v36);
  if ( (v233 & 1) == 0 )
  {
    v36 = 40LL * v235;
    sub_C7D6A0(v234, v36, 8);
  }
  sub_FFCE90(v262);
  sub_FFD870(v262);
  sub_FFBC40(v262);
  v124 = v277;
  v125 = v276;
  if ( v277 != v276 )
  {
    do
    {
      v126 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v125[7];
      *v125 = &unk_49E5048;
      if ( v126 )
      {
        v36 = (__int64)(v125 + 5);
        v126(v125 + 5, v125 + 5, 3);
      }
      *v125 = &unk_49DB368;
      v127 = v125[3];
      if ( v127 != 0 && v127 != -4096 && v127 != -8192 )
        sub_BD60C0(v125 + 1);
      v125 += 9;
    }
    while ( v124 != v125 );
    v125 = v276;
  }
  if ( v125 )
  {
    v36 = v278 - (_QWORD)v125;
    j_j___libc_free_0(v125, v278 - (_QWORD)v125);
  }
  if ( !v273 )
    _libc_free(v270, v36);
  if ( (_BYTE *)v262[0] != v263 )
    _libc_free(v262[0], v36);
  nullsub_61();
  v230 = &unk_49DA100;
  nullsub_63();
  if ( v215 != (unsigned int *)v217 )
    _libc_free(v215, v36);
  result = v198;
  if ( v198 )
  {
    v129 = *(_QWORD *)(v198 + 728);
    while ( v129 )
    {
      v130 = v129;
      sub_F6BD50(*(_QWORD **)(v129 + 24));
      v131 = *(_QWORD *)(v129 + 48);
      v129 = *(_QWORD *)(v129 + 16);
      if ( v131 != 0 && v131 != -4096 && v131 != -8192 )
        sub_BD60C0((_QWORD *)(v130 + 32));
      v36 = 56;
      j_j___libc_free_0(v130, 56);
    }
    v132 = *(_QWORD **)(v198 + 504);
    v133 = &v132[3 * *(unsigned int *)(v198 + 512)];
    if ( v132 != v133 )
    {
      do
      {
        v134 = *(v133 - 1);
        v133 -= 3;
        if ( v134 != -4096 && v134 != 0 && v134 != -8192 )
          sub_BD60C0(v133);
      }
      while ( v132 != v133 );
      v133 = *(_QWORD **)(v198 + 504);
    }
    if ( v133 != (_QWORD *)(v198 + 520) )
      _libc_free(v133, v36);
    if ( !*(_BYTE *)(v198 + 436) )
      _libc_free(*(_QWORD *)(v198 + 416), v36);
    v135 = *(_QWORD **)(v198 + 8);
    v136 = &v135[3 * *(unsigned int *)(v198 + 16)];
    if ( v135 != v136 )
    {
      do
      {
        v137 = *(v136 - 1);
        v136 -= 3;
        if ( v137 != 0 && v137 != -4096 && v137 != -8192 )
          sub_BD60C0(v136);
      }
      while ( v135 != v136 );
      v136 = *(_QWORD **)(v198 + 8);
    }
    if ( v136 != (_QWORD *)(v198 + 24) )
      _libc_free(v136, v36);
    return j_j___libc_free_0(v198, 760);
  }
  return result;
}
