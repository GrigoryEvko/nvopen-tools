// Function: sub_2708220
// Address: 0x2708220
//
__int64 __fastcall sub_2708220(__int64 a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __m128i a5)
{
  __int64 **v5; // rbx
  __int64 v6; // rax
  bool v7; // zf
  __int64 v8; // rax
  unsigned __int64 v9; // r12
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 *v13; // rdi
  __int64 v14; // rax
  __int64 *v15; // rdi
  __int64 v16; // rax
  __int64 *v17; // rdi
  __int64 v18; // rax
  __int64 *v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 *v23; // rdi
  __int64 *v24; // rax
  char v25; // al
  char v26; // bl
  unsigned __int64 v27; // rdi
  __int64 *v28; // r15
  __int64 *v29; // r13
  __int64 k; // rax
  __int64 v31; // rdi
  unsigned int v32; // ecx
  __int64 v33; // rsi
  __int64 *v34; // r13
  unsigned __int64 v35; // r14
  __int64 v36; // rsi
  __int64 v37; // rdi
  unsigned __int64 v38; // rdi
  __int64 v39; // rax
  _QWORD *v40; // r13
  _QWORD *v41; // r15
  unsigned __int64 v42; // r14
  unsigned __int64 v43; // r12
  unsigned __int64 v44; // rdi
  unsigned __int64 v45; // rdx
  unsigned __int64 v46; // rax
  __int64 *v47; // rdi
  __int64 v48; // rax
  __int64 *v49; // rdi
  __int64 v50; // rax
  __int64 *v51; // rdi
  __int64 v52; // rax
  __int64 *v53; // rdi
  __int64 v54; // rax
  __int64 v55; // rsi
  __int64 v56; // rax
  __int64 *v57; // rdi
  __int64 *v58; // rax
  char v59; // al
  char v60; // bl
  __int64 v61; // rsi
  __int64 v62; // rdx
  __int64 v64; // rax
  __int64 *v65; // r15
  __int64 *v66; // r13
  __int64 m; // rax
  __int64 v68; // rdi
  unsigned int v69; // ecx
  __int64 v70; // rsi
  __int64 *v71; // r13
  unsigned __int64 v72; // r14
  __int64 v73; // rsi
  __int64 v74; // rdi
  unsigned __int64 v75; // rdi
  unsigned __int64 v76; // r8
  __int64 v77; // r14
  __int64 v78; // r14
  __int64 v79; // r13
  _QWORD *v80; // rdi
  __int64 v81; // rax
  __int64 v82; // rbx
  __m128i *v83; // rax
  __m128i *v84; // rsi
  __int64 v85; // rcx
  __int64 v86; // r8
  __int64 v87; // r9
  char v88; // al
  __int64 v89; // rax
  unsigned __int64 v90; // rdx
  _QWORD *v91; // r13
  _QWORD *v92; // r15
  unsigned __int64 v93; // r14
  unsigned __int64 v94; // r12
  unsigned __int64 v95; // rdi
  __m128i *v96; // rax
  __int64 v97; // rax
  __int64 v98; // rax
  __int64 v99; // rsi
  __int64 v100; // rdx
  __int64 v101; // rcx
  __int64 v102; // r8
  __int64 v103; // r9
  __int64 v104; // rsi
  __int64 v105; // rdx
  __int64 v106; // rcx
  __int64 v107; // r8
  __int64 v108; // r9
  char v109; // dl
  char v110; // al
  unsigned __int64 v111; // rax
  unsigned __int64 v112; // rdi
  __int64 *v113; // r15
  __int64 *v114; // r14
  __int64 i; // rdx
  __int64 v116; // rdi
  unsigned int v117; // ecx
  __int64 v118; // rsi
  __int64 *v119; // r15
  unsigned __int64 v120; // r14
  __int64 v121; // rsi
  __int64 v122; // rdi
  unsigned __int64 v123; // rdi
  __int64 v124; // rax
  _QWORD *v125; // r14
  _QWORD *v126; // r15
  unsigned __int64 v127; // rbx
  unsigned __int64 v128; // r12
  unsigned __int64 v129; // rdi
  __int64 v130; // rax
  __int64 *v131; // r15
  __int64 *v132; // r14
  __int64 j; // rdx
  __int64 v134; // rdi
  unsigned int v135; // ecx
  __int64 v136; // rsi
  __int64 *v137; // r15
  unsigned __int64 v138; // r14
  __int64 v139; // rsi
  __int64 v140; // rdi
  unsigned __int64 v141; // rdi
  unsigned __int64 v142; // r8
  __int64 v143; // rax
  __int64 v144; // r15
  __int64 v145; // r14
  _QWORD *v146; // rdi
  __int64 v147; // r14
  int v148; // eax
  int v149; // eax
  __int64 v150; // rax
  unsigned __int64 v151; // r14
  unsigned __int64 v152; // rdi
  __int64 *v153; // r12
  __int64 *v154; // r15
  __int64 v155; // rdi
  unsigned int v156; // ecx
  __int64 v157; // rsi
  unsigned __int64 v158; // rdi
  __int64 *v159; // r12
  __int64 *v160; // r15
  __int64 v161; // rsi
  __int64 v162; // rdi
  unsigned __int64 v163; // rdi
  __int64 v164; // rax
  _QWORD *v165; // r15
  _QWORD *v166; // r12
  unsigned __int64 v167; // rbx
  unsigned __int64 v168; // r13
  unsigned __int64 v169; // rdi
  __int64 v170; // rax
  _QWORD *v171; // r14
  _QWORD *v172; // r15
  unsigned __int64 v173; // rbx
  unsigned __int64 v174; // r12
  unsigned __int64 v175; // rdi
  unsigned __int64 v176; // rax
  unsigned __int64 v177; // rax
  _QWORD *v178; // r15
  __int64 v179; // rsi
  unsigned int v180; // eax
  __int64 v181; // rdx
  __int64 v182; // rax
  unsigned __int64 v183; // r8
  __int64 (__fastcall **v184)(); // r14
  __int64 v185; // rax
  __int64 *v186; // rax
  __int64 *v187; // rcx
  __int64 *v188; // r12
  __int64 v189; // rax
  _QWORD *v190; // r15
  _QWORD *v191; // r12
  unsigned __int64 v192; // rbx
  unsigned __int64 v193; // r13
  unsigned __int64 v194; // rdi
  __int64 *v195; // r12
  __int64 *v196; // r15
  __int64 v197; // rdi
  unsigned int v198; // ecx
  __int64 v199; // rsi
  unsigned __int64 v200; // rdi
  __int64 *v201; // r12
  __int64 *v202; // r15
  __int64 v203; // rsi
  __int64 v204; // rdi
  unsigned __int64 v205; // rdi
  __int64 v206; // rax
  __int64 v207; // r12
  __int64 v208; // r15
  _QWORD *v209; // rdi
  __int64 **v210; // [rsp+8h] [rbp-438h]
  __int64 **v211; // [rsp+8h] [rbp-438h]
  unsigned __int64 v212; // [rsp+10h] [rbp-430h]
  unsigned __int64 v213; // [rsp+10h] [rbp-430h]
  unsigned __int64 v214; // [rsp+18h] [rbp-428h]
  unsigned __int64 v215; // [rsp+18h] [rbp-428h]
  _QWORD *v216; // [rsp+20h] [rbp-420h]
  __int64 *v217; // [rsp+28h] [rbp-418h]
  __int64 **v218; // [rsp+38h] [rbp-408h]
  __int64 **v219; // [rsp+38h] [rbp-408h]
  unsigned __int64 v220; // [rsp+50h] [rbp-3F0h]
  unsigned __int64 v221; // [rsp+50h] [rbp-3F0h]
  __int64 v223; // [rsp+68h] [rbp-3D8h] BYREF
  __int64 v224; // [rsp+70h] [rbp-3D0h] BYREF
  __int64 v225; // [rsp+78h] [rbp-3C8h] BYREF
  __int64 v226; // [rsp+80h] [rbp-3C0h] BYREF
  __int64 v227; // [rsp+88h] [rbp-3B8h] BYREF
  __int64 v228; // [rsp+90h] [rbp-3B0h] BYREF
  unsigned __int64 v229; // [rsp+98h] [rbp-3A8h]
  unsigned __int64 v230; // [rsp+A0h] [rbp-3A0h] BYREF
  __int64 v231; // [rsp+A8h] [rbp-398h] BYREF
  __int64 v232; // [rsp+B0h] [rbp-390h] BYREF
  __int64 (__fastcall **v233)(); // [rsp+B8h] [rbp-388h]
  __m128i *v234; // [rsp+C0h] [rbp-380h] BYREF
  __int64 v235; // [rsp+C8h] [rbp-378h]
  __m128i v236; // [rsp+D0h] [rbp-370h] BYREF
  _DWORD v237[4]; // [rsp+E0h] [rbp-360h] BYREF
  __int64 (__fastcall *v238)(_QWORD *, _DWORD *, int); // [rsp+F0h] [rbp-350h]
  __int64 (__fastcall *v239)(unsigned int *); // [rsp+F8h] [rbp-348h]
  __m128i *v240; // [rsp+100h] [rbp-340h] BYREF
  __int64 v241; // [rsp+108h] [rbp-338h]
  __m128i v242; // [rsp+110h] [rbp-330h] BYREF
  _DWORD v243[4]; // [rsp+120h] [rbp-320h] BYREF
  __int64 (__fastcall *v244)(_QWORD *, _DWORD *, int); // [rsp+130h] [rbp-310h]
  __int64 (__fastcall *v245)(unsigned int *); // [rsp+138h] [rbp-308h]
  __m128i v246; // [rsp+160h] [rbp-2E0h] BYREF
  __m128i v247; // [rsp+170h] [rbp-2D0h] BYREF
  __int64 *v248; // [rsp+180h] [rbp-2C0h]
  unsigned __int64 v249; // [rsp+188h] [rbp-2B8h]
  unsigned __int64 v250; // [rsp+190h] [rbp-2B0h]
  __int64 v251; // [rsp+198h] [rbp-2A8h]
  __int64 v252; // [rsp+1A0h] [rbp-2A0h]
  __int64 v253; // [rsp+1A8h] [rbp-298h]
  __int64 v254; // [rsp+1B0h] [rbp-290h]
  __int64 v255; // [rsp+1B8h] [rbp-288h]
  _QWORD *v256; // [rsp+1C0h] [rbp-280h]
  char v257; // [rsp+1C8h] [rbp-278h]
  __int64 (__fastcall *v258)(__int64 *, __int64); // [rsp+1D0h] [rbp-270h]
  __int64 *v259; // [rsp+1D8h] [rbp-268h]
  __int64 v260; // [rsp+1E0h] [rbp-260h]
  __int64 v261; // [rsp+1E8h] [rbp-258h]
  __int64 v262; // [rsp+1F0h] [rbp-250h]
  int v263; // [rsp+1F8h] [rbp-248h]
  __int64 *v264; // [rsp+200h] [rbp-240h]
  __int64 v265; // [rsp+208h] [rbp-238h]
  __int64 v266; // [rsp+210h] [rbp-230h] BYREF
  _BYTE *v267; // [rsp+218h] [rbp-228h]
  __int64 v268; // [rsp+220h] [rbp-220h]
  int v269; // [rsp+228h] [rbp-218h]
  char v270; // [rsp+22Ch] [rbp-214h]
  _BYTE v271[64]; // [rsp+230h] [rbp-210h] BYREF
  _BYTE *v272; // [rsp+270h] [rbp-1D0h]
  __int64 v273; // [rsp+278h] [rbp-1C8h]
  _BYTE v274[72]; // [rsp+280h] [rbp-1C0h] BYREF
  int v275; // [rsp+2C8h] [rbp-178h] BYREF
  __int64 v276; // [rsp+2D0h] [rbp-170h]
  int *v277; // [rsp+2D8h] [rbp-168h]
  int *v278; // [rsp+2E0h] [rbp-160h]
  __int64 v279; // [rsp+2E8h] [rbp-158h]
  unsigned __int64 v280; // [rsp+2F0h] [rbp-150h] BYREF
  __int64 v281; // [rsp+2F8h] [rbp-148h]
  __int64 v282; // [rsp+300h] [rbp-140h]

  v5 = (__int64 **)a3;
  v6 = sub_BC0510(a4, &unk_4F82418, a3);
  v7 = *((_BYTE *)a2 + 16) == 0;
  v223 = *(_QWORD *)(v6 + 8);
  v224 = v223;
  v225 = v223;
  if ( v7 )
  {
    v45 = *a2;
    v46 = a2[1];
    v246.m128i_i64[0] = (__int64)v5;
    v246.m128i_i64[1] = (__int64)sub_26F60C0;
    v47 = *v5;
    v249 = v45;
    v247.m128i_i64[0] = (__int64)&v223;
    v247.m128i_i64[1] = (__int64)sub_26F6100;
    v250 = v46;
    v248 = &v225;
    v48 = sub_BCB2B0(v47);
    v49 = *v5;
    v251 = v48;
    v50 = sub_BCE3C0(v49, 0);
    v51 = *v5;
    v252 = v50;
    v52 = sub_BCB2D0(v51);
    v53 = *v5;
    v253 = v52;
    v54 = sub_BCB2E0(v53);
    v55 = (__int64)*v5;
    v254 = v54;
    v56 = sub_AE4420((__int64)(v5 + 39), v55, 0);
    v57 = *v5;
    v255 = v56;
    v58 = (__int64 *)sub_BCB2B0(v57);
    v256 = sub_BCD420(v58, 0);
    v59 = sub_26F7D90(v246.m128i_i64[0]);
    v260 = 0;
    v257 = v59;
    v258 = sub_26F60E0;
    v261 = 0;
    v259 = &v224;
    v264 = &v266;
    v267 = v271;
    v272 = v274;
    v273 = 0x800000000LL;
    v277 = &v275;
    v278 = &v275;
    v262 = 0;
    v263 = 0;
    v265 = 0;
    v266 = 0;
    v268 = 8;
    v269 = 0;
    v270 = 1;
    v275 = 0;
    v276 = 0;
    v279 = 0;
    v280 = 0;
    v281 = 0;
    v282 = 0;
    sub_2707FF0(&v280);
    if ( v249 && *(_BYTE *)(v249 + 346) || v250 && *(_BYTE *)(v250 + 346) )
    {
      sub_26F71B0((__int64)&v246);
      v61 = a1 + 32;
      v62 = a1 + 80;
    }
    else
    {
      v60 = sub_2703170((__int64)&v246, a5);
      sub_26F71B0((__int64)&v246);
      v61 = a1 + 32;
      v62 = a1 + 80;
      if ( v60 )
        goto LABEL_45;
    }
    v82 = a1;
    v81 = a1;
    goto LABEL_71;
  }
  v8 = sub_22077B0(0x248u);
  v9 = v8;
  if ( v8 )
  {
    *(_DWORD *)(v8 + 8) = 0;
    v10 = v8 + 8;
    *(_QWORD *)(v10 + 8) = 0;
    *(_QWORD *)(v9 + 24) = v10;
    *(_QWORD *)(v9 + 32) = v10;
    *(_QWORD *)(v9 + 136) = v9 + 152;
    *(_QWORD *)(v9 + 64) = 0x2000000000LL;
    *(_QWORD *)(v9 + 168) = v9 + 72;
    *(_QWORD *)(v9 + 88) = v9 + 104;
    *(_QWORD *)(v9 + 232) = v9 + 216;
    *(_QWORD *)(v9 + 240) = v9 + 216;
    *(_QWORD *)(v9 + 96) = 0x400000000LL;
    *(_QWORD *)(v9 + 280) = v9 + 264;
    *(_QWORD *)(v9 + 288) = v9 + 264;
    *(_QWORD *)(v9 + 40) = 0;
    *(_QWORD *)(v9 + 48) = 0;
    *(_QWORD *)(v9 + 56) = 0;
    *(_QWORD *)(v9 + 72) = 0;
    *(_QWORD *)(v9 + 80) = 0;
    *(_QWORD *)(v9 + 144) = 0;
    *(_QWORD *)(v9 + 152) = 0;
    *(_QWORD *)(v9 + 160) = 1;
    *(_QWORD *)(v9 + 176) = 0;
    *(_QWORD *)(v9 + 184) = 0;
    *(_QWORD *)(v9 + 192) = 0;
    *(_DWORD *)(v9 + 200) = 0;
    *(_DWORD *)(v9 + 216) = 0;
    *(_QWORD *)(v9 + 224) = 0;
    *(_QWORD *)(v9 + 248) = 0;
    *(_DWORD *)(v9 + 264) = 0;
    *(_QWORD *)(v9 + 272) = 0;
    *(_QWORD *)(v9 + 296) = 0;
    *(_QWORD *)(v9 + 304) = 0;
    *(_QWORD *)(v9 + 312) = 0;
    *(_QWORD *)(v9 + 440) = 0x400000000LL;
    *(_QWORD *)(v9 + 480) = v9 + 496;
    *(_QWORD *)(v9 + 320) = 0;
    *(_DWORD *)(v9 + 328) = 0;
    *(_QWORD *)(v9 + 336) = 0;
    *(_DWORD *)(v9 + 344) = 0;
    *(_QWORD *)(v9 + 352) = 0;
    *(_QWORD *)(v9 + 360) = 0;
    *(_QWORD *)(v9 + 368) = 0;
    *(_DWORD *)(v9 + 376) = 0;
    *(_QWORD *)(v9 + 384) = 0;
    *(_QWORD *)(v9 + 392) = 0;
    *(_QWORD *)(v9 + 400) = 0;
    *(_DWORD *)(v9 + 408) = 0;
    *(_QWORD *)(v9 + 416) = 0;
    *(_QWORD *)(v9 + 424) = 0;
    *(_QWORD *)(v9 + 432) = v9 + 448;
    *(_QWORD *)(v9 + 488) = 0;
    *(_QWORD *)(v9 + 496) = 0;
    *(_QWORD *)(v9 + 504) = 1;
    *(_QWORD *)(v9 + 512) = v9 + 416;
    *(_QWORD *)(v9 + 520) = 0;
    *(_QWORD *)(v9 + 528) = 0;
    *(_QWORD *)(v9 + 536) = 0;
    *(_QWORD *)(v9 + 544) = 0;
    *(_QWORD *)(v9 + 552) = 0;
    *(_QWORD *)(v9 + 560) = 0;
    *(_QWORD *)(v9 + 568) = 0;
    *(_DWORD *)(v9 + 576) = 0;
  }
  if ( qword_4FF9750 )
  {
    sub_8FD6D0((__int64)&v234, "-wholeprogramdevirt-read-summary: ", &qword_4FF9748);
    if ( v235 == 0x3FFFFFFFFFFFFFFFLL || v235 == 4611686018427387902LL )
      goto LABEL_291;
    v83 = (__m128i *)sub_2241490((unsigned __int64 *)&v234, ": ", 2u);
    v246.m128i_i64[0] = (__int64)&v247;
    if ( (__m128i *)v83->m128i_i64[0] == &v83[1] )
    {
      a5 = _mm_loadu_si128(v83 + 1);
      v247 = a5;
    }
    else
    {
      v246.m128i_i64[0] = v83->m128i_i64[0];
      v247.m128i_i64[0] = v83[1].m128i_i64[0];
    }
    v246.m128i_i64[1] = v83->m128i_i64[1];
    v83->m128i_i64[0] = (__int64)v83[1].m128i_i64;
    v83->m128i_i64[1] = 0;
    v83[1].m128i_i8[0] = 0;
    v240 = &v242;
    if ( (__m128i *)v246.m128i_i64[0] == &v247 )
    {
      v242 = _mm_load_si128(&v247);
    }
    else
    {
      v240 = (__m128i *)v246.m128i_i64[0];
      v242.m128i_i64[0] = v247.m128i_i64[0];
    }
    v243[0] = 1;
    v241 = v246.m128i_i64[1];
    v245 = sub_226E290;
    v244 = sub_226EF00;
    if ( v234 != &v236 )
      j_j___libc_free_0((unsigned __int64)v234);
    LOWORD(v248) = 260;
    v84 = &v246;
    v246.m128i_i64[0] = (__int64)&qword_4FF9748;
    sub_C7EA90((__int64)&v234, v246.m128i_i64, 0, 1u, 0, 0);
    v88 = v236.m128i_i8[0] & 1;
    if ( (v236.m128i_i8[0] & 1) != 0 && (v84 = (__m128i *)(unsigned int)v234, v87 = v235, (_DWORD)v234) )
    {
      sub_C63CA0(&v232, (int)v234, v235);
      v89 = v232;
      v90 = v232 | 1;
      v232 |= 1uLL;
      if ( (v89 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_26F95C0((__int64)&v240, &v232, v90);
      v216 = 0;
      v88 = v236.m128i_i8[0] & 1;
    }
    else
    {
      v90 = (unsigned __int64)v234;
      v234 = 0;
      v216 = (_QWORD *)v90;
    }
    if ( !v88 && v234 )
      (*(void (__fastcall **)(__m128i *, __m128i *, unsigned __int64, __int64, __int64, __int64))(v234->m128i_i64[0] + 8))(
        v234,
        v84,
        v90,
        v85,
        v86,
        v87);
    v104 = (__int64)v216;
    sub_C7EC60(&v246, v216);
    sub_9F1E00(
      (__int64)&v234,
      (__int64)v216,
      v105,
      v106,
      v107,
      v108,
      a5,
      (const __m128i *)v246.m128i_i64[0],
      v246.m128i_u64[1]);
    v109 = v235 & 1;
    v110 = (2 * (v235 & 1)) | v235 & 0xFD;
    LOBYTE(v235) = v110;
    if ( v109 )
    {
      v226 = 0;
      LOBYTE(v235) = v110 & 0xFD;
      v176 = (unsigned __int64)v234;
      v227 = 0;
      v234 = 0;
      v177 = v176 & 0xFFFFFFFFFFFFFFFELL;
      v178 = (_QWORD *)v177;
      if ( v177 )
      {
        v228 = 0;
        v104 = (__int64)&unk_4F84052;
        if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v177 + 48LL))(v177, &unk_4F84052) )
        {
          v186 = (__int64 *)v178[2];
          v187 = (__int64 *)v178[1];
          v229 = 1;
          v217 = v186;
          if ( v187 != v186 )
          {
            v215 = v9;
            v188 = v187;
            do
            {
              v232 = *v188;
              *v188 = 0;
              sub_26F9660(&v231, &v232);
              v104 = (__int64)&v246;
              v246.m128i_i64[0] = v229 | 1;
              sub_9CDB40(&v230, (unsigned __int64 *)&v246, (unsigned __int64 *)&v231);
              v229 = v230 | 1;
              if ( (v246.m128i_i8[0] & 1) != 0 || (v246.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
                sub_C63C30(&v246, (__int64)&v246);
              if ( (v231 & 1) != 0 || (v231 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                sub_C63C30(&v231, (__int64)&v246);
              if ( v232 )
                (*(void (__fastcall **)(__int64))(*(_QWORD *)v232 + 8LL))(v232);
              ++v188;
            }
            while ( v217 != v188 );
            v9 = v215;
          }
          v232 = v229 | 1;
          (*(void (__fastcall **)(_QWORD *))(*v178 + 8LL))(v178);
        }
        else
        {
          v104 = (__int64)&v246;
          v246.m128i_i64[0] = (__int64)v178;
          sub_26F9660(&v232, &v246);
          if ( v246.m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v246.m128i_i64[0] + 8LL))(v246.m128i_i64[0]);
        }
        if ( (v232 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          BUG();
        if ( (v228 & 1) != 0 || (v228 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_C63C30(&v228, v104);
      }
      if ( (v227 & 1) != 0 || (v227 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v227, v104);
      if ( (v226 & 1) != 0 || (v226 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v226, v104);
      v179 = v216[1];
      sub_CB0A90((__int64)&v246, v179, v216[2] - v179, 0, 0, 0);
      sub_CB4D10((__int64)&v246, v179);
      sub_CB0300((__int64)&v246);
      sub_2633C40((__int64)&v246, v9);
      sub_CB1A30((__int64)&v246);
      v180 = sub_CB0000((__int64)&v246);
      v104 = v180;
      sub_C63CA0(&v231, v180, v181);
      v182 = v231;
      v231 = 0;
      v232 = v182 | 1;
      if ( (v182 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_26F95C0((__int64)&v240, &v232, v182 | 1);
      sub_CB34B0((__int64)&v246, v104);
      v214 = v9;
    }
    else
    {
      v111 = (unsigned __int64)v234;
      v234 = 0;
      v214 = v111;
      if ( v9 )
      {
        sub_C7D6A0(*(_QWORD *)(v9 + 560), 16LL * *(unsigned int *)(v9 + 576), 8);
        v112 = *(_QWORD *)(v9 + 528);
        if ( v112 )
          j_j___libc_free_0(v112);
        v113 = *(__int64 **)(v9 + 432);
        v114 = &v113[*(unsigned int *)(v9 + 440)];
        if ( v113 != v114 )
        {
          for ( i = *(_QWORD *)(v9 + 432); ; i = *(_QWORD *)(v9 + 432) )
          {
            v116 = *v113;
            v117 = (unsigned int)(((__int64)v113 - i) >> 3) >> 7;
            v118 = 4096LL << v117;
            if ( v117 >= 0x1E )
              v118 = 0x40000000000LL;
            ++v113;
            sub_C7D6A0(v116, v118, 16);
            if ( v114 == v113 )
              break;
          }
        }
        v119 = *(__int64 **)(v9 + 480);
        v120 = (unsigned __int64)&v119[2 * *(unsigned int *)(v9 + 488)];
        if ( v119 != (__int64 *)v120 )
        {
          do
          {
            v121 = v119[1];
            v122 = *v119;
            v119 += 2;
            sub_C7D6A0(v122, v121, 16);
          }
          while ( (__int64 *)v120 != v119 );
          v120 = *(_QWORD *)(v9 + 480);
        }
        if ( v120 != v9 + 496 )
          _libc_free(v120);
        v123 = *(_QWORD *)(v9 + 432);
        if ( v123 != v9 + 448 )
          _libc_free(v123);
        v124 = *(unsigned int *)(v9 + 408);
        if ( (_DWORD)v124 )
        {
          v125 = *(_QWORD **)(v9 + 392);
          v212 = v9;
          v210 = v5;
          v126 = &v125[7 * v124];
          do
          {
            if ( *v125 <= 0xFFFFFFFFFFFFFFFDLL )
            {
              v127 = v125[3];
              while ( v127 )
              {
                v128 = v127;
                sub_26F87B0(*(_QWORD **)(v127 + 24));
                v129 = *(_QWORD *)(v127 + 32);
                v127 = *(_QWORD *)(v127 + 16);
                if ( v129 != v128 + 48 )
                  j_j___libc_free_0(v129);
                j_j___libc_free_0(v128);
              }
            }
            v125 += 7;
          }
          while ( v126 != v125 );
          v9 = v212;
          v5 = v210;
          v124 = *(unsigned int *)(v212 + 408);
        }
        sub_C7D6A0(*(_QWORD *)(v9 + 392), 56 * v124, 8);
        v130 = *(unsigned int *)(v9 + 376);
        if ( (_DWORD)v130 )
        {
          v171 = *(_QWORD **)(v9 + 360);
          v213 = v9;
          v211 = v5;
          v172 = &v171[7 * v130];
          do
          {
            if ( *v171 <= 0xFFFFFFFFFFFFFFFDLL )
            {
              v173 = v171[3];
              while ( v173 )
              {
                v174 = v173;
                sub_26F87B0(*(_QWORD **)(v173 + 24));
                v175 = *(_QWORD *)(v173 + 32);
                v173 = *(_QWORD *)(v173 + 16);
                if ( v175 != v174 + 48 )
                  j_j___libc_free_0(v175);
                j_j___libc_free_0(v174);
              }
            }
            v171 += 7;
          }
          while ( v172 != v171 );
          v9 = v213;
          v5 = v211;
          v130 = *(unsigned int *)(v213 + 376);
        }
        sub_C7D6A0(*(_QWORD *)(v9 + 360), 56 * v130, 8);
        sub_C7D6A0(*(_QWORD *)(v9 + 312), 16LL * *(unsigned int *)(v9 + 328), 8);
        sub_26F7F10(*(_QWORD **)(v9 + 272));
        sub_26F8560(*(_QWORD *)(v9 + 224));
        sub_C7D6A0(*(_QWORD *)(v9 + 184), 16LL * *(unsigned int *)(v9 + 200), 8);
        v131 = *(__int64 **)(v9 + 88);
        v132 = &v131[*(unsigned int *)(v9 + 96)];
        if ( v131 != v132 )
        {
          for ( j = *(_QWORD *)(v9 + 88); ; j = *(_QWORD *)(v9 + 88) )
          {
            v134 = *v131;
            v135 = (unsigned int)(((__int64)v131 - j) >> 3) >> 7;
            v136 = 4096LL << v135;
            if ( v135 >= 0x1E )
              v136 = 0x40000000000LL;
            ++v131;
            sub_C7D6A0(v134, v136, 16);
            if ( v132 == v131 )
              break;
          }
        }
        v137 = *(__int64 **)(v9 + 136);
        v138 = (unsigned __int64)&v137[2 * *(unsigned int *)(v9 + 144)];
        if ( v137 != (__int64 *)v138 )
        {
          do
          {
            v139 = v137[1];
            v140 = *v137;
            v137 += 2;
            sub_C7D6A0(v140, v139, 16);
          }
          while ( (__int64 *)v138 != v137 );
          v138 = *(_QWORD *)(v9 + 136);
        }
        if ( v138 != v9 + 152 )
          _libc_free(v138);
        v141 = *(_QWORD *)(v9 + 88);
        if ( v141 != v9 + 104 )
          _libc_free(v141);
        v142 = *(_QWORD *)(v9 + 48);
        if ( *(_DWORD *)(v9 + 60) )
        {
          v143 = *(unsigned int *)(v9 + 56);
          if ( (_DWORD)v143 )
          {
            v144 = 8 * v143;
            v145 = 0;
            do
            {
              v146 = *(_QWORD **)(v142 + v145);
              if ( v146 && v146 != (_QWORD *)-8LL )
              {
                sub_C7D6A0((__int64)v146, *v146 + 33LL, 8);
                v142 = *(_QWORD *)(v9 + 48);
              }
              v145 += 8;
            }
            while ( v145 != v144 );
          }
        }
        _libc_free(v142);
        sub_26F6A90(*(_QWORD **)(v9 + 16));
        v104 = 584;
        j_j___libc_free_0(v9);
      }
      if ( dword_4FF9848 != 1 )
      {
        v147 = *(_QWORD *)(v214 + 48) + 8LL * *(unsigned int *)(v214 + 56);
        v148 = sub_C92610();
        v104 = (__int64)"[Regular LTO]";
        v149 = sub_C92860((__int64 *)(v214 + 48), "[Regular LTO]", 0xDu, v148);
        v150 = v149 == -1
             ? *(_QWORD *)(v214 + 48) + 8LL * *(unsigned int *)(v214 + 56)
             : *(_QWORD *)(v214 + 48) + 8LL * v149;
        if ( v147 == v150 )
        {
          v184 = sub_2241E50();
          v246.m128i_i64[0] = (__int64)&v247;
          sub_26F6410(v246.m128i_i64, "combined summary should contain Regular LTO module", (__int64)"");
          v104 = (__int64)&v246;
          sub_C63F00(&v232, (__int64)&v246, 0x16u, (__int64)v184);
          if ( (__m128i *)v246.m128i_i64[0] != &v247 )
          {
            v104 = v247.m128i_i64[0] + 1;
            j_j___libc_free_0(v246.m128i_u64[0]);
          }
          v185 = v232;
          v232 = 0;
          v246.m128i_i64[0] = v185 | 1;
          if ( (v185 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_26F95C0((__int64)&v240, v246.m128i_i64, v185 | 1);
        }
      }
    }
    if ( (v235 & 2) != 0 )
      sub_25CE240(&v234, v104);
    v151 = (unsigned __int64)v234;
    if ( (v235 & 1) != 0 )
    {
      if ( v234 )
        (*(void (__fastcall **)(__m128i *, __int64))(v234->m128i_i64[0] + 8))(v234, v104);
    }
    else if ( v234 )
    {
      sub_C7D6A0(v234[35].m128i_i64[0], 16LL * v234[36].m128i_u32[0], 8);
      v152 = *(_QWORD *)(v151 + 528);
      if ( v152 )
        j_j___libc_free_0(v152);
      v153 = *(__int64 **)(v151 + 432);
      v154 = &v153[*(unsigned int *)(v151 + 440)];
      while ( v154 != v153 )
      {
        v155 = *v153;
        v156 = (unsigned int)(((__int64)v153 - *(_QWORD *)(v151 + 432)) >> 3) >> 7;
        v157 = 4096LL << v156;
        if ( v156 >= 0x1E )
          v157 = 0x40000000000LL;
        ++v153;
        sub_C7D6A0(v155, v157, 16);
      }
      v158 = *(_QWORD *)(v151 + 480);
      v159 = (__int64 *)v158;
      v160 = (__int64 *)(v158 + 16LL * *(unsigned int *)(v151 + 488));
      if ( (__int64 *)v158 != v160 )
      {
        do
        {
          v161 = v159[1];
          v162 = *v159;
          v159 += 2;
          sub_C7D6A0(v162, v161, 16);
        }
        while ( v160 != v159 );
        v158 = *(_QWORD *)(v151 + 480);
      }
      if ( v158 != v151 + 496 )
        _libc_free(v158);
      v163 = *(_QWORD *)(v151 + 432);
      if ( v163 != v151 + 448 )
        _libc_free(v163);
      v164 = *(unsigned int *)(v151 + 408);
      if ( (_DWORD)v164 )
      {
        v165 = *(_QWORD **)(v151 + 392);
        v218 = v5;
        v166 = &v165[7 * v164];
        do
        {
          if ( *v165 <= 0xFFFFFFFFFFFFFFFDLL )
          {
            v167 = v165[3];
            while ( v167 )
            {
              v168 = v167;
              sub_26F87B0(*(_QWORD **)(v167 + 24));
              v169 = *(_QWORD *)(v167 + 32);
              v167 = *(_QWORD *)(v167 + 16);
              if ( v169 != v168 + 48 )
                j_j___libc_free_0(v169);
              j_j___libc_free_0(v168);
            }
          }
          v165 += 7;
        }
        while ( v166 != v165 );
        v5 = v218;
      }
      sub_C7D6A0(*(_QWORD *)(v151 + 392), 56LL * *(unsigned int *)(v151 + 408), 8);
      v189 = *(unsigned int *)(v151 + 376);
      if ( (_DWORD)v189 )
      {
        v190 = *(_QWORD **)(v151 + 360);
        v219 = v5;
        v191 = &v190[7 * v189];
        do
        {
          if ( *v190 <= 0xFFFFFFFFFFFFFFFDLL )
          {
            v192 = v190[3];
            while ( v192 )
            {
              v193 = v192;
              sub_26F87B0(*(_QWORD **)(v192 + 24));
              v194 = *(_QWORD *)(v192 + 32);
              v192 = *(_QWORD *)(v192 + 16);
              if ( v194 != v193 + 48 )
                j_j___libc_free_0(v194);
              j_j___libc_free_0(v193);
            }
          }
          v190 += 7;
        }
        while ( v191 != v190 );
        v5 = v219;
      }
      sub_C7D6A0(*(_QWORD *)(v151 + 360), 56LL * *(unsigned int *)(v151 + 376), 8);
      sub_C7D6A0(*(_QWORD *)(v151 + 312), 16LL * *(unsigned int *)(v151 + 328), 8);
      sub_26F7F10(*(_QWORD **)(v151 + 272));
      sub_26F8560(*(_QWORD *)(v151 + 224));
      sub_C7D6A0(*(_QWORD *)(v151 + 184), 16LL * *(unsigned int *)(v151 + 200), 8);
      v195 = *(__int64 **)(v151 + 88);
      v196 = &v195[*(unsigned int *)(v151 + 96)];
      while ( v196 != v195 )
      {
        v197 = *v195;
        v198 = (unsigned int)(((__int64)v195 - *(_QWORD *)(v151 + 88)) >> 3) >> 7;
        v199 = 4096LL << v198;
        if ( v198 >= 0x1E )
          v199 = 0x40000000000LL;
        ++v195;
        sub_C7D6A0(v197, v199, 16);
      }
      v200 = *(_QWORD *)(v151 + 136);
      v201 = (__int64 *)v200;
      v202 = (__int64 *)(v200 + 16LL * *(unsigned int *)(v151 + 144));
      if ( (__int64 *)v200 != v202 )
      {
        do
        {
          v203 = v201[1];
          v204 = *v201;
          v201 += 2;
          sub_C7D6A0(v204, v203, 16);
        }
        while ( v202 != v201 );
        v200 = *(_QWORD *)(v151 + 136);
      }
      if ( v200 != v151 + 152 )
        _libc_free(v200);
      v205 = *(_QWORD *)(v151 + 88);
      if ( v205 != v151 + 104 )
        _libc_free(v205);
      if ( *(_DWORD *)(v151 + 60) )
      {
        v206 = *(unsigned int *)(v151 + 56);
        v183 = *(_QWORD *)(v151 + 48);
        if ( (_DWORD)v206 )
        {
          v207 = 8 * v206;
          v208 = 0;
          do
          {
            v209 = *(_QWORD **)(v183 + v208);
            if ( v209 && v209 != (_QWORD *)-8LL )
            {
              sub_C7D6A0((__int64)v209, *v209 + 33LL, 8);
              v183 = *(_QWORD *)(v151 + 48);
            }
            v208 += 8;
          }
          while ( v208 != v207 );
        }
      }
      else
      {
        v183 = *(_QWORD *)(v151 + 48);
      }
      _libc_free(v183);
      sub_26F6A90(*(_QWORD **)(v151 + 16));
      v104 = 584;
      j_j___libc_free_0(v151);
    }
    if ( v216 )
      (*(void (__fastcall **)(_QWORD *, __int64))(*v216 + 8LL))(v216, v104);
    if ( v244 )
      v244(v243, v243, 3);
    if ( v240 != &v242 )
      j_j___libc_free_0((unsigned __int64)v240);
    v9 = v214;
  }
  if ( dword_4FF9848 == 1 )
  {
    v12 = v9;
    v11 = 0;
  }
  else
  {
    v11 = 0;
    if ( dword_4FF9848 == 2 )
      v11 = v9;
    v12 = 0;
  }
  v13 = *v5;
  v250 = v12;
  v246.m128i_i64[1] = (__int64)sub_26F60C0;
  v246.m128i_i64[0] = (__int64)v5;
  v247.m128i_i64[0] = (__int64)&v223;
  v247.m128i_i64[1] = (__int64)sub_26F6100;
  v249 = v11;
  v248 = &v225;
  v14 = sub_BCB2B0(v13);
  v15 = *v5;
  v251 = v14;
  v16 = sub_BCE3C0(v15, 0);
  v17 = *v5;
  v252 = v16;
  v18 = sub_BCB2D0(v17);
  v19 = *v5;
  v253 = v18;
  v20 = sub_BCB2E0(v19);
  v21 = (__int64)*v5;
  v254 = v20;
  v22 = sub_AE4420((__int64)(v5 + 39), v21, 0);
  v23 = *v5;
  v255 = v22;
  v24 = (__int64 *)sub_BCB2B0(v23);
  v256 = sub_BCD420(v24, 0);
  v25 = sub_26F7D90(v246.m128i_i64[0]);
  v260 = 0;
  v257 = v25;
  v258 = sub_26F60E0;
  v261 = 0;
  v259 = &v224;
  v264 = &v266;
  v267 = v271;
  v272 = v274;
  v273 = 0x800000000LL;
  v277 = &v275;
  v278 = &v275;
  v262 = 0;
  v263 = 0;
  v265 = 0;
  v266 = 0;
  v268 = 8;
  v269 = 0;
  v270 = 1;
  v275 = 0;
  v276 = 0;
  v279 = 0;
  v280 = 0;
  v281 = 0;
  v282 = 0;
  sub_2707FF0(&v280);
  if ( !v249 || (v26 = 0, !*(_BYTE *)(v249 + 346)) )
  {
    if ( !v250 || (v26 = 0, !*(_BYTE *)(v250 + 346)) )
      v26 = sub_2703170((__int64)&v246, a5);
  }
  sub_26F71B0((__int64)&v246);
  if ( qword_4FF9650 )
  {
    sub_8FD6D0((__int64)&v246, "-wholeprogramdevirt-write-summary: ", &qword_4FF9648);
    if ( v246.m128i_i64[1] != 0x3FFFFFFFFFFFFFFFLL && v246.m128i_i64[1] != 4611686018427387902LL )
    {
      v96 = (__m128i *)sub_2241490((unsigned __int64 *)&v246, ": ", 2u);
      v240 = &v242;
      if ( (__m128i *)v96->m128i_i64[0] == &v96[1] )
      {
        v242 = _mm_loadu_si128(v96 + 1);
      }
      else
      {
        v240 = (__m128i *)v96->m128i_i64[0];
        v242.m128i_i64[0] = v96[1].m128i_i64[0];
      }
      v241 = v96->m128i_i64[1];
      v96->m128i_i64[0] = (__int64)v96[1].m128i_i64;
      v96->m128i_i64[1] = 0;
      v96[1].m128i_i8[0] = 0;
      v234 = &v236;
      if ( v240 == &v242 )
      {
        v236 = _mm_load_si128(&v242);
      }
      else
      {
        v234 = v240;
        v236.m128i_i64[0] = v242.m128i_i64[0];
      }
      v237[0] = 1;
      v235 = v241;
      v239 = sub_226E290;
      v238 = sub_226EF00;
      if ( (__m128i *)v246.m128i_i64[0] != &v247 )
        j_j___libc_free_0(v246.m128i_u64[0]);
      LODWORD(v232) = 0;
      v233 = sub_2241E40();
      if ( (unsigned __int64)qword_4FF9650 > 2
        && (v97 = qword_4FF9648 + qword_4FF9650 - 3, *(_WORD *)v97 == 25134)
        && *(_BYTE *)(v97 + 2) == 99 )
      {
        sub_CB7060((__int64)&v246, (_BYTE *)qword_4FF9648, qword_4FF9650, (__int64)&v232, 0);
        sub_C63CA0(&v231, v232, (__int64)v233);
        v170 = v231;
        v231 = 0;
        v240 = (__m128i *)(v170 | 1);
        if ( (v170 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_26F95C0((__int64)&v234, (__int64 *)&v240, v170 | 1);
        sub_A3C720(v9, (__int64)&v246, 0, 0);
        sub_CB5B00(v246.m128i_i32, (__int64)&v246);
      }
      else
      {
        sub_CB7060((__int64)&v240, (_BYTE *)qword_4FF9648, qword_4FF9650, (__int64)&v232, 3u);
        sub_C63CA0(&v231, v232, (__int64)v233);
        v98 = v231;
        v231 = 0;
        v246.m128i_i64[0] = v98 | 1;
        if ( (v98 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_26F95C0((__int64)&v234, v246.m128i_i64, v98 | 1);
        sub_CB1A80((__int64)&v246, (__int64)&v240, 0, 70);
        sub_CB2850((__int64)&v246);
        v99 = 0;
        if ( (unsigned __int8)sub_CB2870((__int64)&v246, 0) )
        {
          sub_CB05C0(&v246, 0, v100, v101, v102, v103);
          v99 = v9;
          sub_2633C40((__int64)&v246, v9);
          sub_CB2220(&v246);
          nullsub_173();
        }
        sub_CB1B70((__int64)&v246);
        sub_CB0A00(&v246, v99);
        sub_CB5B00((int *)&v240, v99);
      }
      if ( v238 )
        v238(v237, v237, 3);
      if ( v234 != &v236 )
        j_j___libc_free_0((unsigned __int64)v234);
      goto LABEL_15;
    }
LABEL_291:
    sub_4262D8((__int64)"basic_string::append");
  }
LABEL_15:
  if ( v9 )
  {
    sub_C7D6A0(*(_QWORD *)(v9 + 560), 16LL * *(unsigned int *)(v9 + 576), 8);
    v27 = *(_QWORD *)(v9 + 528);
    if ( v27 )
      j_j___libc_free_0(v27);
    v28 = *(__int64 **)(v9 + 432);
    v29 = &v28[*(unsigned int *)(v9 + 440)];
    if ( v28 != v29 )
    {
      for ( k = *(_QWORD *)(v9 + 432); ; k = *(_QWORD *)(v9 + 432) )
      {
        v31 = *v28;
        v32 = (unsigned int)(((__int64)v28 - k) >> 3) >> 7;
        v33 = 4096LL << v32;
        if ( v32 >= 0x1E )
          v33 = 0x40000000000LL;
        ++v28;
        sub_C7D6A0(v31, v33, 16);
        if ( v29 == v28 )
          break;
      }
    }
    v34 = *(__int64 **)(v9 + 480);
    v35 = (unsigned __int64)&v34[2 * *(unsigned int *)(v9 + 488)];
    if ( v34 != (__int64 *)v35 )
    {
      do
      {
        v36 = v34[1];
        v37 = *v34;
        v34 += 2;
        sub_C7D6A0(v37, v36, 16);
      }
      while ( (__int64 *)v35 != v34 );
      v35 = *(_QWORD *)(v9 + 480);
    }
    if ( v35 != v9 + 496 )
      _libc_free(v35);
    v38 = *(_QWORD *)(v9 + 432);
    if ( v38 != v9 + 448 )
      _libc_free(v38);
    v39 = *(unsigned int *)(v9 + 408);
    if ( (_DWORD)v39 )
    {
      v40 = *(_QWORD **)(v9 + 392);
      v220 = v9;
      v41 = &v40[7 * v39];
      do
      {
        if ( *v40 <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v42 = v40[3];
          while ( v42 )
          {
            v43 = v42;
            sub_26F87B0(*(_QWORD **)(v42 + 24));
            v44 = *(_QWORD *)(v42 + 32);
            v42 = *(_QWORD *)(v42 + 16);
            if ( v44 != v43 + 48 )
              j_j___libc_free_0(v44);
            j_j___libc_free_0(v43);
          }
        }
        v40 += 7;
      }
      while ( v41 != v40 );
      v9 = v220;
      v39 = *(unsigned int *)(v220 + 408);
    }
    sub_C7D6A0(*(_QWORD *)(v9 + 392), 56 * v39, 8);
    v64 = *(unsigned int *)(v9 + 376);
    if ( (_DWORD)v64 )
    {
      v91 = *(_QWORD **)(v9 + 360);
      v221 = v9;
      v92 = &v91[7 * v64];
      do
      {
        if ( *v91 <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v93 = v91[3];
          while ( v93 )
          {
            v94 = v93;
            sub_26F87B0(*(_QWORD **)(v93 + 24));
            v95 = *(_QWORD *)(v93 + 32);
            v93 = *(_QWORD *)(v93 + 16);
            if ( v95 != v94 + 48 )
              j_j___libc_free_0(v95);
            j_j___libc_free_0(v94);
          }
        }
        v91 += 7;
      }
      while ( v92 != v91 );
      v9 = v221;
      v64 = *(unsigned int *)(v221 + 376);
    }
    sub_C7D6A0(*(_QWORD *)(v9 + 360), 56 * v64, 8);
    sub_C7D6A0(*(_QWORD *)(v9 + 312), 16LL * *(unsigned int *)(v9 + 328), 8);
    sub_26F7F10(*(_QWORD **)(v9 + 272));
    sub_26F8560(*(_QWORD *)(v9 + 224));
    sub_C7D6A0(*(_QWORD *)(v9 + 184), 16LL * *(unsigned int *)(v9 + 200), 8);
    v65 = *(__int64 **)(v9 + 88);
    v66 = &v65[*(unsigned int *)(v9 + 96)];
    if ( v65 != v66 )
    {
      for ( m = *(_QWORD *)(v9 + 88); ; m = *(_QWORD *)(v9 + 88) )
      {
        v68 = *v65;
        v69 = (unsigned int)(((__int64)v65 - m) >> 3) >> 7;
        v70 = 4096LL << v69;
        if ( v69 >= 0x1E )
          v70 = 0x40000000000LL;
        ++v65;
        sub_C7D6A0(v68, v70, 16);
        if ( v66 == v65 )
          break;
      }
    }
    v71 = *(__int64 **)(v9 + 136);
    v72 = (unsigned __int64)&v71[2 * *(unsigned int *)(v9 + 144)];
    if ( v71 != (__int64 *)v72 )
    {
      do
      {
        v73 = v71[1];
        v74 = *v71;
        v71 += 2;
        sub_C7D6A0(v74, v73, 16);
      }
      while ( (__int64 *)v72 != v71 );
      v72 = *(_QWORD *)(v9 + 136);
    }
    if ( v72 != v9 + 152 )
      _libc_free(v72);
    v75 = *(_QWORD *)(v9 + 88);
    if ( v75 != v9 + 104 )
      _libc_free(v75);
    v76 = *(_QWORD *)(v9 + 48);
    if ( *(_DWORD *)(v9 + 60) )
    {
      v77 = *(unsigned int *)(v9 + 56);
      if ( (_DWORD)v77 )
      {
        v78 = 8 * v77;
        v79 = 0;
        do
        {
          v80 = *(_QWORD **)(v76 + v79);
          if ( v80 != (_QWORD *)-8LL && v80 )
          {
            sub_C7D6A0((__int64)v80, *v80 + 33LL, 8);
            v76 = *(_QWORD *)(v9 + 48);
          }
          v79 += 8;
        }
        while ( v79 != v78 );
      }
    }
    _libc_free(v76);
    sub_26F6A90(*(_QWORD **)(v9 + 16));
    j_j___libc_free_0(v9);
  }
  v81 = a1;
  v61 = a1 + 32;
  v62 = a1 + 80;
  if ( !v26 )
  {
    v82 = a1;
LABEL_71:
    *(_QWORD *)(v81 + 56) = v62;
    *(_QWORD *)(v81 + 8) = v61;
    *(_QWORD *)(v81 + 48) = 0;
    *(_QWORD *)(v81 + 64) = 2;
    *(_DWORD *)(v81 + 72) = 0;
    *(_BYTE *)(v81 + 76) = 1;
    *(_QWORD *)(v81 + 16) = 0x100000002LL;
    *(_DWORD *)(v82 + 24) = 0;
    *(_BYTE *)(v82 + 28) = 1;
    *(_QWORD *)(v82 + 32) = &qword_4F82400;
    *(_QWORD *)v82 = 1;
    return a1;
  }
LABEL_45:
  memset((void *)a1, 0, 0x60u);
  *(_QWORD *)(a1 + 8) = v61;
  *(_DWORD *)(a1 + 16) = 2;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 56) = v62;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  return a1;
}
