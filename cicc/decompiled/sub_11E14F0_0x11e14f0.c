// Function: sub_11E14F0
// Address: 0x11e14f0
//
__int64 __fastcall sub_11E14F0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // r12
  unsigned __int64 v8; // rax
  __m128i *v9; // rsi
  __int64 v10; // r14
  __int64 v11; // rdi
  __int64 v12; // rax
  unsigned int v13; // edx
  __int64 v14; // rdi
  int v15; // eax
  bool v16; // al
  bool v17; // al
  __int64 v18; // rcx
  size_t v19; // rdx
  _QWORD *v20; // rsi
  _BYTE *v21; // rbx
  _BYTE *v22; // rax
  __int64 v23; // r14
  _BYTE *v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // rdi
  __int64 v27; // r12
  __int64 v28; // rax
  _BYTE *v29; // rax
  _QWORD *v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdi
  __int64 v37; // r14
  __int64 v38; // rax
  char v39; // bl
  _QWORD *v40; // rax
  __int64 v41; // r12
  unsigned int *v42; // r14
  __int64 v43; // rbx
  __int64 v44; // rdx
  unsigned int v45; // esi
  _QWORD *v46; // rdi
  __int64 **v47; // rbx
  __int64 v48; // rdi
  __int64 (__fastcall *v49)(__int64, unsigned int, _BYTE *, __int64); // rax
  _BYTE *v50; // r9
  __int64 v51; // rax
  _QWORD *v52; // rax
  __int64 v53; // r14
  char *v54; // r12
  size_t v55; // r14
  char *v56; // rdx
  char *v57; // rax
  unsigned __int64 i; // rcx
  __int64 v59; // rsi
  unsigned __int8 *v60; // rax
  unsigned __int8 *v61; // r8
  unsigned __int64 v62; // rcx
  __int64 v63; // rdi
  __int64 v64; // rdi
  __int64 (__fastcall *v65)(__int64, unsigned int, _BYTE *, __int64); // rax
  _BYTE *v66; // rbx
  _BYTE *v67; // rax
  __int64 v68; // rax
  unsigned __int8 *v69; // r14
  __int64 v70; // rax
  __int64 v71; // rdi
  unsigned __int8 *v72; // r11
  __int64 (__fastcall *v73)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v74; // rax
  __int64 v75; // r10
  _QWORD *v76; // rdi
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rax
  unsigned __int8 *v80; // rbx
  _BYTE *v81; // rax
  __int64 v82; // rax
  __int64 v83; // rdi
  unsigned __int8 *v84; // r12
  __int64 (__fastcall *v85)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v86; // r14
  unsigned int *v87; // r12
  __int64 v88; // r14
  __int64 v89; // rbx
  __int64 v90; // rdx
  unsigned int v91; // esi
  __int64 v92; // rax
  unsigned int *v93; // r12
  __int64 v94; // rbx
  __int64 v95; // rdx
  unsigned int v96; // esi
  __int64 v97; // r14
  unsigned int *v98; // r12
  __int64 v99; // rdx
  unsigned int v100; // esi
  __int64 v101; // r12
  unsigned int *v102; // rbx
  __int64 v103; // r14
  __int64 v104; // rdx
  unsigned int v105; // esi
  __int64 v106; // rax
  __int64 **v107; // r12
  unsigned int v108; // r14d
  unsigned int v109; // eax
  _QWORD *v110; // rdi
  __int64 v111; // rax
  __int64 v112; // rax
  __int64 v113; // rdi
  unsigned __int8 *v114; // r14
  __int64 (__fastcall *v115)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  unsigned __int8 *v116; // r12
  unsigned int v117; // r14d
  _QWORD *v118; // rdi
  __int64 v119; // rax
  _BYTE *v120; // rax
  __int64 v121; // rax
  _QWORD *v122; // rdi
  __int64 v123; // rax
  __int64 v124; // rax
  __int64 v125; // rdi
  unsigned __int8 *v126; // r14
  __int64 (__fastcall *v127)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  unsigned __int8 *v128; // r10
  __int64 v129; // rdi
  __int64 (__fastcall *v130)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v131; // rsi
  __int64 v132; // rax
  __int64 v133; // r12
  _BYTE *v134; // rax
  __int64 v135; // rax
  __int64 **v136; // r14
  __int64 v137; // r12
  __int64 v138; // rax
  unsigned __int64 v139; // rax
  __int64 v140; // rax
  _QWORD *v141; // rdi
  __int64 **v142; // r12
  __int64 v143; // rdi
  __int64 (__fastcall *v144)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v145; // r12
  char *v146; // r14
  _QWORD *v147; // rdi
  unsigned __int8 v148; // bl
  __int64 v149; // rax
  _BYTE *v150; // rax
  __int64 v151; // rbx
  __int64 v152; // r8
  __int64 v153; // r9
  __int64 v154; // rax
  unsigned __int64 v155; // rdx
  unsigned __int32 v156; // eax
  __int64 **v157; // r12
  unsigned __int8 *v158; // rax
  _QWORD *v159; // rdi
  __int64 v160; // r14
  unsigned int *v161; // rbx
  __int64 v162; // rdx
  unsigned int *v163; // r12
  __int64 v164; // r14
  __int64 v165; // rbx
  __int64 v166; // rdx
  unsigned int v167; // esi
  __int64 v168; // r14
  unsigned int *v169; // rbx
  __int64 v170; // rdx
  unsigned int v171; // esi
  __int64 v172; // rdi
  __int64 (__fastcall *v173)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v174; // rax
  char *v175; // r14
  unsigned __int64 v176; // rcx
  char *v177; // r12
  unsigned __int64 v178; // rax
  char *j; // rdi
  char v180; // cl
  char v181; // dl
  char *v182; // rsi
  char *v183; // rax
  char *v184; // r8
  char *v185; // rax
  unsigned int v186; // ecx
  bool v187; // dl
  __int64 v188; // rax
  __int64 v189; // rdi
  __int64 (__fastcall *v190)(__int64, unsigned int, _BYTE *, __int64); // rax
  unsigned int *v191; // r14
  __int64 v192; // r12
  __int64 v193; // rdx
  unsigned int v194; // esi
  _QWORD *v195; // rdi
  _QWORD *v196; // rax
  __int64 v197; // rsi
  unsigned int *v198; // r12
  __int64 v199; // rax
  __int64 v200; // r14
  __int64 v201; // rbx
  __int64 v202; // rdx
  unsigned int v203; // esi
  unsigned int *v204; // r12
  __int64 v205; // r14
  __int64 v206; // rdx
  unsigned int v207; // esi
  __int64 v208; // [rsp-10h] [rbp-150h]
  _BYTE *v209; // [rsp+8h] [rbp-138h]
  unsigned int v210; // [rsp+10h] [rbp-130h]
  __int64 v211; // [rsp+10h] [rbp-130h]
  __int64 v212; // [rsp+10h] [rbp-130h]
  __int64 dest; // [rsp+18h] [rbp-128h]
  void *desta; // [rsp+18h] [rbp-128h]
  __int64 **v216; // [rsp+20h] [rbp-120h]
  __int64 v217; // [rsp+20h] [rbp-120h]
  __int64 v218; // [rsp+28h] [rbp-118h]
  unsigned __int8 *v219; // [rsp+28h] [rbp-118h]
  __int64 v220; // [rsp+28h] [rbp-118h]
  __int64 v221; // [rsp+28h] [rbp-118h]
  __int64 v222; // [rsp+28h] [rbp-118h]
  __int64 v223; // [rsp+28h] [rbp-118h]
  __int64 v224; // [rsp+28h] [rbp-118h]
  __int64 v225; // [rsp+28h] [rbp-118h]
  unsigned __int8 *v226; // [rsp+28h] [rbp-118h]
  unsigned __int8 *v227; // [rsp+28h] [rbp-118h]
  __int64 v228; // [rsp+28h] [rbp-118h]
  __int64 v229; // [rsp+28h] [rbp-118h]
  __int64 v230; // [rsp+28h] [rbp-118h]
  unsigned __int8 *v231; // [rsp+28h] [rbp-118h]
  _QWORD *v232; // [rsp+28h] [rbp-118h]
  _QWORD *v233; // [rsp+28h] [rbp-118h]
  __int64 v234; // [rsp+28h] [rbp-118h]
  __int64 v235; // [rsp+30h] [rbp-110h]
  unsigned __int8 *v236; // [rsp+30h] [rbp-110h]
  char *v237; // [rsp+30h] [rbp-110h]
  __int64 v238; // [rsp+30h] [rbp-110h]
  __int64 v239; // [rsp+38h] [rbp-108h]
  unsigned int v240; // [rsp+38h] [rbp-108h]
  __int64 v241; // [rsp+38h] [rbp-108h]
  __int64 v242; // [rsp+38h] [rbp-108h]
  __int64 v243; // [rsp+38h] [rbp-108h]
  __int64 v244; // [rsp+38h] [rbp-108h]
  void *s; // [rsp+40h] [rbp-100h] BYREF
  size_t n; // [rsp+48h] [rbp-F8h]
  __int64 v247; // [rsp+50h] [rbp-F0h] BYREF
  unsigned int v248; // [rsp+58h] [rbp-E8h]
  void *src; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v250; // [rsp+68h] [rbp-D8h]
  _QWORD v251[2]; // [rsp+70h] [rbp-D0h] BYREF
  __int16 v252; // [rsp+80h] [rbp-C0h]
  _BYTE *v253[4]; // [rsp+90h] [rbp-B0h] BYREF
  __int16 v254; // [rsp+B0h] [rbp-90h]
  __m128i v255; // [rsp+C0h] [rbp-80h] BYREF
  _QWORD v256[2]; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v257; // [rsp+E0h] [rbp-60h]
  __int64 v258; // [rsp+E8h] [rbp-58h]
  __int64 v259; // [rsp+F0h] [rbp-50h]
  __int64 v260; // [rsp+F8h] [rbp-48h]
  __int16 v261; // [rsp+100h] [rbp-40h]

  v4 = a2;
  v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v256[0] = 0;
  v256[1] = 0;
  v257 = 0;
  v258 = 0;
  v6 = *(_QWORD *)(a2 - 32 * v5);
  v259 = 0;
  v260 = 0;
  v235 = v6;
  v7 = *(_QWORD *)(a2 + 32 * (2 - v5));
  v8 = a1[2];
  v261 = 257;
  v9 = &v255;
  v255 = (__m128i)v8;
  if ( (unsigned __int8)sub_9B6260(v7, &v255, 0) )
  {
    v255.m128i_i32[0] = 0;
    sub_11DA4B0(v4, v255.m128i_i32, 1);
    v9 = (__m128i *)v235;
    if ( (unsigned __int8)sub_11D9DE0(*(_QWORD *)(v4 + 16), v235) )
      return sub_11DD0A0(v4, v7, (unsigned int **)a3);
  }
  v10 = 0;
  v11 = *(_QWORD *)(v4 + 8);
  v218 = *(_QWORD *)(v4 + 32 * (1LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v218 == 17 )
    v10 = *(_QWORD *)(v4 + 32 * (1LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v7 != 17 )
  {
    v34 = sub_AD6530(v11, (__int64)v9);
    v18 = 0;
    v239 = v34;
    goto LABEL_11;
  }
  v12 = sub_AD6530(v11, (__int64)v9);
  v13 = *(_DWORD *)(v7 + 32);
  v14 = v7 + 24;
  v239 = v12;
  if ( v13 <= 0x40 )
  {
    v16 = *(_QWORD *)(v7 + 24) == 0;
  }
  else
  {
    v210 = *(_DWORD *)(v7 + 32);
    v15 = sub_C444A0(v14);
    v13 = v210;
    v14 = v7 + 24;
    v16 = v210 == v15;
  }
  if ( v16 )
    return v239;
  if ( v13 <= 0x40 )
    v17 = *(_QWORD *)(v7 + 24) == 1;
  else
    v17 = v13 - 1 == (unsigned int)sub_C444A0(v14);
  v18 = v7;
  if ( v17 )
  {
    v35 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
    v36 = *(_QWORD *)(a3 + 48);
    v37 = v35;
    v254 = 259;
    v253[0] = "memchr.char0";
    v38 = sub_AA4E30(v36);
    v39 = sub_AE5020(v38, v37);
    LOWORD(v257) = 257;
    v40 = sub_BD2C40(80, unk_3F10A14);
    v41 = (__int64)v40;
    if ( v40 )
      sub_B4D190((__int64)v40, v37, v235, (__int64)&v255, 0, v39, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
      *(_QWORD *)(a3 + 88),
      v41,
      v253,
      *(_QWORD *)(a3 + 56),
      *(_QWORD *)(a3 + 64));
    v42 = *(unsigned int **)a3;
    v43 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
    if ( *(_QWORD *)a3 != v43 )
    {
      do
      {
        v44 = *((_QWORD *)v42 + 1);
        v45 = *v42;
        v42 += 4;
        sub_B99FD0(v41, v45, v44);
      }
      while ( (unsigned int *)v43 != v42 );
    }
    v46 = *(_QWORD **)(a3 + 72);
    v254 = 257;
    v47 = (__int64 **)sub_BCB2B0(v46);
    if ( v47 == *(__int64 ***)(v218 + 8) )
    {
      v50 = (_BYTE *)v218;
      goto LABEL_35;
    }
    v48 = *(_QWORD *)(a3 + 80);
    v49 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v48 + 120LL);
    if ( v49 == sub_920130 )
    {
      if ( *(_BYTE *)v218 > 0x15u )
        goto LABEL_91;
      if ( (unsigned __int8)sub_AC4810(0x26u) )
        v50 = (_BYTE *)sub_ADAB70(38, v218, v47, 0);
      else
        v50 = (_BYTE *)sub_AA93C0(0x26u, v218, (__int64)v47);
    }
    else
    {
      v50 = (_BYTE *)v49(v48, 38u, (_BYTE *)v218, (__int64)v47);
    }
    if ( v50 )
    {
LABEL_35:
      v255.m128i_i64[0] = (__int64)"memchr.char0cmp";
      LOWORD(v257) = 259;
      v51 = sub_92B530((unsigned int **)a3, 0x20u, v41, v50, (__int64)&v255);
      LOWORD(v257) = 259;
      v255.m128i_i64[0] = (__int64)"memchr.sel";
      return sub_B36550((unsigned int **)a3, v51, v235, v239, (__int64)&v255, 0);
    }
LABEL_91:
    LOWORD(v257) = 257;
    v221 = sub_B51D30(38, v218, (__int64)v47, (__int64)&v255, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
      *(_QWORD *)(a3 + 88),
      v221,
      v253,
      *(_QWORD *)(a3 + 56),
      *(_QWORD *)(a3 + 64));
    v50 = (_BYTE *)v221;
    if ( *(_QWORD *)a3 != *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8) )
    {
      v222 = v41;
      v87 = *(unsigned int **)a3;
      v88 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
      v89 = (__int64)v50;
      do
      {
        v90 = *((_QWORD *)v87 + 1);
        v91 = *v87;
        v87 += 4;
        sub_B99FD0(v89, v91, v90);
      }
      while ( (unsigned int *)v88 != v87 );
      v41 = v222;
      v50 = (_BYTE *)v89;
    }
    goto LABEL_35;
  }
LABEL_11:
  v211 = v18;
  s = 0;
  n = 0;
  if ( !(unsigned __int8)sub_98B0F0(v235, &s, 0) )
    return 0;
  v19 = n;
  if ( v10 )
  {
    v20 = *(_QWORD **)(v10 + 24);
    if ( *(_DWORD *)(v10 + 32) > 0x40u )
      v20 = (_QWORD *)*v20;
    if ( n )
    {
      v21 = s;
      v22 = memchr(s, (char)v20, n);
      if ( v22 )
      {
        v23 = v22 - v21;
        if ( v22 - v21 != -1 )
        {
          v255.m128i_i64[0] = (__int64)"memchr.cmp";
          LOWORD(v257) = 259;
          v24 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v7 + 8), v23, 0);
          v25 = sub_92B530((unsigned int **)a3, 0x25u, v7, v24, (__int64)&v255);
          v26 = *(_QWORD **)(a3 + 72);
          v27 = v25;
          LOWORD(v257) = 259;
          v255.m128i_i64[0] = (__int64)"memchr.ptr";
          v28 = sub_BCB2E0(v26);
          v29 = (_BYTE *)sub_ACD640(v28, v23, 0);
          v30 = *(_QWORD **)(a3 + 72);
          v253[0] = v29;
          v31 = sub_BCB2B0(v30);
          v32 = sub_921130((unsigned int **)a3, v31, v235, v253, 1, (__int64)&v255, 3u);
          LOWORD(v257) = 257;
          return sub_B36550((unsigned int **)a3, v27, v239, v32, (__int64)&v255, 0);
        }
      }
    }
    return v239;
  }
  if ( !n )
    return v239;
  if ( v211 )
  {
    v52 = *(_QWORD **)(v211 + 24);
    if ( *(_DWORD *)(v211 + 32) > 0x40u )
      v52 = (_QWORD *)*v52;
    if ( (unsigned __int64)v52 <= n )
      v19 = (size_t)v52;
    n = v19;
  }
  v53 = sub_C93580(&s, *(_BYTE *)s, 0);
  if ( v53 == -1 )
  {
    desta = *(void **)(v7 + 8);
    v92 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
    v254 = 257;
    v216 = (__int64 **)v92;
    if ( *(_QWORD *)(v218 + 8) == v92 )
    {
      v66 = (_BYTE *)v218;
      goto LABEL_84;
    }
LABEL_70:
    v64 = *(_QWORD *)(a3 + 80);
    v65 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v64 + 120LL);
    if ( v65 == sub_920130 )
    {
      if ( *(_BYTE *)v218 > 0x15u )
      {
LABEL_105:
        LOWORD(v257) = 257;
        v66 = (_BYTE *)sub_B51D30(38, v218, (__int64)v216, (__int64)&v255, 0, 0);
        (*(void (__fastcall **)(_QWORD, _BYTE *, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
          *(_QWORD *)(a3 + 88),
          v66,
          v253,
          *(_QWORD *)(a3 + 56),
          *(_QWORD *)(a3 + 64));
        if ( *(_QWORD *)a3 != *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8) )
        {
          v223 = v53;
          v97 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
          v212 = v7;
          v98 = *(unsigned int **)a3;
          do
          {
            v99 = *((_QWORD *)v98 + 1);
            v100 = *v98;
            v98 += 4;
            sub_B99FD0((__int64)v66, v100, v99);
          }
          while ( (unsigned int *)v97 != v98 );
          v53 = v223;
          v7 = v212;
        }
LABEL_75:
        if ( v53 != -1 )
          goto LABEL_76;
LABEL_84:
        v79 = sub_AD64C0((__int64)v216, *(char *)s, 0);
        LOWORD(v257) = 257;
        v80 = (unsigned __int8 *)sub_92B530((unsigned int **)a3, 0x20u, v79, v66, (__int64)&v255);
        LOWORD(v257) = 257;
        v81 = (_BYTE *)sub_AD64C0((__int64)desta, 0, 0);
        v82 = sub_92B530((unsigned int **)a3, 0x21u, v7, v81, (__int64)&v255);
        v83 = *(_QWORD *)(a3 + 80);
        v84 = (unsigned __int8 *)v82;
        v254 = 257;
        v85 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v83 + 16LL);
        if ( v85 == sub_9202E0 )
        {
          if ( *v84 > 0x15u || *v80 > 0x15u )
            goto LABEL_100;
          if ( (unsigned __int8)sub_AC47B0(28) )
            v86 = sub_AD5570(28, (__int64)v84, v80, 0, 0);
          else
            v86 = sub_AABE40(0x1Cu, v84, v80);
        }
        else
        {
          v86 = v85(v83, 28u, v84, v80);
        }
        if ( v86 )
        {
LABEL_90:
          v255.m128i_i64[0] = (__int64)"memchr.sel2";
          LOWORD(v257) = 259;
          return sub_B36550((unsigned int **)a3, v86, v235, v239, (__int64)&v255, 0);
        }
LABEL_100:
        LOWORD(v257) = 257;
        v86 = sub_B504D0(28, (__int64)v84, (__int64)v80, (__int64)&v255, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
          *(_QWORD *)(a3 + 88),
          v86,
          v253,
          *(_QWORD *)(a3 + 56),
          *(_QWORD *)(a3 + 64));
        v93 = *(unsigned int **)a3;
        v94 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
        if ( *(_QWORD *)a3 != v94 )
        {
          do
          {
            v95 = *((_QWORD *)v93 + 1);
            v96 = *v93;
            v93 += 4;
            sub_B99FD0(v86, v96, v95);
          }
          while ( (unsigned int *)v94 != v93 );
        }
        goto LABEL_90;
      }
      if ( (unsigned __int8)sub_AC4810(0x26u) )
        v66 = (_BYTE *)sub_ADAB70(38, v218, v216, 0);
      else
        v66 = (_BYTE *)sub_AA93C0(0x26u, v218, (__int64)v216);
    }
    else
    {
      v66 = (_BYTE *)v65(v64, 38u, (_BYTE *)v218, (__int64)v216);
    }
    if ( v66 )
      goto LABEL_75;
    goto LABEL_105;
  }
  if ( sub_C93580(&s, *((_BYTE *)s + v53), v53) == -1 )
  {
    desta = *(void **)(v7 + 8);
    v216 = (__int64 **)sub_BCB2B0(*(_QWORD **)(a3 + 72));
    v254 = 257;
    if ( *(__int64 ***)(v218 + 8) == v216 )
    {
      v66 = (_BYTE *)v218;
LABEL_76:
      src = (void *)sub_AD64C0((__int64)desta, v53, 0);
      v67 = (_BYTE *)sub_AD64C0((__int64)v216, *((char *)s + v53), 0);
      LOWORD(v257) = 257;
      v68 = sub_92B530((unsigned int **)a3, 0x20u, (__int64)v66, v67, (__int64)&v255);
      LOWORD(v257) = 257;
      v69 = (unsigned __int8 *)v68;
      v70 = sub_92B530((unsigned int **)a3, 0x22u, v7, src, (__int64)&v255);
      v71 = *(_QWORD *)(a3 + 80);
      v254 = 257;
      v72 = (unsigned __int8 *)v70;
      v73 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v71 + 16LL);
      if ( v73 == sub_9202E0 )
      {
        if ( *v69 > 0x15u || *v72 > 0x15u )
          goto LABEL_111;
        v219 = v72;
        if ( (unsigned __int8)sub_AC47B0(28) )
          v74 = sub_AD5570(28, (__int64)v69, v219, 0, 0);
        else
          v74 = sub_AABE40(0x1Cu, v69, v219);
        v72 = v219;
        v75 = v74;
      }
      else
      {
        v227 = v72;
        v140 = v73(v71, 28u, v69, v72);
        v72 = v227;
        v75 = v140;
      }
      if ( v75 )
      {
LABEL_83:
        v76 = *(_QWORD **)(a3 + 72);
        v220 = v75;
        LOWORD(v257) = 257;
        v77 = sub_BCB2B0(v76);
        v78 = sub_921130((unsigned int **)a3, v77, v235, (_BYTE **)&src, 1, (__int64)&v255, 3u);
        LOWORD(v257) = 259;
        v255.m128i_i64[0] = (__int64)"memchr.sel1";
        v239 = sub_B36550((unsigned int **)a3, v220, v78, v239, (__int64)&v255, 0);
        goto LABEL_84;
      }
LABEL_111:
      LOWORD(v257) = 257;
      v224 = sub_B504D0(28, (__int64)v69, (__int64)v72, (__int64)&v255, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
        *(_QWORD *)(a3 + 88),
        v224,
        v253,
        *(_QWORD *)(a3 + 56),
        *(_QWORD *)(a3 + 64));
      v75 = v224;
      if ( *(_QWORD *)a3 != *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8) )
      {
        v225 = v7;
        v101 = v75;
        v209 = v66;
        v102 = *(unsigned int **)a3;
        v103 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
        do
        {
          v104 = *((_QWORD *)v102 + 1);
          v105 = *v102;
          v102 += 4;
          sub_B99FD0(v101, v105, v104);
        }
        while ( (unsigned int *)v103 != v102 );
        v75 = v101;
        v66 = v209;
        v7 = v225;
      }
      goto LABEL_83;
    }
    goto LABEL_70;
  }
  if ( !v211 )
  {
    if ( (unsigned __int8)sub_11D9DE0(*(_QWORD *)(v4 + 16), v235) )
      return sub_11DD0A0(v4, v7, (unsigned int **)a3);
    return 0;
  }
  if ( (unsigned __int8)sub_11F3070(*(_QWORD *)(v4 + 40), a1[9], a1[8], 0) || !n || !(unsigned __int8)sub_988330(v4) )
    return 0;
  v54 = (char *)s;
  v55 = n;
  v56 = (char *)s + n;
  if ( s == (char *)s + n )
  {
    i = *(unsigned __int8 *)s;
  }
  else
  {
    v57 = (char *)s + 1;
    for ( i = *(unsigned __int8 *)s; v56 != v57; ++v57 )
    {
      if ( (unsigned __int8)i < (unsigned __int8)*v57 )
        i = (unsigned __int8)*v57;
    }
  }
  v59 = a1[2];
  v60 = *(unsigned __int8 **)(v59 + 32);
  v61 = &v60[*(_QWORD *)(v59 + 40)];
  if ( v61 == v60 )
  {
LABEL_187:
    v255.m128i_i64[0] = n;
    src = v251;
    if ( n > 0xF )
    {
      src = (void *)sub_22409D0(&src, &v255, 0);
      v195 = src;
      v251[0] = v255.m128i_i64[0];
    }
    else
    {
      if ( n == 1 )
      {
        LOBYTE(v251[0]) = *(_BYTE *)s;
        goto LABEL_190;
      }
      if ( !n )
      {
LABEL_190:
        v250 = v255.m128i_i64[0];
        *((_BYTE *)src + v255.m128i_i64[0]) = 0;
        v175 = (char *)src;
        v176 = v250;
        v177 = (char *)src + v250;
        if ( src != (char *)src + v250 )
        {
          _BitScanReverse64(&v178, v250);
          v244 = v250;
          sub_11D9E20((__int64)src, (char *)src + v250, 2LL * (int)(63 - (v178 ^ 0x3F)));
          if ( v244 <= 16 )
          {
            sub_11D9C50(v175, v177);
          }
          else
          {
            sub_11D9C50(v175, v175 + 16);
            for ( j = v175 + 16; v177 != j; *v182 = v180 )
            {
              v180 = *j;
              v181 = *(j - 1);
              v182 = j;
              v183 = j - 1;
              if ( v181 > *j )
              {
                do
                {
                  v183[1] = v181;
                  v182 = v183;
                  v181 = *--v183;
                }
                while ( v180 < v181 );
              }
              ++j;
            }
          }
          v176 = v250;
        }
        if ( v176 <= 1 )
          goto LABEL_147;
        v159 = src;
        v184 = (char *)src + v176 - 1;
        v185 = (char *)src;
        v186 = 1;
        do
        {
          v187 = v185[1] > *v185 + 1;
          ++v185;
          v186 += v187;
        }
        while ( v184 != v185 );
        v239 = 0;
        if ( v186 <= 2 )
          goto LABEL_147;
        goto LABEL_162;
      }
      v195 = v251;
    }
    memcpy(v195, v54, v55);
    goto LABEL_190;
  }
  do
  {
    if ( *v60 >= (unsigned int)(unsigned __int8)i + 1 )
    {
      if ( (unsigned __int8)i > 7u )
      {
        v240 = (unsigned __int8)((((((i | (i >> 1)) >> 2) | i | (i >> 1)) >> 4) | ((i | (i >> 1)) >> 2) | i | (i >> 1))
                               + 1);
        v248 = v240;
        dest = (unsigned __int8)((((((i | (i >> 1)) >> 2) | i | (i >> 1)) >> 4) | ((i | (i >> 1)) >> 2) | i | (i >> 1))
                               + 1);
        if ( v240 > 0x40 )
        {
          sub_C43690((__int64)&v247, 0, 0);
          v54 = (char *)s;
          v56 = (char *)s + n;
LABEL_61:
          while ( v56 != v54 )
          {
            v62 = (unsigned __int8)*v54;
            v63 = 1LL << v62;
            if ( v248 <= 0x40 )
              v247 |= v63;
            else
              *(_QWORD *)(v247 + ((v62 >> 3) & 0x18)) |= v63;
            ++v54;
          }
          v106 = sub_ACCFD0(*(__int64 **)(a3 + 72), (__int64)&v247);
          v254 = 257;
          v107 = *(__int64 ***)(v106 + 8);
          v236 = (unsigned __int8 *)v106;
          v217 = *(_QWORD *)(v218 + 8);
          v108 = sub_BCB060(v217);
          v109 = sub_BCB060((__int64)v107);
          if ( v108 >= v109 )
          {
            if ( v107 == (__int64 **)v217 || v108 == v109 )
              goto LABEL_120;
            v189 = *(_QWORD *)(a3 + 80);
            v190 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v189 + 120LL);
            if ( v190 == sub_920130 )
            {
              if ( *(_BYTE *)v218 > 0x15u )
                goto LABEL_212;
              if ( (unsigned __int8)sub_AC4810(0x26u) )
                v174 = sub_ADAB70(38, v218, v107, 0);
              else
                v174 = sub_AA93C0(0x26u, v218, (__int64)v107);
            }
            else
            {
              v174 = v190(v189, 38u, (_BYTE *)v218, (__int64)v107);
            }
            if ( !v174 )
            {
LABEL_212:
              LOWORD(v257) = 257;
              v218 = sub_B51D30(38, v218, (__int64)v107, (__int64)&v255, 0, 0);
              (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
                *(_QWORD *)(a3 + 88),
                v218,
                v253,
                *(_QWORD *)(a3 + 56),
                *(_QWORD *)(a3 + 64));
              v191 = *(unsigned int **)a3;
              v192 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
              if ( *(_QWORD *)a3 != v192 )
              {
                do
                {
                  v193 = *((_QWORD *)v191 + 1);
                  v194 = *v191;
                  v191 += 4;
                  sub_B99FD0(v218, v194, v193);
                }
                while ( (unsigned int *)v192 != v191 );
              }
              goto LABEL_120;
            }
LABEL_184:
            v218 = v174;
            goto LABEL_120;
          }
          if ( v107 == (__int64 **)v217 )
            goto LABEL_120;
          v172 = *(_QWORD *)(a3 + 80);
          v173 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v172 + 120LL);
          if ( v173 == sub_920130 )
          {
            if ( *(_BYTE *)v218 > 0x15u )
              goto LABEL_222;
            if ( (unsigned __int8)sub_AC4810(0x27u) )
              v174 = sub_ADAB70(39, v218, v107, 0);
            else
              v174 = sub_AA93C0(0x27u, v218, (__int64)v107);
          }
          else
          {
            v174 = v173(v172, 39u, (_BYTE *)v218, (__int64)v107);
          }
          if ( v174 )
            goto LABEL_184;
LABEL_222:
          LOWORD(v257) = 257;
          v196 = sub_BD2C40(72, unk_3F10A14);
          if ( v196 )
          {
            v197 = v218;
            v232 = v196;
            sub_B515B0((__int64)v196, v197, (__int64)v107, (__int64)&v255, 0, 0);
            v196 = v232;
          }
          v233 = v196;
          (*(void (__fastcall **)(_QWORD, _QWORD *, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
            *(_QWORD *)(a3 + 88),
            v196,
            v253,
            *(_QWORD *)(a3 + 56),
            *(_QWORD *)(a3 + 64));
          v198 = *(unsigned int **)a3;
          v199 = (__int64)v233;
          v200 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
          if ( *(_QWORD *)a3 != v200 )
          {
            v234 = v4;
            v201 = v199;
            do
            {
              v202 = *((_QWORD *)v198 + 1);
              v203 = *v198;
              v198 += 4;
              sub_B99FD0(v201, v203, v202);
            }
            while ( (unsigned int *)v200 != v198 );
            v199 = v201;
            v4 = v234;
          }
          v218 = v199;
LABEL_120:
          v110 = *(_QWORD **)(a3 + 72);
          v254 = 257;
          v111 = sub_BCD140(v110, v240);
          v112 = sub_ACD640(v111, 255, 0);
          v113 = *(_QWORD *)(a3 + 80);
          v114 = (unsigned __int8 *)v112;
          v115 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v113 + 16LL);
          if ( v115 == sub_9202E0 )
          {
            if ( *(_BYTE *)v218 > 0x15u || *v114 > 0x15u )
            {
LABEL_172:
              LOWORD(v257) = 257;
              v116 = (unsigned __int8 *)sub_B504D0(28, v218, (__int64)v114, (__int64)&v255, 0, 0);
              (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
                *(_QWORD *)(a3 + 88),
                v116,
                v253,
                *(_QWORD *)(a3 + 56),
                *(_QWORD *)(a3 + 64));
              v168 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
              if ( *(_QWORD *)a3 != v168 )
              {
                v230 = v4;
                v169 = *(unsigned int **)a3;
                do
                {
                  v170 = *((_QWORD *)v169 + 1);
                  v171 = *v169;
                  v169 += 4;
                  sub_B99FD0((__int64)v116, v171, v170);
                }
                while ( (unsigned int *)v168 != v169 );
                v4 = v230;
              }
LABEL_126:
              v117 = v240;
              v118 = *(_QWORD **)(a3 + 72);
              v255.m128i_i64[0] = (__int64)"memchr.bounds";
              LOWORD(v257) = 259;
              v119 = sub_BCD140(v118, v240);
              v120 = (_BYTE *)sub_ACD640(v119, dest, 0);
              v121 = sub_92B530((unsigned int **)a3, 0x24u, (__int64)v116, v120, (__int64)&v255);
              v122 = *(_QWORD **)(a3 + 72);
              v241 = v121;
              v254 = 257;
              v123 = sub_BCD140(v122, v117);
              v124 = sub_ACD640(v123, 1, 0);
              v125 = *(_QWORD *)(a3 + 80);
              v126 = (unsigned __int8 *)v124;
              v127 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v125 + 32LL);
              if ( v127 == sub_9201A0 )
              {
                if ( *v126 > 0x15u || *v116 > 0x15u )
                {
LABEL_168:
                  LOWORD(v257) = 257;
                  v228 = sub_B504D0(25, (__int64)v126, (__int64)v116, (__int64)&v255, 0, 0);
                  (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
                    *(_QWORD *)(a3 + 88),
                    v228,
                    v253,
                    *(_QWORD *)(a3 + 56),
                    *(_QWORD *)(a3 + 64));
                  v163 = *(unsigned int **)a3;
                  v128 = (unsigned __int8 *)v228;
                  v164 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
                  if ( *(_QWORD *)a3 != v164 )
                  {
                    v229 = v4;
                    v165 = (__int64)v128;
                    do
                    {
                      v166 = *((_QWORD *)v163 + 1);
                      v167 = *v163;
                      v163 += 4;
                      sub_B99FD0(v165, v167, v166);
                    }
                    while ( (unsigned int *)v164 != v163 );
                    v128 = (unsigned __int8 *)v165;
                    v4 = v229;
                  }
LABEL_132:
                  v129 = *(_QWORD *)(a3 + 80);
                  v253[0] = "memchr.bits";
                  v254 = 259;
                  v252 = 257;
                  v130 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v129 + 16LL);
                  if ( v130 == sub_9202E0 )
                  {
                    if ( *v128 > 0x15u || *v236 > 0x15u )
                      goto LABEL_164;
                    v226 = v128;
                    v131 = (__int64)v128;
                    if ( (unsigned __int8)sub_AC47B0(28) )
                      v132 = sub_AD5570(28, (__int64)v226, v236, 0, 0);
                    else
                      v132 = sub_AABE40(0x1Cu, v226, v236);
                    v128 = v226;
                    v133 = v132;
                  }
                  else
                  {
                    v231 = v128;
                    v131 = 28;
                    v188 = v130(v129, 28u, v128, v236);
                    v128 = v231;
                    v133 = v188;
                  }
                  if ( v133 )
                  {
LABEL_139:
                    v134 = (_BYTE *)sub_AD6530(*(_QWORD *)(v133 + 8), v131);
                    v135 = sub_92B530((unsigned int **)a3, 0x21u, v133, v134, (__int64)v253);
                    v136 = *(__int64 ***)(v4 + 8);
                    v137 = v135;
                    LOWORD(v257) = 257;
                    v253[0] = "memchr";
                    v254 = 259;
                    v138 = sub_AD6530(*(_QWORD *)(v135 + 8), 257);
                    v139 = sub_B36550((unsigned int **)a3, v241, v137, v138, (__int64)v253, 0);
                    v239 = sub_11DB4B0((__int64 *)a3, 0x30u, v139, v136, (__int64)&v255, 0, (int)src, 0);
                    if ( v248 > 0x40 && v247 )
                      j_j___libc_free_0_0(v247);
                    return v239;
                  }
LABEL_164:
                  LOWORD(v257) = 257;
                  v133 = sub_B504D0(28, (__int64)v128, (__int64)v236, (__int64)&v255, 0, 0);
                  v131 = v133;
                  (*(void (__fastcall **)(_QWORD, __int64, void **, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
                    *(_QWORD *)(a3 + 88),
                    v133,
                    &src,
                    *(_QWORD *)(a3 + 56),
                    *(_QWORD *)(a3 + 64));
                  v160 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
                  if ( *(_QWORD *)a3 != v160 )
                  {
                    v238 = v4;
                    v161 = *(unsigned int **)a3;
                    do
                    {
                      v162 = *((_QWORD *)v161 + 1);
                      v131 = *v161;
                      v161 += 4;
                      sub_B99FD0(v133, v131, v162);
                    }
                    while ( (unsigned int *)v160 != v161 );
                    v4 = v238;
                  }
                  goto LABEL_139;
                }
                if ( (unsigned __int8)sub_AC47B0(25) )
                  v128 = (unsigned __int8 *)sub_AD5570(25, (__int64)v126, v116, 0, 0);
                else
                  v128 = (unsigned __int8 *)sub_AABE40(0x19u, v126, v116);
              }
              else
              {
                v128 = (unsigned __int8 *)v127(v125, 25u, v126, v116, 0, 0);
              }
              if ( v128 )
                goto LABEL_132;
              goto LABEL_168;
            }
            if ( (unsigned __int8)sub_AC47B0(28) )
              v116 = (unsigned __int8 *)sub_AD5570(28, v218, v114, 0, 0);
            else
              v116 = (unsigned __int8 *)sub_AABE40(0x1Cu, (unsigned __int8 *)v218, v114);
          }
          else
          {
            v116 = (unsigned __int8 *)v115(v113, 28u, (_BYTE *)v218, v114);
          }
          if ( v116 )
            goto LABEL_126;
          goto LABEL_172;
        }
      }
      else
      {
        v248 = 8;
        dest = 8;
        v240 = 8;
      }
      v247 = 0;
      goto LABEL_61;
    }
    ++v60;
  }
  while ( v61 != v60 );
  if ( s )
    goto LABEL_187;
  LOBYTE(v251[0]) = 0;
  src = v251;
  v250 = 0;
LABEL_147:
  v141 = *(_QWORD **)(a3 + 72);
  v254 = 257;
  v142 = (__int64 **)sub_BCB2B0(v141);
  if ( v142 == *(__int64 ***)(v218 + 8) )
  {
    v242 = v218;
  }
  else
  {
    v143 = *(_QWORD *)(a3 + 80);
    v144 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v143 + 120LL);
    if ( v144 != sub_920130 )
    {
      v242 = v144(v143, 38u, (_BYTE *)v218, (__int64)v142);
      goto LABEL_152;
    }
    if ( *(_BYTE *)v218 > 0x15u )
      goto LABEL_234;
    v242 = (unsigned __int8)sub_AC4810(0x26u) ? sub_ADAB70(38, v218, v142, 0) : sub_AA93C0(0x26u, v218, (__int64)v142);
LABEL_152:
    if ( !v242 )
    {
LABEL_234:
      LOWORD(v257) = 257;
      v242 = sub_B51D30(38, v218, (__int64)v142, (__int64)&v255, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
        *(_QWORD *)(a3 + 88),
        v242,
        v253,
        *(_QWORD *)(a3 + 56),
        *(_QWORD *)(a3 + 64));
      v204 = *(unsigned int **)a3;
      v205 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
      if ( *(_QWORD *)a3 != v205 )
      {
        do
        {
          v206 = *((_QWORD *)v204 + 1);
          v207 = *v204;
          v204 += 4;
          sub_B99FD0(v242, v207, v206);
        }
        while ( (unsigned int *)v205 != v204 );
      }
    }
  }
  v255.m128i_i64[0] = (__int64)v256;
  v255.m128i_i64[1] = 0x600000000LL;
  if ( (char *)src + v250 == src )
  {
    v156 = 0;
  }
  else
  {
    v145 = v242;
    v243 = v4;
    v237 = (char *)src + v250;
    v146 = (char *)src;
    do
    {
      v147 = *(_QWORD **)(a3 + 72);
      v148 = *v146;
      v254 = 257;
      v149 = sub_BCB2B0(v147);
      v150 = (_BYTE *)sub_ACD640(v149, v148, 0);
      v151 = sub_92B530((unsigned int **)a3, 0x20u, v145, v150, (__int64)v253);
      v154 = v255.m128i_u32[2];
      v155 = v255.m128i_u32[2] + 1LL;
      if ( v155 > v255.m128i_u32[3] )
      {
        sub_C8D5F0((__int64)&v255, v256, v155, 8u, v152, v153);
        v154 = v255.m128i_u32[2];
      }
      ++v146;
      *(_QWORD *)(v255.m128i_i64[0] + 8 * v154) = v151;
      v156 = ++v255.m128i_i32[2];
    }
    while ( v237 != v146 );
    v4 = v243;
  }
  v254 = 257;
  v157 = *(__int64 ***)(v4 + 8);
  v158 = sub_11DBC90((__int64 *)a3, (unsigned __int8 **)v255.m128i_i64[0], v156);
  v239 = sub_11DB4B0((__int64 *)a3, 0x30u, (unsigned __int64)v158, v157, (__int64)v253, 0, v247, 0);
  if ( (_QWORD *)v255.m128i_i64[0] != v256 )
    _libc_free(v255.m128i_i64[0], v208);
  v159 = src;
LABEL_162:
  if ( v159 != v251 )
    j_j___libc_free_0(v159, v251[0] + 1LL);
  return v239;
}
