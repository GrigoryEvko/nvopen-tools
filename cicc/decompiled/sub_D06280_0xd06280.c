// Function: sub_D06280
// Address: 0xd06280
//
__int64 __fastcall sub_D06280(
        __int64 *a1,
        unsigned __int8 *a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        char *a6,
        __int64 a7,
        __int64 a8)
{
  unsigned __int64 v12; // rsi
  __int64 v13; // r8
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rax
  unsigned int v17; // r15d
  __int64 v18; // rax
  bool v19; // al
  unsigned int v20; // r13d
  unsigned __int8 v21; // al
  __int64 v22; // rdi
  __int64 v23; // rbx
  __int64 v24; // r12
  __int64 v25; // rdi
  __int64 v26; // rbx
  __int64 v27; // r12
  __int64 v28; // rdi
  __int64 v30; // rax
  unsigned int v31; // r15d
  __int64 v32; // rax
  bool v33; // al
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // r13
  int v37; // eax
  int v38; // eax
  int v39; // eax
  unsigned int v40; // edx
  __int64 v41; // rax
  unsigned __int32 v42; // ecx
  __int32 v43; // eax
  unsigned int v44; // eax
  __int32 v45; // eax
  __int32 v46; // eax
  __int64 v47; // rbx
  unsigned int v48; // edx
  unsigned int v50; // esi
  __int64 v51; // rsi
  __int64 v52; // rcx
  unsigned __int64 v53; // rsi
  unsigned __int64 v54; // rdx
  __int32 v55; // eax
  __int64 v56; // rdi
  __int64 v57; // rdi
  unsigned int v58; // eax
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  int v63; // eax
  __int64 *v64; // r13
  __int64 *v65; // r14
  __int64 v66; // rdi
  __int64 v67; // rax
  unsigned __int64 v68; // rbx
  __int64 v69; // rax
  __int64 v70; // r13
  __int64 v71; // r13
  void *v72; // rax
  __int64 v73; // rbx
  __int64 v74; // rax
  unsigned int v75; // edx
  unsigned __int64 v76; // r12
  unsigned __int64 v77; // rcx
  bool v78; // r12
  int v79; // eax
  __int32 v80; // eax
  unsigned __int64 *v81; // r15
  unsigned __int64 v82; // rax
  bool v83; // r14
  const void *v84; // rdi
  int v85; // edx
  __int64 v86; // rdx
  int v87; // eax
  __int32 v88; // eax
  unsigned __int8 **v89; // rbx
  unsigned int v90; // r13d
  __int64 v92; // rsi
  __int64 v93; // rdx
  __int64 v94; // rcx
  __int64 v95; // r8
  unsigned __int64 v96; // rdx
  __int64 *v97; // rax
  __int64 v98; // rbx
  unsigned int v99; // ebx
  unsigned int v100; // eax
  __int64 v101; // rax
  unsigned __int64 v102; // rdx
  __int64 *v103; // rbx
  __int64 v104; // r13
  int v105; // eax
  __int64 *v106; // rbx
  __int64 v107; // r13
  int v108; // eax
  __int32 v109; // edx
  __int64 v110; // rsi
  __int64 v111; // rcx
  __int32 v112; // eax
  __int32 v113; // eax
  int v114; // eax
  unsigned int v115; // eax
  __int64 v116; // r8
  __int32 v117; // eax
  __int32 v118; // eax
  __int64 v119; // rdx
  __int64 v120; // rdi
  unsigned __int64 v121; // rsi
  unsigned __int64 v122; // rdx
  __int64 v123; // rax
  unsigned int v124; // r13d
  unsigned __int64 v125; // rbx
  unsigned __int64 v126; // rax
  unsigned __int64 v127; // rax
  _QWORD *v128; // rax
  unsigned __int32 v129; // eax
  unsigned int v130; // ebx
  __int64 v131; // rax
  __int64 v132; // rax
  const void **v133; // rdi
  __int64 i; // rcx
  __int64 v135; // rax
  unsigned int v137; // eax
  __int64 v138; // rdx
  __int64 v139; // rcx
  __int64 v140; // r8
  unsigned __int64 v141; // rsi
  unsigned __int64 v142; // rdx
  bool v143; // al
  __int64 *v144; // r10
  __int64 v145; // rax
  __int64 v146; // rdx
  __int64 v147; // rcx
  char v148; // al
  unsigned int v149; // eax
  __int64 v150; // rdx
  unsigned int v151; // ecx
  __int64 v152; // rsi
  unsigned int v153; // r8d
  bool v154; // zf
  __int64 v155; // rdx
  __int64 *v156; // r10
  unsigned int v157; // r8d
  bool v158; // zf
  __int64 v159; // rdx
  __int64 v160; // rbx
  __int64 v161; // rax
  __int64 v162; // rdx
  __int64 v163; // rcx
  __int64 *v164; // rsi
  __int64 v165; // [rsp+8h] [rbp-508h]
  __int32 v166; // [rsp+10h] [rbp-500h]
  unsigned __int32 v167; // [rsp+10h] [rbp-500h]
  unsigned int v168; // [rsp+18h] [rbp-4F8h]
  unsigned int v169; // [rsp+2Ch] [rbp-4E4h]
  __int64 v170; // [rsp+40h] [rbp-4D0h]
  __int64 *v171; // [rsp+40h] [rbp-4D0h]
  __int64 v172; // [rsp+48h] [rbp-4C8h]
  __int64 *v173; // [rsp+48h] [rbp-4C8h]
  __int64 v174; // [rsp+48h] [rbp-4C8h]
  __int64 *v175; // [rsp+48h] [rbp-4C8h]
  void *v176; // [rsp+50h] [rbp-4C0h]
  const void **v177; // [rsp+58h] [rbp-4B8h]
  __int64 v178; // [rsp+58h] [rbp-4B8h]
  __int64 v179; // [rsp+60h] [rbp-4B0h]
  unsigned int v180; // [rsp+60h] [rbp-4B0h]
  __int64 *v181; // [rsp+60h] [rbp-4B0h]
  __int64 v182; // [rsp+68h] [rbp-4A8h]
  int v183; // [rsp+70h] [rbp-4A0h]
  unsigned int v184; // [rsp+70h] [rbp-4A0h]
  unsigned __int64 v185; // [rsp+70h] [rbp-4A0h]
  unsigned int v186; // [rsp+78h] [rbp-498h]
  __int64 v188; // [rsp+80h] [rbp-490h]
  __int64 v189; // [rsp+80h] [rbp-490h]
  __int64 v190; // [rsp+80h] [rbp-490h]
  unsigned __int64 v191; // [rsp+80h] [rbp-490h]
  __int64 v192; // [rsp+88h] [rbp-488h]
  unsigned __int64 v193; // [rsp+88h] [rbp-488h]
  __int64 v194; // [rsp+88h] [rbp-488h]
  unsigned int v195; // [rsp+88h] [rbp-488h]
  unsigned int v196; // [rsp+88h] [rbp-488h]
  __int64 v197; // [rsp+90h] [rbp-480h] BYREF
  __int64 v198; // [rsp+98h] [rbp-478h] BYREF
  void *v199; // [rsp+A0h] [rbp-470h] BYREF
  unsigned __int32 v200; // [rsp+A8h] [rbp-468h]
  _QWORD *v201; // [rsp+B0h] [rbp-460h] BYREF
  unsigned int v202; // [rsp+B8h] [rbp-458h]
  __int64 v203; // [rsp+C0h] [rbp-450h] BYREF
  unsigned __int32 v204; // [rsp+C8h] [rbp-448h]
  _QWORD *v205; // [rsp+D0h] [rbp-440h] BYREF
  unsigned int v206; // [rsp+D8h] [rbp-438h]
  _QWORD *v207; // [rsp+E0h] [rbp-430h] BYREF
  unsigned int v208; // [rsp+E8h] [rbp-428h]
  __int64 v209; // [rsp+F0h] [rbp-420h] BYREF
  unsigned int v210; // [rsp+F8h] [rbp-418h]
  __int64 v211; // [rsp+100h] [rbp-410h] BYREF
  unsigned int v212; // [rsp+108h] [rbp-408h]
  const void *v213; // [rsp+110h] [rbp-400h]
  unsigned int v214; // [rsp+118h] [rbp-3F8h]
  void *v215; // [rsp+120h] [rbp-3F0h] BYREF
  unsigned int v216; // [rsp+128h] [rbp-3E8h]
  __int64 v217; // [rsp+130h] [rbp-3E0h] BYREF
  unsigned int v218; // [rsp+138h] [rbp-3D8h]
  _QWORD *v219; // [rsp+140h] [rbp-3D0h] BYREF
  unsigned int v220; // [rsp+148h] [rbp-3C8h]
  __int64 v221; // [rsp+150h] [rbp-3C0h]
  unsigned int v222; // [rsp+158h] [rbp-3B8h]
  void *v223; // [rsp+160h] [rbp-3B0h] BYREF
  unsigned __int32 v224; // [rsp+168h] [rbp-3A8h]
  __int64 v225; // [rsp+170h] [rbp-3A0h] BYREF
  unsigned int v226; // [rsp+178h] [rbp-398h]
  void *v227; // [rsp+180h] [rbp-390h] BYREF
  __int64 v228; // [rsp+188h] [rbp-388h]
  __int64 v229; // [rsp+190h] [rbp-380h] BYREF
  __int64 v230; // [rsp+198h] [rbp-378h]
  __int64 v231; // [rsp+1A0h] [rbp-370h]
  __int64 v232; // [rsp+1A8h] [rbp-368h]
  __m128i v233; // [rsp+1B0h] [rbp-360h] BYREF
  unsigned int v234; // [rsp+1C0h] [rbp-350h]
  __int64 v235; // [rsp+1C8h] [rbp-348h] BYREF
  unsigned int v236; // [rsp+1D0h] [rbp-340h]
  char v237; // [rsp+1D8h] [rbp-338h] BYREF
  int v238; // [rsp+2B8h] [rbp-258h]
  char *v239; // [rsp+2C0h] [rbp-250h] BYREF
  __int64 v240; // [rsp+2C8h] [rbp-248h]
  __int64 v241; // [rsp+2D0h] [rbp-240h]
  __int64 v242; // [rsp+2D8h] [rbp-238h] BYREF
  __int64 v243; // [rsp+2E0h] [rbp-230h]
  __int64 v244; // [rsp+2E8h] [rbp-228h] BYREF
  int v245; // [rsp+3C8h] [rbp-148h]
  __m128i v246; // [rsp+3D0h] [rbp-140h] BYREF
  const void *v247; // [rsp+3E0h] [rbp-130h] BYREF
  __int64 *v248; // [rsp+3E8h] [rbp-128h] BYREF
  __int64 v249; // [rsp+3F0h] [rbp-120h]
  __int64 v250; // [rsp+3F8h] [rbp-118h] BYREF
  __int64 v251; // [rsp+400h] [rbp-110h]
  __int64 v252; // [rsp+408h] [rbp-108h]
  __int16 v253; // [rsp+410h] [rbp-100h]
  int v254; // [rsp+4D8h] [rbp-38h]

  v198 = a3;
  v197 = a5;
  if ( (a3 == -1 || a3 == 0xBFFFFFFFFFFFFFFELL) && (v197 == -1 || v197 == 0xBFFFFFFFFFFFFFFELL) )
  {
    v21 = *(_BYTE *)a4;
    if ( *(_BYTE *)a4 <= 0x1Cu )
    {
      if ( v21 != 5 || *(_WORD *)(a4 + 2) != 34 )
        return 1;
    }
    else if ( v21 != 63 && v21 != 60 )
    {
      return 1;
    }
    v246.m128i_i64[1] = -1;
    v247 = 0;
    v56 = *(_QWORD *)a8;
    v248 = 0;
    v249 = 0;
    v246.m128i_i64[0] = a7;
    v250 = 0;
    v239 = a6;
    v240 = -1;
    v241 = 0;
    v242 = 0;
    v243 = 0;
    v244 = 0;
    return (unsigned __int8)sub_CF4D50(v56, (__int64)&v239, (__int64)&v246, a8, 0) != 0;
  }
  if ( *(_BYTE *)(a8 + 513) )
    v192 = a1[4];
  else
    v192 = 0;
  sub_D05650((__int64)&v233, a2, *a1);
  v12 = a4;
  sub_D05650((__int64)&v239, (unsigned __int8 *)a4, *a1);
  v14 = v233.m128i_i64[0];
  if ( (unsigned __int8 *)v233.m128i_i64[0] == a2 && v239 == (char *)a4 )
    goto LABEL_195;
  if ( v234 != (_DWORD)v241 )
  {
    v12 = (unsigned __int64)&v227;
    v246.m128i_i64[1] = -1;
    v247 = 0;
    v22 = *(_QWORD *)a8;
    v248 = 0;
    v249 = 0;
    v246.m128i_i64[0] = a7;
    v250 = 0;
    v227 = a6;
    v228 = -1;
    v229 = 0;
    v230 = 0;
    v231 = 0;
    v232 = 0;
    v20 = (unsigned __int8)sub_CF4D50(v22, (__int64)&v227, (__int64)&v246, a8, 0) != 0;
    goto LABEL_27;
  }
  v15 = v236;
  if ( v236 < (unsigned int)v243 )
  {
    LODWORD(v247) = v234;
    v246 = v233;
    v248 = &v250;
    v234 = 0;
    v249 = 0x400000000LL;
    if ( v236 )
    {
      sub_D01180((__int64)&v248, (__int64)&v235, v233.m128i_i64[0], v236, v13, (__int64)&v235);
      v254 = v238;
      v233.m128i_i64[0] = (__int64)v239;
      if ( v234 > 0x40 && v233.m128i_i64[1] )
        j_j___libc_free_0_0(v233.m128i_i64[1]);
    }
    else
    {
      v254 = v238;
      v233.m128i_i64[0] = (__int64)v239;
    }
    v233.m128i_i64[1] = v240;
    v58 = v241;
    LODWORD(v241) = 0;
    v234 = v58;
    sub_D01180((__int64)&v235, (__int64)&v242, v14, v15, v13, (__int64)&v235);
    v238 = v245;
    v239 = (char *)v246.m128i_i64[0];
    if ( (unsigned int)v241 > 0x40 && v240 )
      j_j___libc_free_0_0(v240);
    v240 = v246.m128i_i64[1];
    v63 = (int)v247;
    LODWORD(v247) = 0;
    LODWORD(v241) = v63;
    sub_D01180((__int64)&v242, (__int64)&v248, v59, v60, v61, v62);
    v64 = v248;
    v245 = v254;
    v65 = &v248[7 * (unsigned int)v249];
    if ( v248 != v65 )
    {
      do
      {
        v65 -= 7;
        if ( *((_DWORD *)v65 + 8) > 0x40u )
        {
          v66 = v65[3];
          if ( v66 )
            j_j___libc_free_0_0(v66);
        }
      }
      while ( v64 != v65 );
      v65 = v248;
    }
    if ( v65 != &v250 )
      _libc_free(v65, &v248);
    if ( (unsigned int)v247 > 0x40 && v246.m128i_i64[1] )
      j_j___libc_free_0_0(v246.m128i_i64[1]);
    v67 = v197;
    v197 = a3;
    v198 = v67;
  }
  v12 = (unsigned __int64)&v233;
  sub_D043E0((__int64)a1, (__int64)&v233, (__int64)&v239, a8);
  if ( (v238 & 1) != 0 && !v236 && v197 != -1 && v197 != 0xBFFFFFFFFFFFFFFELL && (v197 & 0x4000000000000000LL) == 0 )
  {
    v246.m128i_i8[8] = 0;
    v246.m128i_i64[0] = v197 & 0x3FFFFFFFFFFFFFFFLL;
    v16 = sub_CA1930(&v246);
    v17 = v234;
    v188 = v16;
    if ( v234 > 0x40 )
    {
      v103 = (__int64 *)v233.m128i_i64[1];
      v104 = *(_QWORD *)(v233.m128i_i64[1] + 8LL * ((v234 - 1) >> 6)) & (1LL << ((unsigned __int8)v234 - 1));
      if ( v104 )
        v105 = sub_C44500((__int64)&v233.m128i_i64[1]);
      else
        v105 = sub_C444A0((__int64)&v233.m128i_i64[1]);
      if ( v17 + 1 - v105 > 0x40 )
      {
        v19 = v104 != 0;
        goto LABEL_17;
      }
      v18 = *v103;
    }
    else
    {
      v18 = 0;
      if ( v234 )
        v18 = v233.m128i_i64[1] << (64 - (unsigned __int8)v234) >> (64 - (unsigned __int8)v234);
    }
    v19 = v188 > v18;
LABEL_17:
    if ( !v19 )
    {
      v20 = 0;
      if ( sub_CF74D0(v239) )
        goto LABEL_27;
    }
  }
  if ( (v245 & 1) == 0 || v236 || v198 == -1 || v198 == 0xBFFFFFFFFFFFFFFELL || (v198 & 0x4000000000000000LL) != 0 )
    goto LABEL_61;
  v246.m128i_i8[8] = 0;
  v246.m128i_i64[0] = v198 & 0x3FFFFFFFFFFFFFFFLL;
  v30 = sub_CA1930(&v246);
  v31 = v234;
  v189 = -v30;
  if ( v234 > 0x40 )
  {
    v106 = (__int64 *)v233.m128i_i64[1];
    v107 = *(_QWORD *)(v233.m128i_i64[1] + 8LL * ((v234 - 1) >> 6)) & (1LL << ((unsigned __int8)v234 - 1));
    if ( v107 )
      v108 = sub_C44500((__int64)&v233.m128i_i64[1]);
    else
      v108 = sub_C444A0((__int64)&v233.m128i_i64[1]);
    if ( v31 + 1 - v108 > 0x40 )
    {
      v33 = v107 == 0;
      goto LABEL_59;
    }
    v32 = *v106;
  }
  else
  {
    v32 = 0;
    if ( v234 )
      v32 = v233.m128i_i64[1] << (64 - (unsigned __int8)v234) >> (64 - (unsigned __int8)v234);
  }
  v33 = v189 < v32;
LABEL_59:
  if ( v33 )
    goto LABEL_62;
  v20 = 0;
  if ( sub_CF74D0((char *)v233.m128i_i64[0]) )
    goto LABEL_27;
LABEL_61:
  v31 = v234;
LABEL_62:
  if ( v31 <= 0x40 )
  {
    v34 = v233.m128i_i64[1];
  }
  else
  {
    if ( v31 - (unsigned int)sub_C444A0((__int64)&v233.m128i_i64[1]) > 0x40 )
      goto LABEL_66;
    v34 = *(_QWORD *)v233.m128i_i64[1];
  }
  if ( !v34 && !v236 )
  {
    v12 = (unsigned __int64)&v227;
    v247 = 0;
    v248 = 0;
    v57 = *(_QWORD *)a8;
    v249 = 0;
    v250 = 0;
    v246.m128i_i64[0] = (__int64)v239;
    v229 = 0;
    v246.m128i_i64[1] = v197;
    v230 = 0;
    v227 = (void *)v233.m128i_i64[0];
    v231 = 0;
    v228 = v198;
    v232 = 0;
    v20 = sub_CF4D50(v57, (__int64)&v227, (__int64)&v246, a8, 0);
    goto LABEL_27;
  }
LABEL_66:
  v246.m128i_i64[1] = -1;
  v35 = *(_QWORD *)a8;
  v12 = (unsigned __int64)&v227;
  v247 = 0;
  v248 = 0;
  v246.m128i_i64[0] = (__int64)v239;
  v249 = 0;
  v250 = 0;
  v227 = (void *)v233.m128i_i64[0];
  v228 = -1;
  v229 = 0;
  v230 = 0;
  v231 = 0;
  v232 = 0;
  v20 = sub_CF4D50(v35, (__int64)&v227, (__int64)&v246, a8, 0);
  if ( (_BYTE)v20 == 3 )
  {
    if ( v236 )
    {
      if ( v236 != 1 )
      {
LABEL_69:
        if ( (v238 & 4) == 0
          || (v179 = v197, v12 = v197, v197 == 0xBFFFFFFFFFFFFFFELL)
          || v197 == -1
          || (v197 & 0x4000000000000000LL) != 0 )
        {
LABEL_73:
          v36 = v198;
          if ( (v198 & 0x4000000000000000LL) != 0 )
            goto LABEL_195;
          v179 = v197;
          v12 = v197;
          if ( (v197 & 0x4000000000000000LL) != 0 )
            goto LABEL_195;
LABEL_75:
          if ( v179 != 0xBFFFFFFFFFFFFFFELL && v36 != 0xBFFFFFFFFFFFFFFELL )
          {
            v200 = 1;
            v199 = 0;
            v246.m128i_i32[2] = v234;
            if ( v234 > 0x40 )
              sub_C43780((__int64)&v246, (const void **)&v233.m128i_i64[1]);
            else
              v246.m128i_i64[0] = v233.m128i_i64[1];
            sub_AADBC0((__int64)&v211, v246.m128i_i64);
            if ( v246.m128i_i32[2] > 0x40u && v246.m128i_i64[0] )
              j_j___libc_free_0_0(v246.m128i_i64[0]);
            v169 = v236;
            if ( v236 )
            {
              v170 = 0;
              v183 = 0;
              while ( 1 )
              {
                v47 = v235 + v170;
                v48 = *(_DWORD *)(v235 + v170 + 32);
                v177 = (const void **)(v235 + v170 + 24);
                v206 = v48;
                if ( v48 <= 0x40 )
                  break;
                sub_C43780((__int64)&v205, v177);
                v48 = v206;
                if ( *(_BYTE *)(v47 + 48) )
                  goto LABEL_395;
                v48 = *(_DWORD *)(v47 + 32);
                if ( v48 <= 0x40 )
                {
                  _RAX = *(_QWORD *)(v47 + 24);
                  v50 = v206;
LABEL_481:
                  v246.m128i_i32[2] = v48;
                  v246.m128i_i64[0] = 0;
                  __asm { tzcnt   rcx, rax }
                  v154 = _RAX == 0;
                  v137 = 64;
                  if ( !v154 )
                    v137 = _RCX;
                  if ( v137 > v48 )
                    LOBYTE(v137) = v48;
                  v119 = 1LL << v137;
                  goto LABEL_486;
                }
                v166 = *(_DWORD *)(v47 + 32);
                v168 = sub_C44590((__int64)v177);
                v246.m128i_i32[2] = v166;
                sub_C43690((__int64)&v246, 0, 0);
                v119 = 1LL << v168;
                if ( v246.m128i_i32[2] > 0x40u )
                {
                  *(_QWORD *)(v246.m128i_i64[0] + 8LL * (v168 >> 6)) |= v119;
                  v50 = v206;
                  goto LABEL_391;
                }
                v50 = v206;
LABEL_486:
                v246.m128i_i64[0] |= v119;
LABEL_391:
                if ( v50 > 0x40 && v205 )
                  j_j___libc_free_0_0(v205);
                v48 = v246.m128i_u32[2];
                v205 = (_QWORD *)v246.m128i_i64[0];
                v206 = v246.m128i_u32[2];
LABEL_395:
                _RAX = (__int64)v205;
                if ( v183 )
                {
                  v51 = 1LL << ((unsigned __int8)v48 - 1);
                  if ( v48 > 0x40 )
                  {
                    if ( (v205[(v48 - 1) >> 6] & v51) == 0 )
                    {
                      v224 = v48;
                      sub_C43780((__int64)&v223, (const void **)&v205);
                      goto LABEL_178;
                    }
                    v246.m128i_i32[2] = v48;
                    sub_C43780((__int64)&v246, (const void **)&v205);
                    v48 = v246.m128i_u32[2];
                    if ( v246.m128i_i32[2] > 0x40u )
                    {
                      sub_C43D10((__int64)&v246);
LABEL_177:
                      sub_C46250((__int64)&v246);
                      v224 = v246.m128i_u32[2];
                      v223 = (void *)v246.m128i_i64[0];
LABEL_178:
                      LODWORD(v228) = v200;
                      if ( v200 > 0x40 )
                        sub_C43780((__int64)&v227, (const void **)&v199);
                      else
                        v227 = v199;
                      sub_C49E90((__int64)&v246, (__int64)&v227, (__int64)&v223);
                      if ( v200 > 0x40 && v199 )
                        j_j___libc_free_0_0(v199);
                      v199 = (void *)v246.m128i_i64[0];
                      v200 = v246.m128i_u32[2];
                      if ( (unsigned int)v228 > 0x40 && v227 )
                        j_j___libc_free_0_0(v227);
                      if ( v224 > 0x40 && v223 )
                        j_j___libc_free_0_0(v223);
                      goto LABEL_189;
                    }
                    v52 = v246.m128i_i64[0];
LABEL_174:
                    v53 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v48;
                    v154 = v48 == 0;
                    v54 = 0;
                    if ( !v154 )
                      v54 = v53;
                    v246.m128i_i64[0] = v54 & ~v52;
                    goto LABEL_177;
                  }
LABEL_172:
                  v52 = _RAX;
                  if ( (v51 & _RAX) == 0 )
                  {
                    v224 = v48;
                    v223 = (void *)_RAX;
                    goto LABEL_178;
                  }
                  v246.m128i_i32[2] = v48;
                  goto LABEL_174;
                }
LABEL_325:
                v109 = v206;
                v110 = 1LL << ((unsigned __int8)v206 - 1);
                if ( v206 <= 0x40 )
                {
                  v111 = _RAX;
                  if ( (v110 & _RAX) == 0 )
                  {
                    LODWORD(v228) = v206;
                    v227 = (void *)_RAX;
                    goto LABEL_328;
                  }
                  v246.m128i_i32[2] = v206;
                  goto LABEL_435;
                }
                if ( (*(_QWORD *)(_RAX + 8LL * ((v206 - 1) >> 6)) & v110) == 0 )
                {
                  LODWORD(v228) = v206;
                  sub_C43780((__int64)&v227, (const void **)&v205);
                  goto LABEL_328;
                }
                v246.m128i_i32[2] = v206;
                sub_C43780((__int64)&v246, (const void **)&v205);
                v109 = v246.m128i_i32[2];
                if ( v246.m128i_i32[2] <= 0x40u )
                {
                  v111 = v246.m128i_i64[0];
LABEL_435:
                  v121 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v109;
                  v154 = v109 == 0;
                  v122 = 0;
                  if ( !v154 )
                    v122 = v121;
                  v246.m128i_i64[0] = v122 & ~v111;
                  goto LABEL_438;
                }
                sub_C43D10((__int64)&v246);
LABEL_438:
                sub_C46250((__int64)&v246);
                LODWORD(v228) = v246.m128i_i32[2];
                v227 = (void *)v246.m128i_i64[0];
LABEL_328:
                if ( v200 > 0x40 && v199 )
                  j_j___libc_free_0_0(v199);
                v199 = v227;
                v200 = v228;
LABEL_189:
                sub_99D930((__int64)&v215, *(unsigned __int8 **)v47, 0, 1u, a1[3], *(_QWORD *)(v47 + 40), 0, 0);
                sub_9AC3E0((__int64)&v219, *(_QWORD *)v47, *a1, 0, a1[3], *(_QWORD *)(v47 + 40), v192, 1);
                sub_AAF050((__int64)&v227, (__int64)&v219, 1);
                sub_AB2160((__int64)&v246, (__int64)&v215, (__int64)&v227, 2u);
                if ( v216 > 0x40 && v215 )
                  j_j___libc_free_0_0(v215);
                v215 = (void *)v246.m128i_i64[0];
                v55 = v246.m128i_i32[2];
                v246.m128i_i32[2] = 0;
                v216 = v55;
                if ( v218 > 0x40 && v217 )
                {
                  j_j___libc_free_0_0(v217);
                  v217 = (__int64)v247;
                  v218 = (unsigned int)v248;
                  if ( v246.m128i_i32[2] > 0x40u && v246.m128i_i64[0] )
                    j_j___libc_free_0_0(v246.m128i_i64[0]);
                }
                else
                {
                  v217 = (__int64)v247;
                  v218 = (unsigned int)v248;
                }
                if ( (unsigned int)v230 > 0x40 && v229 )
                  j_j___libc_free_0_0(v229);
                if ( (unsigned int)v228 > 0x40 && v227 )
                  j_j___libc_free_0_0(v227);
                v224 = v216;
                if ( v216 > 0x40 )
                  sub_C43780((__int64)&v223, (const void **)&v215);
                else
                  v223 = v215;
                v226 = v218;
                if ( v218 > 0x40 )
                  sub_C43780((__int64)&v225, (const void **)&v217);
                else
                  v225 = v217;
                v37 = *(_DWORD *)(v47 + 16);
                if ( v37 )
                {
                  sub_AB4490((__int64)&v246, (__int64)&v223, v224 - v37);
                  if ( v224 > 0x40 && v223 )
                    j_j___libc_free_0_0(v223);
                  v223 = (void *)v246.m128i_i64[0];
                  v118 = v246.m128i_i32[2];
                  v246.m128i_i32[2] = 0;
                  v224 = v118;
                  if ( v226 > 0x40 && v225 )
                  {
                    j_j___libc_free_0_0(v225);
                    v225 = (__int64)v247;
                    v226 = (unsigned int)v248;
                    if ( v246.m128i_i32[2] > 0x40u && v246.m128i_i64[0] )
                      j_j___libc_free_0_0(v246.m128i_i64[0]);
                  }
                  else
                  {
                    v225 = (__int64)v247;
                    v226 = (unsigned int)v248;
                  }
                }
                if ( *(_BYTE *)(v47 + 20) && !sub_AB0760((__int64)&v223) )
                {
                  v115 = v224;
                  v210 = v224;
                  v116 = 1LL << ((unsigned __int8)v224 - 1);
                  if ( v224 <= 0x40 )
                  {
                    v209 = 0;
                    goto LABEL_353;
                  }
                  v165 = 1LL << ((unsigned __int8)v224 - 1);
                  v167 = v224 - 1;
                  sub_C43690((__int64)&v209, 0, 0);
                  v116 = v165;
                  if ( v210 <= 0x40 )
                  {
                    v115 = v224;
LABEL_353:
                    v209 |= v116;
                  }
                  else
                  {
                    *(_QWORD *)(v209 + 8LL * (v167 >> 6)) |= v165;
                    v115 = v224;
                  }
                  v208 = v115;
                  if ( v115 > 0x40 )
                    sub_C43690((__int64)&v207, 0, 0);
                  else
                    v207 = 0;
                  sub_AADC30((__int64)&v227, (__int64)&v207, &v209);
                  sub_AB2160((__int64)&v246, (__int64)&v223, (__int64)&v227, 0);
                  if ( v224 > 0x40 && v223 )
                    j_j___libc_free_0_0(v223);
                  v223 = (void *)v246.m128i_i64[0];
                  v117 = v246.m128i_i32[2];
                  v246.m128i_i32[2] = 0;
                  v224 = v117;
                  if ( v226 > 0x40 && v225 )
                    j_j___libc_free_0_0(v225);
                  v225 = (__int64)v247;
                  v226 = (unsigned int)v248;
                  if ( v246.m128i_i32[2] > 0x40u && v246.m128i_i64[0] )
                    j_j___libc_free_0_0(v246.m128i_i64[0]);
                  if ( (unsigned int)v230 > 0x40 && v229 )
                    j_j___libc_free_0_0(v229);
                  if ( (unsigned int)v228 > 0x40 && v227 )
                    j_j___libc_free_0_0(v227);
                  if ( v208 > 0x40 && v207 )
                    j_j___libc_free_0_0(v207);
                  if ( v210 > 0x40 && v209 )
                    j_j___libc_free_0_0(v209);
                }
                v38 = *(_DWORD *)(v47 + 12);
                if ( v38 )
                {
                  sub_AB41D0((__int64)&v246, (__int64)&v223, v224 + v38);
                  if ( v224 > 0x40 && v223 )
                    j_j___libc_free_0_0(v223);
                  v223 = (void *)v246.m128i_i64[0];
                  v113 = v246.m128i_i32[2];
                  v246.m128i_i32[2] = 0;
                  v224 = v113;
                  if ( v226 > 0x40 && v225 )
                  {
                    j_j___libc_free_0_0(v225);
                    v225 = (__int64)v247;
                    v226 = (unsigned int)v248;
                    if ( v246.m128i_i32[2] > 0x40u && v246.m128i_i64[0] )
                      j_j___libc_free_0_0(v246.m128i_i64[0]);
                  }
                  else
                  {
                    v225 = (__int64)v247;
                    v226 = (unsigned int)v248;
                  }
                }
                v39 = *(_DWORD *)(v47 + 8);
                if ( !v39 )
                  goto LABEL_102;
                sub_AB3F90((__int64)&v246, (__int64)&v223, v224 + v39);
                if ( v224 > 0x40 && v223 )
                  j_j___libc_free_0_0(v223);
                v223 = (void *)v246.m128i_i64[0];
                v112 = v246.m128i_i32[2];
                v246.m128i_i32[2] = 0;
                v224 = v112;
                if ( v226 <= 0x40 || !v225 )
                {
                  v41 = (__int64)v247;
                  v40 = (unsigned int)v248;
                  v225 = (__int64)v247;
                  goto LABEL_103;
                }
                j_j___libc_free_0_0(v225);
                v41 = (__int64)v247;
                v40 = (unsigned int)v248;
                v225 = (__int64)v247;
                v226 = (unsigned int)v248;
                if ( v246.m128i_i32[2] > 0x40u && v246.m128i_i64[0] )
                {
                  j_j___libc_free_0_0(v246.m128i_i64[0]);
LABEL_102:
                  v40 = v226;
                  v41 = v225;
                }
LABEL_103:
                v42 = v224;
                LODWORD(v230) = v40;
                v224 = 0;
                LODWORD(v228) = v42;
                v229 = v41;
                v227 = v223;
                v226 = 0;
                sub_AB4E00((__int64)&v246, (__int64)&v227, v212);
                if ( v216 > 0x40 && v215 )
                  j_j___libc_free_0_0(v215);
                v215 = (void *)v246.m128i_i64[0];
                v43 = v246.m128i_i32[2];
                v246.m128i_i32[2] = 0;
                v216 = v43;
                if ( v218 > 0x40 && v217 )
                {
                  j_j___libc_free_0_0(v217);
                  v217 = (__int64)v247;
                  v218 = (unsigned int)v248;
                  if ( v246.m128i_i32[2] > 0x40u && v246.m128i_i64[0] )
                    j_j___libc_free_0_0(v246.m128i_i64[0]);
                }
                else
                {
                  v217 = (__int64)v247;
                  v218 = (unsigned int)v248;
                }
                if ( (unsigned int)v230 > 0x40 && v229 )
                  j_j___libc_free_0_0(v229);
                if ( (unsigned int)v228 > 0x40 && v227 )
                  j_j___libc_free_0_0(v227);
                if ( v226 > 0x40 && v225 )
                  j_j___libc_free_0_0(v225);
                if ( v224 > 0x40 && v223 )
                  j_j___libc_free_0_0(v223);
                v44 = *(_DWORD *)(v47 + 32);
                v154 = *(_BYTE *)(v47 + 48) == 0;
                v224 = v44;
                if ( v154 )
                {
                  if ( v44 > 0x40 )
                    sub_C43780((__int64)&v223, v177);
                  else
                    v223 = *(void **)(v47 + 24);
                  sub_AADBC0((__int64)&v227, (__int64 *)&v223);
                  sub_AB5C70((__int64)&v246, (__int64)&v215, (__int64)&v227);
                  if ( v216 <= 0x40 )
                    goto LABEL_129;
                }
                else
                {
                  if ( v44 > 0x40 )
                    sub_C43780((__int64)&v223, v177);
                  else
                    v223 = *(void **)(v47 + 24);
                  sub_AADBC0((__int64)&v227, (__int64 *)&v223);
                  sub_ABAB70((__int64)&v246, (__int64)&v215, (__int64)&v227);
                  if ( v216 <= 0x40 )
                    goto LABEL_129;
                }
                if ( v215 )
                  j_j___libc_free_0_0(v215);
LABEL_129:
                v215 = (void *)v246.m128i_i64[0];
                v45 = v246.m128i_i32[2];
                v246.m128i_i32[2] = 0;
                v216 = v45;
                if ( v218 > 0x40 && v217 )
                {
                  j_j___libc_free_0_0(v217);
                  v217 = (__int64)v247;
                  v218 = (unsigned int)v248;
                  if ( v246.m128i_i32[2] > 0x40u && v246.m128i_i64[0] )
                    j_j___libc_free_0_0(v246.m128i_i64[0]);
                }
                else
                {
                  v217 = (__int64)v247;
                  v218 = (unsigned int)v248;
                }
                if ( (unsigned int)v230 > 0x40 && v229 )
                  j_j___libc_free_0_0(v229);
                if ( (unsigned int)v228 > 0x40 && v227 )
                  j_j___libc_free_0_0(v227);
                if ( v224 > 0x40 && v223 )
                  j_j___libc_free_0_0(v223);
                if ( *(_BYTE *)(v47 + 49) )
                {
                  sub_AB51C0((__int64)&v246, (__int64)&v211, (__int64)&v215);
                  if ( v212 > 0x40 )
                    goto LABEL_145;
                }
                else
                {
                  sub_AB4F10((__int64)&v246, (__int64)&v211, (__int64)&v215);
                  if ( v212 > 0x40 )
                  {
LABEL_145:
                    if ( v211 )
                      j_j___libc_free_0_0(v211);
                  }
                }
                v211 = v246.m128i_i64[0];
                v46 = v246.m128i_i32[2];
                v246.m128i_i32[2] = 0;
                v212 = v46;
                if ( v214 > 0x40 && v213 )
                {
                  j_j___libc_free_0_0(v213);
                  v213 = v247;
                  v214 = (unsigned int)v248;
                  if ( v246.m128i_i32[2] > 0x40u && v246.m128i_i64[0] )
                    j_j___libc_free_0_0(v246.m128i_i64[0]);
                }
                else
                {
                  v213 = v247;
                  v214 = (unsigned int)v248;
                }
                if ( v222 > 0x40 && v221 )
                  j_j___libc_free_0_0(v221);
                if ( v220 > 0x40 && v219 )
                  j_j___libc_free_0_0(v219);
                if ( v218 > 0x40 && v217 )
                  j_j___libc_free_0_0(v217);
                if ( v216 > 0x40 && v215 )
                  j_j___libc_free_0_0(v215);
                if ( v206 > 0x40 && v205 )
                  j_j___libc_free_0_0(v205);
                ++v183;
                v170 += 56;
                if ( v183 == v169 )
                  goto LABEL_444;
              }
              _RAX = *(_QWORD *)(v47 + 24);
              v50 = v48;
              v205 = (_QWORD *)_RAX;
              if ( *(_BYTE *)(v47 + 48) )
              {
                if ( v183 )
                {
                  v51 = 1LL << ((unsigned __int8)v48 - 1);
                  goto LABEL_172;
                }
                goto LABEL_325;
              }
              goto LABEL_481;
            }
LABEL_444:
            sub_C4B8A0((__int64)&v201, (__int64)&v233.m128i_i64[1], (__int64)&v199);
            if ( v202 > 0x40 )
              v127 = v201[(v202 - 1) >> 6];
            else
              v127 = (unsigned __int64)v201;
            if ( (v127 & (1LL << ((unsigned __int8)v202 - 1))) != 0 )
              sub_C45EE0((__int64)&v201, (__int64 *)&v199);
            LOBYTE(v220) = 0;
            v178 = v179 & 0x3FFFFFFFFFFFFFFFLL;
            v219 = (_QWORD *)(v179 & 0x3FFFFFFFFFFFFFFFLL);
            v185 = sub_CA1930(&v219);
            v180 = v202;
            if ( v202 > 0x40 )
            {
              if ( v180 - (unsigned int)sub_C444A0((__int64)&v201) > 0x40 )
              {
LABEL_451:
                sub_9865C0((__int64)&v223, (__int64)&v199);
                sub_C46B40((__int64)&v223, (__int64 *)&v201);
                v129 = v224;
                v224 = 0;
                v246.m128i_i8[8] = 0;
                LODWORD(v228) = v129;
                v227 = v223;
                v246.m128i_i64[0] = v36 & 0x3FFFFFFFFFFFFFFFLL;
                v12 = sub_CA1930(&v246);
                if ( !sub_986EE0((__int64)&v227, v12) )
                {
                  sub_969240((__int64 *)&v227);
                  v20 = 0;
                  sub_969240((__int64 *)&v223);
                  goto LABEL_468;
                }
                sub_969240((__int64 *)&v227);
                sub_969240((__int64 *)&v223);
LABEL_453:
                v130 = v212;
                LOBYTE(v220) = 0;
                v176 = (void *)(v36 & 0x3FFFFFFFFFFFFFFFLL);
                v219 = (_QWORD *)(v36 & 0x3FFFFFFFFFFFFFFFLL);
                v131 = sub_CA1930(&v219);
                sub_9691E0((__int64)&v227, v130, v131, 0, 0);
                sub_9691E0((__int64)&v215, v130, 0, 0, 0);
                sub_AADC30((__int64)&v246, (__int64)&v215, (__int64 *)&v227);
                sub_AB4F10((__int64)&v223, (__int64)&v211, (__int64)&v246);
                sub_969240((__int64 *)&v247);
                sub_969240(v246.m128i_i64);
                sub_969240((__int64 *)&v215);
                sub_969240((__int64 *)&v227);
                LOBYTE(v220) = 0;
                v219 = (_QWORD *)v178;
                v132 = sub_CA1930(&v219);
                sub_9691E0((__int64)&v246, v130, v132, 0, 0);
                sub_9691E0((__int64)&v215, v130, 0, 0, 0);
                sub_AADC30((__int64)&v227, (__int64)&v215, v246.m128i_i64);
                sub_969240((__int64 *)&v215);
                sub_969240(v246.m128i_i64);
                v12 = (unsigned __int64)&v223;
                sub_AB2160((__int64)&v246, (__int64)&v223, (__int64)&v227, 0);
                v20 = 0;
                LOBYTE(v130) = sub_AAF7D0((__int64)&v246);
                sub_969240((__int64 *)&v247);
                sub_969240(v246.m128i_i64);
                if ( (_BYTE)v130 )
                {
LABEL_467:
                  sub_969240(&v229);
                  sub_969240((__int64 *)&v227);
                  sub_969240(&v225);
                  sub_969240((__int64 *)&v223);
LABEL_468:
                  if ( v202 > 0x40 && v201 )
                    j_j___libc_free_0_0(v201);
                  if ( v214 > 0x40 && v213 )
                    j_j___libc_free_0_0(v213);
                  if ( v212 > 0x40 && v211 )
                    j_j___libc_free_0_0(v211);
                  if ( v200 <= 0x40 )
                    goto LABEL_27;
                  v120 = (__int64)v199;
                  if ( !v199 )
                    goto LABEL_27;
                  goto LABEL_424;
                }
                v133 = (const void **)&v219;
                for ( i = 6; i; --i )
                {
                  *(_DWORD *)v133 = 0;
                  v133 = (const void **)((char *)v133 + 4);
                }
                if ( v236 == 1 )
                {
                  v160 = v235;
                  if ( *(_DWORD *)(v235 + 16) )
                    goto LABEL_459;
                  v161 = *(_QWORD *)(v235 + 40);
                  v162 = a1[3];
                  v163 = *a1;
                  v247 = 0;
                  v249 = v162;
                  v248 = (__int64 *)v192;
                  v246 = (__m128i)(unsigned __int64)v163;
                  v250 = v161;
                  v251 = 0;
                  v252 = 0;
                  v253 = 257;
                  if ( !(unsigned __int8)sub_9B6260(*(_QWORD *)v235, &v246, 0) )
                    goto LABEL_459;
                  v164 = (__int64 *)(v160 + 24);
                  if ( !(unsigned __int8)sub_D015A0(v160) )
                    goto LABEL_459;
                }
                else
                {
                  if ( v236 != 2 )
                    goto LABEL_459;
                  v172 = v235;
                  v181 = (__int64 *)(v235 + 56);
                  v143 = sub_D016D0(v235, v235 + 56);
                  v144 = (__int64 *)v172;
                  if ( !v143 || *(_DWORD *)(v172 + 16) || !sub_D00280(v172, (__int64)v181) || *(_BYTE *)(a8 + 512) )
                    goto LABEL_524;
                  v145 = v144[5];
                  if ( !v145 )
                    v145 = v144[12];
                  v146 = a1[3];
                  v147 = *a1;
                  v247 = 0;
                  v246 = (__m128i)(unsigned __int64)v147;
                  v249 = v146;
                  v248 = (__int64 *)v192;
                  v250 = v145;
                  v251 = 0;
                  v252 = 0;
                  v253 = 257;
                  v173 = v144;
                  v148 = sub_9B6040(*v144, v144[7], &v246, 0);
                  v144 = v173;
                  if ( !v148 )
                  {
LABEL_524:
                    v174 = (__int64)v144;
                    if ( sub_D016D0((__int64)v144, (__int64)v181) )
                      goto LABEL_459;
                    v149 = *(_DWORD *)(v174 + 32);
                    v150 = *(_QWORD *)(v174 + 24);
                    v151 = v149 - 1;
                    v152 = 1LL << ((unsigned __int8)v149 - 1);
                    if ( *(_BYTE *)(v174 + 49) == *(_BYTE *)(v174 + 105) )
                    {
                      if ( v149 > 0x40 )
                        v150 = *(_QWORD *)(v150 + 8LL * (v151 >> 6));
                      v157 = *(_DWORD *)(v174 + 88);
                      v158 = (v150 & v152) == 0;
                      v159 = *(_QWORD *)(v174 + 80);
                      if ( v157 > 0x40 )
                        v159 = *(_QWORD *)(v159 + 8LL * ((v157 - 1) >> 6));
                      if ( ((v159 & (1LL << ((unsigned __int8)v157 - 1))) != 0) != !v158 )
                        goto LABEL_531;
                    }
                    else
                    {
                      if ( v149 > 0x40 )
                        v150 = *(_QWORD *)(v150 + 8LL * (v151 >> 6));
                      v153 = *(_DWORD *)(v174 + 88);
                      v154 = (v150 & v152) == 0;
                      v155 = *(_QWORD *)(v174 + 80);
                      if ( v153 > 0x40 )
                        v155 = *(_QWORD *)(v155 + 8LL * ((v153 - 1) >> 6));
                      if ( ((v155 & (1LL << ((unsigned __int8)v153 - 1))) != 0) == !v154 )
                      {
LABEL_531:
                        if ( !*(_DWORD *)(v174 + 16) && sub_D00280(v174, (__int64)v181) && !*(_BYTE *)(a8 + 512) )
                        {
                          v175 = v156;
                          v171 = v156 + 10;
                          sub_9692E0((__int64)&v246, v156 + 10);
                          sub_9692E0((__int64)&v215, v175 + 3);
                          sub_C49E90((__int64)&v207, (__int64)&v215, (__int64)&v246);
                          sub_969240((__int64 *)&v215);
                          sub_969240(v246.m128i_i64);
                          sub_9692E0((__int64)&v246, v175 + 3);
                          sub_C4A1D0((__int64)&v209, (__int64)&v246, (__int64)&v207);
                          sub_969240(v246.m128i_i64);
                          sub_9692E0((__int64)&v246, v171);
                          sub_C4A1D0((__int64)&v215, (__int64)&v246, (__int64)&v207);
                          sub_969240(v246.m128i_i64);
                          if ( (unsigned __int8)sub_D01B00(v175, (__int64)&v209, v181, (__int64)&v215, *a1, a1[3], v192) )
                          {
                            if ( (_BYTE)v221 )
                            {
                              if ( v220 <= 0x40 && v208 <= 0x40 )
                              {
                                v220 = v208;
                                v219 = v207;
                              }
                              else
                              {
                                sub_C43990((__int64)&v219, (__int64)&v207);
                              }
                            }
                            else
                            {
                              v220 = v208;
                              if ( v208 > 0x40 )
                                sub_C43780((__int64)&v219, (const void **)&v207);
                              else
                                v219 = v207;
                              LOBYTE(v221) = 1;
                            }
                          }
                          sub_969240((__int64 *)&v215);
                          sub_969240(&v209);
                          sub_969240((__int64 *)&v207);
                        }
                      }
                    }
LABEL_459:
                    if ( (_BYTE)v221 )
                    {
                      sub_9865C0((__int64)&v246, (__int64)&v233.m128i_i64[1]);
                      sub_C46B40((__int64)&v246, (__int64 *)&v219);
                      v204 = v246.m128i_u32[2];
                      v203 = v246.m128i_i64[0];
                      sub_9865C0((__int64)&v246, (__int64)&v233.m128i_i64[1]);
                      sub_C45EE0((__int64)&v246, (__int64 *)&v219);
                      v206 = v246.m128i_u32[2];
                      v205 = (_QWORD *)v246.m128i_i64[0];
                      if ( v204 > 0x40 )
                        v135 = *(_QWORD *)(v203 + 8LL * ((v204 - 1) >> 6));
                      else
                        v135 = v203;
                      if ( (v135 & (1LL << ((unsigned __int8)v204 - 1))) != 0 )
                      {
                        sub_9865C0((__int64)&v207, (__int64)&v203);
                        sub_AADAA0((__int64)&v209, (__int64)&v207, v138, v139, v140);
                        LOBYTE(v216) = 0;
                        v215 = v176;
                        v141 = sub_CA1930(&v215);
                        if ( !sub_986EE0((__int64)&v209, v141) )
                        {
                          v142 = (unsigned __int64)v205;
                          if ( v206 > 0x40 )
                            v142 = v205[(v206 - 1) >> 6];
                          if ( (v142 & (1LL << ((unsigned __int8)v206 - 1))) == 0 )
                          {
                            v246.m128i_i8[8] = 0;
                            v246.m128i_i64[0] = v178;
                            v12 = sub_CA1930(&v246);
                            if ( !sub_986EE0((__int64)&v205, v12) )
                            {
                              sub_969240(&v209);
                              v20 = 0;
                              sub_969240((__int64 *)&v207);
                              sub_969240((__int64 *)&v205);
                              sub_969240(&v203);
LABEL_465:
                              if ( (_BYTE)v221 )
                              {
                                LOBYTE(v221) = 0;
                                sub_969240((__int64 *)&v219);
                              }
                              goto LABEL_467;
                            }
                          }
                        }
                        sub_969240(&v209);
                        sub_969240((__int64 *)&v207);
                      }
                      sub_969240((__int64 *)&v205);
                      sub_969240(&v203);
                    }
                    v12 = (unsigned __int64)&v233;
                    v20 = (unsigned __int8)sub_D049D0((__int64)a1, (__int64)&v233, v198, v197, a1[3], v192, a8) == 0;
                    goto LABEL_465;
                  }
                  v164 = v173 + 3;
                }
                sub_9692E0((__int64)&v246, v164);
                sub_D00D90((__int64 *)&v219, v246.m128i_i64);
                sub_969240(v246.m128i_i64);
                goto LABEL_459;
              }
              v128 = (_QWORD *)*v201;
            }
            else
            {
              v128 = v201;
            }
            if ( v185 > (unsigned __int64)v128 )
              goto LABEL_453;
            goto LABEL_451;
          }
LABEL_195:
          v20 = 1;
          goto LABEL_27;
        }
        v246.m128i_i8[8] = 0;
        v246.m128i_i64[0] = v197 & 0x3FFFFFFFFFFFFFFFLL;
        v123 = sub_CA1930(&v246);
        v124 = v234;
        v125 = v123;
        if ( v234 > 0x40 )
        {
          if ( v124 - (unsigned int)sub_C444A0((__int64)&v233.m128i_i64[1]) > 0x40 )
            goto LABEL_324;
          v126 = *(_QWORD *)v233.m128i_i64[1];
        }
        else
        {
          v126 = v233.m128i_u64[1];
        }
        if ( v125 > v126 )
        {
          v36 = v198;
          if ( (v198 & 0x4000000000000000LL) != 0 )
            goto LABEL_195;
          goto LABEL_75;
        }
        goto LABEL_324;
      }
LABEL_264:
      v89 = (unsigned __int8 **)v235;
      v12 = *(unsigned int *)(v235 + 16);
      if ( (_DWORD)v12 )
        goto LABEL_69;
      v90 = v234;
      if ( !(v234 <= 0x40 ? v233.m128i_i64[1] == 0 : v90 == (unsigned int)sub_C444A0((__int64)&v233.m128i_i64[1])) )
        goto LABEL_69;
      if ( (unsigned __int8)sub_D033B0(*v89) )
      {
        v92 = v235 + 24;
        if ( *(_BYTE *)(v235 + 49) )
        {
          sub_9865C0((__int64)&v246, v92);
          sub_AADAA0((__int64)&v219, (__int64)&v246, v93, v94, v95);
          sub_969240(v246.m128i_i64);
        }
        else
        {
          sub_9865C0((__int64)&v219, v92);
        }
        v96 = (unsigned __int64)v219;
        v12 = 1LL << ((unsigned __int8)v220 - 1);
        if ( v220 > 0x40 )
          v96 = v219[(v220 - 1) >> 6];
        v97 = &v198;
        if ( (v96 & v12) == 0 )
          v97 = &v197;
        v98 = *v97;
        LOBYTE(v215) = *(_BYTE *)(v235 + 48) ^ 1;
        if ( !(_BYTE)v215 )
          goto LABEL_277;
        sub_988CD0((__int64)&v246, a1[1], v220);
        sub_AB13A0((__int64)&v223, (__int64)&v246);
        v12 = (unsigned __int64)&v223;
        sub_C4A7C0((__int64)&v227, (__int64)&v223, (__int64)&v219, (bool *)&v215);
        sub_969240((__int64 *)&v227);
        sub_969240((__int64 *)&v223);
        sub_969240((__int64 *)&v247);
        sub_969240(v246.m128i_i64);
        if ( !(_BYTE)v215 )
        {
LABEL_277:
          if ( v98 != 0xBFFFFFFFFFFFFFFELL && v98 != -1 )
          {
            sub_9692E0((__int64)&v246, (__int64 *)&v219);
            v12 = v98 & 0x3FFFFFFFFFFFFFFFLL;
            if ( !sub_986EE0((__int64)&v246, v98 & 0x3FFFFFFFFFFFFFFFLL) )
            {
              sub_969240(v246.m128i_i64);
              v20 = 0;
              sub_969240((__int64 *)&v219);
              goto LABEL_27;
            }
            sub_969240(v246.m128i_i64);
          }
        }
        if ( v220 > 0x40 && v219 )
          j_j___libc_free_0_0(v219);
      }
      v100 = v236;
LABEL_303:
      if ( !v100 )
        goto LABEL_73;
      goto LABEL_69;
    }
    v12 = v234;
    v68 = v197;
    v190 = v198;
    v69 = v233.m128i_i64[1];
    v70 = 1LL << ((unsigned __int8)v234 - 1);
    if ( v234 > 0x40 )
    {
      v71 = *(_QWORD *)(v233.m128i_i64[1] + 8LL * ((v234 - 1) >> 6)) & v70;
      if ( !v71 )
        goto LABEL_222;
      v246.m128i_i32[2] = v234;
      sub_C43780((__int64)&v246, (const void **)&v233.m128i_i64[1]);
      v12 = v246.m128i_u32[2];
      if ( v246.m128i_i32[2] > 0x40u )
      {
        sub_C43D10((__int64)&v246);
LABEL_258:
        sub_C46250((__int64)&v246);
        v88 = v246.m128i_i32[2];
        v246.m128i_i32[2] = 0;
        if ( v234 > 0x40 && v233.m128i_i64[1] )
        {
          v182 = v246.m128i_i64[0];
          v184 = v88;
          j_j___libc_free_0_0(v233.m128i_i64[1]);
          v233.m128i_i64[1] = v182;
          v234 = v184;
          if ( v246.m128i_i32[2] > 0x40u && v246.m128i_i64[0] )
            j_j___libc_free_0_0(v246.m128i_i64[0]);
        }
        else
        {
          v233.m128i_i64[1] = v246.m128i_i64[0];
          v234 = v88;
        }
        v68 = v198;
        v190 = v197;
LABEL_222:
        if ( v68 == -1 || v68 == 0xBFFFFFFFFFFFFFFELL )
          goto LABEL_195;
        v72 = (void *)(v68 & 0x3FFFFFFFFFFFFFFFLL);
        v73 = (v68 >> 62) & 1;
        v215 = v72;
        LOBYTE(v216) = v73;
        if ( (_BYTE)v73 )
        {
          sub_988CD0((__int64)&v246, a1[1], v234);
          sub_AB0910((__int64)&v223, (__int64)&v246);
          LODWORD(v228) = v234;
          if ( v234 > 0x40 )
            sub_C43690((__int64)&v227, (__int64)v215, 0);
          else
            v227 = v215;
          v12 = (unsigned __int64)&v223;
          sub_C49BE0((__int64)&v219, (__int64)&v223, (__int64)&v227, (bool *)&v211);
          if ( (unsigned int)v228 > 0x40 && v227 )
            j_j___libc_free_0_0(v227);
          if ( v224 > 0x40 && v223 )
            j_j___libc_free_0_0(v223);
          v99 = v220;
          if ( !(_BYTE)v211 )
          {
            v12 = (unsigned __int64)&v219;
            if ( (int)sub_C49970((__int64)&v233.m128i_i64[1], (unsigned __int64 *)&v219) >= 0 )
            {
              v20 = 0;
              if ( v99 > 0x40 && v219 )
                j_j___libc_free_0_0(v219);
              if ( (unsigned int)v248 > 0x40 && v247 )
                j_j___libc_free_0_0(v247);
              if ( v246.m128i_i32[2] <= 0x40u )
                goto LABEL_27;
              v120 = v246.m128i_i64[0];
              if ( !v246.m128i_i64[0] )
                goto LABEL_27;
LABEL_424:
              j_j___libc_free_0_0(v120);
              goto LABEL_27;
            }
          }
          if ( v99 > 0x40 && v219 )
            j_j___libc_free_0_0(v219);
          if ( (unsigned int)v248 > 0x40 && v247 )
            j_j___libc_free_0_0(v247);
          if ( v246.m128i_i32[2] > 0x40u && v246.m128i_i64[0] )
            j_j___libc_free_0_0(v246.m128i_i64[0]);
          v100 = v236;
          if ( v236 != 1 )
            goto LABEL_303;
          goto LABEL_264;
        }
        v74 = sub_CA1930(&v215);
        v75 = v234;
        v76 = v74;
        if ( v234 > 0x40 )
        {
          v196 = v234;
          v114 = sub_C444A0((__int64)&v233.m128i_i64[1]);
          v75 = v196;
          if ( v196 - v114 > 0x40 )
            goto LABEL_324;
          v77 = *(_QWORD *)v233.m128i_i64[1];
        }
        else
        {
          v77 = v233.m128i_u64[1];
        }
        if ( v76 <= v77 )
        {
LABEL_324:
          v20 = 0;
          goto LABEL_27;
        }
        v12 = v190;
        v78 = v190 != -1 && v190 != 0xBFFFFFFFFFFFFFFELL;
        if ( !v78 )
          goto LABEL_407;
        if ( (v190 & 0x4000000000000000LL) != 0 )
          goto LABEL_407;
        v186 = v75;
        if ( v75 > 0x40 )
        {
          v193 = v77;
          v79 = sub_C444A0((__int64)&v233.m128i_i64[1]);
          v77 = v193;
          if ( v186 - v79 > 0x40 )
            goto LABEL_407;
        }
        if ( v77 > 0x7FFFFFFF )
          goto LABEL_407;
        LOBYTE(v224) = 0;
        v223 = (void *)(v190 & 0x3FFFFFFFFFFFFFFFLL);
        v194 = sub_CA1930(&v223);
        sub_9865C0((__int64)&v227, (__int64)&v233.m128i_i64[1]);
        v12 = v194;
        sub_C46A40((__int64)&v227, v194);
        v80 = v228;
        v81 = (unsigned __int64 *)v227;
        LODWORD(v228) = 0;
        v195 = v80;
        v246.m128i_i32[2] = v80;
        v246.m128i_i64[0] = (__int64)v227;
        v82 = sub_CA1930(&v215);
        if ( v195 <= 0x40 )
        {
          if ( v82 < (unsigned __int64)v81 )
          {
            if ( (unsigned int)v228 <= 0x40 )
              goto LABEL_407;
            v84 = v227;
            v83 = 0;
            if ( !v227 )
              goto LABEL_407;
LABEL_241:
            j_j___libc_free_0_0(v84);
            goto LABEL_242;
          }
          if ( (unsigned int)v228 <= 0x40 )
          {
LABEL_243:
            v85 = v234;
            if ( v234 > 0x40 )
            {
              v86 = *(_QWORD *)v233.m128i_i64[1];
            }
            else
            {
              if ( !v234 )
                goto LABEL_489;
              v86 = v233.m128i_i64[1] << (64 - (unsigned __int8)v234) >> (64 - (unsigned __int8)v234);
            }
            v85 = -(int)v86;
            v87 = 0;
            if ( (unsigned __int64)(v85 + 0x400000LL) > 0x7FFFFF )
            {
LABEL_247:
              LODWORD(v73) = (unsigned __int8)v73;
              if ( v71 && (_BYTE)v73 )
              {
                if ( v87 == -4194304 )
                {
                  v87 = -4194304;
                }
                else
                {
                  LODWORD(v73) = (unsigned __int8)v73;
                  v87 = (-512 * v87) >> 9;
                }
              }
              goto LABEL_251;
            }
LABEL_489:
            LOBYTE(v73) = v78;
            v87 = v85 << 9 >> 9;
            goto LABEL_247;
          }
          v83 = v190 != -1 && v190 != 0xBFFFFFFFFFFFFFFELL;
        }
        else
        {
          v191 = v82;
          v83 = 0;
          if ( v195 - (unsigned int)sub_C444A0((__int64)&v246) <= 0x40 && v191 >= *v81 )
            v83 = v78;
          if ( v246.m128i_i64[0] )
            j_j___libc_free_0_0(v246.m128i_i64[0]);
          if ( (unsigned int)v228 <= 0x40 )
          {
LABEL_242:
            if ( v83 )
              goto LABEL_243;
LABEL_407:
            v87 = 0;
            LODWORD(v73) = 0;
LABEL_251:
            v20 = (v87 << 9) | ((_DWORD)v73 << 8) | 2;
            goto LABEL_27;
          }
        }
        v84 = v227;
        if ( !v227 )
          goto LABEL_242;
        goto LABEL_241;
      }
      v69 = v246.m128i_i64[0];
    }
    else
    {
      v71 = v233.m128i_i64[1] & v70;
      if ( !v71 )
        goto LABEL_222;
      v246.m128i_i32[2] = v234;
    }
    v101 = ~v69;
    v102 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v12;
    if ( !(_DWORD)v12 )
      v102 = 0;
    v246.m128i_i64[0] = v102 & v101;
    goto LABEL_258;
  }
LABEL_27:
  v23 = v242;
  v24 = v242 + 56LL * (unsigned int)v243;
  if ( v242 != v24 )
  {
    do
    {
      v24 -= 56;
      if ( *(_DWORD *)(v24 + 32) > 0x40u )
      {
        v25 = *(_QWORD *)(v24 + 24);
        if ( v25 )
          j_j___libc_free_0_0(v25);
      }
    }
    while ( v23 != v24 );
    v24 = v242;
  }
  if ( (__int64 *)v24 != &v244 )
    _libc_free(v24, v12);
  if ( (unsigned int)v241 > 0x40 && v240 )
    j_j___libc_free_0_0(v240);
  v26 = v235;
  v27 = v235 + 56LL * v236;
  if ( v235 != v27 )
  {
    do
    {
      v27 -= 56;
      if ( *(_DWORD *)(v27 + 32) > 0x40u )
      {
        v28 = *(_QWORD *)(v27 + 24);
        if ( v28 )
          j_j___libc_free_0_0(v28);
      }
    }
    while ( v26 != v27 );
    v27 = v235;
  }
  if ( (char *)v27 != &v237 )
    _libc_free(v27, v12);
  if ( v234 > 0x40 && v233.m128i_i64[1] )
    j_j___libc_free_0_0(v233.m128i_i64[1]);
  return v20;
}
