// Function: sub_2DE22E0
// Address: 0x2de22e0
//
__int64 __fastcall sub_2DE22E0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 *v5; // rbx
  __int64 *v6; // r15
  unsigned int v7; // r14d
  __int64 v8; // rsi
  unsigned int *v9; // rax
  char v10; // r9
  __int64 v11; // r13
  char v12; // r8
  __int64 v13; // r9
  __int64 v14; // r13
  unsigned int *v15; // rbx
  _QWORD *v16; // r15
  __m128i v17; // xmm1
  __m128i v18; // xmm0
  __m128i v19; // xmm2
  __m128i v20; // xmm3
  __m128i v21; // xmm4
  __m128i v22; // xmm5
  _QWORD *v23; // r15
  __int64 v24; // rax
  _QWORD *v25; // rax
  __int64 v26; // rbx
  __int64 v27; // r9
  __int64 v28; // rcx
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __int64 v31; // r8
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rcx
  char *v34; // r15
  char v35; // r10
  unsigned __int64 v36; // rax
  int v37; // edx
  __int64 v38; // rsi
  __int64 v39; // rax
  char v40; // al
  char v41; // r10
  char v42; // al
  unsigned int v43; // esi
  __int64 v44; // rax
  __int64 v45; // rcx
  unsigned __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rdx
  __int64 v49; // rbx
  __int64 v50; // rbx
  char v51; // dh
  __int64 v52; // r15
  char v53; // al
  __int64 v54; // rax
  __int64 v55; // rdi
  __int64 v56; // rsi
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // r15
  unsigned int *v60; // rax
  int v61; // ecx
  unsigned int *v62; // rdx
  __int64 v63; // rax
  __int64 v64; // r15
  int v65; // edx
  int v66; // edx
  unsigned int v67; // ecx
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // rcx
  int v71; // edx
  int v72; // edx
  unsigned int v73; // ecx
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // rcx
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // rcx
  __int64 v80; // r14
  _QWORD *v81; // r15
  char *v82; // r14
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // rsi
  _QWORD *v86; // rax
  _QWORD *v87; // rdx
  __int64 *v88; // rbx
  __int64 *v89; // r13
  __int64 v90; // rdi
  __int64 v92; // rax
  unsigned __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // r9
  __int64 v96; // r8
  unsigned __int64 v97; // rax
  int v98; // edx
  __int64 v99; // rax
  bool v100; // cf
  __int64 v101; // rdx
  char v102; // al
  char v103; // cl
  __int64 v104; // rax
  unsigned __int64 v105; // r9
  __int64 v106; // rax
  char *v107; // rsi
  char *v108; // rcx
  __int64 v109; // rdx
  unsigned int v110; // r10d
  __int64 v111; // rdi
  int v112; // eax
  char v113; // r8
  unsigned int v114; // r8d
  int v115; // eax
  bool v116; // al
  int v117; // eax
  bool v118; // al
  _QWORD **v119; // rdx
  int v120; // ecx
  __int32 v121; // eax
  __int64 *v122; // rax
  __int64 v123; // rsi
  unsigned int *v124; // r14
  unsigned int *v125; // rbx
  __int64 v126; // rdx
  unsigned int v127; // esi
  __int64 v128; // rax
  __int64 v129; // rdx
  __int64 v130; // rdx
  _QWORD *v131; // rax
  _QWORD *v132; // rdx
  int v133; // eax
  bool v134; // al
  unsigned int v135; // [rsp+Ch] [rbp-524h]
  __int64 v136; // [rsp+10h] [rbp-520h]
  unsigned int v137; // [rsp+10h] [rbp-520h]
  char v138; // [rsp+10h] [rbp-520h]
  char *v139; // [rsp+18h] [rbp-518h]
  char *v140; // [rsp+18h] [rbp-518h]
  unsigned int v141; // [rsp+18h] [rbp-518h]
  __int64 v142; // [rsp+20h] [rbp-510h]
  unsigned __int64 v143; // [rsp+20h] [rbp-510h]
  unsigned __int64 v144; // [rsp+20h] [rbp-510h]
  __int64 v145; // [rsp+20h] [rbp-510h]
  unsigned int v146; // [rsp+20h] [rbp-510h]
  __int64 *v147; // [rsp+28h] [rbp-508h]
  char v148; // [rsp+28h] [rbp-508h]
  char *v149; // [rsp+28h] [rbp-508h]
  char *v150; // [rsp+28h] [rbp-508h]
  __int64 v151; // [rsp+30h] [rbp-500h]
  __int64 v152; // [rsp+30h] [rbp-500h]
  __int64 v153; // [rsp+40h] [rbp-4F0h]
  __int64 v154; // [rsp+40h] [rbp-4F0h]
  char v155; // [rsp+48h] [rbp-4E8h]
  __int64 v156; // [rsp+48h] [rbp-4E8h]
  char v157; // [rsp+50h] [rbp-4E0h]
  __int64 v158; // [rsp+50h] [rbp-4E0h]
  __int64 v159; // [rsp+50h] [rbp-4E0h]
  __int64 v160; // [rsp+50h] [rbp-4E0h]
  __int64 v161; // [rsp+50h] [rbp-4E0h]
  unsigned __int64 v162; // [rsp+50h] [rbp-4E0h]
  unsigned __int64 v163; // [rsp+50h] [rbp-4E0h]
  __int64 v164; // [rsp+58h] [rbp-4D8h]
  __int64 v165; // [rsp+58h] [rbp-4D8h]
  char v166; // [rsp+58h] [rbp-4D8h]
  char v167; // [rsp+60h] [rbp-4D0h]
  char v168; // [rsp+60h] [rbp-4D0h]
  __int64 v169; // [rsp+60h] [rbp-4D0h]
  int v170; // [rsp+60h] [rbp-4D0h]
  __int64 v171; // [rsp+68h] [rbp-4C8h]
  char v172; // [rsp+68h] [rbp-4C8h]
  __int64 v173; // [rsp+68h] [rbp-4C8h]
  __int64 v174; // [rsp+70h] [rbp-4C0h]
  void (__fastcall *v176)(_QWORD **, _QWORD **, __int64); // [rsp+78h] [rbp-4B8h]
  _QWORD *v177; // [rsp+98h] [rbp-498h] BYREF
  __m128i v178; // [rsp+A0h] [rbp-490h] BYREF
  void (__fastcall *v179)(__m128i *, __m128i *, __int64); // [rsp+B0h] [rbp-480h]
  char (__fastcall *v180)(__int64 *, __int64 *); // [rsp+B8h] [rbp-478h]
  char *v181; // [rsp+C0h] [rbp-470h] BYREF
  __m128i v182; // [rsp+C8h] [rbp-468h] BYREF
  __int64 (__fastcall *v183)(char *, __m128i *, int); // [rsp+D8h] [rbp-458h]
  char (__fastcall *v184)(__int64 *, __int64 *); // [rsp+E0h] [rbp-450h]
  __int64 v185[4]; // [rsp+F0h] [rbp-440h] BYREF
  __int64 v186; // [rsp+110h] [rbp-420h]
  __int64 v187; // [rsp+118h] [rbp-418h]
  char v188; // [rsp+120h] [rbp-410h]
  char v189; // [rsp+121h] [rbp-40Fh]
  char v190; // [rsp+122h] [rbp-40Eh]
  _QWORD *v191; // [rsp+130h] [rbp-400h] BYREF
  __int64 v192; // [rsp+138h] [rbp-3F8h]
  void (__fastcall *v193)(_QWORD **, _QWORD **, __int64); // [rsp+140h] [rbp-3F0h] BYREF
  __m128i v194; // [rsp+148h] [rbp-3E8h] BYREF
  __m128i v195; // [rsp+158h] [rbp-3D8h] BYREF
  __m128i v196; // [rsp+168h] [rbp-3C8h] BYREF
  __m128i v197; // [rsp+178h] [rbp-3B8h] BYREF
  __int64 v198; // [rsp+188h] [rbp-3A8h]
  unsigned int *v199; // [rsp+190h] [rbp-3A0h] BYREF
  __int64 v200; // [rsp+198h] [rbp-398h]
  const char *v201; // [rsp+1A0h] [rbp-390h] BYREF
  char v202; // [rsp+1A8h] [rbp-388h]
  __int64 v203; // [rsp+1B0h] [rbp-380h]
  __int64 v204; // [rsp+1B8h] [rbp-378h]
  __int64 v205; // [rsp+1C0h] [rbp-370h]
  __int64 v206; // [rsp+1C8h] [rbp-368h]
  __int64 v207; // [rsp+1D0h] [rbp-360h]
  __int64 v208; // [rsp+1D8h] [rbp-358h]
  void **v209; // [rsp+1E0h] [rbp-350h]
  _QWORD *v210; // [rsp+1E8h] [rbp-348h]
  __int64 v211; // [rsp+1F0h] [rbp-340h]
  __int64 v212; // [rsp+1F8h] [rbp-338h]
  __int64 v213; // [rsp+200h] [rbp-330h]
  __int64 v214; // [rsp+208h] [rbp-328h]
  void *v215; // [rsp+210h] [rbp-320h] BYREF
  _QWORD v216[2]; // [rsp+218h] [rbp-318h] BYREF
  int v217; // [rsp+228h] [rbp-308h]
  char v218; // [rsp+22Ch] [rbp-304h]
  char v219; // [rsp+230h] [rbp-300h] BYREF
  __int64 v220; // [rsp+2B0h] [rbp-280h]
  __int64 v221; // [rsp+2B8h] [rbp-278h]
  __int64 v222; // [rsp+2C0h] [rbp-270h]
  int v223; // [rsp+2C8h] [rbp-268h]
  char *v224; // [rsp+2D0h] [rbp-260h]
  __int64 v225; // [rsp+2D8h] [rbp-258h]
  char v226; // [rsp+2E0h] [rbp-250h] BYREF
  __int64 v227; // [rsp+310h] [rbp-220h]
  __int64 v228; // [rsp+318h] [rbp-218h]
  __int64 v229; // [rsp+320h] [rbp-210h]
  int v230; // [rsp+328h] [rbp-208h]
  __int64 v231; // [rsp+330h] [rbp-200h]
  char *v232; // [rsp+338h] [rbp-1F8h]
  __int64 v233; // [rsp+340h] [rbp-1F0h]
  int v234; // [rsp+348h] [rbp-1E8h]
  char v235; // [rsp+34Ch] [rbp-1E4h]
  char v236; // [rsp+350h] [rbp-1E0h] BYREF
  __int64 v237; // [rsp+360h] [rbp-1D0h]
  __int64 v238; // [rsp+368h] [rbp-1C8h]
  __int64 v239; // [rsp+370h] [rbp-1C0h]
  __int64 v240; // [rsp+378h] [rbp-1B8h]
  __int64 v241; // [rsp+380h] [rbp-1B0h]
  __int64 v242; // [rsp+388h] [rbp-1A8h]
  __int16 v243; // [rsp+390h] [rbp-1A0h]
  char v244; // [rsp+392h] [rbp-19Eh]
  char *v245; // [rsp+398h] [rbp-198h]
  __int64 v246; // [rsp+3A0h] [rbp-190h]
  char v247; // [rsp+3A8h] [rbp-188h] BYREF
  __int64 v248; // [rsp+3C8h] [rbp-168h]
  __int64 v249; // [rsp+3D0h] [rbp-160h]
  __int16 v250; // [rsp+3D8h] [rbp-158h]
  __int64 v251; // [rsp+3E0h] [rbp-150h]
  _QWORD *v252; // [rsp+3E8h] [rbp-148h]
  void **v253; // [rsp+3F0h] [rbp-140h]
  __int64 v254; // [rsp+3F8h] [rbp-138h]
  int v255; // [rsp+400h] [rbp-130h]
  __int16 v256; // [rsp+404h] [rbp-12Ch]
  char v257; // [rsp+406h] [rbp-12Ah]
  __int64 v258; // [rsp+408h] [rbp-128h]
  __int64 v259; // [rsp+410h] [rbp-120h]
  _QWORD v260[3]; // [rsp+418h] [rbp-118h] BYREF
  __m128i v261; // [rsp+430h] [rbp-100h]
  __m128i v262; // [rsp+440h] [rbp-F0h]
  __m128i v263; // [rsp+450h] [rbp-E0h]
  __m128i v264; // [rsp+460h] [rbp-D0h]
  __int64 v265; // [rsp+470h] [rbp-C0h]
  void *v266; // [rsp+478h] [rbp-B8h] BYREF
  char v267[16]; // [rsp+480h] [rbp-B0h] BYREF
  __int64 (__fastcall *v268)(char *, __m128i *, int); // [rsp+490h] [rbp-A0h]
  char (__fastcall *v269)(__int64 *, __int64 *); // [rsp+498h] [rbp-98h]
  char *v270; // [rsp+4A0h] [rbp-90h]
  __int64 v271; // [rsp+4A8h] [rbp-88h]
  char v272; // [rsp+4B0h] [rbp-80h] BYREF
  const char *v273; // [rsp+4F0h] [rbp-40h]

  v5 = *(__int64 **)(a2 + 8);
  v6 = *(__int64 **)(a2 + 16);
  if ( v5 != v6 )
  {
    v7 = 0;
    do
    {
      v8 = *v5++;
      v7 |= sub_2DE22E0(a1, v8, a3);
    }
    while ( v6 != v5 );
    if ( (_BYTE)v7 )
    {
      sub_2DE1C70("nested hardware-loops not supported", 0x23u, (__int64)"HWLoopNested", 12, *(__int64 **)(a1 + 64), a2);
      return v7;
    }
  }
  sub_DF8ED0((__int64)v185, a2);
  v7 = sub_DFF220(v185, *(_QWORD *)(a1 + 8));
  if ( !(_BYTE)v7 )
  {
    sub_2DE1C70(
      "cannot analyze loop, irreducible control flow",
      0x2Du,
      (__int64)"HWLoopCannotAnalyze",
      19,
      *(__int64 **)(a1 + 64),
      a2);
    return v7;
  }
  v9 = *(unsigned int **)(a1 + 72);
  if ( *((_BYTE *)v9 + 17) )
  {
    if ( !*((_BYTE *)v9 + 12) )
      goto LABEL_8;
LABEL_132:
    v186 = sub_BCCE00(a3, v9[2]);
    v9 = *(unsigned int **)(a1 + 72);
    if ( !*((_BYTE *)v9 + 4) )
      goto LABEL_9;
    goto LABEL_133;
  }
  v7 = sub_DF9C60(*(_QWORD *)(a1 + 40));
  if ( !(_BYTE)v7 )
  {
    sub_2DE1C70(
      "it's not profitable to create a hardware-loop",
      0x2Du,
      (__int64)"HWLoopNotProfitable",
      19,
      *(__int64 **)(a1 + 64),
      a2);
    return v7;
  }
  v9 = *(unsigned int **)(a1 + 72);
  if ( *((_BYTE *)v9 + 12) )
    goto LABEL_132;
LABEL_8:
  if ( !*((_BYTE *)v9 + 4) )
    goto LABEL_9;
LABEL_133:
  v187 = sub_ACD640(v186, *v9, 0);
  v9 = *(unsigned int **)(a1 + 72);
LABEL_9:
  v10 = 0;
  v11 = v185[0];
  if ( *((_BYTE *)v9 + 19) )
    v10 = *((_BYTE *)v9 + 18);
  v12 = 0;
  if ( *((_BYTE *)v9 + 21) )
    v12 = *((_BYTE *)v9 + 20);
  if ( (unsigned __int8)sub_DF8F50(v185, *(_QWORD *)a1, *(_QWORD *)(a1 + 8), *(_QWORD *)(a1 + 24), v12, v10) )
  {
    if ( sub_D4B130(v11)
      || sub_F67CB0(v11, *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), 0, *(unsigned __int8 *)(a1 + 16), v13) )
    {
      v14 = v185[0];
      v15 = *(unsigned int **)a1;
      v153 = *(_QWORD *)(a1 + 72);
      v176 = *(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(a1 + 32);
      v147 = *(__int64 **)(a1 + 64);
      sub_AA4B30(**(_QWORD **)(v185[0] + 32));
      v199 = v15;
      v202 = 1;
      v16 = (_QWORD *)v185[3];
      v171 = v186;
      v200 = (__int64)v176;
      v174 = v185[2];
      v203 = 0;
      v151 = v187;
      v204 = 0;
      v155 = v189;
      v205 = 0;
      v157 = v190;
      v201 = "loopcnt";
      v216[0] = &v219;
      LODWORD(v206) = 0;
      v207 = 0;
      v208 = 0;
      v209 = 0;
      v210 = 0;
      v211 = 0;
      v212 = 0;
      v213 = 0;
      v214 = 0;
      v215 = 0;
      v216[1] = 16;
      v217 = 0;
      v218 = 1;
      v17 = _mm_loadu_si128(&v182);
      v224 = &v226;
      v232 = &v236;
      v178.m128i_i64[0] = (__int64)&v199;
      v18 = _mm_load_si128(&v178);
      v225 = 0x200000000LL;
      v181 = (char *)&unk_49DA0D8;
      v243 = 1;
      v178 = v17;
      v182 = v18;
      v220 = 0;
      v221 = 0;
      v222 = 0;
      v223 = 0;
      v227 = 0;
      v228 = 0;
      v229 = 0;
      v230 = 0;
      v231 = 0;
      v233 = 2;
      v234 = 0;
      v235 = 1;
      v237 = 0;
      v238 = 0;
      v239 = 0;
      v240 = 0;
      v241 = 0;
      v242 = 0;
      v244 = 0;
      v179 = 0;
      v193 = (void (__fastcall *)(_QWORD **, _QWORD **, __int64))v176;
      v183 = (__int64 (__fastcall *)(char *, __m128i *, int))sub_27BFDD0;
      v194 = (__m128i)(unsigned __int64)v176;
      v180 = v184;
      v184 = sub_27BFD20;
      v195 = 0u;
      v191 = &unk_49E5698;
      v196 = 0u;
      v192 = (__int64)&unk_49D94D0;
      v197 = 0u;
      LOWORD(v198) = 257;
      v251 = sub_B2BE50(*(_QWORD *)v15);
      v19 = _mm_loadu_si128(&v194);
      v252 = v260;
      v20 = _mm_loadu_si128(&v195);
      v253 = &v266;
      v21 = _mm_loadu_si128(&v196);
      v256 = 512;
      v22 = _mm_loadu_si128(&v197);
      v250 = 0;
      v245 = &v247;
      v260[2] = v193;
      v246 = 0x200000000LL;
      v265 = v198;
      v254 = 0;
      v255 = 0;
      v257 = 7;
      v258 = 0;
      v259 = 0;
      v248 = 0;
      v249 = 0;
      v260[0] = &unk_49E5698;
      v260[1] = &unk_49D94D0;
      v266 = &unk_49DA0D8;
      v261 = v19;
      v262 = v20;
      v263 = v21;
      v264 = v22;
      v268 = 0;
      if ( v183 )
      {
        v183(v267, &v182, 2);
        v269 = v184;
        v268 = (__int64 (__fastcall *)(_QWORD *, _QWORD *, int))v183;
      }
      v191 = &unk_49E5698;
      v192 = (__int64)&unk_49D94D0;
      nullsub_63();
      nullsub_63();
      sub_B32BF0(&v181);
      if ( v179 )
        v179(&v178, &v178, 3);
      v270 = &v272;
      v271 = 0x800000000LL;
      v273 = byte_3F871B3;
      if ( *(_BYTE *)(sub_D95540((__int64)v16) + 8) != 14 && v171 != sub_D95540((__int64)v16) )
        v16 = sub_DC2B70((__int64)v15, (__int64)v16, v171, 0);
      v194.m128i_i64[0] = (__int64)sub_DA2C50((__int64)v15, v171, 1, 0);
      v191 = &v193;
      v193 = (void (__fastcall *)(_QWORD **, _QWORD **, __int64))v16;
      v192 = 0x200000002LL;
      v23 = sub_DC7EB0((__int64 *)v15, (__int64)&v191, 0, 0);
      if ( v191 != &v193 )
        _libc_free((unsigned __int64)v191);
      v24 = sub_D95540((__int64)v23);
      v25 = sub_DA2C50((__int64)v15, v24, 0, 0);
      if ( (unsigned __int8)sub_DDD5B0((__int64 *)v15, v14, 33, (__int64)v23, (__int64)v25) )
      {
        if ( *(_BYTE *)(v153 + 23) )
        {
          v167 = 1;
          v26 = sub_D4B130(v14);
        }
        else
        {
          v167 = v157;
          v26 = sub_D4B130(v14);
          v28 = v26 + 48;
          if ( !v157 )
            goto LABEL_24;
        }
        v92 = sub_AA54C0(v26);
        v28 = v26 + 48;
        if ( v92 )
        {
          v93 = *(_QWORD *)(v26 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          v30 = v93;
          if ( v93 == v28 )
            goto LABEL_199;
          if ( !v93 )
            goto LABEL_261;
          if ( (unsigned int)*(unsigned __int8 *)(v93 - 24) - 30 > 0xA )
LABEL_199:
            BUG();
          if ( (*(_DWORD *)(v93 - 20) & 0x7FFFFFF) != 1 )
            goto LABEL_26;
          v94 = sub_AA54C0(v26);
          v95 = v94 + 48;
          v96 = v94;
          v97 = *(_QWORD *)(v94 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v95 == v97 )
          {
            v101 = 0;
          }
          else
          {
            if ( !v97 )
              goto LABEL_261;
            v98 = *(unsigned __int8 *)(v97 - 24);
            v99 = v97 - 24;
            v100 = (unsigned int)(v98 - 30) < 0xB;
            v101 = 0;
            if ( v100 )
              v101 = v99;
          }
          v142 = v26 + 48;
          v159 = v95;
          v165 = v96;
          v102 = sub_F80650((__int64 *)&v199, (__int64)v23, v101, v26 + 48, v96, v95);
          v103 = v167;
          v27 = v159;
          if ( v102 )
            v26 = v165;
          else
            v103 = 0;
          v167 = v103;
          v28 = v142;
          if ( v102 )
            v28 = v159;
        }
      }
      else
      {
        v167 = 0;
        v26 = sub_D4B130(v14);
        v28 = v26 + 48;
      }
LABEL_24:
      v29 = *(_QWORD *)(v26 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      v30 = v29;
      if ( v29 == v28 )
      {
LABEL_166:
        v31 = 0;
LABEL_27:
        v164 = v28;
        if ( (unsigned __int8)sub_F80650((__int64 *)&v199, (__int64)v23, v31, v28, v31, v27) )
        {
          v32 = *(_QWORD *)(v26 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v32 == v164 )
          {
            v33 = 0;
          }
          else
          {
            if ( !v32 )
              goto LABEL_261;
            v33 = v32 - 24;
            if ( (unsigned int)*(unsigned __int8 *)(v32 - 24) - 30 >= 0xB )
              v33 = 0;
          }
          v34 = (char *)sub_F8DB90((__int64)&v199, (__int64)v23, v171, v33 + 24, 0);
          if ( v167 )
          {
            v173 = sub_D4B130(v14);
            if ( sub_AA54C0(v173) )
            {
              v104 = sub_AA54C0(v173);
              v105 = *(_QWORD *)(v104 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v105 == v104 + 48 || !v105 || (unsigned int)*(unsigned __int8 *)(v105 - 24) - 30 > 0xA )
                goto LABEL_261;
              if ( *(_BYTE *)(v105 - 24) == 31 && (*(_DWORD *)(v105 - 20) & 0x7FFFFFF) != 1 )
              {
                v106 = *(_QWORD *)(v105 - 120);
                if ( *(_BYTE *)v106 == 82 )
                {
                  v170 = *(_WORD *)(v106 + 2) & 0x3F;
                  if ( (unsigned int)(v170 - 32) <= 1 )
                  {
                    v107 = 0;
                    if ( *v34 == 68 )
                      v107 = (char *)*((_QWORD *)v34 - 4);
                    v108 = *(char **)(v106 - 64);
                    v109 = *(_QWORD *)(v106 - 32);
                    v166 = *v108;
                    if ( *v108 != 17 )
                    {
                      if ( *(_BYTE *)v109 != 17 )
                        goto LABEL_194;
LABEL_183:
                      v114 = *(_DWORD *)(v109 + 32);
                      if ( v114 <= 0x40 )
                      {
                        v116 = *(_QWORD *)(v109 + 24) == 0;
                      }
                      else
                      {
                        v137 = *(_DWORD *)(v109 + 32);
                        v140 = v108;
                        v144 = v105;
                        v161 = v109;
                        v115 = sub_C444A0(v109 + 24);
                        v114 = v137;
                        v109 = v161;
                        v105 = v144;
                        v108 = v140;
                        v116 = v137 == v115;
                      }
                      if ( v34 == v108 && v116 )
                        goto LABEL_250;
                      if ( v166 != 17 )
                      {
LABEL_246:
                        if ( v114 <= 0x40 )
                        {
                          v134 = *(_QWORD *)(v109 + 24) == 0;
                        }
                        else
                        {
                          v146 = v114;
                          v150 = v108;
                          v163 = v105;
                          v133 = sub_C444A0(v109 + 24);
                          v105 = v163;
                          v108 = v150;
                          v134 = v146 == v133;
                        }
                        if ( v107 != v108 || !v134 )
                          goto LABEL_194;
LABEL_250:
                        if ( v173 == *(_QWORD *)(v105 - 32LL * (v170 != 33) - 56) )
                        {
                          sub_27C20B0((__int64)&v199);
                          v35 = 1;
                          goto LABEL_34;
                        }
                        goto LABEL_194;
                      }
                      v110 = *((_DWORD *)v108 + 8);
                      v111 = (__int64)(v108 + 24);
                      v113 = 17;
LABEL_189:
                      if ( v110 <= 0x40 )
                      {
                        v118 = *((_QWORD *)v108 + 3) == 0;
                      }
                      else
                      {
                        v138 = v113;
                        v141 = v110;
                        v145 = v109;
                        v149 = v108;
                        v162 = v105;
                        v117 = sub_C444A0(v111);
                        v105 = v162;
                        v108 = v149;
                        v109 = v145;
                        v113 = v138;
                        v118 = v141 == v117;
                      }
                      if ( v107 == (char *)v109 && v118 )
                        goto LABEL_250;
                      if ( v113 != 17 )
                      {
LABEL_194:
                        v26 = sub_D4B130(v14);
                        sub_27C20B0((__int64)&v199);
                        v35 = 0;
                        goto LABEL_34;
                      }
                      v114 = *(_DWORD *)(v109 + 32);
                      goto LABEL_246;
                    }
                    v110 = *((_DWORD *)v108 + 8);
                    v111 = (__int64)(v108 + 24);
                    if ( v110 <= 0x40 )
                    {
                      if ( *((_QWORD *)v108 + 3) )
                        goto LABEL_182;
                    }
                    else
                    {
                      v135 = *((_DWORD *)v108 + 8);
                      v136 = *(_QWORD *)(v106 - 32);
                      v139 = *(char **)(v106 - 64);
                      v143 = v105;
                      v160 = (__int64)(v108 + 24);
                      v112 = sub_C444A0(v111);
                      v110 = v135;
                      v111 = v160;
                      v105 = v143;
                      v108 = v139;
                      v109 = v136;
                      if ( v135 != v112 )
                        goto LABEL_182;
                    }
                    if ( v34 == (char *)v109 && v109 )
                      goto LABEL_250;
LABEL_182:
                    v113 = *(_BYTE *)v109;
                    if ( *(_BYTE *)v109 != 17 )
                      goto LABEL_189;
                    goto LABEL_183;
                  }
                }
              }
            }
          }
          v26 = sub_D4B130(v14);
          sub_27C20B0((__int64)&v199);
          v35 = 0;
          if ( v34 )
          {
LABEL_34:
            v177 = v34;
            v36 = *(_QWORD *)(v26 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v26 + 48 == v36 )
            {
              v38 = 0;
            }
            else
            {
              if ( !v36 )
                goto LABEL_261;
              v37 = *(unsigned __int8 *)(v36 - 24);
              v38 = 0;
              v39 = v36 - 24;
              if ( (unsigned int)(v37 - 30) < 0xB )
                v38 = v39;
            }
            v172 = v35;
            sub_23D0AB0((__int64)&v199, v38, 0, 0, 0);
            v191 = *(_QWORD **)(*(_QWORD *)(v26 + 72) + 120LL);
            v40 = sub_A73ED0(&v191, 72);
            v41 = v172;
            if ( v40 )
              BYTE4(v212) = 1;
            v178.m128i_i64[0] = v177[1];
            if ( v155 )
            {
              if ( v172 )
              {
                v43 = 351;
              }
              else
              {
                v43 = 344;
                v172 = v155;
              }
            }
            else
            {
              v42 = *(_BYTE *)(v153 + 19);
              v172 = v42;
              if ( v42 )
              {
                if ( v41 )
                  v42 = v41;
                v172 = v42;
                v43 = v41 == 0 ? 344 : 351;
              }
              else
              {
                v43 = v41 == 0 ? 322 : 350;
              }
            }
            v194.m128i_i16[4] = 257;
            HIDWORD(v181) = 0;
            v168 = v41;
            v44 = sub_B33D10((__int64)&v199, v43, (__int64)&v178, 1, (int)&v177, 1, (unsigned int)v181, (__int64)&v191);
            v158 = v44;
            if ( !v168 )
            {
              if ( !v172 )
LABEL_59:
                v158 = (__int64)v177;
LABEL_60:
              nullsub_61();
              v215 = &unk_49DA100;
              nullsub_63();
              if ( v199 != (unsigned int *)&v201 )
                _libc_free((unsigned __int64)v199);
              if ( !v155 && !*(_BYTE *)(v153 + 19) )
              {
                sub_23D0AB0((__int64)&v199, v174, 0, 0, 0);
                v191 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v174 + 40) + 72LL) + 120LL);
                if ( (unsigned __int8)sub_A73ED0(&v191, 72) )
                  BYTE4(v212) = 1;
                HIDWORD(v181) = 0;
                v194.m128i_i16[4] = 257;
                v177 = (_QWORD *)v151;
                v178.m128i_i64[0] = *(_QWORD *)(v151 + 8);
                v128 = sub_B33D10(
                         (__int64)&v199,
                         0xDDu,
                         (__int64)&v178,
                         1,
                         (int)&v177,
                         1,
                         (unsigned int)v181,
                         (__int64)&v191);
                v82 = *(char **)(v174 - 96);
                if ( v82 )
                {
                  v129 = *(_QWORD *)(v174 - 88);
                  **(_QWORD **)(v174 - 80) = v129;
                  if ( v129 )
                    *(_QWORD *)(v129 + 16) = *(_QWORD *)(v174 - 80);
                }
                *(_QWORD *)(v174 - 96) = v128;
                if ( v128 )
                {
                  v130 = *(_QWORD *)(v128 + 16);
                  *(_QWORD *)(v174 - 88) = v130;
                  if ( v130 )
                    *(_QWORD *)(v130 + 16) = v174 - 88;
                  *(_QWORD *)(v174 - 80) = v128 + 16;
                  *(_QWORD *)(v128 + 16) = v174 - 96;
                }
                v85 = *(_QWORD *)(v174 - 32);
                if ( *(_BYTE *)(v14 + 84) )
                {
                  v131 = *(_QWORD **)(v14 + 64);
                  v132 = &v131[*(unsigned int *)(v14 + 76)];
                  if ( v131 == v132 )
                    goto LABEL_204;
                  while ( v85 != *v131 )
                  {
                    if ( v132 == ++v131 )
                      goto LABEL_204;
                  }
                  goto LABEL_119;
                }
LABEL_203:
                if ( sub_C8CA60(v14 + 56, v85) )
                  goto LABEL_119;
                goto LABEL_204;
              }
              sub_23D0AB0((__int64)&v199, v174, 0, 0, 0);
              v191 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v174 + 40) + 72LL) + 120LL);
              if ( (unsigned __int8)sub_A73ED0(&v191, 72) )
                BYTE4(v212) = 1;
              v178.m128i_i32[1] = 0;
              v181 = v34;
              v182.m128i_i64[0] = v151;
              v194.m128i_i16[4] = 257;
              v177 = (_QWORD *)*((_QWORD *)v34 + 1);
              v50 = sub_B33D10(
                      (__int64)&v199,
                      0xDEu,
                      (__int64)&v177,
                      1,
                      (int)&v181,
                      2,
                      v178.m128i_u32[0],
                      (__int64)&v191);
              nullsub_61();
              v215 = &unk_49DA100;
              nullsub_63();
              if ( v199 != (unsigned int *)&v201 )
                _libc_free((unsigned __int64)v199);
              v156 = sub_D4B130(v14);
              v152 = **(_QWORD **)(v14 + 32);
              v154 = *(_QWORD *)(v174 + 40);
              v52 = sub_AA4FF0(v152);
              v53 = 0;
              if ( v52 )
                v53 = v51;
              v148 = v53;
              v54 = sub_AA48A0(v152);
              v211 = 0;
              v208 = v54;
              v199 = (unsigned int *)&v201;
              v209 = &v215;
              v200 = 0x200000000LL;
              v210 = v216;
              WORD2(v212) = 512;
              v205 = v152;
              LODWORD(v212) = 0;
              v215 = &unk_49DA100;
              BYTE6(v212) = 7;
              v216[0] = &unk_49DA0B0;
              LOBYTE(v54) = 1;
              BYTE1(v54) = v148;
              v206 = v52;
              v213 = 0;
              v214 = 0;
              LOWORD(v207) = v54;
              if ( v52 != v152 + 48 )
              {
                v55 = v52 - 24;
                if ( !v52 )
                  v55 = 0;
                v56 = *(_QWORD *)sub_B46C60(v55);
                v191 = (_QWORD *)v56;
                if ( v56 && (sub_B96E90((__int64)&v191, v56, 1), (v59 = (__int64)v191) != 0) )
                {
                  v60 = v199;
                  v61 = v200;
                  v62 = &v199[4 * (unsigned int)v200];
                  if ( v199 != v62 )
                  {
                    while ( *v60 )
                    {
                      v60 += 4;
                      if ( v62 == v60 )
                        goto LABEL_206;
                    }
                    *((_QWORD *)v60 + 1) = v191;
                    goto LABEL_80;
                  }
LABEL_206:
                  if ( (unsigned int)v200 >= (unsigned __int64)HIDWORD(v200) )
                  {
                    if ( HIDWORD(v200) < (unsigned __int64)(unsigned int)v200 + 1 )
                    {
                      sub_C8D5F0((__int64)&v199, &v201, (unsigned int)v200 + 1LL, 0x10u, v57, v58);
                      v62 = &v199[4 * (unsigned int)v200];
                    }
                    *(_QWORD *)v62 = 0;
                    *((_QWORD *)v62 + 1) = v59;
                    v59 = (__int64)v191;
                    LODWORD(v200) = v200 + 1;
                  }
                  else
                  {
                    if ( v62 )
                    {
                      *v62 = 0;
                      *((_QWORD *)v62 + 1) = v59;
                      v61 = v200;
                      v59 = (__int64)v191;
                    }
                    LODWORD(v200) = v61 + 1;
                  }
                }
                else
                {
                  sub_93FB40((__int64)&v199, 0);
                  v59 = (__int64)v191;
                }
                if ( v59 )
LABEL_80:
                  sub_B91220((__int64)&v191, v59);
              }
              v194.m128i_i16[4] = 257;
              v63 = sub_D5C860((__int64 *)&v199, *(_QWORD *)(v158 + 8), 2, (__int64)&v191);
              v64 = v63;
              v65 = *(_DWORD *)(v63 + 4) & 0x7FFFFFF;
              if ( v65 == *(_DWORD *)(v63 + 72) )
              {
                sub_B48D90(v63);
                v65 = *(_DWORD *)(v64 + 4) & 0x7FFFFFF;
              }
              v66 = (v65 + 1) & 0x7FFFFFF;
              v67 = v66 | *(_DWORD *)(v64 + 4) & 0xF8000000;
              v68 = *(_QWORD *)(v64 - 8) + 32LL * (unsigned int)(v66 - 1);
              *(_DWORD *)(v64 + 4) = v67;
              if ( *(_QWORD *)v68 )
              {
                v69 = *(_QWORD *)(v68 + 8);
                **(_QWORD **)(v68 + 16) = v69;
                if ( v69 )
                  *(_QWORD *)(v69 + 16) = *(_QWORD *)(v68 + 16);
              }
              *(_QWORD *)v68 = v158;
              v70 = *(_QWORD *)(v158 + 16);
              *(_QWORD *)(v68 + 8) = v70;
              if ( v70 )
                *(_QWORD *)(v70 + 16) = v68 + 8;
              *(_QWORD *)(v68 + 16) = v158 + 16;
              *(_QWORD *)(v158 + 16) = v68;
              *(_QWORD *)(*(_QWORD *)(v64 - 8)
                        + 32LL * *(unsigned int *)(v64 + 72)
                        + 8LL * ((*(_DWORD *)(v64 + 4) & 0x7FFFFFFu) - 1)) = v156;
              v71 = *(_DWORD *)(v64 + 4) & 0x7FFFFFF;
              if ( v71 == *(_DWORD *)(v64 + 72) )
              {
                sub_B48D90(v64);
                v71 = *(_DWORD *)(v64 + 4) & 0x7FFFFFF;
              }
              v72 = (v71 + 1) & 0x7FFFFFF;
              v73 = v72 | *(_DWORD *)(v64 + 4) & 0xF8000000;
              v74 = *(_QWORD *)(v64 - 8) + 32LL * (unsigned int)(v72 - 1);
              *(_DWORD *)(v64 + 4) = v73;
              if ( *(_QWORD *)v74 )
              {
                v75 = *(_QWORD *)(v74 + 8);
                **(_QWORD **)(v74 + 16) = v75;
                if ( v75 )
                  *(_QWORD *)(v75 + 16) = *(_QWORD *)(v74 + 16);
              }
              *(_QWORD *)v74 = v50;
              if ( v50 )
              {
                v76 = *(_QWORD *)(v50 + 16);
                *(_QWORD *)(v74 + 8) = v76;
                if ( v76 )
                  *(_QWORD *)(v76 + 16) = v74 + 8;
                *(_QWORD *)(v74 + 16) = v50 + 16;
                *(_QWORD *)(v50 + 16) = v74;
              }
              *(_QWORD *)(*(_QWORD *)(v64 - 8)
                        + 32LL * *(unsigned int *)(v64 + 72)
                        + 8LL * ((*(_DWORD *)(v64 + 4) & 0x7FFFFFFu) - 1)) = v154;
              nullsub_61();
              v215 = &unk_49DA100;
              nullsub_63();
              if ( v199 != (unsigned int *)&v201 )
                _libc_free((unsigned __int64)v199);
              if ( (*(_BYTE *)(v50 + 7) & 0x40) != 0 )
                v77 = *(_QWORD *)(v50 - 8);
              else
                v77 = v50 - 32LL * (*(_DWORD *)(v50 + 4) & 0x7FFFFFF);
              if ( *(_QWORD *)v77 )
              {
                v78 = *(_QWORD *)(v77 + 8);
                **(_QWORD **)(v77 + 16) = v78;
                if ( v78 )
                  *(_QWORD *)(v78 + 16) = *(_QWORD *)(v77 + 16);
              }
              *(_QWORD *)v77 = v64;
              v79 = *(_QWORD *)(v64 + 16);
              *(_QWORD *)(v77 + 8) = v79;
              if ( v79 )
                *(_QWORD *)(v79 + 16) = v77 + 8;
              *(_QWORD *)(v77 + 16) = v64 + 16;
              *(_QWORD *)(v64 + 16) = v77;
              sub_23D0AB0((__int64)&v199, v174, 0, 0, 0);
              LOWORD(v184) = 257;
              v80 = sub_AD64C0(*(_QWORD *)(v50 + 8), 0, 0);
              v81 = (_QWORD *)(*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v209 + 7))(
                                v209,
                                33,
                                v50,
                                v80);
              if ( v81 )
              {
                v82 = *(char **)(v174 - 96);
                if ( !v82 || (v83 = *(_QWORD *)(v174 - 88), (**(_QWORD **)(v174 - 80) = v83) == 0) )
                {
                  *(_QWORD *)(v174 - 96) = v81;
LABEL_111:
                  v84 = v81[2];
                  *(_QWORD *)(v174 - 88) = v84;
                  if ( v84 )
                    *(_QWORD *)(v84 + 16) = v174 - 88;
                  *(_QWORD *)(v174 - 80) = v81 + 2;
                  v81[2] = v174 - 96;
LABEL_114:
                  v85 = *(_QWORD *)(v174 - 32);
                  if ( *(_BYTE *)(v14 + 84) )
                  {
                    v86 = *(_QWORD **)(v14 + 64);
                    v87 = &v86[*(unsigned int *)(v14 + 76)];
                    if ( v86 == v87 )
                    {
LABEL_204:
                      sub_B4CC70(v174);
                      goto LABEL_119;
                    }
                    while ( v85 != *v86 )
                    {
                      if ( v87 == ++v86 )
                        goto LABEL_204;
                    }
LABEL_119:
                    v193 = 0;
                    sub_F5CAB0(v82, 0, 0, (__int64)&v191);
                    if ( v193 )
                      v193(&v191, &v191, 3);
                    nullsub_61();
                    v215 = &unk_49DA100;
                    nullsub_63();
                    if ( v199 != (unsigned int *)&v201 )
                      _libc_free((unsigned __int64)v199);
                    v88 = *(__int64 **)(v14 + 32);
                    v89 = *(__int64 **)(v14 + 40);
                    if ( v88 != v89 )
                    {
                      do
                      {
                        v90 = *v88++;
                        sub_F39260(v90, 0, 0);
                      }
                      while ( v89 != v88 );
                      *(_BYTE *)(a1 + 80) = 1;
                      goto LABEL_126;
                    }
LABEL_163:
                    *(_BYTE *)(a1 + 80) = 1;
                    goto LABEL_126;
                  }
                  goto LABEL_203;
                }
              }
              else
              {
                v194.m128i_i16[4] = 257;
                v81 = sub_BD2C40(72, unk_3F10FD0);
                if ( v81 )
                {
                  v119 = *(_QWORD ***)(v50 + 8);
                  v120 = *((unsigned __int8 *)v119 + 8);
                  if ( (unsigned int)(v120 - 17) > 1 )
                  {
                    v123 = sub_BCB2A0(*v119);
                  }
                  else
                  {
                    v121 = *((_DWORD *)v119 + 8);
                    v178.m128i_i8[4] = (_BYTE)v120 == 18;
                    v178.m128i_i32[0] = v121;
                    v122 = (__int64 *)sub_BCB2A0(*v119);
                    v123 = sub_BCE1B0(v122, v178.m128i_i64[0]);
                  }
                  sub_B523C0((__int64)v81, v123, 53, 33, v50, v80, (__int64)&v191, 0, 0, 0);
                }
                (*(void (__fastcall **)(_QWORD *, _QWORD *, char **, __int64, __int64))(*v210 + 16LL))(
                  v210,
                  v81,
                  &v181,
                  v206,
                  v207);
                v124 = v199;
                v125 = &v199[4 * (unsigned int)v200];
                if ( v199 != v125 )
                {
                  do
                  {
                    v126 = *((_QWORD *)v124 + 1);
                    v127 = *v124;
                    v124 += 4;
                    sub_B99FD0((__int64)v81, v127, v126);
                  }
                  while ( v125 != v124 );
                }
                v82 = *(char **)(v174 - 96);
                if ( !v82 || (v83 = *(_QWORD *)(v174 - 88), (**(_QWORD **)(v174 - 80) = v83) == 0) )
                {
LABEL_110:
                  *(_QWORD *)(v174 - 96) = v81;
                  if ( !v81 )
                    goto LABEL_114;
                  goto LABEL_111;
                }
              }
              *(_QWORD *)(v83 + 16) = *(_QWORD *)(v174 - 80);
              goto LABEL_110;
            }
            v45 = v44;
            if ( v172 )
            {
              v194.m128i_i16[4] = 257;
              LODWORD(v181) = 1;
              v45 = sub_94D3D0(&v199, v44, (__int64)&v181, 1, (__int64)&v191);
            }
            v46 = *(_QWORD *)(v26 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v26 + 48 == v46 )
LABEL_260:
              BUG();
            if ( v46 )
            {
              v169 = v46 - 24;
              if ( (unsigned int)*(unsigned __int8 *)(v46 - 24) - 30 <= 0xA )
              {
                if ( *(_QWORD *)(v46 - 120) )
                {
                  v47 = *(_QWORD *)(v46 - 112);
                  **(_QWORD **)(v46 - 104) = v47;
                  if ( v47 )
                    *(_QWORD *)(v47 + 16) = *(_QWORD *)(v46 - 104);
                }
                *(_QWORD *)(v46 - 120) = v45;
                if ( v45 )
                {
                  v48 = *(_QWORD *)(v45 + 16);
                  *(_QWORD *)(v46 - 112) = v48;
                  if ( v48 )
                    *(_QWORD *)(v48 + 16) = v46 - 112;
                  *(_QWORD *)(v46 - 104) = v45 + 16;
                  *(_QWORD *)(v45 + 16) = v46 - 120;
                }
                v49 = *(_QWORD *)(v46 - 56);
                if ( v49 != sub_D4B130(v14) )
                  sub_B4CC70(v169);
                if ( !v172 )
                  goto LABEL_59;
                v194.m128i_i16[4] = 257;
                LODWORD(v181) = 0;
                v158 = sub_94D3D0(&v199, v158, (__int64)&v181, 1, (__int64)&v191);
                goto LABEL_60;
              }
              goto LABEL_260;
            }
LABEL_261:
            BUG();
          }
        }
        else
        {
          sub_27C20B0((__int64)&v199);
        }
        sub_2DE1C70("could not safely create a loop count expression", 0x2Fu, (__int64)"HWLoopNotSafe", 13, v147, v14);
        goto LABEL_163;
      }
      if ( !v29 )
        goto LABEL_261;
LABEL_26:
      v31 = v30 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v30 - 24) - 30 <= 0xA )
        goto LABEL_27;
      goto LABEL_166;
    }
  }
  else
  {
    sub_2DE1C70("loop is not a candidate", 0x17u, (__int64)"HWLoopNoCandidate", 17, *(__int64 **)(a1 + 64), v11);
  }
  if ( !*(_BYTE *)(a1 + 80) )
    return 0;
LABEL_126:
  if ( v188 )
    return 0;
  return *(unsigned __int8 *)(*(_QWORD *)(a1 + 72) + 21LL) ^ 1u;
}
