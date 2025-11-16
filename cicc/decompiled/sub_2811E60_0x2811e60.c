// Function: sub_2811E60
// Address: 0x2811e60
//
__int64 __fastcall sub_2811E60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  _QWORD *v15; // r12
  _QWORD *v16; // r14
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 *v19; // r13
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 **v26; // rax
  _QWORD *v27; // rbx
  __int64 v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 **v32; // r12
  __int64 *v33; // rax
  char v34; // r13
  _QWORD **v35; // r15
  __int64 v36; // r14
  unsigned int v37; // eax
  _QWORD *v38; // r15
  unsigned int v39; // eax
  _QWORD *v40; // rdx
  unsigned __int64 v41; // rax
  int v42; // edx
  unsigned __int64 v43; // rax
  unsigned int v44; // eax
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 **v47; // rdx
  __int64 v48; // rcx
  void **v49; // rax
  int v50; // eax
  void **v51; // rax
  unsigned int v53; // r14d
  unsigned int v54; // eax
  __m128i v55; // xmm1
  __m128i v56; // xmm0
  __int64 v57; // rax
  __m128i v58; // xmm2
  __m128i v59; // xmm3
  __m128i v60; // xmm4
  __m128i v61; // xmm5
  char *v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  char v66; // r14
  unsigned int *v67; // r15
  unsigned int *v68; // r12
  __int64 v69; // rax
  __int64 *v70; // rax
  __int64 v71; // r12
  __int64 v72; // r12
  __int64 v73; // r13
  unsigned int v74; // eax
  unsigned __int8 *v75; // rax
  unsigned __int8 *v76; // rdi
  __int64 v77; // r13
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // r8
  __int64 v81; // r9
  unsigned __int64 v82; // r12
  __int64 v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rdx
  _QWORD *v86; // rbx
  _QWORD *v87; // r12
  __int64 v88; // rax
  _QWORD *v89; // rbx
  _QWORD *v90; // r12
  __int64 v91; // rax
  __int64 v92; // r13
  __int64 v93; // r14
  __int64 v94; // rbx
  __int64 v95; // rdi
  unsigned int v96; // r15d
  __int64 *v97; // rax
  __int64 v98; // r13
  __int64 v99; // rbx
  __int64 v100; // rdi
  unsigned int v101; // r14d
  __int64 *v102; // rax
  __int64 v103; // rdi
  char *v104; // rax
  unsigned int v105; // [rsp+10h] [rbp-960h]
  __int64 v107; // [rsp+20h] [rbp-950h]
  _QWORD *v108; // [rsp+20h] [rbp-950h]
  __int64 v109; // [rsp+28h] [rbp-948h]
  __int64 *v110; // [rsp+28h] [rbp-948h]
  char v112; // [rsp+3Fh] [rbp-931h]
  __int64 v113; // [rsp+40h] [rbp-930h]
  __int64 v114; // [rsp+40h] [rbp-930h]
  __int64 v115; // [rsp+48h] [rbp-928h]
  __int64 v116; // [rsp+48h] [rbp-928h]
  __int64 v117; // [rsp+50h] [rbp-920h]
  __int64 *v118; // [rsp+50h] [rbp-920h]
  __int64 *v119; // [rsp+60h] [rbp-910h]
  __int64 v120; // [rsp+78h] [rbp-8F8h]
  __int64 *v121; // [rsp+80h] [rbp-8F0h]
  __int64 v122; // [rsp+88h] [rbp-8E8h]
  __int64 **v123; // [rsp+90h] [rbp-8E0h]
  __int64 **v124; // [rsp+98h] [rbp-8D8h]
  __int64 v125; // [rsp+A0h] [rbp-8D0h] BYREF
  __int64 v126; // [rsp+A8h] [rbp-8C8h] BYREF
  __m128i v127; // [rsp+B0h] [rbp-8C0h] BYREF
  void (__fastcall *v128)(__m128i *, __m128i *, __int64); // [rsp+C0h] [rbp-8B0h]
  char (__fastcall *v129)(__int64 *, __int64 *); // [rsp+C8h] [rbp-8A8h]
  const char *v130; // [rsp+D0h] [rbp-8A0h] BYREF
  __m128i v131; // [rsp+D8h] [rbp-898h] BYREF
  __int64 (__fastcall *v132)(__int64 *, __m128i *, int); // [rsp+E8h] [rbp-888h]
  char (__fastcall *v133)(__int64 *, __int64 *); // [rsp+F0h] [rbp-880h]
  _QWORD v134[10]; // [rsp+100h] [rbp-870h] BYREF
  unsigned int *v135; // [rsp+150h] [rbp-820h] BYREF
  __int64 v136; // [rsp+158h] [rbp-818h]
  __int64 v137; // [rsp+160h] [rbp-810h] BYREF
  __m128i v138; // [rsp+168h] [rbp-808h] BYREF
  __m128i v139; // [rsp+178h] [rbp-7F8h] BYREF
  __m128i v140; // [rsp+188h] [rbp-7E8h] BYREF
  __m128i v141; // [rsp+198h] [rbp-7D8h] BYREF
  void **v142; // [rsp+1A8h] [rbp-7C8h]
  __int64 v143; // [rsp+1B0h] [rbp-7C0h]
  int v144; // [rsp+1B8h] [rbp-7B8h]
  __int16 v145; // [rsp+1BCh] [rbp-7B4h]
  char v146; // [rsp+1BEh] [rbp-7B2h]
  __int64 v147; // [rsp+1C0h] [rbp-7B0h]
  __int64 v148; // [rsp+1C8h] [rbp-7A8h]
  void *v149; // [rsp+1D0h] [rbp-7A0h] BYREF
  void *v150; // [rsp+1D8h] [rbp-798h] BYREF
  __int64 v151; // [rsp+1E0h] [rbp-790h] BYREF
  __int64 *v152; // [rsp+1E8h] [rbp-788h]
  __int64 v153; // [rsp+1F0h] [rbp-780h]
  const char *v154; // [rsp+1F8h] [rbp-778h]
  __int64 v155; // [rsp+200h] [rbp-770h]
  __int64 v156; // [rsp+208h] [rbp-768h]
  __int64 v157; // [rsp+210h] [rbp-760h]
  __int64 *v158; // [rsp+218h] [rbp-758h]
  __int64 v159; // [rsp+220h] [rbp-750h]
  int v160; // [rsp+228h] [rbp-748h]
  char v161; // [rsp+22Ch] [rbp-744h]
  char v162; // [rsp+230h] [rbp-740h] BYREF
  __int64 v163; // [rsp+250h] [rbp-720h]
  __int64 v164; // [rsp+258h] [rbp-718h]
  __int64 v165; // [rsp+260h] [rbp-710h]
  __int64 v166; // [rsp+268h] [rbp-708h]
  __int64 v167; // [rsp+270h] [rbp-700h] BYREF
  char *v168; // [rsp+278h] [rbp-6F8h]
  __int64 v169; // [rsp+280h] [rbp-6F0h]
  int v170; // [rsp+288h] [rbp-6E8h]
  char v171; // [rsp+28Ch] [rbp-6E4h]
  char v172; // [rsp+290h] [rbp-6E0h] BYREF
  char v173; // [rsp+2B0h] [rbp-6C0h]
  __int64 v174; // [rsp+2B8h] [rbp-6B8h]
  const char *v175; // [rsp+2C0h] [rbp-6B0h]
  __int64 v176; // [rsp+2C8h] [rbp-6A8h]
  _QWORD v177[96]; // [rsp+2D0h] [rbp-6A0h] BYREF
  __m128i v178; // [rsp+5D0h] [rbp-3A0h] BYREF
  __int64 v179; // [rsp+5E0h] [rbp-390h]
  __int64 v180; // [rsp+5E8h] [rbp-388h] BYREF
  __int64 v181; // [rsp+5F0h] [rbp-380h] BYREF
  unsigned __int64 v182; // [rsp+5F8h] [rbp-378h]
  __int64 v183; // [rsp+600h] [rbp-370h] BYREF
  void **v184; // [rsp+608h] [rbp-368h]
  __int64 v185; // [rsp+610h] [rbp-360h]
  __int64 v186; // [rsp+618h] [rbp-358h]
  _QWORD v187[9]; // [rsp+620h] [rbp-350h] BYREF
  int v188; // [rsp+668h] [rbp-308h]
  char v189; // [rsp+66Ch] [rbp-304h]
  char v190; // [rsp+670h] [rbp-300h] BYREF
  __int64 v191; // [rsp+6F0h] [rbp-280h]
  __int64 v192; // [rsp+6F8h] [rbp-278h]
  __int64 v193; // [rsp+700h] [rbp-270h]
  int v194; // [rsp+708h] [rbp-268h]
  char *v195; // [rsp+710h] [rbp-260h]
  __int64 v196; // [rsp+718h] [rbp-258h]
  char v197; // [rsp+720h] [rbp-250h] BYREF
  __int64 v198; // [rsp+750h] [rbp-220h]
  __int64 v199; // [rsp+758h] [rbp-218h]
  __int64 v200; // [rsp+760h] [rbp-210h]
  __int64 v201; // [rsp+768h] [rbp-208h] BYREF
  int *v202; // [rsp+770h] [rbp-200h]
  __int64 v203; // [rsp+778h] [rbp-1F8h]
  __int64 v204; // [rsp+780h] [rbp-1F0h]
  int v205; // [rsp+788h] [rbp-1E8h] BYREF
  char v206; // [rsp+78Ch] [rbp-1E4h]
  char v207; // [rsp+790h] [rbp-1E0h] BYREF
  __int64 v208; // [rsp+7A0h] [rbp-1D0h]
  __int64 v209; // [rsp+7A8h] [rbp-1C8h]
  __int64 v210; // [rsp+7B0h] [rbp-1C0h]
  __int64 v211; // [rsp+7B8h] [rbp-1B8h]
  __int64 v212; // [rsp+7C0h] [rbp-1B0h]
  _QWORD *v213; // [rsp+7C8h] [rbp-1A8h] BYREF
  __int64 v214; // [rsp+7D0h] [rbp-1A0h]
  _QWORD v215[2]; // [rsp+7D8h] [rbp-198h] BYREF
  char v216; // [rsp+7E8h] [rbp-188h] BYREF
  __int64 v217; // [rsp+808h] [rbp-168h]
  __int64 v218; // [rsp+810h] [rbp-160h]
  __int16 v219; // [rsp+818h] [rbp-158h]
  __int64 v220; // [rsp+820h] [rbp-150h]
  _QWORD *v221; // [rsp+828h] [rbp-148h]
  __m128i **v222; // [rsp+830h] [rbp-140h]
  __int64 v223; // [rsp+838h] [rbp-138h]
  int v224; // [rsp+840h] [rbp-130h]
  __int16 v225; // [rsp+844h] [rbp-12Ch]
  char v226; // [rsp+846h] [rbp-12Ah]
  __int64 v227; // [rsp+848h] [rbp-128h]
  __int64 v228; // [rsp+850h] [rbp-120h]
  _QWORD v229[3]; // [rsp+858h] [rbp-118h] BYREF
  __m128i v230; // [rsp+870h] [rbp-100h]
  __m128i v231; // [rsp+880h] [rbp-F0h]
  __m128i v232; // [rsp+890h] [rbp-E0h]
  __m128i v233; // [rsp+8A0h] [rbp-D0h] BYREF
  void **v234; // [rsp+8B0h] [rbp-C0h]
  __m128i *v235; // [rsp+8B8h] [rbp-B8h] BYREF
  __int64 v236; // [rsp+8C0h] [rbp-B0h] BYREF
  __int64 (__fastcall *v237)(__int64 *, __m128i *, int); // [rsp+8D0h] [rbp-A0h]
  char (__fastcall *v238)(__int64 *, __int64 *); // [rsp+8D8h] [rbp-98h]
  char *v239; // [rsp+8E0h] [rbp-90h]
  __int64 v240; // [rsp+8E8h] [rbp-88h]
  char v241; // [rsp+8F0h] [rbp-80h] BYREF
  const char *v242; // [rsp+930h] [rbp-40h]

  memset(v177, 0, sizeof(v177));
  v10 = a5[9];
  if ( v10 )
  {
    v178.m128i_i64[0] = a5[9];
    v178.m128i_i64[1] = (__int64)&v180;
    v179 = 0x1000000000LL;
    v177[0] = v10;
    v177[2] = 0x1000000000LL;
    v177[1] = &v177[3];
    v201 = 0;
    v202 = &v205;
    v203 = 8;
    LODWORD(v204) = 0;
    BYTE4(v204) = 1;
    v213 = v215;
    v214 = 0x800000000LL;
    v233.m128i_i32[0] = 0;
    v233.m128i_i64[1] = 0;
    v234 = (void **)&v233;
    v235 = &v233;
    v236 = 0;
    sub_C8CF70((__int64)&v177[51], &v177[55], 8, (__int64)&v205, (__int64)&v201);
    v177[64] = 0x800000000LL;
    v177[63] = &v177[65];
    if ( (_DWORD)v214 )
      sub_28119A0((__int64)&v177[63], (__int64)&v213, v11, v12, v13, v14);
    if ( v233.m128i_i64[1] )
    {
      v177[91] = v233.m128i_i64[1];
      LODWORD(v177[90]) = v233.m128i_i32[0];
      v177[92] = v234;
      v177[93] = v235;
      *(_QWORD *)(v233.m128i_i64[1] + 8) = &v177[90];
      v233.m128i_i64[1] = 0;
      v177[94] = v236;
      v234 = (void **)&v233;
      v235 = &v233;
      v236 = 0;
    }
    else
    {
      LODWORD(v177[90]) = 0;
      v177[91] = 0;
      v177[92] = &v177[90];
      v177[93] = &v177[90];
      v177[94] = 0;
    }
    LOBYTE(v177[95]) = 1;
    sub_280F210(0);
    v15 = v213;
    v16 = &v213[3 * (unsigned int)v214];
    if ( v213 != v16 )
    {
      do
      {
        v17 = *(v16 - 1);
        v16 -= 3;
        if ( v17 != -4096 && v17 != 0 && v17 != -8192 )
          sub_BD60C0(v16);
      }
      while ( v15 != v16 );
      v16 = v213;
    }
    if ( v16 != v215 )
      _libc_free((unsigned __int64)v16);
    if ( !BYTE4(v204) )
      _libc_free((unsigned __int64)v202);
    v18 = v178.m128i_i64[1];
    v19 = (__int64 *)(v178.m128i_i64[1] + 24LL * (unsigned int)v179);
    if ( (__int64 *)v178.m128i_i64[1] != v19 )
    {
      do
      {
        v20 = *(v19 - 1);
        v19 -= 3;
        if ( v20 != 0 && v20 != -4096 && v20 != -8192 )
          sub_BD60C0(v19);
      }
      while ( (__int64 *)v18 != v19 );
      v19 = (__int64 *)v178.m128i_i64[1];
    }
    if ( v19 != &v180 )
      _libc_free((unsigned __int64)v19);
    if ( byte_4F8F8E8[0] )
      nullsub_390();
  }
  v21 = a5[6];
  v22 = a5[3];
  v112 = 0;
  v23 = a5[2];
  v24 = *a5;
  memset(v134, 0, 32);
  v134[7] = v22;
  v25 = *(unsigned int *)(a3 + 16);
  v134[8] = v21;
  v26 = *(__int64 ***)(a3 + 8);
  v27 = a5;
  v28 = a5[4];
  v134[5] = v24;
  v134[6] = v23;
  v29 = (__int64)v134;
  v134[4] = v28;
  v134[9] = 0;
  v123 = &v26[v25];
  v124 = v26;
  if ( v123 == v26 )
    goto LABEL_126;
  do
  {
    v30 = **v124;
    if ( !v30 )
      goto LABEL_29;
    v152 = *v124;
    v158 = (__int64 *)&v162;
    v151 = v30;
    v153 = 0;
    v154 = 0;
    v155 = 0;
    v156 = 0;
    v157 = 0;
    v159 = 4;
    v160 = 0;
    v161 = 1;
    v163 = 0;
    v164 = 0;
    v165 = 0;
    v166 = 0;
    v167 = 0;
    v168 = &v172;
    v169 = 4;
    v170 = 0;
    v171 = 1;
    v173 = 0;
    v174 = 0;
    v175 = 0;
    v176 = 0;
    v31 = sub_D440B0((__int64)v134, v30);
    v32 = (__int64 **)v27[6];
    v115 = v31;
    v33 = 0;
    v30 = v27[4];
    if ( LOBYTE(v177[95]) )
      v33 = v177;
    v121 = (__int64 *)v27[4];
    v119 = v33;
    v117 = v27[1];
    v120 = v27[3];
    v122 = v27[2];
    v34 = sub_280FC50((__int64)&v151, (__int64 *)v30, v32);
    if ( v34 )
    {
      if ( !(_BYTE)qword_4FFF128 )
        goto LABEL_38;
      v35 = *(_QWORD ***)(*(_QWORD *)(*(_QWORD *)v152[4] + 72LL) + 40LL);
      v109 = (__int64)(v35 + 39);
      v113 = *(_QWORD *)(v153 + 8);
      v36 = *((_QWORD *)v154 + 1);
      v37 = sub_AE44F0((__int64)(v35 + 39));
      v38 = *v35;
      v105 = v37;
      v39 = sub_AE44F0(v109);
      v107 = 0;
      v30 = v39;
      if ( v39 )
        v107 = sub_BCD140(v38, v39);
      if ( v113 != v36
        || (v53 = sub_BCB060(v113), v105 <= v53)
        || (v54 = sub_BCB060(v107), v25 = 2 * v53, v54 < (unsigned int)v25) )
      {
LABEL_38:
        if ( v173 )
          goto LABEL_39;
        goto LABEL_44;
      }
      v179 = (__int64)"loopflatten";
      v187[7] = &v190;
      v195 = &v197;
      v196 = 0x200000000LL;
      v178.m128i_i64[1] = v109;
      v178.m128i_i64[0] = (__int64)v121;
      LOBYTE(v180) = 1;
      v181 = 0;
      v182 = 0;
      v183 = 0;
      LODWORD(v184) = 0;
      v185 = 0;
      v186 = 0;
      memset(v187, 0, 56);
      v187[8] = 16;
      v188 = 0;
      v189 = 1;
      v191 = 0;
      v192 = 0;
      v193 = 0;
      v194 = 0;
      v198 = 0;
      v199 = 0;
      v200 = 0;
      v203 = (__int64)&v207;
      v127.m128i_i64[0] = (__int64)&v178;
      v55 = _mm_loadu_si128(&v131);
      v56 = _mm_loadu_si128(&v127);
      v135 = (unsigned int *)&unk_49E5698;
      v130 = (const char *)&unk_49DA0D8;
      v132 = (__int64 (__fastcall *)(__int64 *, __m128i *, int))sub_27BFDD0;
      v129 = v133;
      LOWORD(v214) = 1;
      v133 = sub_27BFD20;
      v127 = v55;
      v131 = v56;
      LODWORD(v201) = 0;
      v202 = 0;
      v204 = 2;
      v205 = 0;
      v206 = 1;
      v208 = 0;
      v209 = 0;
      v210 = 0;
      v211 = 0;
      v212 = 0;
      v213 = 0;
      BYTE2(v214) = 0;
      v128 = 0;
      v137 = v109;
      v138 = (__m128i)(unsigned __int64)v109;
      v136 = (__int64)&unk_49D94D0;
      v139 = 0u;
      v140 = 0u;
      v141 = 0u;
      LOWORD(v142) = 257;
      v57 = sub_B2BE50(*v121);
      v58 = _mm_loadu_si128(&v138);
      v220 = v57;
      v59 = _mm_loadu_si128(&v139);
      v221 = v229;
      v60 = _mm_loadu_si128(&v140);
      v222 = &v235;
      v61 = _mm_loadu_si128(&v141);
      v225 = 512;
      v219 = 0;
      v215[1] = 0x200000000LL;
      v229[2] = v137;
      v215[0] = &v216;
      v223 = 0;
      v224 = 0;
      v226 = 7;
      v227 = 0;
      v228 = 0;
      v217 = 0;
      v218 = 0;
      v229[0] = &unk_49E5698;
      v229[1] = &unk_49D94D0;
      v234 = v142;
      v235 = (__m128i *)&unk_49DA0D8;
      v237 = 0;
      v230 = v58;
      v231 = v59;
      v232 = v60;
      v233 = v61;
      if ( v132 )
      {
        v132(&v236, &v131, 2);
        v238 = v133;
        v237 = v132;
      }
      v135 = (unsigned int *)&unk_49E5698;
      v136 = (__int64)&unk_49D94D0;
      nullsub_63();
      nullsub_63();
      sub_B32BF0(&v130);
      if ( v128 )
        v128(&v127, &v127, 3);
      v30 = v120;
      v239 = &v241;
      v240 = 0x800000000LL;
      v242 = byte_3F871B3;
      v135 = (unsigned int *)&v137;
      v136 = 0x400000000LL;
      v130 = (const char *)v153;
      LODWORD(v126) = 0;
      v127.m128i_i32[0] = 0;
      v131.m128i_i64[0] = v107;
      v131.m128i_i8[8] = 0;
      if ( !sub_2A7DE70(
              (unsigned int)&v130,
              v120,
              (_DWORD)v121,
              (unsigned int)&v178,
              v122,
              (unsigned int)&v135,
              (__int64)&v126,
              (__int64)&v127,
              1,
              1) )
      {
        v66 = 0;
LABEL_83:
        v67 = v135;
        v68 = &v135[6 * (unsigned int)v136];
        if ( v135 != v68 )
        {
          do
          {
            v69 = *((_QWORD *)v68 - 1);
            v68 -= 6;
            if ( v69 != 0 && v69 != -4096 && v69 != -8192 )
              sub_BD60C0(v68);
          }
          while ( v67 != v68 );
          v68 = v135;
        }
        if ( v68 != (unsigned int *)&v137 )
          _libc_free((unsigned __int64)v68);
        sub_27C20B0((__int64)&v178);
        if ( v173 )
        {
          if ( v66 )
            goto LABEL_51;
LABEL_39:
          v112 = v34;
          goto LABEL_40;
        }
        if ( v66 )
          goto LABEL_51;
LABEL_44:
        v114 = sub_B2BEC0(*(_QWORD *)(**(_QWORD **)(v151 + 32) + 72LL));
        if ( (_BYTE)qword_4FFF208 )
          goto LABEL_51;
        v40 = (_QWORD *)(sub_D4B130(v151) + 48);
        v41 = *v40 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (_QWORD *)v41 == v40 )
        {
          v43 = 0;
        }
        else
        {
          if ( !v41 )
            BUG();
          v42 = *(unsigned __int8 *)(v41 - 24);
          v43 = v41 - 24;
          if ( (unsigned int)(v42 - 30) >= 0xB )
            v43 = 0;
        }
        v182 = v43;
        v178 = (__m128i)(unsigned __int64)v114;
        v180 = v122;
        v30 = v156;
        v181 = v117;
        v179 = 0;
        v183 = 0;
        v184 = 0;
        LOWORD(v185) = 257;
        v44 = sub_9AC590(v155, v156, &v178, 0);
        if ( v44 != 2 )
        {
          if ( v44 > 1 )
            goto LABEL_51;
          goto LABEL_40;
        }
        v70 = v158;
        if ( v161 )
        {
          v25 = HIDWORD(v159);
          v30 = (__int64)&v158[HIDWORD(v159)];
        }
        else
        {
          v25 = (unsigned int)v159;
          v30 = (__int64)&v158[(unsigned int)v159];
        }
        v118 = (__int64 *)v30;
        v29 = v30;
        if ( v158 == (__int64 *)v30 )
          goto LABEL_99;
        while ( 1 )
        {
          v71 = *v70;
          v25 = (__int64)v70;
          if ( (unsigned __int64)*v70 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( (__int64 *)v30 == ++v70 )
            goto LABEL_99;
        }
        v110 = v70;
        if ( v70 == (__int64 *)v30 )
        {
LABEL_99:
          if ( !(_BYTE)qword_4FFF048 )
            goto LABEL_40;
          v72 = v151;
          v73 = *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(v151 + 32) + 72LL) + 40LL);
          v74 = sub_BCB060(*(_QWORD *)(v156 + 8));
          v29 = *(_QWORD *)(v73 + 352);
          v25 = v74;
          v75 = *(unsigned __int8 **)(v73 + 344);
          v30 = v29 >> 2;
          v76 = &v75[v29];
          if ( v29 >> 2 > 0 )
          {
            v30 = (__int64)&v75[4 * v30];
            while ( 1 )
            {
              v29 = *v75;
              if ( v25 == v29 )
                break;
              v29 = v75[1];
              if ( v25 == v29 )
              {
                ++v75;
                break;
              }
              v29 = v75[2];
              if ( v25 == v29 )
              {
                v75 += 2;
                break;
              }
              v29 = v75[3];
              if ( v25 == v29 )
              {
                v75 += 3;
                break;
              }
              v75 += 4;
              if ( (unsigned __int8 *)v30 == v75 )
              {
                v29 = v76 - v75;
                goto LABEL_198;
              }
            }
LABEL_107:
            if ( v76 != v75 )
            {
              v77 = sub_D4B130(v72);
              sub_2A28870(&v178, v115, 0, 0, v151, v120, v122, v121);
              sub_F6D5D0((__int64)&v135, v178.m128i_i64[0], v78, v79, v80, v81);
              sub_2A28FB0(&v178, &v135);
              if ( v135 != (unsigned int *)&v137 )
                _libc_free((unsigned __int64)v135);
              v82 = *(_QWORD *)(v77 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v82 == v77 + 48 )
                goto LABEL_222;
              if ( !v82 )
                BUG();
              if ( (unsigned int)*(unsigned __int8 *)(v82 - 24) - 30 > 0xA )
              {
LABEL_222:
                v6 = sub_BD5C60(0);
                v143 = 0;
                v141.m128i_i64[0] = v6;
                v141.m128i_i64[1] = (__int64)&v149;
                v142 = &v150;
                v135 = (unsigned int *)&v137;
                v144 = 0;
                v149 = &unk_49DA100;
                v136 = 0x200000000LL;
                v145 = 512;
                v150 = &unk_49DA0B0;
                v146 = 7;
                v147 = 0;
                v148 = 0;
                v139.m128i_i64[1] = 0;
                v140.m128i_i64[0] = 0;
                v140.m128i_i16[4] = 0;
                sub_D5F1F0((__int64)&v135, 0);
                v130 = "flatten.mul";
                BYTE4(v126) = 0;
                v127.m128i_i64[1] = v155;
                LOWORD(v133) = 259;
                v127.m128i_i64[0] = v156;
                v125 = *(_QWORD *)(v156 + 8);
                v7 = sub_B33D10((__int64)&v135, 0x171u, (__int64)&v125, 1, (int)&v127, 2, v126, (__int64)&v130);
                v130 = "flatten.tripcount";
                LOWORD(v133) = 259;
                v127.m128i_i32[0] = 0;
                v176 = sub_94D3D0(&v135, v7, (__int64)&v127, 1, (__int64)&v130);
                v130 = "flatten.overflow";
                LOWORD(v133) = 259;
                v127.m128i_i32[0] = 1;
                sub_94D3D0(&v135, v7, (__int64)&v127, 1, (__int64)&v130);
                BUG();
              }
              v141.m128i_i64[0] = sub_BD5C60(v82 - 24);
              v141.m128i_i64[1] = (__int64)&v149;
              v145 = 512;
              v135 = (unsigned int *)&v137;
              v136 = 0x200000000LL;
              v149 = &unk_49DA100;
              v140.m128i_i16[4] = 0;
              v142 = &v150;
              v150 = &unk_49DA0B0;
              v143 = 0;
              v144 = 0;
              v146 = 7;
              v147 = 0;
              v148 = 0;
              v139.m128i_i64[1] = 0;
              v140.m128i_i64[0] = 0;
              sub_D5F1F0((__int64)&v135, v82 - 24);
              v130 = "flatten.mul";
              BYTE4(v126) = 0;
              v127.m128i_i64[1] = v155;
              LOWORD(v133) = 259;
              v127.m128i_i64[0] = v156;
              v125 = *(_QWORD *)(v156 + 8);
              v116 = sub_B33D10((__int64)&v135, 0x171u, (__int64)&v125, 1, (int)&v127, 2, v126, (__int64)&v130);
              v130 = "flatten.tripcount";
              LOWORD(v133) = 259;
              v127.m128i_i32[0] = 0;
              v176 = sub_94D3D0(&v135, v116, (__int64)&v127, 1, (__int64)&v130);
              v130 = "flatten.overflow";
              LOWORD(v133) = 259;
              v127.m128i_i32[0] = 1;
              v83 = sub_94D3D0(&v135, v116, (__int64)&v127, 1, (__int64)&v130);
              if ( *(_QWORD *)(v82 - 120) )
              {
                v84 = *(_QWORD *)(v82 - 112);
                **(_QWORD **)(v82 - 104) = v84;
                if ( v84 )
                  *(_QWORD *)(v84 + 16) = *(_QWORD *)(v82 - 104);
              }
              *(_QWORD *)(v82 - 120) = v83;
              if ( v83 )
              {
                v85 = *(_QWORD *)(v83 + 16);
                *(_QWORD *)(v82 - 112) = v85;
                if ( v85 )
                  *(_QWORD *)(v85 + 16) = v82 - 112;
                *(_QWORD *)(v82 - 104) = v83 + 16;
                *(_QWORD *)(v83 + 16) = v82 - 120;
              }
              nullsub_61();
              v149 = &unk_49DA100;
              nullsub_63();
              if ( v135 != (unsigned int *)&v137 )
                _libc_free((unsigned __int64)v135);
              sub_2808D60((__int64)&v178);
LABEL_51:
              v30 = v122;
              v112 |= sub_2810CB0(&v151, v122, v120, (__int64)v121, a6, v119);
            }
            goto LABEL_40;
          }
LABEL_198:
          if ( v29 != 2 )
          {
            if ( v29 != 3 )
            {
              if ( v29 != 1 )
                goto LABEL_40;
              goto LABEL_201;
            }
            v29 = *v75;
            if ( v25 == v29 )
              goto LABEL_107;
            ++v75;
          }
          v29 = *v75;
          if ( v25 == v29 )
            goto LABEL_107;
          ++v75;
LABEL_201:
          v29 = *v75;
          if ( v25 != v29 )
            goto LABEL_40;
          goto LABEL_107;
        }
        v108 = v27;
        if ( *(_BYTE *)v71 != 63 )
          goto LABEL_149;
        while ( (*(_DWORD *)(v71 + 4) & 0x7FFFFFF) == 2 )
        {
          v98 = *(_QWORD *)(v71 + 16);
          v99 = *(_QWORD *)(v71 - 32);
          if ( v98 )
          {
            while ( 1 )
            {
              v100 = *(_QWORD *)(v98 + 24);
              if ( *(_BYTE *)v100 != 61 )
              {
                if ( *(_BYTE *)v100 != 62 )
                  goto LABEL_176;
                if ( (*(_BYTE *)(v100 + 7) & 0x40) != 0 )
                {
                  v25 = *(_QWORD *)(v100 - 8);
                  if ( v71 != *(_QWORD *)(v25 + 32) )
                    goto LABEL_176;
                }
                else
                {
                  v25 = v100 - 32LL * (*(_DWORD *)(v100 + 4) & 0x7FFFFFF);
                  if ( v71 != *(_QWORD *)(v25 + 32) )
                    goto LABEL_176;
                }
              }
              v30 = (__int64)v152;
              if ( (unsigned __int8)sub_98D040(v100, (__int64)v152) )
              {
                if ( sub_B4DE30(v71) )
                {
                  v30 = *(_QWORD *)(v71 + 8);
                  v101 = *(_DWORD *)(*(_QWORD *)(v99 + 8) + 8LL) >> 8;
                  if ( v101 >= (unsigned int)sub_AE43A0(v114, v30) )
                    goto LABEL_164;
                }
              }
LABEL_176:
              v98 = *(_QWORD *)(v98 + 8);
              if ( !v98 )
                goto LABEL_149;
            }
          }
LABEL_165:
          v97 = v110 + 1;
          if ( v110 + 1 == v118 )
            goto LABEL_169;
          v29 = (__int64)v118;
          while ( 1 )
          {
            v71 = *v97;
            v25 = (__int64)v97;
            if ( (unsigned __int64)*v97 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v118 == ++v97 )
              goto LABEL_169;
          }
          v30 = (__int64)v118;
          v110 = v97;
          if ( v97 == v118 )
          {
LABEL_169:
            v27 = v108;
            goto LABEL_99;
          }
          if ( *(_BYTE *)v71 != 63 )
            break;
        }
LABEL_149:
        v92 = *(_QWORD *)(v71 + 16);
        if ( !v92 )
          goto LABEL_165;
        while ( 1 )
        {
          v93 = *(_QWORD *)(v92 + 24);
          if ( *(_BYTE *)v93 == 63 )
          {
            v94 = *(_QWORD *)(v93 + 16);
            if ( v94 )
              break;
          }
LABEL_151:
          v92 = *(_QWORD *)(v92 + 8);
          if ( !v92 )
            goto LABEL_165;
        }
        while ( 1 )
        {
          v95 = *(_QWORD *)(v94 + 24);
          if ( *(_BYTE *)v95 != 61 )
          {
            if ( *(_BYTE *)v95 != 62 )
              goto LABEL_156;
            if ( (*(_BYTE *)(v95 + 7) & 0x40) != 0 )
            {
              v25 = *(_QWORD *)(v95 - 8);
              if ( v93 != *(_QWORD *)(v25 + 32) )
                goto LABEL_156;
            }
            else
            {
              v25 = v95 - 32LL * (*(_DWORD *)(v95 + 4) & 0x7FFFFFF);
              if ( v93 != *(_QWORD *)(v25 + 32) )
                goto LABEL_156;
            }
          }
          v30 = (__int64)v152;
          if ( (unsigned __int8)sub_98D040(v95, (__int64)v152) )
          {
            if ( sub_B4DE30(v93) )
            {
              v30 = *(_QWORD *)(v93 + 8);
              v96 = *(_DWORD *)(*(_QWORD *)(v71 + 8) + 8LL) >> 8;
              if ( v96 >= (unsigned int)sub_AE43A0(v114, v30) )
              {
LABEL_164:
                v27 = v108;
                goto LABEL_51;
              }
            }
          }
LABEL_156:
          v94 = *(_QWORD *)(v94 + 8);
          if ( !v94 )
            goto LABEL_151;
        }
      }
      if ( !(unsigned __int8)sub_F5CB10((__int64)v130, 0, 0) )
      {
        if ( !v171 )
          goto LABEL_220;
        v104 = v168;
        v63 = HIDWORD(v169);
        v62 = &v168[8 * HIDWORD(v169)];
        if ( v168 != v62 )
        {
          while ( v153 != *(_QWORD *)v104 )
          {
            v104 += 8;
            if ( v62 == v104 )
              goto LABEL_218;
          }
          goto LABEL_81;
        }
LABEL_218:
        if ( HIDWORD(v169) < (unsigned int)v169 )
        {
          ++HIDWORD(v169);
          *(_QWORD *)v62 = v153;
          ++v167;
        }
        else
        {
LABEL_220:
          sub_C8CC70((__int64)&v167, v153, (__int64)v62, v63, v64, v65);
        }
      }
LABEL_81:
      v30 = v120;
      v66 = 0;
      v130 = v154;
      v131.m128i_i8[8] = 0;
      v131.m128i_i64[0] = v107;
      if ( sub_2A7DE70(
             (unsigned int)&v130,
             v120,
             (_DWORD)v121,
             (unsigned int)&v178,
             v122,
             (unsigned int)&v135,
             (__int64)&v126,
             (__int64)&v127,
             1,
             1) )
      {
        sub_F5CB10((__int64)v130, 0, 0);
        v30 = (__int64)v121;
        v173 = 1;
        v174 = v153;
        v175 = v154;
        v66 = sub_280FC50((__int64)&v151, v121, v32);
      }
      goto LABEL_83;
    }
LABEL_40:
    if ( v171 )
    {
      if ( v161 )
        goto LABEL_29;
    }
    else
    {
      _libc_free((unsigned __int64)v168);
      if ( v161 )
        goto LABEL_29;
    }
    _libc_free((unsigned __int64)v158);
LABEL_29:
    ++v124;
  }
  while ( v123 != v124 );
  if ( v112 )
  {
    if ( v27[9] && byte_4F8F8E8[0] )
    {
      v30 = 0;
      nullsub_390();
    }
    sub_22D0390((__int64)&v178, v30, v25, v29, (__int64)a5, a6);
    if ( v27[9] )
    {
      if ( BYTE4(v186) )
      {
        v47 = (__int64 **)&v184[HIDWORD(v185)];
        v48 = HIDWORD(v185);
        if ( v184 == (void **)v47 )
        {
LABEL_185:
          v50 = v186;
        }
        else
        {
          v49 = v184;
          while ( *v49 != &unk_4F8F810 )
          {
            if ( v47 == (__int64 **)++v49 )
              goto LABEL_185;
          }
          --HIDWORD(v185);
          v47 = (__int64 **)v184[HIDWORD(v185)];
          *v49 = v47;
          v48 = HIDWORD(v185);
          ++v183;
          v50 = v186;
        }
      }
      else
      {
        v102 = sub_C8CA60((__int64)&v183, (__int64)&unk_4F8F810);
        if ( v102 )
        {
          *v102 = -2;
          ++v183;
          v48 = HIDWORD(v185);
          v50 = v186 + 1;
          LODWORD(v186) = v186 + 1;
        }
        else
        {
          v48 = HIDWORD(v185);
          v50 = v186;
        }
      }
      if ( (_DWORD)v48 != v50 )
        goto LABEL_64;
      if ( BYTE4(v180) )
      {
        v51 = (void **)v178.m128i_i64[1];
        v103 = v178.m128i_i64[1] + 8LL * HIDWORD(v179);
        v48 = HIDWORD(v179);
        v47 = (__int64 **)v178.m128i_i64[1];
        if ( v178.m128i_i64[1] == v103 )
          goto LABEL_188;
        while ( *v47 != &qword_4F82400 )
        {
          if ( (__int64 **)v103 == ++v47 )
          {
LABEL_68:
            while ( *v51 != &unk_4F8F810 )
            {
              if ( v47 == (__int64 **)++v51 )
                goto LABEL_188;
            }
            break;
          }
        }
      }
      else if ( !sub_C8CA60((__int64)&v178, (__int64)&qword_4F82400) )
      {
LABEL_64:
        if ( !BYTE4(v180) )
        {
LABEL_190:
          sub_C8CC70((__int64)&v178, (__int64)&unk_4F8F810, (__int64)v47, v48, v45, v46);
          goto LABEL_69;
        }
        v51 = (void **)v178.m128i_i64[1];
        v48 = HIDWORD(v179);
        v47 = (__int64 **)(v178.m128i_i64[1] + 8LL * HIDWORD(v179));
        if ( v47 != (__int64 **)v178.m128i_i64[1] )
          goto LABEL_68;
LABEL_188:
        if ( (unsigned int)v179 > (unsigned int)v48 )
        {
          HIDWORD(v179) = v48 + 1;
          *v47 = (__int64 *)&unk_4F8F810;
          ++v178.m128i_i64[0];
          goto LABEL_69;
        }
        goto LABEL_190;
      }
    }
LABEL_69:
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)&v181, (__int64)&v178);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v187, (__int64)&v183);
    if ( BYTE4(v186) )
    {
      if ( BYTE4(v180) )
        goto LABEL_71;
    }
    else
    {
      _libc_free((unsigned __int64)v184);
      if ( BYTE4(v180) )
        goto LABEL_71;
    }
    _libc_free(v178.m128i_u64[1]);
    goto LABEL_71;
  }
LABEL_126:
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
LABEL_71:
  sub_233F0E0((__int64)v134);
  if ( LOBYTE(v177[95]) )
  {
    LOBYTE(v177[95]) = 0;
    sub_280F210((_QWORD *)v177[91]);
    v86 = (_QWORD *)v177[63];
    v87 = (_QWORD *)(v177[63] + 24LL * LODWORD(v177[64]));
    if ( (_QWORD *)v177[63] != v87 )
    {
      do
      {
        v88 = *(v87 - 1);
        v87 -= 3;
        if ( v88 != 0 && v88 != -4096 && v88 != -8192 )
          sub_BD60C0(v87);
      }
      while ( v86 != v87 );
      v87 = (_QWORD *)v177[63];
    }
    if ( v87 != &v177[65] )
      _libc_free((unsigned __int64)v87);
    if ( !BYTE4(v177[54]) )
      _libc_free(v177[52]);
    v89 = (_QWORD *)v177[1];
    v90 = (_QWORD *)(v177[1] + 24LL * LODWORD(v177[2]));
    if ( (_QWORD *)v177[1] != v90 )
    {
      do
      {
        v91 = *(v90 - 1);
        v90 -= 3;
        if ( v91 != 0 && v91 != -4096 && v91 != -8192 )
          sub_BD60C0(v90);
      }
      while ( v89 != v90 );
      v90 = (_QWORD *)v177[1];
    }
    if ( v90 != &v177[3] )
      _libc_free((unsigned __int64)v90);
  }
  return a1;
}
