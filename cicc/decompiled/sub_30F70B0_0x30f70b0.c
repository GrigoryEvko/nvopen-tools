// Function: sub_30F70B0
// Address: 0x30f70b0
//
void __fastcall sub_30F70B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __m128i v15; // xmm3
  __m128i v16; // xmm4
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __m128i v22; // xmm5
  __m128i v23; // xmm6
  __m128i v24; // xmm7
  __m128i v25; // xmm0
  __m128i v26; // xmm1
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __m128i v32; // xmm2
  __m128i v33; // xmm3
  __m128i v34; // xmm4
  __m128i v35; // xmm5
  __m128i v36; // xmm6
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __m128i v42; // xmm7
  __m128i v43; // xmm0
  __m128i v44; // xmm1
  __m128i v45; // xmm2
  __m128i v46; // xmm3
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // r8
  __int64 v60; // r9
  _QWORD *v61; // rsi
  __int64 v62; // rcx
  __int64 v63; // rdx
  unsigned __int64 v64; // rdi
  unsigned __int64 *v65; // rbx
  unsigned __int64 v66; // r12
  unsigned __int64 v67; // rdi
  unsigned __int64 v68; // rdi
  unsigned __int64 *v69; // rbx
  unsigned __int64 v70; // r12
  unsigned __int64 v71; // rdi
  unsigned __int64 v72; // rdi
  unsigned __int64 *v73; // rbx
  unsigned __int64 v74; // r12
  unsigned __int64 v75; // rdi
  unsigned __int64 v76; // rdi
  unsigned __int64 *v77; // rbx
  unsigned __int64 v78; // r12
  unsigned __int64 v79; // rdi
  unsigned __int64 v80; // rdi
  unsigned __int64 *v81; // rbx
  unsigned __int64 v82; // r12
  unsigned __int64 v83; // rdi
  unsigned __int64 v84; // rdi
  unsigned __int64 *v85; // rbx
  unsigned __int64 v86; // r12
  unsigned __int64 v87; // rdi
  unsigned __int64 v88; // rdi
  unsigned __int64 *v89; // rbx
  unsigned __int64 v90; // r12
  unsigned __int64 v91; // rdi
  unsigned __int64 v92; // rdi
  unsigned __int64 *v93; // rbx
  unsigned __int64 v94; // r12
  unsigned __int64 v95; // rdi
  unsigned __int64 v96; // rdi
  unsigned __int64 *v97; // rbx
  unsigned __int64 v98; // r12
  unsigned __int64 v99; // rdi
  unsigned __int64 v100; // rdi
  unsigned __int64 *v101; // rbx
  unsigned __int64 v102; // r12
  unsigned __int64 v103; // rdi
  unsigned __int64 v104; // rdi
  unsigned __int64 *v105; // rbx
  unsigned __int64 v106; // r12
  unsigned __int64 v107; // rdi
  unsigned __int64 v108; // rdi
  unsigned __int64 *v109; // rbx
  unsigned __int64 v110; // r12
  unsigned __int64 v111; // rdi
  __int64 v112[4]; // [rsp+20h] [rbp-990h] BYREF
  __int64 v113[4]; // [rsp+40h] [rbp-970h] BYREF
  __int64 v114[4]; // [rsp+60h] [rbp-950h] BYREF
  _BYTE v115[8]; // [rsp+80h] [rbp-930h] BYREF
  unsigned __int64 v116; // [rsp+88h] [rbp-928h]
  char v117; // [rsp+9Ch] [rbp-914h]
  _BYTE v118[64]; // [rsp+A0h] [rbp-910h] BYREF
  __m128i v119; // [rsp+E0h] [rbp-8D0h] BYREF
  __m128i v120; // [rsp+F0h] [rbp-8C0h] BYREF
  __m128i v121; // [rsp+100h] [rbp-8B0h] BYREF
  __m128i v122; // [rsp+110h] [rbp-8A0h] BYREF
  __m128i v123; // [rsp+120h] [rbp-890h] BYREF
  int v124; // [rsp+130h] [rbp-880h]
  _BYTE v125[8]; // [rsp+140h] [rbp-870h] BYREF
  unsigned __int64 v126; // [rsp+148h] [rbp-868h]
  char v127; // [rsp+15Ch] [rbp-854h]
  _BYTE v128[64]; // [rsp+160h] [rbp-850h] BYREF
  __m128i v129; // [rsp+1A0h] [rbp-810h] BYREF
  __m128i v130; // [rsp+1B0h] [rbp-800h]
  __m128i v131; // [rsp+1C0h] [rbp-7F0h]
  __m128i v132; // [rsp+1D0h] [rbp-7E0h]
  __m128i v133; // [rsp+1E0h] [rbp-7D0h]
  int v134; // [rsp+1F0h] [rbp-7C0h]
  _BYTE v135[8]; // [rsp+200h] [rbp-7B0h] BYREF
  unsigned __int64 v136; // [rsp+208h] [rbp-7A8h]
  char v137; // [rsp+21Ch] [rbp-794h]
  _BYTE v138[64]; // [rsp+220h] [rbp-790h] BYREF
  __m128i v139; // [rsp+260h] [rbp-750h] BYREF
  __m128i v140; // [rsp+270h] [rbp-740h] BYREF
  __m128i v141; // [rsp+280h] [rbp-730h] BYREF
  __m128i v142; // [rsp+290h] [rbp-720h] BYREF
  __m128i v143; // [rsp+2A0h] [rbp-710h] BYREF
  int v144; // [rsp+2B0h] [rbp-700h]
  _BYTE v145[8]; // [rsp+2C0h] [rbp-6F0h] BYREF
  unsigned __int64 v146; // [rsp+2C8h] [rbp-6E8h]
  char v147; // [rsp+2DCh] [rbp-6D4h]
  _BYTE v148[64]; // [rsp+2E0h] [rbp-6D0h] BYREF
  __m128i v149; // [rsp+320h] [rbp-690h] BYREF
  __m128i v150; // [rsp+330h] [rbp-680h]
  __m128i v151; // [rsp+340h] [rbp-670h]
  __m128i v152; // [rsp+350h] [rbp-660h]
  __m128i v153; // [rsp+360h] [rbp-650h]
  int v154; // [rsp+370h] [rbp-640h]
  _BYTE v155[8]; // [rsp+380h] [rbp-630h] BYREF
  unsigned __int64 v156; // [rsp+388h] [rbp-628h]
  char v157; // [rsp+39Ch] [rbp-614h]
  _BYTE v158[64]; // [rsp+3A0h] [rbp-610h] BYREF
  __m128i v159; // [rsp+3E0h] [rbp-5D0h] BYREF
  __m128i v160; // [rsp+3F0h] [rbp-5C0h]
  __m128i v161; // [rsp+400h] [rbp-5B0h]
  __m128i v162; // [rsp+410h] [rbp-5A0h]
  __m128i v163; // [rsp+420h] [rbp-590h]
  int v164; // [rsp+430h] [rbp-580h]
  _BYTE v165[8]; // [rsp+440h] [rbp-570h] BYREF
  unsigned __int64 v166; // [rsp+448h] [rbp-568h]
  char v167; // [rsp+45Ch] [rbp-554h]
  _BYTE v168[64]; // [rsp+460h] [rbp-550h] BYREF
  __m128i v169; // [rsp+4A0h] [rbp-510h] BYREF
  __m128i v170; // [rsp+4B0h] [rbp-500h] BYREF
  __m128i v171; // [rsp+4C0h] [rbp-4F0h] BYREF
  __m128i v172; // [rsp+4D0h] [rbp-4E0h] BYREF
  __m128i v173; // [rsp+4E0h] [rbp-4D0h] BYREF
  int v174; // [rsp+4F0h] [rbp-4C0h]
  _BYTE v175[8]; // [rsp+500h] [rbp-4B0h] BYREF
  unsigned __int64 v176; // [rsp+508h] [rbp-4A8h]
  char v177; // [rsp+51Ch] [rbp-494h]
  _BYTE v178[64]; // [rsp+520h] [rbp-490h] BYREF
  __m128i v179; // [rsp+560h] [rbp-450h] BYREF
  __m128i v180; // [rsp+570h] [rbp-440h]
  __m128i v181; // [rsp+580h] [rbp-430h]
  __m128i v182; // [rsp+590h] [rbp-420h]
  __m128i v183; // [rsp+5A0h] [rbp-410h]
  int v184; // [rsp+5B0h] [rbp-400h]
  _BYTE v185[8]; // [rsp+5C0h] [rbp-3F0h] BYREF
  unsigned __int64 v186; // [rsp+5C8h] [rbp-3E8h]
  char v187; // [rsp+5DCh] [rbp-3D4h]
  _BYTE v188[64]; // [rsp+5E0h] [rbp-3D0h] BYREF
  __m128i v189; // [rsp+620h] [rbp-390h] BYREF
  __m128i v190; // [rsp+630h] [rbp-380h] BYREF
  __m128i v191; // [rsp+640h] [rbp-370h] BYREF
  __m128i v192; // [rsp+650h] [rbp-360h] BYREF
  __m128i v193; // [rsp+660h] [rbp-350h] BYREF
  int v194; // [rsp+670h] [rbp-340h]
  _BYTE v195[8]; // [rsp+680h] [rbp-330h] BYREF
  unsigned __int64 v196; // [rsp+688h] [rbp-328h]
  char v197; // [rsp+69Ch] [rbp-314h]
  unsigned __int64 v198; // [rsp+6E0h] [rbp-2D0h]
  unsigned __int64 v199; // [rsp+708h] [rbp-2A8h]
  __int64 v200; // [rsp+728h] [rbp-288h]
  _BYTE v201[8]; // [rsp+740h] [rbp-270h] BYREF
  unsigned __int64 v202; // [rsp+748h] [rbp-268h]
  char v203; // [rsp+75Ch] [rbp-254h]
  unsigned __int64 v204; // [rsp+7A0h] [rbp-210h]
  unsigned __int64 v205; // [rsp+7C8h] [rbp-1E8h]
  __int64 v206; // [rsp+7E8h] [rbp-1C8h]
  _BYTE v207[8]; // [rsp+800h] [rbp-1B0h] BYREF
  unsigned __int64 v208; // [rsp+808h] [rbp-1A8h]
  char v209; // [rsp+81Ch] [rbp-194h]
  unsigned __int64 v210; // [rsp+860h] [rbp-150h]
  __int64 v211; // [rsp+870h] [rbp-140h]
  __int64 v212; // [rsp+878h] [rbp-138h]
  __int64 v213; // [rsp+880h] [rbp-130h]
  unsigned __int64 v214; // [rsp+888h] [rbp-128h]
  __int64 v215; // [rsp+890h] [rbp-120h]
  __int64 v216; // [rsp+898h] [rbp-118h]
  __int64 v217; // [rsp+8A8h] [rbp-108h]
  _BYTE v218[8]; // [rsp+8C0h] [rbp-F0h] BYREF
  unsigned __int64 v219; // [rsp+8C8h] [rbp-E8h]
  char v220; // [rsp+8DCh] [rbp-D4h]
  unsigned __int64 v221; // [rsp+920h] [rbp-90h]
  __int64 v222; // [rsp+930h] [rbp-80h]
  __int64 v223; // [rsp+938h] [rbp-78h]
  __int64 v224; // [rsp+940h] [rbp-70h]
  unsigned __int64 v225; // [rsp+948h] [rbp-68h]
  __int64 v226; // [rsp+950h] [rbp-60h]
  __int64 v227; // [rsp+958h] [rbp-58h]
  __int64 v228; // [rsp+960h] [rbp-50h]
  __int64 v229; // [rsp+968h] [rbp-48h]

  v6 = a3;
  sub_D53210((__int64)v135, a2, a3, a4, a5, a6);
  sub_C8CF70((__int64)v145, v148, 8, (__int64)v138, (__int64)v135);
  v149 = 0u;
  v150 = 0u;
  v151 = 0u;
  v152 = 0u;
  v153 = 0u;
  sub_D53100(v149.m128i_i64, 0, v7);
  if ( v139.m128i_i64[0] )
  {
    v11 = v150.m128i_i64[1];
    v10 = v151.m128i_i64[0];
    v9 = v152.m128i_i64[1];
    v8 = v153.m128i_i64[0];
    v12 = _mm_loadu_si128(&v139);
    v13 = _mm_loadu_si128(&v140);
    v139 = v149;
    v14 = _mm_loadu_si128(&v141);
    v15 = _mm_loadu_si128(&v142);
    v16 = _mm_loadu_si128(&v143);
    v140 = v150;
    v141 = v151;
    v142 = v152;
    v143 = v153;
    v149 = v12;
    v150 = v13;
    v151 = v14;
    v152 = v15;
    v153 = v16;
  }
  v154 = v144;
  sub_D53210((__int64)v115, a1, v8, v9, v10, v11);
  sub_C8CF70((__int64)v125, v128, 8, (__int64)v118, (__int64)v115);
  v129 = 0u;
  v130 = 0u;
  v131 = 0u;
  v132 = 0u;
  v133 = 0u;
  sub_D53100(v129.m128i_i64, 0, v17);
  if ( v119.m128i_i64[0] )
  {
    v21 = v130.m128i_i64[1];
    v20 = v131.m128i_i64[0];
    v19 = v132.m128i_i64[1];
    v18 = v133.m128i_i64[0];
    v22 = _mm_loadu_si128(&v119);
    v23 = _mm_loadu_si128(&v120);
    v119 = v129;
    v24 = _mm_loadu_si128(&v121);
    v25 = _mm_loadu_si128(&v122);
    v26 = _mm_loadu_si128(&v123);
    v120 = v130;
    v121 = v131;
    v122 = v132;
    v123 = v133;
    v129 = v22;
    v130 = v23;
    v131 = v24;
    v132 = v25;
    v133 = v26;
  }
  v134 = v124;
  sub_D53210((__int64)v165, (__int64)v145, v18, v19, v20, v21);
  sub_C8CF70((__int64)v155, v158, 8, (__int64)v168, (__int64)v165);
  v159 = 0u;
  v160 = 0u;
  v161 = 0u;
  v162 = 0u;
  v163 = 0u;
  sub_D53100(v159.m128i_i64, 0, v27);
  if ( v169.m128i_i64[0] )
  {
    v31 = v160.m128i_i64[1];
    v30 = v161.m128i_i64[0];
    v29 = v162.m128i_i64[1];
    v28 = v163.m128i_i64[0];
    v32 = _mm_loadu_si128(&v169);
    v33 = _mm_loadu_si128(&v170);
    v169 = v159;
    v34 = _mm_loadu_si128(&v171);
    v35 = _mm_loadu_si128(&v172);
    v36 = _mm_loadu_si128(&v173);
    v170 = v160;
    v171 = v161;
    v172 = v162;
    v173 = v163;
    v159 = v32;
    v160 = v33;
    v161 = v34;
    v162 = v35;
    v163 = v36;
  }
  v164 = v174;
  sub_D53210((__int64)v185, (__int64)v125, v28, v29, v30, v31);
  sub_C8CF70((__int64)v175, v178, 8, (__int64)v188, (__int64)v185);
  v179 = 0u;
  v180 = 0u;
  v181 = 0u;
  v182 = 0u;
  v183 = 0u;
  sub_D53100(v179.m128i_i64, 0, v37);
  if ( v189.m128i_i64[0] )
  {
    v41 = v180.m128i_i64[1];
    v40 = v181.m128i_i64[0];
    v39 = v182.m128i_i64[1];
    v38 = v183.m128i_i64[0];
    v42 = _mm_loadu_si128(&v189);
    v43 = _mm_loadu_si128(&v190);
    v189 = v179;
    v44 = _mm_loadu_si128(&v191);
    v45 = _mm_loadu_si128(&v192);
    v46 = _mm_loadu_si128(&v193);
    v190 = v180;
    v191 = v181;
    v192 = v182;
    v193 = v183;
    v179 = v42;
    v180 = v43;
    v181 = v44;
    v182 = v45;
    v183 = v46;
  }
  v184 = v194;
  sub_D53210((__int64)v195, (__int64)v155, v38, v39, v40, v41);
  sub_D53210((__int64)v201, (__int64)v175, v47, v48, v49, v50);
  sub_D53210((__int64)v207, (__int64)v195, v51, v52, v53, v54);
  sub_D53210((__int64)v218, (__int64)v201, v55, v56, v57, v58);
  while ( 1 )
  {
    v59 = v213;
    v60 = v211;
    v61 = (_QWORD *)v222;
    v62 = ((v215 - v216) >> 5) + 16 * (((__int64)(v217 - v214) >> 3) - 1) + ((v213 - v211) >> 5);
    v63 = (v224 - v222) >> 5;
    if ( v62 != v63 + 16 * (((__int64)(v229 - v225) >> 3) - 1) + ((v226 - v227) >> 5) )
      goto LABEL_10;
    v112[0] = v222;
    v114[3] = v214;
    v113[2] = v228;
    v113[0] = v226;
    v112[1] = v223;
    v113[1] = v227;
    v113[3] = v229;
    v112[2] = v224;
    v112[3] = v225;
    v114[0] = v211;
    v114[1] = v212;
    v114[2] = v213;
    if ( (unsigned __int8)sub_D542B0(v112, v113, v114) )
      break;
    v61 = (_QWORD *)v222;
LABEL_10:
    v6 += 8;
    *(_QWORD *)(v6 - 8) = *v61;
    sub_D53E10((__int64)v218, (__int64)v61, (__int64 *)v63, v62, v59, v60);
  }
  v64 = v221;
  if ( v221 )
  {
    v65 = (unsigned __int64 *)v225;
    v66 = v229 + 8;
    if ( v229 + 8 > v225 )
    {
      do
      {
        v67 = *v65++;
        j_j___libc_free_0(v67);
      }
      while ( v66 > (unsigned __int64)v65 );
      v64 = v221;
    }
    j_j___libc_free_0(v64);
  }
  if ( !v220 )
    _libc_free(v219);
  v68 = v210;
  if ( v210 )
  {
    v69 = (unsigned __int64 *)v214;
    v70 = v217 + 8;
    if ( v217 + 8 > v214 )
    {
      do
      {
        v71 = *v69++;
        j_j___libc_free_0(v71);
      }
      while ( v70 > (unsigned __int64)v69 );
      v68 = v210;
    }
    j_j___libc_free_0(v68);
  }
  if ( !v209 )
    _libc_free(v208);
  v72 = v204;
  if ( v204 )
  {
    v73 = (unsigned __int64 *)v205;
    v74 = v206 + 8;
    if ( v206 + 8 > v205 )
    {
      do
      {
        v75 = *v73++;
        j_j___libc_free_0(v75);
      }
      while ( v74 > (unsigned __int64)v73 );
      v72 = v204;
    }
    j_j___libc_free_0(v72);
  }
  if ( !v203 )
    _libc_free(v202);
  v76 = v198;
  if ( v198 )
  {
    v77 = (unsigned __int64 *)v199;
    v78 = v200 + 8;
    if ( v200 + 8 > v199 )
    {
      do
      {
        v79 = *v77++;
        j_j___libc_free_0(v79);
      }
      while ( v78 > (unsigned __int64)v77 );
      v76 = v198;
    }
    j_j___libc_free_0(v76);
  }
  if ( !v197 )
    _libc_free(v196);
  v80 = v179.m128i_i64[0];
  if ( v179.m128i_i64[0] )
  {
    v81 = (unsigned __int64 *)v181.m128i_i64[1];
    v82 = v183.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v183.m128i_i64[1] + 8) > v181.m128i_i64[1] )
    {
      do
      {
        v83 = *v81++;
        j_j___libc_free_0(v83);
      }
      while ( v82 > (unsigned __int64)v81 );
      v80 = v179.m128i_i64[0];
    }
    j_j___libc_free_0(v80);
  }
  if ( !v177 )
    _libc_free(v176);
  v84 = v189.m128i_i64[0];
  if ( v189.m128i_i64[0] )
  {
    v85 = (unsigned __int64 *)v191.m128i_i64[1];
    v86 = v193.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v193.m128i_i64[1] + 8) > v191.m128i_i64[1] )
    {
      do
      {
        v87 = *v85++;
        j_j___libc_free_0(v87);
      }
      while ( v86 > (unsigned __int64)v85 );
      v84 = v189.m128i_i64[0];
    }
    j_j___libc_free_0(v84);
  }
  if ( !v187 )
    _libc_free(v186);
  v88 = v159.m128i_i64[0];
  if ( v159.m128i_i64[0] )
  {
    v89 = (unsigned __int64 *)v161.m128i_i64[1];
    v90 = v163.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v163.m128i_i64[1] + 8) > v161.m128i_i64[1] )
    {
      do
      {
        v91 = *v89++;
        j_j___libc_free_0(v91);
      }
      while ( v90 > (unsigned __int64)v89 );
      v88 = v159.m128i_i64[0];
    }
    j_j___libc_free_0(v88);
  }
  if ( !v157 )
    _libc_free(v156);
  v92 = v169.m128i_i64[0];
  if ( v169.m128i_i64[0] )
  {
    v93 = (unsigned __int64 *)v171.m128i_i64[1];
    v94 = v173.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v173.m128i_i64[1] + 8) > v171.m128i_i64[1] )
    {
      do
      {
        v95 = *v93++;
        j_j___libc_free_0(v95);
      }
      while ( v94 > (unsigned __int64)v93 );
      v92 = v169.m128i_i64[0];
    }
    j_j___libc_free_0(v92);
  }
  if ( !v167 )
    _libc_free(v166);
  v96 = v129.m128i_i64[0];
  if ( v129.m128i_i64[0] )
  {
    v97 = (unsigned __int64 *)v131.m128i_i64[1];
    v98 = v133.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v133.m128i_i64[1] + 8) > v131.m128i_i64[1] )
    {
      do
      {
        v99 = *v97++;
        j_j___libc_free_0(v99);
      }
      while ( v98 > (unsigned __int64)v97 );
      v96 = v129.m128i_i64[0];
    }
    j_j___libc_free_0(v96);
  }
  if ( !v127 )
    _libc_free(v126);
  v100 = v119.m128i_i64[0];
  if ( v119.m128i_i64[0] )
  {
    v101 = (unsigned __int64 *)v121.m128i_i64[1];
    v102 = v123.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v123.m128i_i64[1] + 8) > v121.m128i_i64[1] )
    {
      do
      {
        v103 = *v101++;
        j_j___libc_free_0(v103);
      }
      while ( v102 > (unsigned __int64)v101 );
      v100 = v119.m128i_i64[0];
    }
    j_j___libc_free_0(v100);
  }
  if ( !v117 )
    _libc_free(v116);
  v104 = v149.m128i_i64[0];
  if ( v149.m128i_i64[0] )
  {
    v105 = (unsigned __int64 *)v151.m128i_i64[1];
    v106 = v153.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v153.m128i_i64[1] + 8) > v151.m128i_i64[1] )
    {
      do
      {
        v107 = *v105++;
        j_j___libc_free_0(v107);
      }
      while ( v106 > (unsigned __int64)v105 );
      v104 = v149.m128i_i64[0];
    }
    j_j___libc_free_0(v104);
  }
  if ( !v147 )
    _libc_free(v146);
  v108 = v139.m128i_i64[0];
  if ( v139.m128i_i64[0] )
  {
    v109 = (unsigned __int64 *)v141.m128i_i64[1];
    v110 = v143.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v143.m128i_i64[1] + 8) > v141.m128i_i64[1] )
    {
      do
      {
        v111 = *v109++;
        j_j___libc_free_0(v111);
      }
      while ( v110 > (unsigned __int64)v109 );
      v108 = v139.m128i_i64[0];
    }
    j_j___libc_free_0(v108);
  }
  if ( !v137 )
    _libc_free(v136);
}
