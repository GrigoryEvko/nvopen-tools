// Function: sub_288E950
// Address: 0x288e950
//
__int64 __fastcall sub_288E950(__int64 a1, __int64 a2, __int64 a3, __int64 a4, void **a5)
{
  __int64 v6; // r12
  void *v8; // r14
  __int64 *v9; // r13
  __int64 v10; // rsi
  __int64 v11; // rcx
  unsigned int v12; // ebx
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 *v18; // r12
  void *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // r14
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __m128i v33; // xmm6
  __m128i v34; // xmm7
  __m128i v35; // xmm1
  unsigned __int64 *v36; // r14
  unsigned __int64 *v37; // rbx
  unsigned __int64 *v38; // r12
  unsigned __int64 v39; // rdi
  __int64 v40; // r13
  __int64 v41; // rdx
  int v42; // r14d
  __int64 v43; // rsi
  __int64 v44; // rcx
  _QWORD *v45; // rax
  _QWORD *v46; // rdx
  __int64 v47; // rdx
  __int64 *v48; // rax
  unsigned __int64 *v49; // r12
  __int64 v50; // rax
  unsigned __int64 v51; // rdi
  __int64 v52; // r13
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  _QWORD *v57; // rdx
  unsigned int v58; // eax
  __int64 v59; // rax
  __int64 v60; // r12
  __int64 v61; // r13
  __int64 *v62; // rax
  __int64 v63; // rbx
  __int64 v64; // rcx
  __int64 v65; // r12
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r8
  __m128i v70; // xmm3
  __m128i v71; // xmm4
  __m128i v72; // xmm5
  unsigned __int64 *v73; // r13
  unsigned __int64 *v74; // rbx
  unsigned __int64 v75; // rdi
  __int64 *v76; // rax
  unsigned __int64 v77; // rsi
  unsigned int v78; // eax
  __m128i v79; // xmm0
  __int64 *v80; // rax
  __int64 *v81; // rax
  unsigned __int64 *v82; // r13
  unsigned __int64 *v83; // rbx
  int v84; // ecx
  __int64 v85; // rdi
  int v86; // ecx
  unsigned int v87; // edx
  __int64 *v88; // rax
  __int64 v89; // r10
  unsigned __int32 v90; // eax
  _QWORD *v91; // rbx
  __int64 v92; // r13
  __int64 v93; // rax
  __int64 *v94; // rbx
  __int64 *v95; // r12
  unsigned __int64 v96; // rdx
  _QWORD *v97; // r14
  _QWORD *v98; // r15
  __int64 v99; // rax
  unsigned __int64 v100; // rdi
  __int64 v101; // r13
  __int64 v102; // rax
  __int64 v103; // rdx
  __int64 v104; // rcx
  __int64 v105; // r8
  __int64 v106; // r9
  __int64 v107; // r13
  __int64 v108; // rdx
  __int64 v109; // r8
  __int64 v110; // r9
  __m128i v111; // xmm6
  __m128i v112; // xmm7
  __m128i v113; // xmm0
  unsigned __int64 *v114; // r14
  __int64 v115; // r8
  unsigned __int64 *v116; // r13
  unsigned __int64 v117; // rdi
  unsigned __int64 *v118; // rbx
  unsigned __int64 *v119; // r13
  unsigned __int64 v120; // rdi
  __int64 v121; // rdx
  __int64 v122; // rcx
  __int64 v123; // r8
  __int64 v124; // r9
  __int64 v125; // rcx
  __int64 v126; // r8
  __int64 v127; // r9
  __int64 v128; // rcx
  __int64 v129; // r8
  __int64 v130; // r9
  __int64 v131; // rcx
  __int64 v132; // r8
  __int64 v133; // r9
  _QWORD *v134; // rdx
  unsigned __int64 v135; // rax
  int v136; // edx
  __int64 v137; // rdi
  __int64 v138; // rax
  __int64 v139; // rax
  __int64 v140; // rax
  __int64 *v141; // r14
  __int64 v142; // rcx
  __int64 v143; // rax
  __int64 v144; // rbx
  __int64 i; // r13
  __int64 v146; // r12
  __int64 *v147; // r15
  __int64 *v148; // rax
  __int64 v149; // rdi
  __int64 v150; // r14
  __int64 v151; // rax
  __int64 *v152; // r14
  __int64 *v153; // r15
  __int64 *v154; // rax
  __int64 v155; // rdi
  __int64 v156; // r14
  __int64 v157; // rax
  __int64 v158; // rsi
  _QWORD *v159; // r13
  unsigned __int64 v160; // r12
  __int64 v161; // rax
  __int64 v162; // rdx
  __int64 v163; // rax
  __int64 v164; // rsi
  __int64 v165; // rdx
  __int64 v166; // rcx
  __int64 v167; // r8
  __int64 v168; // r9
  __int64 *v169; // rdx
  int v170; // eax
  int v171; // edi
  unsigned __int8 v172; // cl
  __int64 *v173; // rax
  __int64 v174; // r10
  __int64 *v175; // rcx
  __int64 *v176; // rax
  __int64 v177; // r10
  __int64 v178; // r11
  unsigned __int32 v179; // eax
  _QWORD *v180; // rbx
  __int64 v181; // r12
  __int64 v182; // rsi
  int v183; // eax
  int v184; // r8d
  unsigned __int64 *v185; // r12
  unsigned __int64 v186; // rdi
  __int64 v187; // rax
  __int64 v188; // rdx
  __int64 v189; // rcx
  __int64 v190; // r8
  __int64 v191; // r9
  __int64 v192; // r14
  __int64 v193; // rdx
  __int64 v194; // rcx
  __int64 v195; // r8
  __int64 v196; // r9
  __m128i v197; // xmm0
  __m128i v198; // xmm6
  __m128i v199; // xmm7
  unsigned __int64 *v200; // r8
  unsigned __int64 *v201; // rbx
  unsigned __int64 v202; // rdi
  __int64 v203; // rax
  __int64 v204; // rax
  __int64 v205; // rax
  __int64 v206; // rax
  __int64 v207; // rax
  __int64 v208; // rax
  unsigned __int64 *v209; // r12
  unsigned __int64 *v210; // r14
  unsigned __int64 v211; // rdi
  __int64 v212; // rax
  __int64 v213; // rax
  __int64 v214; // r13
  __int64 v215; // rax
  __int64 v216; // rdx
  __int64 v217; // rcx
  __int64 v218; // r8
  __int64 v219; // r9
  __int64 v220; // rbx
  __int64 v221; // r12
  __int64 v222; // rdx
  __int64 v223; // rcx
  __int64 v224; // r9
  __int64 v225; // r12
  __m128i v226; // xmm1
  __m128i v227; // xmm1
  __int64 v228; // r8
  unsigned __int64 *v229; // r8
  unsigned __int64 *v230; // r14
  unsigned __int64 *v231; // r12
  unsigned __int64 v232; // rdi
  unsigned __int64 *v233; // rbx
  unsigned __int64 v234; // rdi
  __int64 v235; // r13
  __int64 v236; // rax
  __int64 v237; // rdx
  __int64 v238; // rcx
  __int64 v239; // r8
  __int64 v240; // r12
  __int64 v241; // rdx
  __int64 v242; // rcx
  __int64 v243; // r8
  __int64 v244; // r9
  __int64 v245; // r12
  __m128i v246; // xmm7
  __m128i v247; // xmm1
  unsigned __int64 *v248; // r8
  unsigned __int64 *v249; // r14
  unsigned __int64 *v250; // r12
  unsigned __int64 v251; // rdi
  unsigned __int64 *v252; // rbx
  unsigned __int64 v253; // rdi
  __int64 v254; // rax
  __int64 v255; // rax
  __int64 v256; // rax
  __int64 v257; // rax
  __int64 v258; // [rsp+0h] [rbp-630h]
  char v259; // [rsp+Fh] [rbp-621h]
  __int64 v260; // [rsp+10h] [rbp-620h]
  __int64 v261; // [rsp+18h] [rbp-618h]
  __int64 v262; // [rsp+18h] [rbp-618h]
  int v263; // [rsp+20h] [rbp-610h]
  int v264; // [rsp+28h] [rbp-608h]
  unsigned __int8 v265; // [rsp+30h] [rbp-600h]
  __int64 v266; // [rsp+30h] [rbp-600h]
  float v267; // [rsp+38h] [rbp-5F8h]
  __int64 v268; // [rsp+48h] [rbp-5E8h]
  __int64 v269; // [rsp+48h] [rbp-5E8h]
  __int64 *v270; // [rsp+50h] [rbp-5E0h]
  void *v271; // [rsp+58h] [rbp-5D8h]
  __int64 v272; // [rsp+58h] [rbp-5D8h]
  __int64 v273; // [rsp+60h] [rbp-5D0h]
  __int64 v274; // [rsp+60h] [rbp-5D0h]
  void *v275; // [rsp+68h] [rbp-5C8h]
  __int64 v276; // [rsp+68h] [rbp-5C8h]
  __int64 v277; // [rsp+68h] [rbp-5C8h]
  __int64 v278; // [rsp+78h] [rbp-5B8h] BYREF
  __m128i v279; // [rsp+80h] [rbp-5B0h] BYREF
  __int64 v280[2]; // [rsp+90h] [rbp-5A0h] BYREF
  __int64 *v281; // [rsp+A0h] [rbp-590h]
  __int64 v282; // [rsp+B0h] [rbp-580h] BYREF
  __int64 v283; // [rsp+B8h] [rbp-578h]
  __int64 v284; // [rsp+C0h] [rbp-570h]
  __int64 v285; // [rsp+C8h] [rbp-568h]
  __int64 *v286; // [rsp+D0h] [rbp-560h]
  void *v287; // [rsp+D8h] [rbp-558h]
  void *v288; // [rsp+E0h] [rbp-550h]
  __int64 v289; // [rsp+E8h] [rbp-548h]
  __int64 v290; // [rsp+F0h] [rbp-540h]
  __int64 v291; // [rsp+F8h] [rbp-538h]
  __m128i v292; // [rsp+100h] [rbp-530h] BYREF
  _QWORD v293[2]; // [rsp+110h] [rbp-520h] BYREF
  _QWORD *v294; // [rsp+120h] [rbp-510h]
  _QWORD v295[4]; // [rsp+130h] [rbp-500h] BYREF
  __m128i v296; // [rsp+150h] [rbp-4E0h] BYREF
  __int64 v297; // [rsp+160h] [rbp-4D0h] BYREF
  __int64 v298; // [rsp+168h] [rbp-4C8h]
  _BYTE *v299; // [rsp+170h] [rbp-4C0h]
  __int64 v300; // [rsp+178h] [rbp-4B8h]
  _QWORD v301[2]; // [rsp+180h] [rbp-4B0h] BYREF
  __m128i v302; // [rsp+190h] [rbp-4A0h] BYREF
  void **v303; // [rsp+1A0h] [rbp-490h] BYREF
  __int64 v304; // [rsp+1A8h] [rbp-488h] BYREF
  __int64 *v305; // [rsp+1B0h] [rbp-480h] BYREF
  __m128i v306; // [rsp+1B8h] [rbp-478h] BYREF
  __int64 v307; // [rsp+1C8h] [rbp-468h]
  __m128i v308; // [rsp+1D0h] [rbp-460h] BYREF
  __m128i v309; // [rsp+1E0h] [rbp-450h]
  unsigned __int64 *v310; // [rsp+1F0h] [rbp-440h] BYREF
  __int64 v311; // [rsp+1F8h] [rbp-438h]
  _BYTE v312[320]; // [rsp+200h] [rbp-430h] BYREF
  char v313; // [rsp+340h] [rbp-2F0h]
  int v314; // [rsp+344h] [rbp-2ECh]
  __int64 v315; // [rsp+348h] [rbp-2E8h]
  void *v316; // [rsp+350h] [rbp-2E0h] BYREF
  __int64 v317; // [rsp+358h] [rbp-2D8h]
  __int64 *v318; // [rsp+360h] [rbp-2D0h]
  __m128i v319; // [rsp+368h] [rbp-2C8h] BYREF
  __int64 v320; // [rsp+378h] [rbp-2B8h]
  __m128i v321; // [rsp+380h] [rbp-2B0h] BYREF
  __m128i v322; // [rsp+390h] [rbp-2A0h] BYREF
  unsigned __int64 *v323; // [rsp+3A0h] [rbp-290h] BYREF
  unsigned int v324; // [rsp+3A8h] [rbp-288h]
  unsigned __int64 v325[2]; // [rsp+3B0h] [rbp-280h] BYREF
  char v326; // [rsp+3C0h] [rbp-270h] BYREF
  __int64 v327; // [rsp+410h] [rbp-220h]
  unsigned int v328; // [rsp+420h] [rbp-210h]
  __int64 v329; // [rsp+430h] [rbp-200h]
  unsigned int v330; // [rsp+440h] [rbp-1F0h]
  __int64 v331; // [rsp+450h] [rbp-1E0h]
  unsigned int v332; // [rsp+460h] [rbp-1D0h]
  _QWORD v333[2]; // [rsp+4B0h] [rbp-180h] BYREF
  char v334; // [rsp+4C0h] [rbp-170h]
  _BYTE *v335; // [rsp+4C8h] [rbp-168h]
  __int64 v336; // [rsp+4D0h] [rbp-160h]
  _BYTE v337[24]; // [rsp+4D8h] [rbp-158h] BYREF
  char v338; // [rsp+4F0h] [rbp-140h]
  int v339; // [rsp+4F4h] [rbp-13Ch]
  __int64 v340; // [rsp+4F8h] [rbp-138h]
  __int16 v341; // [rsp+558h] [rbp-D8h]
  _QWORD v342[2]; // [rsp+560h] [rbp-D0h] BYREF
  __int64 v343; // [rsp+570h] [rbp-C0h]
  __int64 v344; // [rsp+578h] [rbp-B8h] BYREF
  unsigned int v345; // [rsp+580h] [rbp-B0h]
  char v346; // [rsp+5F8h] [rbp-38h] BYREF

  v6 = a3;
  v8 = *a5;
  v9 = (__int64 *)a5[4];
  v275 = a5[2];
  v271 = *a5;
  v270 = v9;
  v10 = *(_QWORD *)(**(_QWORD **)(a3 + 32) + 72LL);
  sub_1049690(v280, v10);
  v11 = (__int64)a5[3];
  v282 = 0;
  v268 = v11;
  v12 = qword_5003868;
  v289 = v11;
  v283 = 0;
  v284 = 0;
  v285 = 0;
  v286 = v9;
  v287 = v8;
  v288 = v275;
  v290 = 0;
  v291 = 0;
  v267 = *(float *)&qword_5003948;
  if ( (sub_F6E950(v6, v10, v13, v11, v14, v15) & 2) != 0 )
    goto LABEL_2;
  v20 = (void *)sub_D4A110(v6, "llvm.loop.licm_versioning.disable", 0x21u, (__int64)&v282, v16, v17);
  v317 = v21;
  v316 = v20;
  if ( (_BYTE)v21 )
    goto LABEL_2;
  v265 = sub_D4B3D0(v6);
  if ( !v265 )
    goto LABEL_9;
  if ( *(_QWORD *)(v6 + 16) != *(_QWORD *)(v6 + 8) )
    goto LABEL_9;
  v40 = *(_QWORD *)(**(_QWORD **)(v6 + 32) + 16LL);
  if ( !v40 )
    goto LABEL_9;
  while ( 1 )
  {
    v41 = *(_QWORD *)(v40 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v41 - 30) <= 0xAu )
      break;
    v40 = *(_QWORD *)(v40 + 8);
    if ( !v40 )
      goto LABEL_9;
  }
  v42 = 0;
  v43 = *(_QWORD *)(v41 + 40);
  v44 = v6 + 56;
  if ( !*(_BYTE *)(v6 + 84) )
    goto LABEL_36;
LABEL_28:
  v45 = *(_QWORD **)(v6 + 64);
  v46 = &v45[*(unsigned int *)(v6 + 76)];
  if ( v45 != v46 )
  {
    while ( v43 != *v45 )
    {
      if ( v46 == ++v45 )
        goto LABEL_33;
    }
LABEL_32:
    ++v42;
  }
LABEL_33:
  while ( 1 )
  {
    v40 = *(_QWORD *)(v40 + 8);
    if ( !v40 )
      break;
    v47 = *(_QWORD *)(v40 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v47 - 30) <= 0xAu )
    {
      v43 = *(_QWORD *)(v47 + 40);
      if ( *(_BYTE *)(v6 + 84) )
        goto LABEL_28;
LABEL_36:
      v273 = v44;
      v48 = sub_C8CA60(v44, v43);
      v44 = v273;
      if ( v48 )
        goto LABEL_32;
    }
  }
  if ( v42 != 1 )
    goto LABEL_9;
  if ( !sub_D46F00(v6) )
    goto LABEL_9;
  v52 = sub_D46F00(v6);
  if ( v52 != sub_D47930(v6) || (unsigned __int8)sub_D497B0(v6, v43, v53, v54, v55, v56) )
    goto LABEL_9;
  v57 = *(_QWORD **)v6;
  v58 = 1;
  if ( *(_QWORD *)v6 )
  {
    do
    {
      v57 = (_QWORD *)*v57;
      ++v58;
    }
    while ( v57 );
  }
  if ( v12 < v58 || (v59 = sub_DCF3A0(v270, (char *)v6, 0), sub_D96A50(v59)) )
  {
LABEL_9:
    v22 = v280[0];
    v23 = sub_B2BE50(v280[0]);
    if ( !sub_B6EA50(v23) )
    {
      v205 = sub_B2BE50(v22);
      v206 = sub_B6F970(v205);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v206 + 48LL))(v206) )
        goto LABEL_2;
    }
    v28 = **(_QWORD **)(v6 + 32);
    sub_D4BD20(&v292, v6, v24, v25, v26, v27);
    sub_B157E0((__int64)&v296, &v292);
    sub_B17640((__int64)&v316, (__int64)"loop-versioning-licm", (__int64)"IllegalLoopStruct", 17, &v296, v28);
    sub_B18290((__int64)&v316, " Unsafe Loop structure", 0x16u);
    v33 = _mm_loadu_si128(&v319);
    v34 = _mm_loadu_si128(&v321);
    LODWORD(v304) = v317;
    v35 = _mm_loadu_si128(&v322);
    v303 = (void **)&unk_49D9D40;
    BYTE4(v304) = BYTE4(v317);
    v306 = v33;
    v305 = v318;
    v308 = v34;
    v307 = v320;
    v310 = (unsigned __int64 *)v312;
    v311 = 0x400000000LL;
    v309 = v35;
    if ( v324 )
    {
      sub_288E6D0((__int64)&v310, (__int64)&v323, v29, v30, v31, v32);
      v316 = &unk_49D9D40;
      v49 = v323;
      v313 = v338;
      v314 = v339;
      v315 = v340;
      v303 = (void **)&unk_49D9DB0;
      v50 = 10LL * v324;
      v36 = &v323[v50];
      if ( v323 != &v323[v50] )
      {
        do
        {
          v36 -= 10;
          v51 = v36[4];
          if ( (unsigned __int64 *)v51 != v36 + 6 )
            j_j___libc_free_0(v51);
          if ( (unsigned __int64 *)*v36 != v36 + 2 )
            j_j___libc_free_0(*v36);
        }
        while ( v49 != v36 );
        v36 = v323;
      }
    }
    else
    {
      v36 = v323;
      v313 = v338;
      v314 = v339;
      v315 = v340;
      v303 = (void **)&unk_49D9DB0;
    }
    if ( v36 != v325 )
      _libc_free((unsigned __int64)v36);
    if ( v292.m128i_i64[0] )
      sub_B91220((__int64)&v292, v292.m128i_i64[0]);
    sub_1049740(v280, (__int64)&v303);
    v37 = v310;
    v303 = (void **)&unk_49D9D40;
    v38 = &v310[10 * (unsigned int)v311];
    if ( v310 != v38 )
    {
      do
      {
        v38 -= 10;
        v39 = v38[4];
        if ( (unsigned __int64 *)v39 != v38 + 6 )
          j_j___libc_free_0(v39);
        if ( (unsigned __int64 *)*v38 != v38 + 2 )
          j_j___libc_free_0(*v38);
      }
      while ( v37 != v38 );
LABEL_22:
      v38 = v310;
    }
LABEL_23:
    if ( v38 == (unsigned __int64 *)v312 )
      goto LABEL_2;
LABEL_24:
    _libc_free((unsigned __int64)v38);
    goto LABEL_2;
  }
  v260 = *(_QWORD *)(v6 + 40);
  if ( *(_QWORD *)(v6 + 32) != v260 )
  {
    v274 = *(_QWORD *)(v6 + 32);
    v259 = 1;
    v263 = 0;
    v264 = 0;
    v261 = v6;
    while ( 1 )
    {
      v60 = *(_QWORD *)(*(_QWORD *)v274 + 56LL);
      v61 = *(_QWORD *)v274 + 48LL;
      if ( v60 != v61 )
        break;
LABEL_91:
      v274 += 8;
      if ( v260 == v274 )
      {
        v6 = v261;
        goto LABEL_93;
      }
    }
    while ( 1 )
    {
      if ( !v60 )
        BUG();
      v63 = v60 - 24;
      if ( (unsigned __int8)(*(_BYTE *)(v60 - 24) - 34) <= 0x33u )
      {
        v64 = 0x8000000000041LL;
        if ( _bittest64(&v64, (unsigned int)*(unsigned __int8 *)(v60 - 24) - 34) )
        {
          if ( (unsigned __int8)sub_A73ED0((_QWORD *)(v60 + 48), 6)
            || (unsigned __int8)sub_B49560(v60 - 24, 6)
            || (unsigned __int8)sub_A73ED0((_QWORD *)(v60 + 48), 27)
            || (unsigned __int8)sub_B49560(v60 - 24, 27)
            || (unsigned int)sub_CF5CA0((__int64)v271, v60 - 24) )
          {
            break;
          }
        }
      }
      if ( (unsigned __int8)sub_B46790((unsigned __int8 *)(v60 - 24), 0) )
        break;
      if ( (unsigned __int8)sub_B46420(v60 - 24) )
      {
        if ( *(_BYTE *)(v60 - 24) != 61 || sub_B46500((unsigned __int8 *)(v60 - 24)) || (*(_BYTE *)(v60 - 22) & 1) != 0 )
          break;
        ++v264;
        v62 = sub_DD8400((__int64)v270, *(_QWORD *)(v60 - 56));
        v263 -= !sub_DADE90((__int64)v270, (__int64)v62, v261) - 1;
      }
      else if ( (unsigned __int8)sub_B46490(v60 - 24) )
      {
        if ( *(_BYTE *)(v60 - 24) != 62 || sub_B46500((unsigned __int8 *)(v60 - 24)) || (*(_BYTE *)(v60 - 22) & 1) != 0 )
          break;
        ++v264;
        v76 = sub_DD8400((__int64)v270, *(_QWORD *)(v60 - 56));
        v259 = 0;
        v263 -= !sub_DADE90((__int64)v270, (__int64)v76, v261) - 1;
      }
      v60 = *(_QWORD *)(v60 + 8);
      if ( v61 == v60 )
        goto LABEL_91;
    }
    v65 = v280[0];
    v66 = sub_B2BE50(v280[0]);
    if ( !sub_B6EA50(v66) )
    {
      v203 = sub_B2BE50(v65);
      v204 = sub_B6F970(v203);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v204 + 48LL))(v204) )
        goto LABEL_2;
    }
    sub_B176B0((__int64)&v316, (__int64)"loop-versioning-licm", (__int64)"IllegalLoopInst", 15, v63);
    sub_B18290((__int64)&v316, " Unsafe Loop Instruction", 0x18u);
    v70 = _mm_loadu_si128(&v319);
    v71 = _mm_loadu_si128(&v321);
    LODWORD(v304) = v317;
    v72 = _mm_loadu_si128(&v322);
    v303 = (void **)&unk_49D9D40;
    BYTE4(v304) = BYTE4(v317);
    v306 = v70;
    v305 = v318;
    v308 = v71;
    v307 = v320;
    v310 = (unsigned __int64 *)v312;
    v311 = 0x400000000LL;
    v309 = v72;
    if ( v324 )
    {
      sub_288E6D0((__int64)&v310, (__int64)&v323, v67, v68, v69, v324);
      v316 = &unk_49D9D40;
      v185 = v323;
      v313 = v338;
      v314 = v339;
      v315 = v340;
      v303 = (void **)&unk_49D9DB0;
      v73 = &v323[10 * v324];
      if ( v323 != v73 )
      {
        do
        {
          v73 -= 10;
          v186 = v73[4];
          if ( (unsigned __int64 *)v186 != v73 + 6 )
            j_j___libc_free_0(v186);
          if ( (unsigned __int64 *)*v73 != v73 + 2 )
            j_j___libc_free_0(*v73);
        }
        while ( v185 != v73 );
        v73 = v323;
      }
    }
    else
    {
      v73 = v323;
      v313 = v338;
      v314 = v339;
      v315 = v340;
      v303 = (void **)&unk_49D9DB0;
    }
    if ( v73 != v325 )
      _libc_free((unsigned __int64)v73);
    sub_1049740(v280, (__int64)&v303);
    v74 = v310;
    v303 = (void **)&unk_49D9D40;
    v38 = &v310[10 * (unsigned int)v311];
    if ( v310 != v38 )
    {
      do
      {
        v38 -= 10;
        v75 = v38[4];
        if ( (unsigned __int64 *)v75 != v38 + 6 )
          j_j___libc_free_0(v75);
        if ( (unsigned __int64 *)*v38 != v38 + 2 )
          j_j___libc_free_0(*v38);
      }
      while ( v74 != v38 );
      goto LABEL_22;
    }
    goto LABEL_23;
  }
  v259 = 1;
  v263 = 0;
  v264 = 0;
LABEL_93:
  v77 = v6;
  v262 = sub_D440B0((__int64)&v282, v6);
  v78 = *(_DWORD *)(*(_QWORD *)(v262 + 8) + 304LL);
  if ( !v78 )
  {
LABEL_2:
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
    goto LABEL_3;
  }
  if ( v78 > dword_4F87348[0] )
  {
    v214 = v280[0];
    v215 = sub_B2BE50(v280[0]);
    if ( !sub_B6EA50(v215) )
    {
      v256 = sub_B2BE50(v214);
      v257 = sub_B6F970(v256);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v257 + 48LL))(v257) )
        goto LABEL_2;
    }
    v220 = **(_QWORD **)(v6 + 32);
    sub_D4BD20(&v278, v6, v216, v217, v218, v219);
    sub_B157E0((__int64)&v279, &v278);
    sub_B17640((__int64)&v316, (__int64)"loop-versioning-licm", (__int64)"RuntimeCheck", 12, &v279, v220);
    sub_B18290((__int64)&v316, "Number of runtime checks ", 0x19u);
    sub_B169E0(v296.m128i_i64, "RuntimeChecks", 13, *(_DWORD *)(*(_QWORD *)(v262 + 8) + 304LL));
    v221 = sub_2445430((__int64)&v316, (__int64)&v296);
    sub_B18290(v221, " exceeds threshold ", 0x13u);
    sub_B169E0(v292.m128i_i64, "Threshold", 9, dword_4F87348[0]);
    v225 = sub_2445430(v221, (__int64)&v292);
    LODWORD(v304) = *(_DWORD *)(v225 + 8);
    BYTE4(v304) = *(_BYTE *)(v225 + 12);
    v305 = *(__int64 **)(v225 + 16);
    v226 = _mm_loadu_si128((const __m128i *)(v225 + 24));
    v303 = (void **)&unk_49D9D40;
    v306 = v226;
    v307 = *(_QWORD *)(v225 + 40);
    v308 = _mm_loadu_si128((const __m128i *)(v225 + 48));
    v227 = _mm_loadu_si128((const __m128i *)(v225 + 64));
    v310 = (unsigned __int64 *)v312;
    v311 = 0x400000000LL;
    v309 = v227;
    v228 = *(unsigned int *)(v225 + 88);
    if ( (_DWORD)v228 )
      sub_288E6D0((__int64)&v310, v225 + 80, v222, v223, v228, v224);
    v313 = *(_BYTE *)(v225 + 416);
    v314 = *(_DWORD *)(v225 + 420);
    v315 = *(_QWORD *)(v225 + 424);
    v303 = (void **)&unk_49D9DB0;
    if ( v294 != v295 )
      j_j___libc_free_0((unsigned __int64)v294);
    if ( (_QWORD *)v292.m128i_i64[0] != v293 )
      j_j___libc_free_0(v292.m128i_u64[0]);
    if ( v299 != (_BYTE *)v301 )
      j_j___libc_free_0((unsigned __int64)v299);
    if ( (__int64 *)v296.m128i_i64[0] != &v297 )
      j_j___libc_free_0(v296.m128i_u64[0]);
    v229 = v323;
    v316 = &unk_49D9D40;
    v230 = v323;
    v231 = &v323[10 * v324];
    if ( v323 != v231 )
    {
      do
      {
        v231 -= 10;
        v232 = v231[4];
        if ( (unsigned __int64 *)v232 != v231 + 6 )
          j_j___libc_free_0(v232);
        if ( (unsigned __int64 *)*v231 != v231 + 2 )
          j_j___libc_free_0(*v231);
      }
      while ( v230 != v231 );
      v229 = v323;
    }
    if ( v229 != v325 )
      _libc_free((unsigned __int64)v229);
    if ( v278 )
      sub_B91220((__int64)&v278, v278);
    sub_1049740(v280, (__int64)&v303);
    v38 = v310;
    v303 = (void **)&unk_49D9D40;
    v233 = &v310[10 * (unsigned int)v311];
    if ( v310 == v233 )
      goto LABEL_272;
    do
    {
      v233 -= 10;
      v234 = v233[4];
      if ( (unsigned __int64 *)v234 != v233 + 6 )
        j_j___libc_free_0(v234);
      if ( (unsigned __int64 *)*v233 != v233 + 2 )
        j_j___libc_free_0(*v233);
    }
    while ( v38 != v233 );
    goto LABEL_271;
  }
  if ( !v263 || v259 )
    goto LABEL_2;
  v79 = 0;
  *(float *)v79.m128i_i32 = (float)v264 * v267;
  if ( *(float *)v79.m128i_i32 > (float)(100 * v263) )
  {
    v235 = v280[0];
    v236 = sub_B2BE50(v280[0]);
    if ( !sub_B6EA50(v236) )
    {
      v254 = sub_B2BE50(v235);
      v255 = sub_B6F970(v254);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v255 + 48LL))(v255) )
        goto LABEL_2;
    }
    v277 = **(_QWORD **)(v6 + 32);
    sub_D4BD20(&v278, v6, v237, v238, v239, v277);
    sub_B157E0((__int64)&v279, &v278);
    sub_B17640((__int64)&v316, (__int64)"loop-versioning-licm", (__int64)"InvariantThreshold", 18, &v279, v277);
    sub_B18290((__int64)&v316, "Invariant load & store ", 0x17u);
    sub_B169E0(v296.m128i_i64, "LoadAndStoreCounter", 19, 100 * v263 / (unsigned int)v264);
    v240 = sub_2445430((__int64)&v316, (__int64)&v296);
    sub_B18290(v240, " are less then defined threshold ", 0x21u);
    sub_B16720((__int64)&v292, "Threshold", 9, v267);
    v245 = sub_2445430(v240, (__int64)&v292);
    LODWORD(v304) = *(_DWORD *)(v245 + 8);
    BYTE4(v304) = *(_BYTE *)(v245 + 12);
    v305 = *(__int64 **)(v245 + 16);
    v246 = _mm_loadu_si128((const __m128i *)(v245 + 24));
    v303 = (void **)&unk_49D9D40;
    v306 = v246;
    v307 = *(_QWORD *)(v245 + 40);
    v308 = _mm_loadu_si128((const __m128i *)(v245 + 48));
    v247 = _mm_loadu_si128((const __m128i *)(v245 + 64));
    v310 = (unsigned __int64 *)v312;
    v309 = v247;
    v311 = 0x400000000LL;
    if ( *(_DWORD *)(v245 + 88) )
      sub_288E6D0((__int64)&v310, v245 + 80, v241, v242, v243, v244);
    v313 = *(_BYTE *)(v245 + 416);
    v314 = *(_DWORD *)(v245 + 420);
    v315 = *(_QWORD *)(v245 + 424);
    v303 = (void **)&unk_49D9DB0;
    if ( v294 != v295 )
      j_j___libc_free_0((unsigned __int64)v294);
    if ( (_QWORD *)v292.m128i_i64[0] != v293 )
      j_j___libc_free_0(v292.m128i_u64[0]);
    if ( v299 != (_BYTE *)v301 )
      j_j___libc_free_0((unsigned __int64)v299);
    if ( (__int64 *)v296.m128i_i64[0] != &v297 )
      j_j___libc_free_0(v296.m128i_u64[0]);
    v248 = v323;
    v316 = &unk_49D9D40;
    v249 = v323;
    v250 = &v323[10 * v324];
    if ( v323 != v250 )
    {
      do
      {
        v250 -= 10;
        v251 = v250[4];
        if ( (unsigned __int64 *)v251 != v250 + 6 )
          j_j___libc_free_0(v251);
        if ( (unsigned __int64 *)*v250 != v250 + 2 )
          j_j___libc_free_0(*v250);
      }
      while ( v249 != v250 );
      v248 = v323;
    }
    if ( v248 != v325 )
      _libc_free((unsigned __int64)v248);
    if ( v278 )
      sub_B91220((__int64)&v278, v278);
    sub_1049740(v280, (__int64)&v303);
    v38 = v310;
    v303 = (void **)&unk_49D9D40;
    v252 = &v310[10 * (unsigned int)v311];
    if ( v310 == v252 )
      goto LABEL_272;
    do
    {
      v252 -= 10;
      v253 = v252[4];
      if ( (unsigned __int64 *)v253 != v252 + 6 )
        j_j___libc_free_0(v253);
      if ( (unsigned __int64 *)*v252 != v252 + 2 )
        j_j___libc_free_0(*v252);
    }
    while ( v38 != v252 );
    goto LABEL_271;
  }
  v318 = 0;
  v319.m128i_i64[0] = 1;
  v316 = v271;
  v317 = (__int64)v271;
  v80 = &v319.m128i_i64[1];
  do
  {
    *v80 = -4;
    v80 += 5;
    *(v80 - 4) = -3;
    *(v80 - 3) = -4;
    *(v80 - 2) = -3;
  }
  while ( v80 != v333 );
  v333[1] = 0;
  v333[0] = v342;
  v335 = v337;
  v336 = 0x400000000LL;
  v334 = 0;
  v341 = 256;
  v342[1] = 0;
  v343 = 1;
  v342[0] = &unk_49DDBE8;
  v81 = &v344;
  do
  {
    *v81 = -4096;
    v81 += 2;
  }
  while ( v81 != (__int64 *)&v346 );
  v82 = *(unsigned __int64 **)(v6 + 40);
  v306 = 0u;
  v83 = *(unsigned __int64 **)(v6 + 32);
  v303 = &v316;
  v305 = &v304;
  v304 = (__int64)&v304 + 4;
  v307 = 0;
  v308.m128i_i32[0] = 0;
  v308.m128i_i32[2] = 0;
  v309.m128i_i64[0] = 0;
  if ( v83 != v82 )
  {
    do
    {
      v84 = *(_DWORD *)(v268 + 24);
      v77 = *v83;
      v85 = *(_QWORD *)(v268 + 8);
      if ( v84 )
      {
        v86 = v84 - 1;
        v87 = v86 & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
        v88 = (__int64 *)(v85 + 16LL * v87);
        v89 = *v88;
        if ( v77 == *v88 )
        {
LABEL_107:
          if ( v6 == v88[1] )
            sub_FD9C40((__int64 **)&v303, v77, v79);
        }
        else
        {
          v183 = 1;
          while ( v89 != -4096 )
          {
            v184 = v183 + 1;
            v87 = v86 & (v183 + v87);
            v88 = (__int64 *)(v85 + 16LL * v87);
            v89 = *v88;
            if ( v77 == *v88 )
              goto LABEL_107;
            v183 = v184;
          }
        }
      }
      ++v83;
    }
    while ( v82 != v83 );
    v169 = v305;
    if ( v305 != &v304 )
    {
      v170 = 0;
      v171 = 0;
      v77 = 0;
      do
      {
        if ( !v169[2] )
        {
          v172 = *((_BYTE *)v169 + 67);
          if ( (v172 & 0x40) == 0 )
            goto LABEL_110;
          v173 = (__int64 *)v169[3];
          v174 = *v173;
          v77 = ((unsigned __int8)((v172 >> 4) & 3) >> 1) | (unsigned int)v77;
          v175 = &v173[6 * *((unsigned int *)v169 + 8)];
          if ( v173 == v175 || (v176 = v173 + 6, v177 = *(_QWORD *)(v174 + 8), v175 == v176) )
          {
LABEL_236:
            v171 = v265;
            v170 = v265;
          }
          else
          {
            while ( 1 )
            {
              v178 = *v176;
              v176 += 6;
              if ( v177 != *(_QWORD *)(v178 + 8) )
                break;
              if ( v175 == v176 )
                goto LABEL_236;
            }
            for ( ; v175 != v176; v176 += 12 )
            {
              if ( v175 == v176 + 6 )
                break;
            }
            v170 = v265;
          }
        }
        v169 = (__int64 *)v169[1];
      }
      while ( v169 != &v304 );
      v77 = v170 & v171 & (unsigned int)v77;
      v259 = v77;
    }
  }
LABEL_110:
  sub_FD6240((__int64)&v303, v77);
  v90 = v308.m128i_i32[0];
  if ( v308.m128i_i32[0] )
  {
    v91 = (_QWORD *)v306.m128i_i64[1];
    v292 = 0u;
    v293[0] = -4096;
    v92 = v306.m128i_i64[1] + 32LL * v308.m128i_u32[0];
    v296 = 0u;
    v297 = -8192;
    do
    {
      v93 = v91[2];
      if ( v93 != 0 && v93 != -4096 && v93 != -8192 )
        sub_BD60C0(v91);
      v91 += 4;
    }
    while ( (_QWORD *)v92 != v91 );
    if ( v297 != 0 && v297 != -4096 && v297 != -8192 )
      sub_BD60C0(&v296);
    if ( v293[0] != -4096 && v293[0] != 0 && v293[0] != -8192 )
      sub_BD60C0(&v292);
    v90 = v308.m128i_i32[0];
  }
  sub_C7D6A0(v306.m128i_i64[1], 32LL * v90, 8);
  v94 = v305;
  if ( v305 != &v304 )
  {
    v266 = a1;
    v258 = v6;
    do
    {
      v95 = v94;
      v94 = (__int64 *)v94[1];
      v96 = *v95 & 0xFFFFFFFFFFFFFFF8LL;
      *v94 = v96 | *v94 & 7;
      *(_QWORD *)(v96 + 8) = v94;
      v97 = (_QWORD *)v95[6];
      v98 = (_QWORD *)v95[5];
      *v95 &= 7uLL;
      v95[1] = 0;
      if ( v97 != v98 )
      {
        do
        {
          v99 = v98[2];
          if ( v99 != -4096 && v99 != 0 && v99 != -8192 )
            sub_BD60C0(v98);
          v98 += 3;
        }
        while ( v97 != v98 );
        v98 = (_QWORD *)v95[5];
      }
      if ( v98 )
        j_j___libc_free_0((unsigned __int64)v98);
      v100 = v95[3];
      if ( v95 + 5 != (__int64 *)v100 )
        _libc_free(v100);
      j_j___libc_free_0((unsigned __int64)v95);
    }
    while ( v94 != &v304 );
    a1 = v266;
    v6 = v258;
  }
  v342[0] = &unk_49DDBE8;
  if ( (v343 & 1) == 0 )
    sub_C7D6A0(v344, 16LL * v345, 8);
  nullsub_184();
  if ( v335 != v337 )
    _libc_free((unsigned __int64)v335);
  if ( (v319.m128i_i8[0] & 1) == 0 )
    sub_C7D6A0(v319.m128i_i64[1], 40LL * (unsigned int)v320, 8);
  v101 = v280[0];
  if ( !v259 )
  {
    v187 = sub_B2BE50(v280[0]);
    if ( !sub_B6EA50(v187) )
    {
      v212 = sub_B2BE50(v101);
      v213 = sub_B6F970(v212);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v213 + 48LL))(v213) )
        goto LABEL_2;
    }
    v192 = **(_QWORD **)(v6 + 32);
    sub_D4BD20(&v292, v6, v188, v189, v190, v191);
    sub_B157E0((__int64)&v296, &v292);
    sub_B17640((__int64)&v316, (__int64)"loop-versioning-licm", (__int64)"IllegalLoopMemoryAccess", 23, &v296, v192);
    sub_B18290((__int64)&v316, " Unsafe Loop memory access", 0x1Au);
    v197 = _mm_loadu_si128(&v319);
    v198 = _mm_loadu_si128(&v321);
    LODWORD(v304) = v317;
    v199 = _mm_loadu_si128(&v322);
    v303 = (void **)&unk_49D9D40;
    BYTE4(v304) = BYTE4(v317);
    v306 = v197;
    v305 = v318;
    v308 = v198;
    v307 = v320;
    v310 = (unsigned __int64 *)v312;
    v311 = 0x400000000LL;
    v309 = v199;
    if ( v324 )
    {
      sub_288E6D0((__int64)&v310, (__int64)&v323, v193, v194, v195, v196);
      v316 = &unk_49D9D40;
      v200 = v323;
      v313 = v338;
      v314 = v339;
      v315 = v340;
      v303 = (void **)&unk_49D9DB0;
      v209 = &v323[10 * v324];
      if ( v323 == v209 )
        goto LABEL_261;
      v210 = v323;
      do
      {
        v209 -= 10;
        v211 = v209[4];
        if ( (unsigned __int64 *)v211 != v209 + 6 )
          j_j___libc_free_0(v211);
        if ( (unsigned __int64 *)*v209 != v209 + 2 )
          j_j___libc_free_0(*v209);
      }
      while ( v210 != v209 );
    }
    else
    {
      v313 = v338;
      v314 = v339;
      v315 = v340;
      v303 = (void **)&unk_49D9DB0;
    }
    v200 = v323;
LABEL_261:
    if ( v200 != v325 )
      _libc_free((unsigned __int64)v200);
    if ( v292.m128i_i64[0] )
      sub_B91220((__int64)&v292, v292.m128i_i64[0]);
    sub_1049740(v280, (__int64)&v303);
    v38 = v310;
    v303 = (void **)&unk_49D9D40;
    v201 = &v310[10 * (unsigned int)v311];
    if ( v310 == v201 )
    {
LABEL_272:
      if ( v38 == (unsigned __int64 *)v312 )
        goto LABEL_2;
      goto LABEL_24;
    }
    do
    {
      v201 -= 10;
      v202 = v201[4];
      if ( (unsigned __int64 *)v202 != v201 + 6 )
        j_j___libc_free_0(v202);
      if ( (unsigned __int64 *)*v201 != v201 + 2 )
        j_j___libc_free_0(*v201);
    }
    while ( v38 != v201 );
LABEL_271:
    v38 = v310;
    goto LABEL_272;
  }
  v102 = sub_B2BE50(v280[0]);
  if ( sub_B6EA50(v102)
    || (v207 = sub_B2BE50(v101),
        v208 = sub_B6F970(v207),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v208 + 48LL))(v208)) )
  {
    v107 = **(_QWORD **)(v6 + 32);
    sub_D4BD20(&v279, v6, v103, v104, v105, v106);
    sub_B157E0((__int64)&v292, &v279);
    sub_B17430((__int64)&v316, (__int64)"loop-versioning-licm", (__int64)"IsLegalForVersioning", 20, &v292, v107);
    sub_B18290((__int64)&v316, " Versioned loop for LICM.", 0x19u);
    sub_B18290((__int64)&v316, " Number of runtime checks we had to insert ", 0x2Bu);
    sub_B169E0(v296.m128i_i64, "RuntimeChecks", 13, *(_DWORD *)(*(_QWORD *)(v262 + 8) + 304LL));
    v303 = (void **)&v305;
    sub_288E0E0((__int64 *)&v303, v296.m128i_i64[0], v296.m128i_i64[0] + v296.m128i_i64[1]);
    v306.m128i_i64[1] = (__int64)&v308;
    sub_288E0E0(&v306.m128i_i64[1], v299, (__int64)&v299[v300]);
    v309 = _mm_loadu_si128(&v302);
    sub_B180C0((__int64)&v316, (unsigned __int64)&v303);
    if ( (__m128i *)v306.m128i_i64[1] != &v308 )
      j_j___libc_free_0(v306.m128i_u64[1]);
    if ( v303 != (void **)&v305 )
      j_j___libc_free_0((unsigned __int64)v303);
    v111 = _mm_loadu_si128(&v319);
    v112 = _mm_loadu_si128(&v321);
    LODWORD(v304) = v317;
    v113 = _mm_loadu_si128(&v322);
    v306 = v111;
    BYTE4(v304) = BYTE4(v317);
    v308 = v112;
    v305 = v318;
    v303 = (void **)&unk_49D9D40;
    v309 = v113;
    v307 = v320;
    v310 = (unsigned __int64 *)v312;
    v311 = 0x400000000LL;
    if ( v324 )
      sub_288E6D0((__int64)&v310, (__int64)&v323, v108, v324, v109, v110);
    v313 = v338;
    v314 = v339;
    v315 = v340;
    v303 = (void **)&unk_49D9D78;
    if ( v299 != (_BYTE *)v301 )
      j_j___libc_free_0((unsigned __int64)v299);
    if ( (__int64 *)v296.m128i_i64[0] != &v297 )
      j_j___libc_free_0(v296.m128i_u64[0]);
    v114 = v323;
    v316 = &unk_49D9D40;
    v115 = 10LL * v324;
    v116 = &v323[v115];
    if ( v323 != &v323[v115] )
    {
      do
      {
        v116 -= 10;
        v117 = v116[4];
        if ( (unsigned __int64 *)v117 != v116 + 6 )
          j_j___libc_free_0(v117);
        if ( (unsigned __int64 *)*v116 != v116 + 2 )
          j_j___libc_free_0(*v116);
      }
      while ( v114 != v116 );
      v116 = v323;
    }
    if ( v116 != v325 )
      _libc_free((unsigned __int64)v116);
    if ( v279.m128i_i64[0] )
      sub_B91220((__int64)&v279, v279.m128i_i64[0]);
    sub_1049740(v280, (__int64)&v303);
    v118 = v310;
    v303 = (void **)&unk_49D9D40;
    v119 = &v310[10 * (unsigned int)v311];
    if ( v310 != v119 )
    {
      do
      {
        v119 -= 10;
        v120 = v119[4];
        if ( (unsigned __int64 *)v120 != v119 + 6 )
          j_j___libc_free_0(v120);
        if ( (unsigned __int64 *)*v119 != v119 + 2 )
          j_j___libc_free_0(*v119);
      }
      while ( v118 != v119 );
      v119 = v310;
    }
    if ( v119 != (unsigned __int64 *)v312 )
      _libc_free((unsigned __int64)v119);
  }
  sub_2A28870(
    &v316,
    v262,
    *(_QWORD *)(*(_QWORD *)(v262 + 8) + 296LL),
    *(unsigned int *)(*(_QWORD *)(v262 + 8) + 304LL),
    v6,
    v268,
    v275,
    v270);
  sub_F6D5D0((__int64)&v303, (__int64)v316, v121, v122, v123, v124);
  sub_2A28FB0(&v316, &v303);
  if ( v303 != (void **)&v305 )
    _libc_free((unsigned __int64)v303);
  sub_F6DC70(v317, "llvm.loop.licm_versioning.disable", 0, v125, v126, v127);
  sub_F6DC70((__int64)v316, "llvm.loop.licm_versioning.disable", 0, v128, v129, v130);
  sub_F6DC70((__int64)v316, "llvm.mem.parallel_loop_access", 0, v131, v132, v133);
  v134 = (_QWORD *)(sub_D47930((__int64)v316) + 48);
  v135 = *v134 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v135 == v134 )
  {
    v137 = 0;
  }
  else
  {
    if ( !v135 )
      BUG();
    v136 = *(unsigned __int8 *)(v135 - 24);
    v137 = 0;
    v138 = v135 - 24;
    if ( (unsigned int)(v136 - 30) < 0xB )
      v137 = v138;
  }
  v292.m128i_i64[0] = sub_BD5C60(v137);
  v139 = sub_B8CD90(&v292, (__int64)"LVDomain", 8, 0);
  v140 = sub_B8CD90(&v292, (__int64)"LVAliasScope", 12, v139);
  v141 = (__int64 *)&v305;
  v296.m128i_i64[0] = (__int64)&v297;
  v142 = *(_QWORD *)(v6 + 40);
  v297 = v140;
  v305 = (__int64 *)v140;
  v143 = *(_QWORD *)(v6 + 32);
  v296.m128i_i64[1] = 0x400000001LL;
  v303 = (void **)&v305;
  v304 = 0x400000001LL;
  v272 = v142;
  if ( v143 != v142 )
  {
    v276 = v143;
    v269 = a1;
    do
    {
      v144 = *(_QWORD *)(*(_QWORD *)v276 + 56LL);
      for ( i = *(_QWORD *)v276 + 48LL; i != v144; v144 = *(_QWORD *)(v144 + 8) )
      {
        v146 = v144 - 24;
        if ( !v144 )
          v146 = 0;
        if ( (unsigned __int8)sub_B46420(v146) || (unsigned __int8)sub_B46490(v146) )
        {
          v147 = (__int64 *)(unsigned int)v304;
          v148 = (__int64 *)sub_BD5C60(v146);
          v149 = 0;
          v150 = sub_B9C770(v148, v141, v147, 0, 1);
          if ( (*(_BYTE *)(v146 + 7) & 0x20) != 0 )
            v149 = sub_B91C10(v146, 8);
          v151 = sub_BA72D0(v149, v150);
          sub_B99FD0(v146, 8u, v151);
          v152 = (__int64 *)v296.m128i_u32[2];
          v153 = (__int64 *)v296.m128i_i64[0];
          v154 = (__int64 *)sub_BD5C60(v146);
          v155 = 0;
          v156 = sub_B9C770(v154, v153, v152, 0, 1);
          if ( (*(_BYTE *)(v146 + 7) & 0x20) != 0 )
            v155 = sub_B91C10(v146, 7);
          v157 = sub_BA72D0(v155, v156);
          sub_B99FD0(v146, 7u, v157);
          v141 = (__int64 *)v303;
        }
      }
      v276 += 8;
    }
    while ( v272 != v276 );
    a1 = v269;
    if ( v141 != (__int64 *)&v305 )
      _libc_free((unsigned __int64)v141);
    if ( (__int64 *)v296.m128i_i64[0] != &v297 )
      _libc_free(v296.m128i_u64[0]);
  }
  sub_C7D6A0(v331, 16LL * v332, 8);
  sub_C7D6A0(v329, 16LL * v330, 8);
  sub_C7D6A0(v327, 16LL * v328, 8);
  if ( (char *)v325[0] != &v326 )
    _libc_free(v325[0]);
  if ( (_BYTE)v323 )
  {
    v179 = v322.m128i_u32[2];
    LOBYTE(v323) = 0;
    if ( v322.m128i_i32[2] )
    {
      v180 = (_QWORD *)v321.m128i_i64[1];
      v181 = v321.m128i_i64[1] + 16LL * v322.m128i_u32[2];
      do
      {
        if ( *v180 != -4096 && *v180 != -8192 )
        {
          v182 = v180[1];
          if ( v182 )
            sub_B91220((__int64)(v180 + 1), v182);
        }
        v180 += 2;
      }
      while ( (_QWORD *)v181 != v180 );
      v179 = v322.m128i_u32[2];
    }
    sub_C7D6A0(v321.m128i_i64[1], 16LL * v179, 8);
  }
  v158 = (unsigned int)v320;
  if ( (_DWORD)v320 )
  {
    v159 = (_QWORD *)v319.m128i_i64[0];
    v296.m128i_i64[1] = 2;
    v297 = 0;
    v160 = v319.m128i_i64[0] + ((unsigned __int64)(unsigned int)v320 << 6);
    v298 = -4096;
    v296.m128i_i64[0] = (__int64)&unk_49DD7B0;
    v303 = (void **)&unk_49DD7B0;
    v161 = -4096;
    v299 = 0;
    v304 = 2;
    v305 = 0;
    v306 = (__m128i)0xFFFFFFFFFFFFE000LL;
    while ( 1 )
    {
      v162 = v159[3];
      if ( v161 != v162 )
      {
        v161 = v306.m128i_i64[0];
        if ( v162 != v306.m128i_i64[0] )
        {
          v163 = v159[7];
          if ( v163 != 0 && v163 != -4096 && v163 != -8192 )
          {
            sub_BD60C0(v159 + 5);
            v162 = v159[3];
          }
          v161 = v162;
        }
      }
      *v159 = &unk_49DB368;
      if ( v161 != -4096 && v161 != 0 && v161 != -8192 )
        sub_BD60C0(v159 + 1);
      v159 += 8;
      if ( (_QWORD *)v160 == v159 )
        break;
      v161 = v298;
    }
    v303 = (void **)&unk_49DB368;
    if ( v306.m128i_i64[0] != 0 && v306.m128i_i64[0] != -4096 && v306.m128i_i64[0] != -8192 )
      sub_BD60C0(&v304);
    v296.m128i_i64[0] = (__int64)&unk_49DB368;
    if ( v298 != 0 && v298 != -4096 && v298 != -8192 )
      sub_BD60C0(&v296.m128i_i64[1]);
    v158 = (unsigned int)v320;
  }
  v164 = v158 << 6;
  sub_C7D6A0(v319.m128i_i64[0], v164, 8);
  sub_22D0390(a1, v164, v165, v166, v167, v168);
LABEL_3:
  sub_288E190((__int64)&v282);
  sub_C7D6A0(v283, 16LL * (unsigned int)v285, 8);
  v18 = v281;
  if ( v281 )
  {
    sub_FDC110(v281);
    j_j___libc_free_0((unsigned __int64)v18);
  }
  return a1;
}
