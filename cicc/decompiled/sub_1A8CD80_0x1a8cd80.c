// Function: sub_1A8CD80
// Address: 0x1a8cd80
//
__int64 __fastcall sub_1A8CD80(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        __m128 a7,
        __m128i a8,
        __m128i a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  void **v14; // r13
  unsigned __int64 *v15; // rdx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rax
  _BYTE *v23; // rax
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  _BYTE *v26; // rsi
  __int64 v27; // rdx
  __m128i *v28; // rcx
  __m128i *v29; // rdi
  unsigned __int64 v30; // r15
  __int64 v31; // rax
  unsigned __int64 v32; // rsi
  unsigned __int64 v33; // rdx
  __m128i *v34; // rax
  __int8 v35; // r8
  __int64 v36; // r8
  int v37; // r9d
  _BYTE *v38; // rcx
  unsigned __int64 v39; // r15
  __int64 v40; // rax
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rdx
  _BYTE *v43; // rax
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // r15
  unsigned __int64 v46; // rdx
  __int64 v47; // r12
  __int64 *v48; // rax
  char v49; // dl
  __int64 v50; // r14
  __int64 *v51; // rax
  __int64 *v52; // rcx
  __int64 *v53; // rsi
  unsigned __int64 v54; // rcx
  char v55; // si
  char v56; // al
  bool v57; // al
  unsigned int v58; // r12d
  __int64 v59; // rdx
  _QWORD *v60; // rax
  __int64 v61; // rsi
  __int64 v62; // rax
  __int64 v64; // rdi
  __int64 v65; // rdx
  __int64 v66; // rax
  int v67; // r9d
  bool v68; // zf
  __int64 v69; // r13
  int v70; // r8d
  int v71; // ecx
  __int64 v72; // rax
  __int64 v73; // rdx
  unsigned __int64 *v74; // rbx
  unsigned __int64 *v75; // rax
  __int64 v76; // rdx
  unsigned __int64 *v77; // r14
  __int64 v78; // r15
  unsigned __int64 v79; // rdx
  unsigned int v80; // eax
  unsigned int *v81; // rbx
  unsigned int *v82; // r13
  unsigned __int64 v83; // rdx
  int v84; // r13d
  unsigned __int64 v85; // rbx
  __int64 v86; // r15
  __int64 v87; // r15
  _QWORD **v88; // rax
  __int64 v89; // r12
  __int64 v90; // rax
  _QWORD *v91; // rax
  __int64 v92; // rdx
  _QWORD *v93; // rdx
  char v94; // cl
  __int64 *v95; // rax
  _QWORD **v96; // rsi
  __int64 *v97; // rbx
  unsigned __int64 v98; // r13
  __int64 v99; // r14
  __int64 v100; // rax
  _QWORD *v101; // rax
  __int64 v102; // rdx
  _QWORD *v103; // rdx
  char v104; // cl
  __int64 *v105; // rax
  unsigned int v106; // eax
  __int64 v107; // rdx
  __int64 v108; // rcx
  int v109; // r8d
  _QWORD *k; // rbx
  __int64 *v111; // rax
  __int64 *v112; // rdx
  __int64 v113; // rax
  _QWORD *v114; // rax
  __int64 v115; // rdx
  _QWORD *v116; // rdx
  char v117; // cl
  __int64 *v118; // rax
  __int64 *v119; // rdi
  unsigned int v120; // edx
  __int64 *v121; // rcx
  _QWORD *v122; // r15
  _QWORD *v123; // r12
  __int64 v124; // rdx
  _QWORD *v125; // rbx
  __int64 v126; // rcx
  __int64 v127; // rax
  __int64 v128; // rdx
  unsigned __int64 v129; // rdi
  unsigned __int64 v130; // rdi
  __int64 *v131; // rdi
  unsigned int v132; // edx
  __int64 *v133; // rcx
  unsigned int v134; // eax
  _QWORD *v135; // rbx
  _QWORD *v136; // r13
  __int64 v137; // rsi
  __int64 *v138; // r8
  unsigned int v139; // ecx
  __int64 *v140; // rdi
  _QWORD *v141; // r15
  _QWORD *v142; // rax
  __int64 v143; // rdx
  __int64 v144; // rbx
  __int64 v145; // rbx
  unsigned __int64 v146; // rsi
  int v147; // r8d
  int v148; // r9d
  unsigned __int64 v149; // rdx
  unsigned int v150; // r13d
  __int64 v151; // rbx
  _BYTE *v152; // rax
  _BYTE *v153; // rdi
  _DWORD *v154; // rax
  unsigned __int64 *v155; // r15
  unsigned int v156; // edi
  unsigned int v157; // r9d
  unsigned __int64 *v158; // rax
  unsigned __int64 v159; // rsi
  _DWORD *v160; // rbx
  int v161; // eax
  int v162; // edx
  unsigned int v163; // r14d
  __int64 *v164; // r12
  unsigned __int64 v165; // rbx
  unsigned __int64 v166; // rdx
  unsigned __int64 v167; // rax
  __int64 *v168; // rax
  __int64 *v169; // rdx
  __int64 *v170; // rax
  __int64 *jj; // rcx
  __int64 *kk; // rsi
  __int64 v173; // r14
  int v174; // r8d
  __int64 *v175; // rdi
  unsigned int v176; // ecx
  __int64 *v177; // rdx
  __int64 v178; // r13
  __int64 *v179; // rcx
  unsigned int v180; // esi
  int v181; // edx
  unsigned int v182; // r10d
  unsigned int v183; // esi
  unsigned __int64 v184; // r9
  unsigned int v185; // r14d
  __int64 *v186; // r12
  unsigned int v187; // edi
  int v188; // ecx
  __int64 *v189; // rdx
  unsigned int v190; // r9d
  __int64 *v191; // rax
  __int64 v192; // rsi
  _QWORD *v193; // rax
  int v194; // r8d
  unsigned __int64 *v195; // rdi
  __int64 v196; // r14
  const __m128i *v197; // rbx
  __int64 v198; // r15
  unsigned int *v199; // rcx
  unsigned int *v200; // r12
  const __m128i *v201; // r15
  __int64 v202; // rax
  unsigned int v203; // ebx
  unsigned int *v204; // r13
  unsigned int v205; // r12d
  int v206; // r8d
  int v207; // r9d
  __int64 v208; // rax
  __int64 v209; // rsi
  int v210; // r8d
  int v211; // r9d
  __int32 v212; // r12d
  const __m128i *v213; // rax
  __int64 v214; // rdx
  __int64 v215; // rcx
  int v216; // r8d
  int v217; // r9d
  _QWORD *v218; // r12
  _QWORD *v219; // rbx
  unsigned __int64 v220; // rdi
  __int64 v221; // rdx
  __int64 v222; // rcx
  __int64 v223; // r8
  _QWORD *v224; // r9
  double v225; // xmm4_8
  double v226; // xmm5_8
  _QWORD *i1; // rbx
  __int64 *v228; // r14
  __int64 v229; // rax
  __int64 v230; // r15
  __m128i **v231; // r15
  unsigned __int64 *v232; // rbx
  unsigned __int64 *v233; // r12
  unsigned __int64 *v234; // rdi
  __m128i **v235; // rbx
  __m128i **v236; // rdi
  __int64 v237; // rax
  __int64 v238; // rax
  size_t v239; // rdx
  unsigned __int64 *v240; // rsi
  unsigned __int64 *p_dest; // rdi
  int v242; // ecx
  unsigned __int64 *v243; // rdx
  unsigned __int64 v244; // rax
  __int64 *v245; // rax
  __int64 *mm; // rdx
  __int64 *v247; // rcx
  int v248; // edi
  __int64 *v249; // rsi
  unsigned int v250; // edx
  __int64 *v251; // rax
  __int64 v252; // r11
  __int64 *v253; // rcx
  unsigned int v254; // esi
  unsigned int v255; // r8d
  unsigned __int64 *v256; // r9
  int v257; // esi
  unsigned int v258; // r13d
  unsigned __int64 v259; // rdi
  __int64 *v260; // rdx
  int v261; // edx
  unsigned __int64 v262; // rax
  __int64 *v263; // rax
  __int64 *m; // rdx
  __int64 *v265; // rcx
  __int64 v266; // r11
  int v267; // edi
  __int64 *v268; // rsi
  unsigned int v269; // edx
  __int64 *v270; // rax
  __int64 v271; // r14
  __int64 *v272; // rcx
  unsigned int v273; // esi
  unsigned int v274; // r8d
  __int64 *v275; // r9
  int v276; // esi
  unsigned int v277; // r13d
  __int64 v278; // rdi
  __int64 *v279; // rdx
  unsigned __int64 v280; // rdx
  unsigned __int64 v281; // rax
  __int64 *v282; // rax
  __int64 *v283; // rdx
  __int64 *v284; // rax
  __int64 *n; // rcx
  __int64 *ii; // rsi
  __int64 v287; // r13
  int v288; // r8d
  __int64 *v289; // rdi
  unsigned int v290; // ecx
  __int64 *v291; // rdx
  __int64 v292; // r14
  __int64 *v293; // rcx
  unsigned int v294; // esi
  unsigned int v295; // r10d
  unsigned int v296; // esi
  __int64 v297; // r9
  int v298; // r8d
  __int64 *v299; // rdi
  unsigned int v300; // [rsp+Ch] [rbp-7C4h]
  __int64 *v306; // [rsp+40h] [rbp-790h]
  __int64 v307; // [rsp+48h] [rbp-788h]
  unsigned int *v308; // [rsp+48h] [rbp-788h]
  __int64 *v309; // [rsp+50h] [rbp-780h]
  unsigned int *v310; // [rsp+50h] [rbp-780h]
  __int64 v311; // [rsp+58h] [rbp-778h]
  __int64 *v312; // [rsp+60h] [rbp-770h]
  int s; // [rsp+80h] [rbp-750h]
  __int64 *v315; // [rsp+88h] [rbp-748h]
  char v316; // [rsp+98h] [rbp-738h]
  __int64 v317; // [rsp+98h] [rbp-738h]
  _QWORD *v318; // [rsp+98h] [rbp-738h]
  _DWORD *v319; // [rsp+98h] [rbp-738h]
  unsigned int *v320; // [rsp+98h] [rbp-738h]
  void **v321; // [rsp+A0h] [rbp-730h]
  __int64 *v322; // [rsp+A0h] [rbp-730h]
  __int64 v323; // [rsp+A8h] [rbp-728h]
  unsigned __int64 *v324; // [rsp+A8h] [rbp-728h]
  const __m128i *nn; // [rsp+A8h] [rbp-728h]
  unsigned __int8 v326; // [rsp+B0h] [rbp-720h]
  unsigned __int64 *v327; // [rsp+C8h] [rbp-708h]
  _QWORD *v328; // [rsp+C8h] [rbp-708h]
  _QWORD *v329; // [rsp+C8h] [rbp-708h]
  __int64 v330; // [rsp+C8h] [rbp-708h]
  void **v331; // [rsp+D8h] [rbp-6F8h]
  __int64 v332; // [rsp+E8h] [rbp-6E8h] BYREF
  __m128i v333[2]; // [rsp+F0h] [rbp-6E0h] BYREF
  _BYTE *v334; // [rsp+110h] [rbp-6C0h] BYREF
  __int64 v335; // [rsp+118h] [rbp-6B8h]
  _BYTE v336[32]; // [rsp+120h] [rbp-6B0h] BYREF
  __int64 v337[2]; // [rsp+140h] [rbp-690h] BYREF
  __int64 v338; // [rsp+150h] [rbp-680h]
  __int64 *v339; // [rsp+158h] [rbp-678h]
  __int64 v340; // [rsp+160h] [rbp-670h]
  __int64 v341; // [rsp+168h] [rbp-668h]
  __int64 *v342; // [rsp+170h] [rbp-660h]
  __int16 v343; // [rsp+178h] [rbp-658h]
  __int64 *v344; // [rsp+180h] [rbp-650h] BYREF
  __int64 v345; // [rsp+188h] [rbp-648h]
  _BYTE v346[64]; // [rsp+190h] [rbp-640h] BYREF
  _QWORD *v347; // [rsp+1D0h] [rbp-600h] BYREF
  _QWORD **v348; // [rsp+1D8h] [rbp-5F8h]
  __int64 v349; // [rsp+1E0h] [rbp-5F0h]
  __int64 v350; // [rsp+1E8h] [rbp-5E8h]
  __int64 *v351; // [rsp+1F0h] [rbp-5E0h]
  __int64 v352; // [rsp+1F8h] [rbp-5D8h]
  unsigned int v353; // [rsp+200h] [rbp-5D0h]
  __int64 v354; // [rsp+208h] [rbp-5C8h]
  __int64 v355; // [rsp+210h] [rbp-5C0h]
  __int64 v356; // [rsp+218h] [rbp-5B8h]
  unsigned __int64 v357[16]; // [rsp+220h] [rbp-5B0h] BYREF
  __m128i v358; // [rsp+2A0h] [rbp-530h] BYREF
  unsigned __int64 src[2]; // [rsp+2B0h] [rbp-520h] BYREF
  int v360; // [rsp+2C0h] [rbp-510h]
  _QWORD v361[8]; // [rsp+2C8h] [rbp-508h] BYREF
  unsigned __int64 v362; // [rsp+308h] [rbp-4C8h] BYREF
  unsigned __int64 v363; // [rsp+310h] [rbp-4C0h]
  unsigned __int64 v364; // [rsp+318h] [rbp-4B8h]
  unsigned __int64 *v365; // [rsp+320h] [rbp-4B0h] BYREF
  __int64 v366; // [rsp+328h] [rbp-4A8h]
  unsigned __int64 v367; // [rsp+330h] [rbp-4A0h] BYREF
  unsigned int v368; // [rsp+338h] [rbp-498h]
  unsigned int v369; // [rsp+33Ch] [rbp-494h]
  int v370; // [rsp+340h] [rbp-490h]
  _BYTE v371[64]; // [rsp+348h] [rbp-488h] BYREF
  unsigned __int64 v372; // [rsp+388h] [rbp-448h] BYREF
  unsigned __int64 v373; // [rsp+390h] [rbp-440h]
  unsigned __int64 v374; // [rsp+398h] [rbp-438h]
  __m128i v375; // [rsp+3B0h] [rbp-420h] BYREF
  unsigned __int64 dest; // [rsp+3C0h] [rbp-410h] BYREF
  __m128 v377; // [rsp+3C8h] [rbp-408h]
  unsigned __int64 v378[2]; // [rsp+3D8h] [rbp-3F8h] BYREF
  __m128 v379; // [rsp+3E8h] [rbp-3E8h] BYREF
  __int64 v380; // [rsp+3F8h] [rbp-3D8h]
  char v381; // [rsp+400h] [rbp-3D0h]
  unsigned __int64 *v382; // [rsp+408h] [rbp-3C8h] BYREF
  __int64 v383; // [rsp+410h] [rbp-3C0h]
  unsigned __int64 v384; // [rsp+418h] [rbp-3B8h] BYREF
  unsigned __int64 v385; // [rsp+420h] [rbp-3B0h]
  unsigned __int64 v386; // [rsp+428h] [rbp-3A8h]
  _QWORD *v387; // [rsp+470h] [rbp-360h]
  unsigned int v388; // [rsp+480h] [rbp-350h]
  char v389; // [rsp+578h] [rbp-258h]
  int v390; // [rsp+57Ch] [rbp-254h]
  __int64 v391; // [rsp+580h] [rbp-250h]
  unsigned __int64 *i; // [rsp+590h] [rbp-240h] BYREF
  __int64 v393; // [rsp+598h] [rbp-238h] BYREF
  unsigned __int64 v394; // [rsp+5A0h] [rbp-230h] BYREF
  __m128i j; // [rsp+5A8h] [rbp-228h] BYREF
  _QWORD v396[2]; // [rsp+5B8h] [rbp-218h] BYREF
  __m128i v397; // [rsp+5C8h] [rbp-208h] BYREF
  __int64 v398; // [rsp+5D8h] [rbp-1F8h]
  char v399; // [rsp+5E0h] [rbp-1F0h]
  __m128i **v400; // [rsp+5E8h] [rbp-1E8h] BYREF
  unsigned int v401; // [rsp+5F0h] [rbp-1E0h]
  __m128i *v402; // [rsp+5F8h] [rbp-1D8h] BYREF
  __m128i *v403; // [rsp+600h] [rbp-1D0h]
  unsigned __int64 v404; // [rsp+608h] [rbp-1C8h]
  _QWORD v405[2]; // [rsp+610h] [rbp-1C0h] BYREF
  unsigned __int64 v406; // [rsp+620h] [rbp-1B0h]
  char v407; // [rsp+638h] [rbp-198h] BYREF
  _BYTE *v408; // [rsp+678h] [rbp-158h]
  _BYTE *v409; // [rsp+680h] [rbp-150h]
  unsigned __int64 v410; // [rsp+688h] [rbp-148h]
  char v411; // [rsp+758h] [rbp-78h]
  int v412; // [rsp+75Ch] [rbp-74h]
  __int64 v413; // [rsp+760h] [rbp-70h]

  v14 = *(void ***)(a2 + 32);
  v344 = (__int64 *)v346;
  v345 = 0x800000000LL;
  v321 = *(void ***)(a2 + 40);
  if ( v14 == v321 )
    return 0;
  v331 = v14;
  do
  {
    v15 = (unsigned __int64 *)*v331;
    v362 = 0;
    memset(v357, 0, sizeof(v357));
    v361[0] = v15;
    v357[1] = (unsigned __int64)&v357[5];
    v357[2] = (unsigned __int64)&v357[5];
    i = v15;
    v358.m128i_i64[1] = (__int64)v361;
    src[0] = (unsigned __int64)v361;
    src[1] = 0x100000008LL;
    LODWORD(v357[3]) = 8;
    v363 = 0;
    v364 = 0;
    v360 = 0;
    v358.m128i_i64[0] = 1;
    LOBYTE(v394) = 0;
    sub_197E9F0(&v362, (__int64)&i);
    sub_16CCEE0(&v375, (__int64)v378, 8, (__int64)v357);
    v16 = v357[13];
    memset(&v357[13], 0, 24);
    v384 = v16;
    v385 = v357[14];
    v386 = v357[15];
    sub_16CCEE0(&v365, (__int64)v371, 8, (__int64)&v358);
    v17 = v362;
    v362 = 0;
    v372 = v17;
    v18 = v363;
    v363 = 0;
    v373 = v18;
    v19 = v364;
    v364 = 0;
    v374 = v19;
    sub_16CCEE0(&i, (__int64)v396, 8, (__int64)&v365);
    v20 = v372;
    v372 = 0;
    v402 = (__m128i *)v20;
    v21 = v373;
    v373 = 0;
    v403 = (__m128i *)v21;
    v22 = v374;
    v374 = 0;
    v404 = v22;
    sub_16CCEE0(v405, (__int64)&v407, 8, (__int64)&v375);
    v23 = (_BYTE *)v384;
    v384 = 0;
    v408 = v23;
    v24 = v385;
    v385 = 0;
    v409 = (_BYTE *)v24;
    v25 = v386;
    v386 = 0;
    v410 = v25;
    if ( v372 )
      j_j___libc_free_0(v372, v374 - v372);
    if ( v367 != v366 )
      _libc_free(v367);
    if ( v384 )
      j_j___libc_free_0(v384, v386 - v384);
    if ( dest != v375.m128i_i64[1] )
      _libc_free(dest);
    if ( v362 )
      j_j___libc_free_0(v362, v364 - v362);
    if ( src[0] != v358.m128i_i64[1] )
      _libc_free(src[0]);
    if ( v357[13] )
      j_j___libc_free_0(v357[13], v357[15] - v357[13]);
    if ( v357[2] != v357[1] )
      _libc_free(v357[2]);
    v26 = v371;
    sub_16CCCB0(&v365, (__int64)v371, (__int64)&i);
    v28 = v403;
    v29 = v402;
    v372 = 0;
    v373 = 0;
    v374 = 0;
    v30 = (char *)v403 - (char *)v402;
    if ( v403 == v402 )
    {
      v30 = 0;
      v32 = 0;
    }
    else
    {
      if ( v30 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_247;
      v31 = sub_22077B0((char *)v403 - (char *)v402);
      v28 = v403;
      v29 = v402;
      v32 = v31;
    }
    v372 = v32;
    v373 = v32;
    v374 = v32 + v30;
    if ( v28 != v29 )
    {
      v33 = v32;
      v34 = v29;
      do
      {
        if ( v33 )
        {
          *(_QWORD *)v33 = v34->m128i_i64[0];
          v35 = v34[1].m128i_i8[0];
          *(_BYTE *)(v33 + 16) = v35;
          if ( v35 )
            *(_QWORD *)(v33 + 8) = v34->m128i_i64[1];
        }
        v34 = (__m128i *)((char *)v34 + 24);
        v33 += 24LL;
      }
      while ( v28 != v34 );
      v32 += 8 * ((unsigned __int64)((char *)&v28[-2].m128i_u64[1] - (char *)v29) >> 3) + 24;
    }
    v373 = v32;
    v29 = &v375;
    sub_16CCCB0(&v375, (__int64)v378, (__int64)v405);
    v38 = v409;
    v26 = v408;
    v384 = 0;
    v385 = 0;
    v386 = 0;
    v39 = v409 - v408;
    if ( v409 == v408 )
    {
      v39 = 0;
      v41 = 0;
    }
    else
    {
      if ( v39 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_247:
        sub_4261EA(v29, v26, v27);
      v40 = sub_22077B0(v409 - v408);
      v38 = v409;
      v26 = v408;
      v41 = v40;
    }
    v384 = v41;
    v385 = v41;
    v386 = v41 + v39;
    if ( v26 == v38 )
    {
      v44 = v41;
    }
    else
    {
      v42 = v41;
      v43 = v26;
      do
      {
        if ( v42 )
        {
          *(_QWORD *)v42 = *(_QWORD *)v43;
          LODWORD(v36) = (unsigned __int8)v43[16];
          *(_BYTE *)(v42 + 16) = v36;
          if ( (_BYTE)v36 )
          {
            v36 = *((_QWORD *)v43 + 1);
            *(_QWORD *)(v42 + 8) = v36;
          }
        }
        v43 += 24;
        v42 += 24LL;
      }
      while ( v43 != v38 );
      v44 = v41 + 8 * ((unsigned __int64)(v43 - 24 - v26) >> 3) + 24;
    }
    v45 = v373;
    v46 = v372;
    v385 = v44;
    if ( v373 - v372 == v44 - v41 )
      goto LABEL_56;
    do
    {
LABEL_40:
      v47 = *(_QWORD *)(v45 - 24);
      if ( *(_QWORD *)(v47 + 16) == *(_QWORD *)(v47 + 8) )
      {
        v62 = (unsigned int)v345;
        if ( (unsigned int)v345 >= HIDWORD(v345) )
        {
          sub_16CD150((__int64)&v344, v346, 0, 8, v36, v37);
          v62 = (unsigned int)v345;
        }
        v344[v62] = v47;
        v45 = v373;
        LODWORD(v345) = v345 + 1;
LABEL_100:
        v47 = *(_QWORD *)(v45 - 24);
      }
      if ( !*(_BYTE *)(v45 - 8) )
      {
        v48 = *(__int64 **)(v47 + 8);
        *(_BYTE *)(v45 - 8) = 1;
        *(_QWORD *)(v45 - 16) = v48;
        goto LABEL_45;
      }
      while ( 1 )
      {
        v48 = *(__int64 **)(v45 - 16);
LABEL_45:
        if ( v48 == *(__int64 **)(v47 + 16) )
          break;
        *(_QWORD *)(v45 - 16) = v48 + 1;
        v50 = *v48;
        v51 = (__int64 *)v366;
        if ( v367 != v366 )
          goto LABEL_43;
        v52 = (__int64 *)(v366 + 8LL * v369);
        if ( (__int64 *)v366 == v52 )
        {
LABEL_97:
          if ( v369 < v368 )
          {
            ++v369;
            *v52 = v50;
            v365 = (unsigned __int64 *)((char *)v365 + 1);
LABEL_54:
            v358.m128i_i64[0] = v50;
            LOBYTE(src[0]) = 0;
            sub_197E9F0(&v372, (__int64)&v358);
            v46 = v372;
            v45 = v373;
            goto LABEL_55;
          }
LABEL_43:
          sub_16CCBA0((__int64)&v365, v50);
          if ( v49 )
            goto LABEL_54;
        }
        else
        {
          v53 = 0;
          while ( v50 != *v51 )
          {
            if ( *v51 == -2 )
            {
              v53 = v51;
              if ( v51 + 1 == v52 )
                goto LABEL_53;
              ++v51;
            }
            else if ( v52 == ++v51 )
            {
              if ( !v53 )
                goto LABEL_97;
LABEL_53:
              *v53 = v50;
              --v370;
              v365 = (unsigned __int64 *)((char *)v365 + 1);
              goto LABEL_54;
            }
          }
        }
      }
      v373 -= 24LL;
      v46 = v372;
      v45 = v373;
      if ( v373 != v372 )
        goto LABEL_100;
LABEL_55:
      v41 = v384;
    }
    while ( v45 - v46 != v385 - v384 );
LABEL_56:
    if ( v45 != v46 )
    {
      v54 = v41;
      while ( *(_QWORD *)v46 == *(_QWORD *)v54 )
      {
        v55 = *(_BYTE *)(v46 + 16);
        v56 = *(_BYTE *)(v54 + 16);
        if ( v55 && v56 )
          v57 = *(_QWORD *)(v46 + 8) == *(_QWORD *)(v54 + 8);
        else
          v57 = v55 == v56;
        if ( !v57 )
          break;
        v46 += 24LL;
        v54 += 24LL;
        if ( v46 == v45 )
          goto LABEL_64;
      }
      goto LABEL_40;
    }
LABEL_64:
    if ( v41 )
      j_j___libc_free_0(v41, v386 - v41);
    if ( dest != v375.m128i_i64[1] )
      _libc_free(dest);
    if ( v372 )
      j_j___libc_free_0(v372, v374 - v372);
    if ( v367 != v366 )
      _libc_free(v367);
    if ( v408 )
      j_j___libc_free_0(v408, v410 - (_QWORD)v408);
    if ( v406 != v405[1] )
      _libc_free(v406);
    if ( v402 )
      j_j___libc_free_0(v402, v404 - (_QWORD)v402);
    if ( v394 != v393 )
      _libc_free(v394);
    ++v331;
  }
  while ( v321 != v331 );
  v315 = &v344[(unsigned int)v345];
  if ( v344 == v315 )
  {
    v58 = 0;
    goto LABEL_108;
  }
  v322 = v344;
  v58 = 0;
  while ( 2 )
  {
    v61 = *v322;
    v339 = 0;
    v343 = 0;
    v337[1] = a1;
    v337[0] = v61;
    v338 = a2;
    v340 = a3;
    v341 = a4;
    v342 = a5;
    sub_1B17B90(&i, v61, "llvm.loop.distribute.enable", 27);
    if ( (_BYTE)v393 )
    {
      v59 = *(_QWORD *)(*i + 136);
      v60 = *(_QWORD **)(v59 + 24);
      if ( *(_DWORD *)(v59 + 32) > 0x40u )
        v60 = (_QWORD *)*v60;
      LOBYTE(v343) = v60 != 0;
      if ( !HIBYTE(v343) )
        HIBYTE(v343) = 1;
LABEL_87:
      if ( !(_BYTE)v343 )
        goto LABEL_88;
      goto LABEL_92;
    }
    if ( HIBYTE(v343) )
      goto LABEL_87;
    if ( !byte_4FB5360 )
      goto LABEL_88;
LABEL_92:
    if ( !sub_13FA090(v337[0]) )
    {
      v58 |= sub_1A8B7A0(v337, (__int64)"MultipleExitBlocks", 18, "multiple exit blocks", 0x14u);
      goto LABEL_88;
    }
    if ( !(unsigned __int8)sub_13FCBF0(v337[0]) )
    {
      v58 |= sub_1A8B7A0(v337, (__int64)"NotLoopSimplifyForm", 19, "loop is not in loop-simplify form", 0x21u);
      goto LABEL_88;
    }
    v64 = v337[0];
    v323 = sub_13FC520(v337[0]);
    if ( !*(_QWORD *)(a6 + 16) )
      sub_4263D6(v64, v337[0], v65);
    v66 = (*(__int64 (__fastcall **)(__int64, __int64))(a6 + 24))(a6, v337[0]);
    v68 = *(_BYTE *)(v66 + 48) == 0;
    v339 = (__int64 *)v66;
    if ( !v68 )
    {
      v58 |= sub_1A8B7A0(
               v337,
               (__int64)"MemOpsCanBeVectorized",
               21,
               "memory operations are safe for vectorization",
               0x2Cu);
      goto LABEL_88;
    }
    v69 = *(_QWORD *)(v66 + 16);
    v326 = *(_BYTE *)(v69 + 218);
    if ( !v326 || (v70 = *(_DWORD *)(v69 + 232)) == 0 )
    {
      v58 |= sub_1A8B7A0(v337, (__int64)"NoUnsafeDeps", 12, "no unsafe dependences to isolate", 0x20u);
      goto LABEL_88;
    }
    v71 = 0;
    v353 = 0;
    v349 = 0;
    v354 = v337[0];
    v350 = 0;
    v355 = v338;
    v351 = 0;
    v356 = v340;
    v352 = 0;
    v72 = *(_QWORD *)(v66 + 16);
    v348 = &v347;
    v347 = &v347;
    v365 = &v367;
    v366 = 0x800000000LL;
    v73 = *(unsigned int *)(v72 + 56);
    v74 = *(unsigned __int64 **)(v72 + 48);
    v75 = &v367;
    v76 = v73;
    v77 = &v74[v76];
    v78 = (v76 * 8) >> 3;
    if ( (unsigned __int64)v76 > 8 )
    {
      sub_16CD150((__int64)&v365, &v367, (v76 * 8) >> 3, 16, v70, v67);
      v71 = v366;
      v75 = &v365[2 * (unsigned int)v366];
    }
    if ( v74 != v77 )
    {
      do
      {
        if ( v75 )
        {
          v79 = *v74;
          *((_DWORD *)v75 + 2) = 0;
          *v75 = v79;
        }
        ++v74;
        v75 += 2;
      }
      while ( v77 != v74 );
      v71 = v366;
    }
    LODWORD(v366) = v71 + v78;
    v80 = v71 + v78;
    v81 = *(unsigned int **)(v69 + 224);
    v82 = &v81[3 * *(unsigned int *)(v69 + 232)];
    if ( v81 != v82 )
    {
      do
      {
        if ( (unsigned __int8)sub_385F840(v81) )
        {
          v83 = (unsigned __int64)v365;
          ++LODWORD(v365[2 * *v81 + 1]);
          --*(_DWORD *)(v83 + 16LL * v81[1] + 8);
        }
        v81 += 3;
      }
      while ( v82 != v81 );
      v80 = v366;
    }
    v84 = 0;
    v85 = (unsigned __int64)v365;
    v86 = 2LL * v80;
    v327 = &v365[v86];
    if ( v365 != &v365[v86] )
    {
      v316 = v58;
      while ( 1 )
      {
        v87 = *(_QWORD *)v85;
        if ( !v84 && !*(_DWORD *)(v85 + 8) )
          break;
        if ( v347 == &v347 || (v88 = v348, !*((_BYTE *)v348 + 120)) )
        {
          v89 = sub_22077B0(304);
          *(_QWORD *)(v89 + 24) = v89 + 56;
          *(_QWORD *)(v89 + 32) = v89 + 56;
          v90 = v354;
          *(_QWORD *)(v89 + 16) = 0;
          *(_QWORD *)(v89 + 128) = v90;
          *(_QWORD *)(v89 + 40) = 8;
          *(_DWORD *)(v89 + 48) = 0;
          *(_BYTE *)(v89 + 120) = 1;
          *(_QWORD *)(v89 + 136) = 0;
          *(_QWORD *)(v89 + 144) = v89 + 160;
          *(_QWORD *)(v89 + 152) = 0x800000000LL;
          *(_QWORD *)(v89 + 224) = 0;
          *(_DWORD *)(v89 + 248) = 128;
          v91 = (_QWORD *)sub_22077B0(0x2000);
          *(_QWORD *)(v89 + 240) = 0;
          *(_QWORD *)(v89 + 232) = v91;
          v393 = 2;
          v92 = *(unsigned int *)(v89 + 248);
          v394 = 0;
          j = (__m128i)0xFFFFFFFFFFFFFFF8LL;
          v93 = &v91[8 * v92];
          for ( i = (unsigned __int64 *)&unk_49E6B50; v93 != v91; v91 += 8 )
          {
            if ( v91 )
            {
              v94 = v393;
              v91[2] = 0;
              v91[3] = -8;
              *v91 = &unk_49E6B50;
              v91[1] = v94 & 6;
              v91[4] = j.m128i_i64[1];
            }
          }
          *(_BYTE *)(v89 + 288) = 0;
          v95 = *(__int64 **)(v89 + 24);
          *(_BYTE *)(v89 + 297) = 1;
          if ( *(__int64 **)(v89 + 32) != v95 )
            goto LABEL_143;
          v119 = &v95[*(unsigned int *)(v89 + 44)];
          v120 = *(_DWORD *)(v89 + 44);
          if ( v95 != v119 )
          {
            v121 = 0;
            while ( v87 != *v95 )
            {
              if ( *v95 == -2 )
                v121 = v95;
              if ( v119 == ++v95 )
                goto LABEL_179;
            }
            goto LABEL_144;
          }
LABEL_188:
          if ( v120 >= *(_DWORD *)(v89 + 40) )
            goto LABEL_143;
          *(_DWORD *)(v89 + 44) = v120 + 1;
          *v119 = v87;
          ++*(_QWORD *)(v89 + 16);
          goto LABEL_144;
        }
        v112 = v348[3];
        if ( v348[4] != v112 )
          goto LABEL_167;
        v138 = &v112[*((unsigned int *)v348 + 11)];
        v139 = *((_DWORD *)v348 + 11);
        if ( v112 != v138 )
        {
          v140 = 0;
          while ( v87 != *v112 )
          {
            if ( *v112 == -2 )
              v140 = v112;
            if ( v138 == ++v112 )
            {
              if ( !v140 )
                goto LABEL_248;
              *v140 = v87;
              --*((_DWORD *)v88 + 12);
              v88[2] = (_QWORD *)((char *)v88[2] + 1);
              goto LABEL_145;
            }
          }
          goto LABEL_145;
        }
LABEL_248:
        if ( v139 < *((_DWORD *)v348 + 10) )
        {
          *((_DWORD *)v348 + 11) = v139 + 1;
          *v138 = v87;
          v88[2] = (_QWORD *)((char *)v88[2] + 1);
        }
        else
        {
LABEL_167:
          sub_16CCBA0((__int64)(v348 + 2), *(_QWORD *)v85);
        }
LABEL_145:
        v84 += *(_DWORD *)(v85 + 8);
        v85 += 16LL;
        if ( v327 == (unsigned __int64 *)v85 )
        {
          LOBYTE(v58) = v316;
          goto LABEL_147;
        }
      }
      v89 = sub_22077B0(304);
      *(_QWORD *)(v89 + 24) = v89 + 56;
      *(_QWORD *)(v89 + 32) = v89 + 56;
      v113 = v354;
      *(_QWORD *)(v89 + 16) = 0;
      *(_QWORD *)(v89 + 128) = v113;
      *(_QWORD *)(v89 + 40) = 8;
      *(_DWORD *)(v89 + 48) = 0;
      *(_BYTE *)(v89 + 120) = 0;
      *(_QWORD *)(v89 + 136) = 0;
      *(_QWORD *)(v89 + 144) = v89 + 160;
      *(_QWORD *)(v89 + 152) = 0x800000000LL;
      *(_QWORD *)(v89 + 224) = 0;
      *(_DWORD *)(v89 + 248) = 128;
      v114 = (_QWORD *)sub_22077B0(0x2000);
      *(_QWORD *)(v89 + 240) = 0;
      *(_QWORD *)(v89 + 232) = v114;
      v393 = 2;
      v115 = *(unsigned int *)(v89 + 248);
      v394 = 0;
      j = (__m128i)0xFFFFFFFFFFFFFFF8LL;
      v116 = &v114[8 * v115];
      for ( i = (unsigned __int64 *)&unk_49E6B50; v116 != v114; v114 += 8 )
      {
        if ( v114 )
        {
          v117 = v393;
          v114[2] = 0;
          v114[3] = -8;
          *v114 = &unk_49E6B50;
          v114[1] = v117 & 6;
          v114[4] = j.m128i_i64[1];
        }
      }
      *(_BYTE *)(v89 + 288) = 0;
      v118 = *(__int64 **)(v89 + 24);
      *(_BYTE *)(v89 + 297) = 1;
      if ( *(__int64 **)(v89 + 32) == v118 )
      {
        v119 = &v118[*(unsigned int *)(v89 + 44)];
        v120 = *(_DWORD *)(v89 + 44);
        if ( v118 != v119 )
        {
          v121 = 0;
          while ( v87 != *v118 )
          {
            if ( *v118 == -2 )
              v121 = v118;
            if ( v119 == ++v118 )
            {
LABEL_179:
              if ( !v121 )
                goto LABEL_188;
              *v121 = v87;
              --*(_DWORD *)(v89 + 48);
              ++*(_QWORD *)(v89 + 16);
              goto LABEL_144;
            }
          }
          goto LABEL_144;
        }
        goto LABEL_188;
      }
LABEL_143:
      sub_16CCBA0(v89 + 16, v87);
LABEL_144:
      sub_2208C80(v89, &v347);
      ++v349;
      goto LABEL_145;
    }
LABEL_147:
    v96 = (_QWORD **)v337[0];
    sub_1B17630(v357, v337[0]);
    v97 = (__int64 *)v357[0];
    v98 = v357[0] + 8LL * LODWORD(v357[1]);
    if ( v357[0] != v98 )
    {
      while ( 1 )
      {
        v317 = *v97;
        v99 = sub_22077B0(304);
        *(_QWORD *)(v99 + 24) = v99 + 56;
        *(_QWORD *)(v99 + 32) = v99 + 56;
        v100 = v354;
        *(_QWORD *)(v99 + 16) = 0;
        *(_QWORD *)(v99 + 128) = v100;
        *(_QWORD *)(v99 + 40) = 8;
        *(_DWORD *)(v99 + 48) = 0;
        *(_BYTE *)(v99 + 120) = 0;
        *(_QWORD *)(v99 + 136) = 0;
        *(_QWORD *)(v99 + 144) = v99 + 160;
        *(_QWORD *)(v99 + 152) = 0x800000000LL;
        *(_QWORD *)(v99 + 224) = 0;
        *(_DWORD *)(v99 + 248) = 128;
        v101 = (_QWORD *)sub_22077B0(0x2000);
        v102 = *(unsigned int *)(v99 + 248);
        *(_QWORD *)(v99 + 240) = 0;
        *(_QWORD *)(v99 + 232) = v101;
        v393 = 2;
        v103 = &v101[8 * v102];
        i = (unsigned __int64 *)&unk_49E6B50;
        v394 = 0;
        for ( j = (__m128i)0xFFFFFFFFFFFFFFF8LL; v103 != v101; v101 += 8 )
        {
          if ( v101 )
          {
            v104 = v393;
            v101[2] = 0;
            v101[3] = -8;
            *v101 = &unk_49E6B50;
            v101[1] = v104 & 6;
            v101[4] = j.m128i_i64[1];
          }
        }
        *(_BYTE *)(v99 + 288) = 0;
        v105 = *(__int64 **)(v99 + 24);
        *(_BYTE *)(v99 + 297) = 1;
        if ( *(__int64 **)(v99 + 32) != v105 )
          goto LABEL_153;
        v131 = &v105[*(unsigned int *)(v99 + 44)];
        v132 = *(_DWORD *)(v99 + 44);
        if ( v105 != v131 )
        {
          v133 = 0;
          while ( v317 != *v105 )
          {
            if ( *v105 == -2 )
              v133 = v105;
            if ( v131 == ++v105 )
            {
              if ( !v133 )
                goto LABEL_245;
              *v133 = v317;
              --*(_DWORD *)(v99 + 48);
              ++*(_QWORD *)(v99 + 16);
              goto LABEL_154;
            }
          }
          goto LABEL_154;
        }
LABEL_245:
        if ( v132 < *(_DWORD *)(v99 + 40) )
        {
          *(_DWORD *)(v99 + 44) = v132 + 1;
          *v131 = v317;
          ++*(_QWORD *)(v99 + 16);
        }
        else
        {
LABEL_153:
          sub_16CCBA0(v99 + 16, v317);
        }
LABEL_154:
        v96 = &v347;
        ++v97;
        sub_2208C80(v99, &v347);
        v106 = ++v349;
        if ( (__int64 *)v98 == v97 )
          goto LABEL_155;
      }
    }
    v106 = v349;
LABEL_155:
    if ( v106 <= 1 )
      goto LABEL_190;
    sub_1A8AAF0(&v347);
    if ( (unsigned int)v349 <= 1 )
      goto LABEL_190;
    for ( k = v347; k != &v347; k = (_QWORD *)*k )
      sub_1A89C20((__int64)(k + 2), (__int64)v96, v107, v108, v109);
    if ( (unsigned __int8)sub_1A8A120(&v347) && (unsigned int)v349 <= 1 )
    {
LABEL_190:
      v326 = v58 | sub_1A8B7A0(v337, (__int64)"CantIsolateUnsafeDeps", 21, "cannot isolate unsafe dependencies", 0x22u);
      goto LABEL_191;
    }
    v311 = sub_1458800(*v339);
    v111 = &qword_4FB5480;
    if ( HIBYTE(v343) && (_BYTE)v343 )
      v111 = &qword_4FB53A0;
    if ( *(_DWORD *)(v311 + 48) > *((_DWORD *)v111 + 40) )
    {
      v326 = v58
           | sub_1A8B7A0(
               v337,
               (__int64)"TooManySCEVRuntimeChecks",
               24,
               "too many SCEV run-time checks needed.\n",
               0x26u);
      goto LABEL_191;
    }
    s = 0;
    v141 = v347;
    if ( v347 != &v347 )
    {
      while ( 1 )
      {
        v142 = (_QWORD *)v141[4];
        v143 = v142 == (_QWORD *)v141[3] ? *((unsigned int *)v141 + 11) : *((unsigned int *)v141 + 10);
        v329 = &v142[v143];
        if ( v142 != v329 )
          break;
LABEL_256:
        ++s;
        v141 = (_QWORD *)*v141;
        if ( v141 == &v347 )
          goto LABEL_257;
      }
      while ( 1 )
      {
        v144 = *v142;
        v318 = v142;
        if ( *v142 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v329 == ++v142 )
          goto LABEL_256;
      }
LABEL_298:
      if ( v318 == v329 )
        goto LABEL_256;
      v185 = v353;
      v186 = v351;
      if ( v353 )
      {
        v187 = v353 - 1;
        v188 = 1;
        v189 = 0;
        v190 = (v353 - 1) & (((unsigned int)v144 >> 9) ^ ((unsigned int)v144 >> 4));
        v191 = &v351[2 * v190];
        v192 = *v191;
        if ( v144 == *v191 )
        {
LABEL_301:
          *((_DWORD *)v191 + 2) = -1;
LABEL_302:
          v193 = v318 + 1;
          if ( v318 + 1 == v329 )
            goto LABEL_256;
          while ( 1 )
          {
            v144 = *v193;
            v318 = v193;
            if ( *v193 < 0xFFFFFFFFFFFFFFFELL )
              goto LABEL_298;
            if ( v329 == ++v193 )
              goto LABEL_256;
          }
        }
        while ( v192 != -8 )
        {
          if ( !v189 && v192 == -16 )
            v189 = v191;
          v190 = v187 & (v188 + v190);
          v191 = &v351[2 * v190];
          v192 = *v191;
          if ( v144 == *v191 )
            goto LABEL_301;
          ++v188;
        }
        if ( v189 )
          v191 = v189;
        ++v350;
        v261 = v352 + 1;
        if ( 4 * ((int)v352 + 1) < 3 * v353 )
        {
          if ( v353 - HIDWORD(v352) - v261 <= v353 >> 3 )
          {
            v262 = (((((((((v187 | ((unsigned __int64)v187 >> 1)) >> 2) | v187 | ((unsigned __int64)v187 >> 1)) >> 4)
                      | ((v187 | ((unsigned __int64)v187 >> 1)) >> 2)
                      | v187
                      | ((unsigned __int64)v187 >> 1)) >> 8)
                    | ((((v187 | ((unsigned __int64)v187 >> 1)) >> 2) | v187 | ((unsigned __int64)v187 >> 1)) >> 4)
                    | ((v187 | ((unsigned __int64)v187 >> 1)) >> 2)
                    | v187
                    | ((unsigned __int64)v187 >> 1)) >> 16)
                  | ((((((v187 | ((unsigned __int64)v187 >> 1)) >> 2) | v187 | ((unsigned __int64)v187 >> 1)) >> 4)
                    | ((v187 | ((unsigned __int64)v187 >> 1)) >> 2)
                    | v187
                    | ((unsigned __int64)v187 >> 1)) >> 8)
                  | ((((v187 | ((unsigned __int64)v187 >> 1)) >> 2) | v187 | ((unsigned __int64)v187 >> 1)) >> 4)
                  | ((v187 | ((unsigned __int64)v187 >> 1)) >> 2)
                  | v187
                  | ((unsigned __int64)v187 >> 1))
                 + 1;
            if ( (unsigned int)v262 < 0x40 )
              LODWORD(v262) = 64;
            v353 = v262;
            v263 = (__int64 *)sub_22077B0(16LL * (unsigned int)v262);
            v351 = v263;
            if ( v186 )
            {
              v352 = 0;
              v312 = &v186[2 * v185];
              for ( m = &v263[2 * v353]; m != v263; v263 += 2 )
              {
                if ( v263 )
                  *v263 = -8;
              }
              v265 = v186;
              do
              {
                v266 = *v265;
                if ( *v265 != -8 && v266 != -16 )
                {
                  if ( !v353 )
                  {
                    MEMORY[0] = *v265;
                    BUG();
                  }
                  v267 = 1;
                  v268 = 0;
                  v269 = (v353 - 1) & (((unsigned int)v266 >> 9) ^ ((unsigned int)v266 >> 4));
                  v270 = &v351[2 * v269];
                  v271 = *v270;
                  if ( v266 != *v270 )
                  {
                    while ( v271 != -8 )
                    {
                      if ( !v268 && v271 == -16 )
                        v268 = v270;
                      v269 = (v353 - 1) & (v267 + v269);
                      v270 = &v351[2 * v269];
                      v271 = *v270;
                      if ( v266 == *v270 )
                        goto LABEL_488;
                      ++v267;
                    }
                    if ( v268 )
                      v270 = v268;
                  }
LABEL_488:
                  *v270 = v266;
                  *((_DWORD *)v270 + 2) = *((_DWORD *)v265 + 2);
                  LODWORD(v352) = v352 + 1;
                }
                v265 += 2;
              }
              while ( v312 != v265 );
              j___libc_free_0(v186);
              v272 = v351;
              v273 = v353;
              v261 = v352 + 1;
            }
            else
            {
              v352 = 0;
              v273 = v353;
              v272 = &v263[2 * v353];
              if ( v263 != v272 )
              {
                v279 = v263;
                do
                {
                  if ( v279 )
                    *v279 = -8;
                  v279 += 2;
                }
                while ( v272 != v279 );
                v272 = v263;
              }
              v261 = 1;
            }
            if ( !v273 )
              goto LABEL_569;
            v274 = v273 - 1;
            v275 = 0;
            v276 = 1;
            v277 = v274 & (((unsigned int)v144 >> 9) ^ ((unsigned int)v144 >> 4));
            v191 = &v272[2 * v277];
            v278 = *v191;
            if ( v144 != *v191 )
            {
              while ( v278 != -8 )
              {
                if ( v278 == -16 && !v275 )
                  v275 = v191;
                v277 = v274 & (v276 + v277);
                v191 = &v272[2 * v277];
                v278 = *v191;
                if ( v144 == *v191 )
                  goto LABEL_473;
                ++v276;
              }
              if ( v275 )
                v191 = v275;
            }
          }
LABEL_473:
          LODWORD(v352) = v261;
          if ( *v191 != -8 )
            --HIDWORD(v352);
          *v191 = v144;
          *((_DWORD *)v191 + 2) = s;
          goto LABEL_302;
        }
      }
      else
      {
        ++v350;
      }
      v280 = ((((((((2 * v353 - 1) | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 2)
                | (2 * v353 - 1)
                | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 4)
              | (((2 * v353 - 1) | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 2)
              | (2 * v353 - 1)
              | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 8)
            | (((((2 * v353 - 1) | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 2)
              | (2 * v353 - 1)
              | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 4)
            | (((2 * v353 - 1) | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 2)
            | (2 * v353 - 1)
            | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 16;
      v281 = (v280
            | (((((((2 * v353 - 1) | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 2)
                | (2 * v353 - 1)
                | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 4)
              | (((2 * v353 - 1) | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 2)
              | (2 * v353 - 1)
              | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 8)
            | (((((2 * v353 - 1) | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 2)
              | (2 * v353 - 1)
              | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 4)
            | (((2 * v353 - 1) | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 2)
            | (2 * v353 - 1)
            | ((unsigned __int64)(2 * v353 - 1) >> 1))
           + 1;
      if ( (unsigned int)v281 < 0x40 )
        LODWORD(v281) = 64;
      v353 = v281;
      v282 = (__int64 *)sub_22077B0(16LL * (unsigned int)v281);
      v351 = v282;
      v283 = v282;
      if ( v186 )
      {
        v352 = 0;
        v284 = &v186[2 * v185];
        for ( n = &v283[2 * v353]; n != v283; v283 += 2 )
        {
          if ( v283 )
            *v283 = -8;
        }
        for ( ii = v186; v284 != ii; ii += 2 )
        {
          v287 = *ii;
          if ( *ii != -8 && v287 != -16 )
          {
            if ( !v353 )
            {
              MEMORY[0] = *ii;
              BUG();
            }
            v288 = 1;
            v289 = 0;
            v290 = (v353 - 1) & (((unsigned int)v287 >> 9) ^ ((unsigned int)v287 >> 4));
            v291 = &v351[2 * v290];
            v292 = *v291;
            if ( v287 != *v291 )
            {
              while ( v292 != -8 )
              {
                if ( !v289 && v292 == -16 )
                  v289 = v291;
                v290 = (v353 - 1) & (v288 + v290);
                v291 = &v351[2 * v290];
                v292 = *v291;
                if ( v287 == *v291 )
                  goto LABEL_531;
                ++v288;
              }
              if ( v289 )
                v291 = v289;
            }
LABEL_531:
            *v291 = v287;
            *((_DWORD *)v291 + 2) = *((_DWORD *)ii + 2);
            LODWORD(v352) = v352 + 1;
          }
        }
        j___libc_free_0(v186);
        v293 = v351;
        v294 = v353;
        v261 = v352 + 1;
      }
      else
      {
        v352 = 0;
        v294 = v353;
        v293 = &v282[2 * v353];
        if ( v282 != v293 )
        {
          do
          {
            if ( v282 )
              *v282 = -8;
            v282 += 2;
          }
          while ( v293 != v282 );
          v293 = v283;
        }
        v261 = 1;
      }
      if ( !v294 )
      {
LABEL_569:
        LODWORD(v352) = v352 + 1;
        BUG();
      }
      v295 = v294 - 1;
      v296 = (v294 - 1) & (((unsigned int)v144 >> 9) ^ ((unsigned int)v144 >> 4));
      v191 = &v293[2 * v296];
      v297 = *v191;
      if ( v144 != *v191 )
      {
        v298 = 1;
        v299 = 0;
        while ( v297 != -8 )
        {
          if ( !v299 && v297 == -16 )
            v299 = v191;
          v296 = v295 & (v298 + v296);
          v191 = &v293[2 * v296];
          v297 = *v191;
          if ( v144 == *v191 )
            goto LABEL_473;
          ++v298;
        }
        if ( v299 )
          v191 = v299;
      }
      goto LABEL_473;
    }
LABEL_257:
    if ( !sub_157F0B0(v323) )
    {
      v146 = sub_157EBA0(v323);
LABEL_261:
      sub_1AA8CA0(v323, v146, v340, v338);
      goto LABEL_262;
    }
    v145 = *(_QWORD *)(v323 + 48);
    if ( v145 )
      v145 -= 24;
    v146 = sub_157EBA0(v323);
    if ( v146 != v145 )
      goto LABEL_261;
LABEL_262:
    v309 = v339;
    v149 = *(unsigned int *)(v339[1] + 16);
    v307 = v339[1];
    v150 = *(_DWORD *)(v307 + 16);
    v151 = 4 * v149;
    v334 = v336;
    v335 = 0x800000000LL;
    if ( v149 <= 8 )
    {
      LODWORD(v335) = v149;
      v152 = &v336[v151];
      v153 = v336;
      if ( &v336[v151] == v336 )
        goto LABEL_265;
      goto LABEL_264;
    }
    sub_16CD150((__int64)&v334, v336, v149, 4, v147, v148);
    v153 = v334;
    LODWORD(v335) = v150;
    v152 = &v334[v151];
    if ( &v334[v151] != v334 )
    {
LABEL_264:
      memset(v153, 0, v152 - v153);
LABEL_265:
      if ( !v150 )
        goto LABEL_321;
    }
    v330 = 0;
    v300 = v150;
    do
    {
      sub_3860240(
        &i,
        v309[2],
        *(_QWORD *)(*(_QWORD *)(v307 + 8) + (v330 << 6) + 16),
        *(unsigned __int8 *)(*(_QWORD *)(v307 + 8) + (v330 << 6) + 40));
      v154 = &v334[4 * v330];
      *v154 = -2;
      v319 = v154;
      v155 = &i[(unsigned int)v393];
      if ( i == v155 )
        goto LABEL_318;
      v324 = i;
      do
      {
        v163 = v353;
        v164 = v351;
        v165 = *v324;
        if ( !v353 )
        {
          ++v350;
LABEL_277:
          v166 = ((((((((2 * v353 - 1) | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 2)
                    | (2 * v353 - 1)
                    | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 4)
                  | (((2 * v353 - 1) | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 2)
                  | (2 * v353 - 1)
                  | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 8)
                | (((((2 * v353 - 1) | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 2)
                  | (2 * v353 - 1)
                  | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 4)
                | (((2 * v353 - 1) | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 2)
                | (2 * v353 - 1)
                | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 16;
          v167 = (v166
                | (((((((2 * v353 - 1) | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 2)
                    | (2 * v353 - 1)
                    | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 4)
                  | (((2 * v353 - 1) | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 2)
                  | (2 * v353 - 1)
                  | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 8)
                | (((((2 * v353 - 1) | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 2)
                  | (2 * v353 - 1)
                  | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 4)
                | (((2 * v353 - 1) | ((unsigned __int64)(2 * v353 - 1) >> 1)) >> 2)
                | (2 * v353 - 1)
                | ((unsigned __int64)(2 * v353 - 1) >> 1))
               + 1;
          if ( (unsigned int)v167 < 0x40 )
            LODWORD(v167) = 64;
          v353 = v167;
          v168 = (__int64 *)sub_22077B0(16LL * (unsigned int)v167);
          v351 = v168;
          v169 = v168;
          if ( v164 )
          {
            v352 = 0;
            v170 = &v164[2 * v163];
            for ( jj = &v169[2 * v353]; jj != v169; v169 += 2 )
            {
              if ( v169 )
                *v169 = -8;
            }
            for ( kk = v164; v170 != kk; kk += 2 )
            {
              v173 = *kk;
              if ( *kk != -16 && v173 != -8 )
              {
                if ( !v353 )
                  goto LABEL_568;
                v174 = 1;
                v175 = 0;
                v176 = (v353 - 1) & (((unsigned int)v173 >> 9) ^ ((unsigned int)v173 >> 4));
                v177 = &v351[2 * v176];
                v178 = *v177;
                if ( v173 != *v177 )
                {
                  while ( v178 != -8 )
                  {
                    if ( !v175 && v178 == -16 )
                      v175 = v177;
                    v176 = (v353 - 1) & (v174 + v176);
                    v177 = &v351[2 * v176];
                    v178 = *v177;
                    if ( v173 == *v177 )
                      goto LABEL_289;
                    ++v174;
                  }
                  if ( v175 )
                    v177 = v175;
                }
LABEL_289:
                *v177 = v173;
                *((_DWORD *)v177 + 2) = *((_DWORD *)kk + 2);
                LODWORD(v352) = v352 + 1;
              }
            }
            j___libc_free_0(v164);
            v179 = v351;
            v180 = v353;
            v181 = v352 + 1;
          }
          else
          {
            v352 = 0;
            v180 = v353;
            v179 = &v168[2 * v353];
            if ( v168 != v179 )
            {
              do
              {
                if ( v168 )
                  *v168 = -8;
                v168 += 2;
              }
              while ( v179 != v168 );
              v179 = v169;
            }
            v181 = 1;
          }
          if ( !v180 )
            goto LABEL_569;
          v182 = v180 - 1;
          v183 = (v180 - 1) & (((unsigned int)v165 >> 9) ^ ((unsigned int)v165 >> 4));
          v158 = (unsigned __int64 *)&v179[2 * v183];
          v184 = *v158;
          if ( *v158 != v165 )
          {
            v194 = 1;
            v195 = 0;
            while ( v184 != -8 )
            {
              if ( !v195 && v184 == -16 )
                v195 = v158;
              v183 = v182 & (v194 + v183);
              v158 = (unsigned __int64 *)&v179[2 * v183];
              v184 = *v158;
              if ( v165 == *v158 )
                goto LABEL_294;
              ++v194;
            }
            if ( v195 )
              v158 = v195;
          }
          goto LABEL_294;
        }
        v156 = v353 - 1;
        v157 = (v353 - 1) & (((unsigned int)v165 >> 9) ^ ((unsigned int)v165 >> 4));
        v158 = (unsigned __int64 *)&v351[2 * v157];
        v159 = *v158;
        if ( *v158 == v165 )
        {
LABEL_270:
          v160 = v319;
          v161 = *((_DWORD *)v158 + 2);
          v162 = *v319;
          if ( *v319 == -2 )
            goto LABEL_297;
          goto LABEL_271;
        }
        v242 = 1;
        v243 = 0;
        while ( v159 != -8 )
        {
          if ( !v243 && v159 == -16 )
            v243 = v158;
          v157 = v156 & (v242 + v157);
          v158 = (unsigned __int64 *)&v351[2 * v157];
          v159 = *v158;
          if ( v165 == *v158 )
            goto LABEL_270;
          ++v242;
        }
        if ( v243 )
          v158 = v243;
        ++v350;
        v181 = v352 + 1;
        if ( 4 * ((int)v352 + 1) >= 3 * v353 )
          goto LABEL_277;
        if ( v353 - HIDWORD(v352) - v181 <= v353 >> 3 )
        {
          v244 = (((((((((v156 | ((unsigned __int64)v156 >> 1)) >> 2) | v156 | ((unsigned __int64)v156 >> 1)) >> 4)
                    | ((v156 | ((unsigned __int64)v156 >> 1)) >> 2)
                    | v156
                    | ((unsigned __int64)v156 >> 1)) >> 8)
                  | ((((v156 | ((unsigned __int64)v156 >> 1)) >> 2) | v156 | ((unsigned __int64)v156 >> 1)) >> 4)
                  | ((v156 | ((unsigned __int64)v156 >> 1)) >> 2)
                  | v156
                  | ((unsigned __int64)v156 >> 1)) >> 16)
                | ((((((v156 | ((unsigned __int64)v156 >> 1)) >> 2) | v156 | ((unsigned __int64)v156 >> 1)) >> 4)
                  | ((v156 | ((unsigned __int64)v156 >> 1)) >> 2)
                  | v156
                  | ((unsigned __int64)v156 >> 1)) >> 8)
                | ((((v156 | ((unsigned __int64)v156 >> 1)) >> 2) | v156 | ((unsigned __int64)v156 >> 1)) >> 4)
                | ((v156 | ((unsigned __int64)v156 >> 1)) >> 2)
                | v156
                | ((unsigned __int64)v156 >> 1))
               + 1;
          if ( (unsigned int)v244 < 0x40 )
            LODWORD(v244) = 64;
          v353 = v244;
          v245 = (__int64 *)sub_22077B0(16LL * (unsigned int)v244);
          v351 = v245;
          if ( v164 )
          {
            v352 = 0;
            v306 = &v164[2 * v163];
            for ( mm = &v245[2 * v353]; mm != v245; v245 += 2 )
            {
              if ( v245 )
                *v245 = -8;
            }
            v247 = v164;
            do
            {
              v173 = *v247;
              if ( *v247 != -8 && v173 != -16 )
              {
                if ( !v353 )
                {
LABEL_568:
                  MEMORY[0] = v173;
                  BUG();
                }
                v248 = 1;
                v249 = 0;
                v250 = (v353 - 1) & (((unsigned int)v173 >> 9) ^ ((unsigned int)v173 >> 4));
                v251 = &v351[2 * v250];
                v252 = *v251;
                if ( *v251 != v173 )
                {
                  while ( v252 != -8 )
                  {
                    if ( !v249 && v252 == -16 )
                      v249 = v251;
                    v250 = (v353 - 1) & (v248 + v250);
                    v251 = &v351[2 * v250];
                    v252 = *v251;
                    if ( v173 == *v251 )
                      goto LABEL_412;
                    ++v248;
                  }
                  if ( v249 )
                    v251 = v249;
                }
LABEL_412:
                *v251 = v173;
                *((_DWORD *)v251 + 2) = *((_DWORD *)v247 + 2);
                LODWORD(v352) = v352 + 1;
              }
              v247 += 2;
            }
            while ( v306 != v247 );
            j___libc_free_0(v164);
            v253 = v351;
            v254 = v353;
            v181 = v352 + 1;
          }
          else
          {
            v352 = 0;
            v254 = v353;
            v253 = &v245[2 * v353];
            if ( v245 != v253 )
            {
              v260 = v245;
              do
              {
                if ( v260 )
                  *v260 = -8;
                v260 += 2;
              }
              while ( v253 != v260 );
              v253 = v245;
            }
            v181 = 1;
          }
          if ( !v254 )
            goto LABEL_569;
          v255 = v254 - 1;
          v256 = 0;
          v257 = 1;
          v258 = v255 & (((unsigned int)v165 >> 9) ^ ((unsigned int)v165 >> 4));
          v158 = (unsigned __int64 *)&v253[2 * v258];
          v259 = *v158;
          if ( v165 != *v158 )
          {
            while ( v259 != -8 )
            {
              if ( !v256 && v259 == -16 )
                v256 = v158;
              v258 = v255 & (v257 + v258);
              v158 = (unsigned __int64 *)&v253[2 * v258];
              v259 = *v158;
              if ( v165 == *v158 )
                goto LABEL_294;
              ++v257;
            }
            if ( v256 )
              v158 = v256;
          }
        }
LABEL_294:
        LODWORD(v352) = v181;
        if ( *v158 != -8 )
          --HIDWORD(v352);
        *v158 = v165;
        v160 = v319;
        *((_DWORD *)v158 + 2) = 0;
        v161 = 0;
        v162 = *v319;
        if ( *v319 == -2 )
        {
LABEL_297:
          *v160 = v161;
          goto LABEL_274;
        }
LABEL_271:
        if ( v162 == -1 )
          break;
        if ( v162 != v161 )
          *v319 = -1;
LABEL_274:
        ++v324;
      }
      while ( v155 != v324 );
      v155 = i;
LABEL_318:
      if ( v155 != &v394 )
        _libc_free((unsigned __int64)v155);
      ++v330;
    }
    while ( v300 > (unsigned int)v330 );
LABEL_321:
    v196 = v339[1];
    v358.m128i_i64[0] = (__int64)src;
    v358.m128i_i64[1] = 0x400000000LL;
    v197 = *(const __m128i **)(v196 + 272);
    v198 = *(unsigned int *)(v196 + 280);
    for ( nn = &v197[v198]; nn != v197; ++v197 )
    {
      v199 = *(unsigned int **)(v197->m128i_i64[0] + 24);
      v200 = v199;
      v320 = &v199[*(unsigned int *)(v197->m128i_i64[0] + 32)];
      if ( v199 != v320 )
      {
        v201 = v197;
        while ( 1 )
        {
          v202 = v201->m128i_i64[1];
          v203 = *v200;
          v204 = *(unsigned int **)(v202 + 24);
          v310 = &v204[*(unsigned int *)(v202 + 32)];
          if ( v204 != v310 )
            break;
LABEL_390:
          if ( v320 == ++v200 )
          {
            v197 = v201;
            goto LABEL_332;
          }
        }
        v308 = v200;
        while ( 1 )
        {
          v205 = *v204;
          if ( (unsigned __int8)sub_385DBB0(v196, v203, *v204) )
          {
            if ( !(unsigned __int8)sub_385DB90(&v334, v203, v205) )
              break;
          }
          if ( v310 == ++v204 )
          {
            v200 = v308;
            goto LABEL_390;
          }
        }
        v208 = v358.m128i_u32[2];
        v197 = v201;
        if ( v358.m128i_i32[2] >= (unsigned __int32)v358.m128i_i32[3] )
        {
          sub_16CD150((__int64)&v358, src, 0, 16, v206, v207);
          v208 = v358.m128i_u32[2];
        }
        a7 = (__m128)_mm_loadu_si128(v201);
        *(__m128 *)(v358.m128i_i64[0] + 16 * v208) = a7;
        ++v358.m128i_i32[2];
      }
LABEL_332:
      ;
    }
    if ( !sub_1452CB0(v311) || (v209 = v358.m128i_u32[2], v358.m128i_i32[2]) )
    {
      sub_1B1E040((unsigned int)&i, (_DWORD)v339, v337[0], v338, v340, v341, 0);
      v212 = v358.m128i_i32[2];
      v375.m128i_i64[0] = (__int64)&dest;
      v375.m128i_i64[1] = 0x400000000LL;
      if ( !v358.m128i_i32[2] )
        goto LABEL_336;
      if ( (unsigned __int64 *)v358.m128i_i64[0] != src )
      {
        v375 = v358;
        v358.m128i_i64[0] = (__int64)src;
        v358.m128i_i64[1] = 0;
        goto LABEL_336;
      }
      if ( v358.m128i_i32[2] <= 4u )
      {
        v239 = 16LL * v358.m128i_u32[2];
        v240 = src;
        p_dest = &dest;
        goto LABEL_387;
      }
      sub_16CD150((__int64)&v375, &dest, v358.m128i_u32[2], 16, v210, v211);
      p_dest = (unsigned __int64 *)v375.m128i_i64[0];
      v240 = (unsigned __int64 *)v358.m128i_i64[0];
      v239 = 16LL * v358.m128i_u32[2];
      if ( v239 )
LABEL_387:
        memcpy(p_dest, v240, v239);
      v375.m128i_i32[2] = v212;
      v358.m128i_i32[2] = 0;
LABEL_336:
      sub_1B1DC30(&i, &v375);
      if ( (unsigned __int64 *)v375.m128i_i64[0] != &dest )
        _libc_free(v375.m128i_u64[0]);
      v213 = (const __m128i *)sub_1458800(*v339);
      sub_197E390(&v375, v213, v214, v215, v216, v217);
      sub_1B1DDA0(&i, &v375);
      v375.m128i_i64[0] = (__int64)&unk_49EC708;
      if ( v388 )
      {
        v218 = v387;
        v219 = &v387[7 * v388];
        do
        {
          if ( *v218 != -16 && *v218 != -8 )
          {
            v220 = v218[1];
            if ( (_QWORD *)v220 != v218 + 3 )
              _libc_free(v220);
          }
          v218 += 7;
        }
        while ( v219 != v218 );
      }
      j___libc_free_0(v387);
      if ( (__m128 *)v378[0] != &v379 )
        _libc_free(v378[0]);
      v209 = (__int64)v357;
      sub_1B1F0F0(&i, v357);
      sub_1B216C0(&i);
      sub_197E0D0((__int64)&i);
    }
    sub_1A8C510(&v347);
    for ( i1 = v347; i1 != &v347; i1 = (_QWORD *)*i1 )
      sub_1A8BF10(
        (__int64)(i1 + 2),
        a7,
        *(double *)a8.m128i_i64,
        *(double *)a9.m128i_i64,
        a10,
        v225,
        v226,
        a13,
        a14,
        v209,
        v221,
        v222,
        v223,
        v224);
    if ( byte_4FB56E0 )
      sub_1403400(v338);
    v228 = v342;
    v229 = sub_15E0530(*v342);
    if ( sub_1602790(v229)
      || (v237 = sub_15E0530(*v228),
          v238 = sub_16033E0(v237),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v238 + 48LL))(v238)) )
    {
      v230 = **(_QWORD **)(v337[0] + 32);
      sub_13FD840(&v332, v337[0]);
      sub_15C9090((__int64)v333, &v332);
      sub_15CA330((__int64)&i, (__int64)"loop-distribute", (__int64)"Distribute", 10, v333, v230);
      sub_15CAB20((__int64)&i, "distributed loop", 0x10u);
      a8 = _mm_loadu_si128(&j);
      a9 = _mm_loadu_si128(&v397);
      v375.m128i_i32[2] = v393;
      v377 = (__m128)a8;
      v375.m128i_i8[12] = BYTE4(v393);
      v379 = (__m128)a9;
      dest = v394;
      v378[0] = v396[0];
      v375.m128i_i64[0] = (__int64)&unk_49ECF68;
      v378[1] = v396[1];
      v381 = v399;
      if ( v399 )
        v380 = v398;
      v382 = &v384;
      v383 = 0x400000000LL;
      if ( v401 )
      {
        sub_1A8B510((__int64)&v382, (__int64)&v400);
        v235 = v400;
        v389 = v411;
        v390 = v412;
        v391 = v413;
        v375.m128i_i64[0] = (__int64)&unk_49ECF98;
        i = (unsigned __int64 *)&unk_49ECF68;
        v231 = &v400[11 * v401];
        if ( v400 != v231 )
        {
          do
          {
            v231 -= 11;
            v236 = (__m128i **)v231[4];
            if ( v236 != v231 + 6 )
              j_j___libc_free_0(v236, &v231[6]->m128i_i8[1]);
            if ( *v231 != (__m128i *)(v231 + 2) )
              j_j___libc_free_0(*v231, &v231[2]->m128i_i8[1]);
          }
          while ( v235 != v231 );
          v231 = v400;
        }
      }
      else
      {
        v231 = v400;
        v389 = v411;
        v390 = v412;
        v391 = v413;
        v375.m128i_i64[0] = (__int64)&unk_49ECF98;
      }
      if ( v231 != &v402 )
        _libc_free((unsigned __int64)v231);
      if ( v332 )
        sub_161E7C0((__int64)&v332, v332);
      sub_143AA50(v228, (__int64)&v375);
      v232 = v382;
      v375.m128i_i64[0] = (__int64)&unk_49ECF68;
      v233 = &v382[11 * (unsigned int)v383];
      if ( v382 != v233 )
      {
        do
        {
          v233 -= 11;
          v234 = (unsigned __int64 *)v233[4];
          if ( v234 != v233 + 6 )
            j_j___libc_free_0(v234, v233[6] + 1);
          if ( (unsigned __int64 *)*v233 != v233 + 2 )
            j_j___libc_free_0(*v233, v233[2] + 1);
        }
        while ( v232 != v233 );
        v233 = v382;
      }
      if ( v233 != &v384 )
        _libc_free((unsigned __int64)v233);
    }
    if ( (unsigned __int64 *)v358.m128i_i64[0] != src )
      _libc_free(v358.m128i_u64[0]);
    if ( v334 != v336 )
      _libc_free((unsigned __int64)v334);
LABEL_191:
    if ( (unsigned __int64 *)v357[0] != &v357[2] )
      _libc_free(v357[0]);
    if ( v365 != &v367 )
      _libc_free((unsigned __int64)v365);
    j___libc_free_0(v351);
    v122 = v347;
    while ( v122 != &v347 )
    {
      v123 = v122;
      v122 = (_QWORD *)*v122;
      if ( *((_BYTE *)v123 + 288) )
      {
        v134 = *((_DWORD *)v123 + 70);
        if ( v134 )
        {
          v135 = (_QWORD *)v123[33];
          v136 = &v135[2 * v134];
          do
          {
            if ( *v135 != -4 && *v135 != -8 )
            {
              v137 = v135[1];
              if ( v137 )
                sub_161E7C0((__int64)(v135 + 1), v137);
            }
            v135 += 2;
          }
          while ( v136 != v135 );
        }
        j___libc_free_0(v123[33]);
      }
      v124 = *((unsigned int *)v123 + 62);
      if ( (_DWORD)v124 )
      {
        v125 = (_QWORD *)v123[29];
        v375.m128i_i64[1] = 2;
        dest = 0;
        v377 = (__m128)0xFFFFFFFFFFFFFFF8LL;
        v328 = &v125[8 * v124];
        v375.m128i_i64[0] = (__int64)&unk_49E6B50;
        i = (unsigned __int64 *)&unk_49E6B50;
        v126 = -8;
        v393 = 2;
        v394 = 0;
        j = (__m128i)0xFFFFFFFFFFFFFFF0LL;
        while ( 1 )
        {
          v127 = v125[3];
          if ( v126 != v127 && v127 != j.m128i_i64[0] )
          {
            v128 = v125[7];
            if ( v128 != 0 && v128 != -8 && v128 != -16 )
            {
              sub_1649B30(v125 + 5);
              v127 = v125[3];
            }
          }
          *v125 = &unk_49EE2B0;
          if ( v127 != -8 && v127 != 0 && v127 != -16 )
            sub_1649B30(v125 + 1);
          v125 += 8;
          if ( v328 == v125 )
            break;
          v126 = v377.m128_u64[0];
        }
        i = (unsigned __int64 *)&unk_49EE2B0;
        if ( j.m128i_i64[0] != -8 && j.m128i_i64[0] != 0 && j.m128i_i64[0] != -16 )
          sub_1649B30(&v393);
        v375.m128i_i64[0] = (__int64)&unk_49EE2B0;
        if ( v377.m128_u64[0] != 0 && v377.m128_u64[0] != -8 && v377.m128_u64[0] != -16 )
          sub_1649B30(&v375.m128i_i64[1]);
      }
      j___libc_free_0(v123[29]);
      v129 = v123[18];
      if ( (_QWORD *)v129 != v123 + 20 )
        _libc_free(v129);
      v130 = v123[4];
      if ( v130 != v123[3] )
        _libc_free(v130);
      j_j___libc_free_0(v123, 304);
    }
    v58 = v326;
LABEL_88:
    if ( v315 != ++v322 )
      continue;
    break;
  }
  v315 = v344;
LABEL_108:
  if ( v315 != (__int64 *)v346 )
    _libc_free((unsigned __int64)v315);
  return v58;
}
