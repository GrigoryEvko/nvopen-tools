// Function: sub_186CA00
// Address: 0x186ca00
//
__int64 __fastcall sub_186CA00(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  signed int v26; // ecx
  int v27; // ecx
  unsigned int v28; // ecx
  __int64 v29; // rcx
  __int64 **v30; // r13
  __int64 **v31; // r12
  __int64 v32; // rsi
  __int64 *v33; // rax
  __int64 *v34; // rdi
  __int64 *v35; // rcx
  __int64 **v36; // rax
  __int64 **v37; // rcx
  __int64 *v38; // rbx
  int v39; // r8d
  int v40; // r9d
  __int64 *v41; // rbx
  __int64 *v42; // r14
  unsigned __int8 v43; // al
  __int64 *v44; // r15
  unsigned __int64 v45; // r13
  unsigned __int64 v46; // r10
  __int64 v47; // rax
  __int64 *v48; // r12
  __int64 v49; // r12
  __int64 *v50; // rax
  __int64 *v51; // r15
  __int64 v52; // rax
  unsigned __int64 *v53; // rax
  __int64 *v54; // r12
  __int64 *v55; // rdx
  char v56; // al
  unsigned __int64 v57; // r10
  __int64 v58; // rax
  __int64 v59; // rax
  unsigned __int64 v60; // r10
  __int64 v61; // r11
  __int64 v62; // r12
  __int64 v63; // r13
  _QWORD *v64; // r13
  _QWORD *v65; // r12
  _QWORD *v66; // rdi
  _QWORD *v67; // r13
  _QWORD *v68; // r15
  _QWORD *v69; // rdi
  __int64 v70; // r14
  unsigned int v71; // r13d
  __int64 *v72; // rax
  __int64 *v73; // rdx
  _BYTE *v74; // r12
  _BYTE *v75; // rax
  __int64 v76; // rdx
  int v77; // ecx
  __int64 v78; // r12
  unsigned __int64 v79; // rax
  unsigned __int64 v80; // rcx
  __int64 *v81; // rax
  __int64 v82; // rbx
  unsigned int v83; // r15d
  __int64 v84; // rbx
  __int64 v85; // r13
  _QWORD *v86; // r14
  __int64 *v87; // rbx
  unsigned __int64 v88; // r12
  __int64 v89; // rax
  __int64 v90; // rsi
  __int64 v91; // rbx
  __int64 v92; // r14
  __int64 v93; // rax
  unsigned __int64 v94; // r8
  int v95; // r9d
  __int64 v96; // rdx
  int v97; // edi
  __int64 v98; // rcx
  int v99; // eax
  int v100; // edi
  __int64 v101; // rsi
  _BYTE *v102; // rax
  unsigned __int64 v103; // rdx
  __int64 *v104; // r12
  _QWORD *v105; // rbx
  _QWORD *v106; // r12
  __int64 v107; // rax
  _QWORD *v108; // rbx
  _QWORD *v109; // r12
  __int64 v110; // rdi
  __int64 *v112; // rcx
  __int64 *v113; // r12
  _QWORD *v114; // rdi
  _QWORD *v115; // rax
  _QWORD *v116; // rsi
  __int64 v117; // rcx
  __int64 v118; // rdx
  _QWORD *v119; // rdx
  _QWORD *v120; // r8
  _QWORD *v121; // rdi
  _QWORD *v122; // rax
  __int64 v123; // rsi
  __int64 v124; // rcx
  _QWORD *v125; // rax
  __int64 v126; // rsi
  __int64 v127; // rcx
  unsigned __int64 *v128; // r12
  unsigned __int64 v129; // rax
  unsigned __int64 v130; // rax
  _QWORD *v131; // rdi
  __int64 v132; // rax
  _QWORD *v133; // rax
  __int64 v134; // r12
  _BYTE *v135; // rax
  _BYTE *v136; // rcx
  __int64 v137; // rsi
  __int64 v138; // rdx
  __int64 v139; // rax
  const char *v140; // rdx
  size_t v141; // rbx
  __int64 v142; // rax
  _QWORD *v143; // rbx
  _QWORD *v144; // r13
  _QWORD *v145; // rdi
  __int64 v146; // r13
  __int64 v147; // rbx
  __int64 v148; // rdi
  __int64 v149; // r14
  __int64 v150; // rax
  _QWORD *v151; // rcx
  __int64 v152; // rdx
  _BYTE *v153; // rsi
  __int64 *v154; // r12
  __int64 *v155; // r14
  __int64 v156; // r15
  __int64 *v157; // rax
  char v158; // dl
  __int64 v159; // r12
  __int64 v160; // rax
  int v161; // edx
  __int64 *v162; // rsi
  __int64 *v163; // rdi
  __int64 v164; // r14
  double v165; // xmm4_8
  double v166; // xmm5_8
  __int64 *v167; // rax
  __int64 v168; // rax
  unsigned int v169; // r14d
  unsigned int v170; // edx
  __int64 v171; // rax
  __int64 v172; // r12
  __int64 v173; // rbx
  _QWORD *v174; // rbx
  _QWORD *v175; // r14
  _QWORD *v176; // rdi
  _QWORD *v177; // rbx
  _QWORD *v178; // r12
  _QWORD *v179; // rdi
  __int64 v180; // rax
  unsigned int v181; // r13d
  unsigned __int64 *v182; // rax
  _BYTE *v183; // rbx
  _BYTE *v184; // r14
  __int64 v185; // rax
  __int64 v186; // r8
  unsigned __int64 v187; // rdx
  __int64 v188; // r13
  __int64 *v189; // rax
  __int64 v190; // rdx
  unsigned __int8 v191; // cl
  __int64 v192; // rax
  __int64 v193; // rax
  __int64 v194; // rax
  __int64 v195; // rax
  char v196; // al
  __int64 v197; // r8
  __int64 *v198; // rax
  __int64 *v199; // rsi
  __int64 *v200; // rcx
  __int64 v201; // rax
  __int64 v202; // rax
  _BYTE *v203; // rdi
  __int64 i; // r14
  _QWORD *v205; // rax
  _QWORD *v206; // rdi
  __int64 v207; // rax
  __int64 v208; // rax
  __int64 v209; // rsi
  __int64 v210; // rax
  unsigned int v211; // eax
  int v212; // r11d
  _QWORD *v213; // rdi
  int v214; // eax
  unsigned int v215; // edx
  __int64 v216; // r8
  int v217; // edi
  _QWORD *v218; // rsi
  _QWORD *v219; // rdx
  unsigned int v220; // r15d
  int v221; // esi
  __int64 v222; // rdi
  unsigned __int64 v223; // [rsp+0h] [rbp-E60h]
  __int64 v224; // [rsp+8h] [rbp-E58h]
  _QWORD *v225; // [rsp+10h] [rbp-E50h]
  __int64 v226; // [rsp+30h] [rbp-E30h]
  __int64 v227; // [rsp+38h] [rbp-E28h]
  _QWORD *v228; // [rsp+58h] [rbp-E08h]
  unsigned int v229; // [rsp+68h] [rbp-DF8h]
  unsigned int v230; // [rsp+6Ch] [rbp-DF4h]
  __int64 **v232; // [rsp+78h] [rbp-DE8h]
  int v233; // [rsp+78h] [rbp-DE8h]
  __int16 v234; // [rsp+78h] [rbp-DE8h]
  unsigned __int8 v235; // [rsp+78h] [rbp-DE8h]
  __int64 *v236; // [rsp+80h] [rbp-DE0h]
  __int64 v237; // [rsp+80h] [rbp-DE0h]
  __int64 v238; // [rsp+80h] [rbp-DE0h]
  char v239; // [rsp+92h] [rbp-DCEh]
  char v240; // [rsp+93h] [rbp-DCDh]
  signed int v241; // [rsp+94h] [rbp-DCCh]
  signed int v242; // [rsp+98h] [rbp-DC8h]
  int v243; // [rsp+9Ch] [rbp-DC4h]
  __int64 v244; // [rsp+A0h] [rbp-DC0h]
  unsigned int v245; // [rsp+A8h] [rbp-DB8h]
  __int64 **v246; // [rsp+B0h] [rbp-DB0h]
  char v247; // [rsp+B0h] [rbp-DB0h]
  __int16 v248; // [rsp+B0h] [rbp-DB0h]
  unsigned int v249; // [rsp+B0h] [rbp-DB0h]
  __int64 *v250; // [rsp+B8h] [rbp-DA8h]
  unsigned __int64 v251; // [rsp+C0h] [rbp-DA0h]
  __int64 v252; // [rsp+C0h] [rbp-DA0h]
  __int64 v253; // [rsp+C0h] [rbp-DA0h]
  __int64 *v254; // [rsp+C0h] [rbp-DA0h]
  __int64 v255; // [rsp+C0h] [rbp-DA0h]
  __int64 v256; // [rsp+C8h] [rbp-D98h]
  unsigned __int64 v257; // [rsp+C8h] [rbp-D98h]
  unsigned __int64 v258; // [rsp+C8h] [rbp-D98h]
  __int64 *v259; // [rsp+C8h] [rbp-D98h]
  __int64 v260; // [rsp+C8h] [rbp-D98h]
  int v261; // [rsp+DCh] [rbp-D84h] BYREF
  __int64 v262; // [rsp+E0h] [rbp-D80h] BYREF
  __int64 v263; // [rsp+E8h] [rbp-D78h] BYREF
  _DWORD v264[2]; // [rsp+F4h] [rbp-D6Ch] BYREF
  char v265; // [rsp+FCh] [rbp-D64h]
  __int64 *v266[2]; // [rsp+100h] [rbp-D60h] BYREF
  __int64 *v267; // [rsp+110h] [rbp-D50h]
  __m128i v268[2]; // [rsp+120h] [rbp-D40h] BYREF
  _QWORD v269[2]; // [rsp+140h] [rbp-D20h] BYREF
  __int64 (__fastcall *v270)(_QWORD *, _QWORD *, int); // [rsp+150h] [rbp-D10h]
  __int64 (__fastcall *v271)(__int64, __int64); // [rsp+158h] [rbp-D08h]
  __int64 v272; // [rsp+160h] [rbp-D00h] BYREF
  _QWORD *v273; // [rsp+168h] [rbp-CF8h]
  __int64 v274; // [rsp+170h] [rbp-CF0h]
  unsigned int v275; // [rsp+178h] [rbp-CE8h]
  __int64 *v276[2]; // [rsp+180h] [rbp-CE0h] BYREF
  __int64 *v277; // [rsp+190h] [rbp-CD0h] BYREF
  __int64 *v278; // [rsp+1A0h] [rbp-CC0h]
  __int64 v279; // [rsp+1B0h] [rbp-CB0h] BYREF
  _QWORD v280[2]; // [rsp+1E0h] [rbp-C80h] BYREF
  _QWORD v281[2]; // [rsp+1F0h] [rbp-C70h] BYREF
  _QWORD *v282; // [rsp+200h] [rbp-C60h]
  _QWORD v283[6]; // [rsp+210h] [rbp-C50h] BYREF
  __int64 v284; // [rsp+240h] [rbp-C20h] BYREF
  __int64 *v285; // [rsp+248h] [rbp-C18h]
  __int64 *v286; // [rsp+250h] [rbp-C10h]
  __int64 v287; // [rsp+258h] [rbp-C08h]
  int v288; // [rsp+260h] [rbp-C00h]
  _BYTE v289[72]; // [rsp+268h] [rbp-BF8h] BYREF
  _BYTE *v290; // [rsp+2B0h] [rbp-BB0h] BYREF
  __int64 v291; // [rsp+2B8h] [rbp-BA8h]
  _BYTE v292[128]; // [rsp+2C0h] [rbp-BA0h] BYREF
  _BYTE *v293; // [rsp+340h] [rbp-B20h] BYREF
  __int64 v294; // [rsp+348h] [rbp-B18h]
  _BYTE v295[256]; // [rsp+350h] [rbp-B10h] BYREF
  _QWORD v296[2]; // [rsp+450h] [rbp-A10h] BYREF
  _QWORD v297[2]; // [rsp+460h] [rbp-A00h] BYREF
  __int64 *v298; // [rsp+470h] [rbp-9F0h]
  _BYTE *v299; // [rsp+478h] [rbp-9E8h]
  __int64 v300; // [rsp+480h] [rbp-9E0h] BYREF
  _BYTE v301[32]; // [rsp+488h] [rbp-9D8h] BYREF
  _BYTE *v302; // [rsp+4A8h] [rbp-9B8h]
  __int64 v303; // [rsp+4B0h] [rbp-9B0h]
  _BYTE v304[192]; // [rsp+4B8h] [rbp-9A8h] BYREF
  _BYTE *v305; // [rsp+578h] [rbp-8E8h]
  __int64 v306; // [rsp+580h] [rbp-8E0h]
  _BYTE v307[72]; // [rsp+588h] [rbp-8D8h] BYREF
  __m128i v308; // [rsp+5D0h] [rbp-890h] BYREF
  __int64 v309; // [rsp+5E0h] [rbp-880h] BYREF
  __m128 v310; // [rsp+5E8h] [rbp-878h]
  __int64 v311; // [rsp+5F8h] [rbp-868h]
  __int64 v312; // [rsp+600h] [rbp-860h] BYREF
  __m128 v313; // [rsp+608h] [rbp-858h]
  __int64 v314; // [rsp+618h] [rbp-848h]
  char v315; // [rsp+620h] [rbp-840h]
  _BYTE *v316; // [rsp+628h] [rbp-838h] BYREF
  __int64 v317; // [rsp+630h] [rbp-830h]
  _BYTE v318[352]; // [rsp+638h] [rbp-828h] BYREF
  char v319; // [rsp+798h] [rbp-6C8h]
  int v320; // [rsp+79Ch] [rbp-6C4h]
  __int64 v321; // [rsp+7A0h] [rbp-6C0h]
  char *v322; // [rsp+7B0h] [rbp-6B0h] BYREF
  __int64 *v323; // [rsp+7B8h] [rbp-6A8h]
  __int64 *v324; // [rsp+7C0h] [rbp-6A0h]
  __int64 v325; // [rsp+7C8h] [rbp-698h]
  int v326; // [rsp+7D0h] [rbp-690h]
  _BYTE v327[48]; // [rsp+7D8h] [rbp-688h] BYREF
  _QWORD *v328; // [rsp+808h] [rbp-658h]
  unsigned int v329; // [rsp+810h] [rbp-650h]
  _BYTE v330[376]; // [rsp+818h] [rbp-648h] BYREF
  __int64 v331[10]; // [rsp+990h] [rbp-4D0h] BYREF
  char v332; // [rsp+9E0h] [rbp-480h]
  __int64 v333; // [rsp+9E8h] [rbp-478h]
  __int64 v334; // [rsp+CB0h] [rbp-1B0h]
  unsigned __int64 v335; // [rsp+CB8h] [rbp-1A8h]
  __int64 v336; // [rsp+D18h] [rbp-148h]
  unsigned __int64 v337; // [rsp+D20h] [rbp-140h]
  char v338; // [rsp+DB8h] [rbp-A8h]
  __int64 v339[12]; // [rsp+DC0h] [rbp-A0h] BYREF
  char v340; // [rsp+E20h] [rbp-40h]

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_479:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F98A8D )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_479;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F98A8D);
  v14 = *(__int64 **)(a1 + 8);
  v228 = *(_QWORD **)(v13 + 160);
  v15 = *v14;
  v16 = v14[1];
  if ( v15 == v16 )
LABEL_480:
    BUG();
  while ( *(_UNKNOWN **)v15 != &unk_4F9D764 )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_480;
  }
  v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(*(_QWORD *)(v15 + 8), &unk_4F9D764);
  v18 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 160) = v17;
  v19 = *v18;
  v20 = v18[1];
  if ( v19 == v20 )
LABEL_481:
    BUG();
  while ( *(_UNKNOWN **)v19 != &unk_4F99CCD )
  {
    v19 += 16;
    if ( v20 == v19 )
      goto LABEL_481;
  }
  v21 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(*(_QWORD *)(v19 + 8), &unk_4F99CCD);
  v22 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 168) = *(_QWORD *)(v21 + 160);
  v23 = *v22;
  v24 = v22[1];
  if ( v23 == v24 )
LABEL_482:
    BUG();
  while ( *(_UNKNOWN **)v23 != &unk_4F9B6E8 )
  {
    v23 += 16;
    if ( v24 == v23 )
      goto LABEL_482;
  }
  v25 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(*(_QWORD *)(v23 + 8), &unk_4F9B6E8);
  v26 = *(_DWORD *)(a1 + 272);
  v340 = 0;
  v331[0] = a1;
  v242 = v26;
  v244 = v25 + 360;
  v241 = *(_DWORD *)(a1 + 268);
  v27 = *(_DWORD *)(a1 + 264);
  v261 = *(_DWORD *)(a1 + 260);
  v243 = v27;
  LOBYTE(v27) = *(_BYTE *)(a1 + 257);
  v262 = a1;
  v239 = v27;
  LOBYTE(v27) = *(_BYTE *)(a1 + 256);
  v269[0] = a1;
  v240 = v27;
  v227 = a1 + 176;
  v28 = *(unsigned __int8 *)(a1 + 153);
  v338 = 0;
  v230 = v28;
  v29 = *(_QWORD *)(a1 + 168);
  v284 = 0;
  v226 = v29;
  v271 = sub_1869E50;
  v270 = sub_1869EB0;
  v285 = (__int64 *)v289;
  v286 = (__int64 *)v289;
  v287 = 8;
  v288 = 0;
  v30 = *(__int64 ***)(a2 + 24);
  v31 = *(__int64 ***)(a2 + 16);
  if ( v31 == v30 )
  {
    v293 = v295;
LABEL_417:
    v235 = 0;
    goto LABEL_188;
  }
  do
  {
LABEL_21:
    v32 = **v31;
    if ( v32 )
    {
      v33 = v285;
      if ( v286 != v285 )
        goto LABEL_19;
      v34 = &v285[HIDWORD(v287)];
      if ( v285 == v34 )
      {
LABEL_386:
        if ( HIDWORD(v287) >= (unsigned int)v287 )
        {
LABEL_19:
          sub_16CCBA0((__int64)&v284, v32);
          goto LABEL_20;
        }
        ++HIDWORD(v287);
        *v34 = v32;
        ++v284;
      }
      else
      {
        v35 = 0;
        while ( v32 != *v33 )
        {
          if ( *v33 == -2 )
            v35 = v33;
          if ( v34 == ++v33 )
          {
            if ( !v35 )
              goto LABEL_386;
            ++v31;
            *v35 = v32;
            --v288;
            ++v284;
            if ( v30 != v31 )
              goto LABEL_21;
            goto LABEL_31;
          }
        }
      }
    }
LABEL_20:
    ++v31;
  }
  while ( v30 != v31 );
LABEL_31:
  v293 = v295;
  v36 = *(__int64 ***)(a2 + 16);
  v294 = 0x1000000000LL;
  v37 = *(__int64 ***)(a2 + 24);
  v290 = v292;
  v232 = v37;
  v291 = 0x800000000LL;
  if ( v37 == v36 )
    goto LABEL_417;
  v246 = v36;
  while ( 2 )
  {
    v38 = (__int64 *)**v246;
    if ( v38 && !sub_15E4F60(**v246) )
    {
      sub_143A950(v276, v38);
      v236 = v38 + 9;
      v250 = (__int64 *)v38[10];
      if ( v250 != v38 + 9 )
      {
        while ( 1 )
        {
          if ( !v250 )
            BUG();
          v41 = v250 + 2;
          v42 = (__int64 *)v250[3];
          if ( v42 != v250 + 2 )
            break;
LABEL_61:
          v250 = (__int64 *)v250[1];
          if ( v236 == v250 )
            goto LABEL_62;
        }
        while ( 1 )
        {
          if ( !v42 )
            BUG();
          v43 = *((_BYTE *)v42 - 8);
          v44 = v42 - 3;
          if ( v43 <= 0x17u )
            goto LABEL_41;
          if ( v43 == 78 )
          {
            v45 = (unsigned __int64)v44 | 4;
            v46 = (unsigned __int64)v44 & 0xFFFFFFFFFFFFFFF8LL;
            if ( ((unsigned __int64)v44 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
              goto LABEL_41;
            v47 = *(v42 - 6);
            if ( !*(_BYTE *)(v47 + 16) && (*(_BYTE *)(v47 + 33) & 0x20) != 0 )
              goto LABEL_41;
          }
          else
          {
            if ( v43 != 29 )
              goto LABEL_41;
            v45 = (unsigned __int64)v44 & 0xFFFFFFFFFFFFFFFBLL;
            v46 = (unsigned __int64)v44 & 0xFFFFFFFFFFFFFFF8LL;
            if ( ((unsigned __int64)v44 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
              goto LABEL_41;
          }
          v48 = (__int64 *)(v46 - 72);
          if ( (v45 & 4) != 0 )
            v48 = (__int64 *)(v46 - 24);
          if ( !*(_BYTE *)(*v48 + 16) )
          {
            v251 = v46;
            v256 = *v48;
            if ( sub_15E4F60(*v48) )
            {
              v56 = sub_186A5D0(v256, *(_QWORD *)(*(_QWORD *)(v251 + 40) + 56LL));
              v57 = v251;
              if ( !v56 )
              {
                v252 = v256;
                v257 = v57;
                v58 = sub_15E0530((__int64)v276[0]);
                v59 = sub_1602790(v58);
                v60 = v257;
                v61 = v252;
                if ( v59
                  || (v194 = sub_15E0530((__int64)v276[0]),
                      v195 = sub_16033E0(v194),
                      v196 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v195 + 48LL))(v195),
                      v60 = v257,
                      v61 = v252,
                      v196) )
                {
                  v258 = v60;
                  v253 = v61;
                  sub_15CA5C0((__int64)&v322, (__int64)"inline", (__int64)"NoDefinition", 12, (__int64)(v42 - 3));
                  sub_15C9340((__int64)v296, "Callee", 6u, v253);
                  v62 = sub_186AF50((__int64)&v322, (__int64)v296);
                  sub_15CAB20(v62, " will not be inlined into ", 0x1Au);
                  sub_15C9340((__int64)v280, "Caller", 6u, *(_QWORD *)(*(_QWORD *)(v258 + 40) + 56LL));
                  v63 = sub_17C21B0(v62, (__int64)v280);
                  sub_15CAB20(v63, " because its definition is unavailable", 0x26u);
                  sub_15CA8C0(v63);
                  v308.m128i_i32[2] = *(_DWORD *)(v63 + 8);
                  v308.m128i_i8[12] = *(_BYTE *)(v63 + 12);
                  v309 = *(_QWORD *)(v63 + 16);
                  a3 = (__m128)_mm_loadu_si128((const __m128i *)(v63 + 24));
                  v310 = a3;
                  v311 = *(_QWORD *)(v63 + 40);
                  v308.m128i_i64[0] = (__int64)&unk_49ECF68;
                  v312 = *(_QWORD *)(v63 + 48);
                  a4 = _mm_loadu_si128((const __m128i *)(v63 + 56));
                  v313 = (__m128)a4;
                  v315 = *(_BYTE *)(v63 + 80);
                  if ( v315 )
                    v314 = *(_QWORD *)(v63 + 72);
                  v316 = v318;
                  v317 = 0x400000000LL;
                  if ( *(_DWORD *)(v63 + 96) )
                    sub_186B280((__int64)&v316, v63 + 88);
                  v319 = *(_BYTE *)(v63 + 456);
                  v320 = *(_DWORD *)(v63 + 460);
                  v321 = *(_QWORD *)(v63 + 464);
                  v308.m128i_i64[0] = (__int64)&unk_49ECFC8;
                  if ( v282 != v283 )
                    j_j___libc_free_0(v282, v283[0] + 1LL);
                  if ( (_QWORD *)v280[0] != v281 )
                    j_j___libc_free_0(v280[0], v281[0] + 1LL);
                  if ( v298 != &v300 )
                    j_j___libc_free_0(v298, v300 + 1);
                  if ( (_QWORD *)v296[0] != v297 )
                    j_j___libc_free_0(v296[0], v297[0] + 1LL);
                  v64 = v328;
                  v322 = (char *)&unk_49ECF68;
                  v65 = &v328[11 * v329];
                  if ( v328 != v65 )
                  {
                    do
                    {
                      v65 -= 11;
                      v66 = (_QWORD *)v65[4];
                      if ( v66 != v65 + 6 )
                        j_j___libc_free_0(v66, v65[6] + 1LL);
                      if ( (_QWORD *)*v65 != v65 + 2 )
                        j_j___libc_free_0(*v65, v65[2] + 1LL);
                    }
                    while ( v64 != v65 );
                    v65 = v328;
                  }
                  if ( v65 != (_QWORD *)v330 )
                    _libc_free((unsigned __int64)v65);
                  sub_143AA50(v276, (__int64)&v308);
                  v67 = v316;
                  v308.m128i_i64[0] = (__int64)&unk_49ECF68;
                  v68 = &v316[88 * (unsigned int)v317];
                  if ( v316 != (_BYTE *)v68 )
                  {
                    do
                    {
                      v68 -= 11;
                      v69 = (_QWORD *)v68[4];
                      if ( v69 != v68 + 6 )
                        j_j___libc_free_0(v69, v68[6] + 1LL);
                      if ( (_QWORD *)*v68 != v68 + 2 )
                        j_j___libc_free_0(*v68, v68[2] + 1LL);
                    }
                    while ( v67 != v68 );
                    v68 = v316;
                  }
                  if ( v68 != (_QWORD *)v318 )
                    _libc_free((unsigned __int64)v68);
                }
              }
LABEL_41:
              v42 = (__int64 *)v42[1];
              if ( v41 == v42 )
                goto LABEL_61;
              continue;
            }
            v49 = *v48;
            if ( !*(_BYTE *)(v49 + 16) )
              break;
          }
LABEL_58:
          v52 = (unsigned int)v294;
          if ( (unsigned int)v294 >= HIDWORD(v294) )
          {
            sub_16CD150((__int64)&v293, v295, 0, 16, v39, v40);
            v52 = (unsigned int)v294;
          }
          v53 = (unsigned __int64 *)&v293[16 * v52];
          *v53 = v45;
          v53[1] = 0xFFFFFFFFLL;
          LODWORD(v294) = v294 + 1;
          v42 = (__int64 *)v42[1];
          if ( v41 == v42 )
            goto LABEL_61;
        }
        v50 = v285;
        if ( v286 == v285 )
        {
          v55 = &v285[HIDWORD(v287)];
          if ( v285 == v55 )
          {
            v51 = v285;
          }
          else
          {
            do
            {
              if ( *v50 == v49 )
                break;
              ++v50;
            }
            while ( v55 != v50 );
            v51 = &v285[HIDWORD(v287)];
          }
        }
        else
        {
          v51 = &v286[(unsigned int)v287];
          v50 = sub_16CC9F0((__int64)&v284, v49);
          if ( *v50 == v49 )
          {
            if ( v286 == v285 )
              v55 = &v286[HIDWORD(v287)];
            else
              v55 = &v286[(unsigned int)v287];
          }
          else
          {
            if ( v286 != v285 )
            {
              v50 = &v286[(unsigned int)v287];
              goto LABEL_57;
            }
            v55 = &v286[HIDWORD(v287)];
            v50 = v55;
          }
        }
        for ( ; v55 != v50; ++v50 )
        {
          if ( (unsigned __int64)*v50 < 0xFFFFFFFFFFFFFFFELL )
            break;
        }
LABEL_57:
        if ( v51 != v50 )
          goto LABEL_41;
        goto LABEL_58;
      }
LABEL_62:
      v54 = v277;
      if ( v277 )
      {
        sub_1368A00(v277);
        j_j___libc_free_0(v54, 8);
      }
    }
    if ( v232 != ++v246 )
      continue;
    break;
  }
  LODWORD(v70) = v294;
  v71 = 0;
  if ( (_DWORD)v294 )
  {
    while ( 1 )
    {
      v78 = 16LL * v71;
      v79 = *(_QWORD *)&v293[v78] & 0xFFFFFFFFFFFFFFF8LL;
      v80 = v79 - 24;
      v81 = (__int64 *)(v79 - 72);
      if ( (*(_QWORD *)&v293[v78] & 4) != 0 )
        v81 = (__int64 *)v80;
      v82 = *v81;
      if ( *(_BYTE *)(*v81 + 16) )
        goto LABEL_121;
      v72 = v285;
      if ( v286 == v285 )
        break;
      v259 = &v286[(unsigned int)v287];
      v72 = sub_16CC9F0((__int64)&v284, v82);
      v73 = v259;
      if ( *v72 == v82 )
      {
        if ( v286 == v285 )
          v112 = &v286[HIDWORD(v287)];
        else
          v112 = &v286[(unsigned int)v287];
        goto LABEL_203;
      }
      if ( v286 == v285 )
      {
        v112 = &v286[HIDWORD(v287)];
        v72 = v112;
        goto LABEL_203;
      }
      v72 = &v286[(unsigned int)v287];
LABEL_116:
      if ( v73 == v72 )
      {
LABEL_121:
        if ( ++v71 >= (unsigned int)v70 )
        {
LABEL_122:
          v272 = 0;
          v273 = 0;
          v83 = v294;
          v296[0] = v228;
          v296[1] = v269;
          v274 = 0;
          v297[0] = v226;
          v299 = v301;
          v300 = 0x400000000LL;
          v302 = v304;
          v275 = 0;
          v297[1] = 0;
          v298 = 0;
          v303 = 0x800000000LL;
          v305 = v307;
          v306 = 0x800000000LL;
          v235 = 0;
          while ( 1 )
          {
            if ( !v83 )
            {
LABEL_164:
              if ( v235 )
                sub_1CCE1F0(v228, a2);
              if ( v305 != v307 )
                _libc_free((unsigned __int64)v305);
              v105 = v302;
              v106 = &v302[24 * (unsigned int)v303];
              if ( v302 != (_BYTE *)v106 )
              {
                do
                {
                  v107 = *(v106 - 1);
                  v106 -= 3;
                  if ( v107 != 0 && v107 != -8 && v107 != -16 )
                    sub_1649B30(v106);
                }
                while ( v105 != v106 );
                v106 = v302;
              }
              if ( v106 != (_QWORD *)v304 )
                _libc_free((unsigned __int64)v106);
              if ( v299 != v301 )
                _libc_free((unsigned __int64)v299);
              if ( v275 )
              {
                v108 = v273;
                v109 = &v273[4 * v275];
                do
                {
                  if ( *v108 != -16 && *v108 != -8 )
                  {
                    v110 = v108[1];
                    if ( v110 )
                      j_j___libc_free_0(v110, v108[3] - v110);
                  }
                  v108 += 4;
                }
                while ( v109 != v108 );
              }
              j___libc_free_0(v273);
              v203 = v290;
              goto LABEL_186;
            }
            v247 = 0;
            v84 = 0;
            v83 = 0;
            do
            {
              v260 = 16 * v84;
              v85 = *(_QWORD *)&v293[16 * v84];
              v86 = (_QWORD *)(v85 & 0xFFFFFFFFFFFFFFF8LL);
              v87 = (__int64 *)((v85 & 0xFFFFFFFFFFFFFFF8LL) - 72);
              if ( (v85 & 4) != 0 )
                v87 = v86 - 3;
              v88 = *v87;
              if ( *(_BYTE *)(*v87 + 16) )
                goto LABEL_215;
              v254 = *(__int64 **)(v86[5] + 56LL);
              if ( sub_15E4F60(*v87) )
                goto LABEL_215;
              if ( (unsigned __int8)sub_1AE9990(v86, v244) )
              {
                sub_143A950(v266, v254);
                sub_186B510(
                  (__int64)v264,
                  v85,
                  (__int64 (__fastcall *)(__int64))sub_1869CF0,
                  (__int64)&v262,
                  (__int64 *)v266,
                  v241,
                  v242,
                  v243,
                  &v261);
                if ( !v265 )
                  goto LABEL_213;
                v114 = v228 + 2;
                v115 = (_QWORD *)v228[3];
                if ( v115 )
                {
                  v116 = v228 + 2;
                  do
                  {
                    while ( 1 )
                    {
                      v117 = v115[2];
                      v118 = v115[3];
                      if ( v115[4] >= (unsigned __int64)v254 )
                        break;
                      v115 = (_QWORD *)v115[3];
                      if ( !v118 )
                        goto LABEL_222;
                    }
                    v116 = v115;
                    v115 = (_QWORD *)v115[2];
                  }
                  while ( v117 );
LABEL_222:
                  if ( v116 != v114 && v116[4] <= (unsigned __int64)v254 )
                    v114 = v116;
                }
                sub_13983A0(v114[5], v85);
                sub_15F20C0(v86);
                if ( !*(_QWORD *)(v88 + 8) )
                  goto LABEL_226;
                goto LABEL_157;
              }
              v89 = *(int *)&v293[v260 + 8];
              v245 = v89;
              if ( (_DWORD)v89 == -1 )
              {
LABEL_131:
                sub_143A950(v266, v254);
                sub_186B510(
                  (__int64)v264,
                  v85,
                  (__int64 (__fastcall *)(__int64))sub_1869CF0,
                  (__int64)&v262,
                  (__int64 *)v266,
                  v241,
                  v242,
                  v243,
                  &v261);
                if ( !v265 )
                  goto LABEL_213;
                v90 = v86[6];
                v263 = v90;
                if ( v90 )
                  sub_1623A60((__int64)&v263, v90, 2);
                v91 = *v87;
                v92 = v86[5];
                if ( *(_BYTE *)(v91 + 16) )
                  v91 = 0;
                v237 = *(_QWORD *)(v92 + 56);
                if ( v240 && byte_4FABE00 && ((v160 = sub_15E44B0(v91), !v161) || !v160)
                  || !v239 && (unsigned __int8)sub_1560180(v91 + 112, 36)
                  || (v93 = sub_1833CC0((__int64)v331, v91), !(unsigned __int8)sub_1ADC640(v85, v296, v93, v230, 0)) )
                {
                  if ( !(unsigned __int8)sub_186A5D0(v88, (__int64)v254) )
                  {
                    v171 = sub_15E0530((__int64)v266[0]);
                    if ( sub_1602790(v171)
                      || (v201 = sub_15E0530((__int64)v266[0]),
                          v202 = sub_16033E0(v201),
                          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v202 + 48LL))(v202)) )
                    {
                      sub_15C9090((__int64)v268, &v263);
                      sub_15CA540((__int64)&v322, (__int64)"inline", (__int64)"NotInlined", 10, v268, v92);
                      sub_15C9340((__int64)v280, "Callee", 6u, v88);
                      v172 = sub_186AF50((__int64)&v322, (__int64)v280);
                      sub_15CAB20(v172, " will not be inlined into ", 0x1Au);
                      sub_15C9340((__int64)v276, "Caller", 6u, (__int64)v254);
                      v173 = sub_17C21B0(v172, (__int64)v276);
                      v308.m128i_i32[2] = *(_DWORD *)(v173 + 8);
                      v308.m128i_i8[12] = *(_BYTE *)(v173 + 12);
                      v309 = *(_QWORD *)(v173 + 16);
                      a5 = _mm_loadu_si128((const __m128i *)(v173 + 24));
                      v310 = (__m128)a5;
                      v311 = *(_QWORD *)(v173 + 40);
                      v308.m128i_i64[0] = (__int64)&unk_49ECF68;
                      v312 = *(_QWORD *)(v173 + 48);
                      a6 = _mm_loadu_si128((const __m128i *)(v173 + 56));
                      v313 = (__m128)a6;
                      v315 = *(_BYTE *)(v173 + 80);
                      if ( v315 )
                        v314 = *(_QWORD *)(v173 + 72);
                      v316 = v318;
                      v317 = 0x400000000LL;
                      if ( *(_DWORD *)(v173 + 96) )
                        sub_186B280((__int64)&v316, v173 + 88);
                      v319 = *(_BYTE *)(v173 + 456);
                      v320 = *(_DWORD *)(v173 + 460);
                      v321 = *(_QWORD *)(v173 + 464);
                      v308.m128i_i64[0] = (__int64)&unk_49ECFC8;
                      if ( v278 != &v279 )
                        j_j___libc_free_0(v278, v279 + 1);
                      if ( (__int64 **)v276[0] != &v277 )
                        j_j___libc_free_0(v276[0], (char *)v277 + 1);
                      if ( v282 != v283 )
                        j_j___libc_free_0(v282, v283[0] + 1LL);
                      if ( (_QWORD *)v280[0] != v281 )
                        j_j___libc_free_0(v280[0], v281[0] + 1LL);
                      v174 = v328;
                      v322 = (char *)&unk_49ECF68;
                      v175 = &v328[11 * v329];
                      if ( v328 != v175 )
                      {
                        do
                        {
                          v175 -= 11;
                          v176 = (_QWORD *)v175[4];
                          if ( v176 != v175 + 6 )
                            j_j___libc_free_0(v176, v175[6] + 1LL);
                          if ( (_QWORD *)*v175 != v175 + 2 )
                            j_j___libc_free_0(*v175, v175[2] + 1LL);
                        }
                        while ( v174 != v175 );
                        v175 = v328;
                      }
                      if ( v175 != (_QWORD *)v330 )
                        _libc_free((unsigned __int64)v175);
                      sub_143AA50(v266, (__int64)&v308);
                      v177 = v316;
                      v308.m128i_i64[0] = (__int64)&unk_49ECF68;
                      v178 = &v316[88 * (unsigned int)v317];
                      if ( v316 != (_BYTE *)v178 )
                      {
                        do
                        {
                          v178 -= 11;
                          v179 = (_QWORD *)v178[4];
                          if ( v179 != v178 + 6 )
                            j_j___libc_free_0(v179, v178[6] + 1LL);
                          if ( (_QWORD *)*v178 != v178 + 2 )
                            j_j___libc_free_0(*v178, v178[2] + 1LL);
                        }
                        while ( v177 != v178 );
                        v178 = v316;
                      }
                      if ( v178 != (_QWORD *)v318 )
                        _libc_free((unsigned __int64)v178);
                    }
                  }
                  if ( v263 )
                    sub_161E7C0((__int64)&v263, v263);
LABEL_213:
                  v113 = v267;
                  if ( v267 )
                  {
                    sub_1368A00(v267);
                    j_j___libc_free_0(v113, 8);
                  }
LABEL_215:
                  ++v83;
                  goto LABEL_162;
                }
                if ( dword_4FAB4A0 )
                  sub_1AD1C40(v227, v237, v91);
                sub_15610A0(v237, v91);
                if ( !byte_4FAB700 )
                {
                  v322 = 0;
                  v323 = (__int64 *)v327;
                  v324 = (__int64 *)v327;
                  v325 = 16;
                  v326 = 0;
                  if ( v245 == -1 )
                  {
                    v146 = 0;
                    v147 = 8LL * (unsigned int)v300;
                    if ( (_DWORD)v300 )
                    {
                      v224 = v92;
                      v229 = v83;
                      v223 = v88;
                      while ( 1 )
                      {
LABEL_297:
                        v148 = *(_QWORD *)&v299[v146];
                        v308.m128i_i64[0] = v148;
                        v149 = *(_QWORD *)(v148 + 56);
                        if ( *(_BYTE *)(v149 + 8) != 14 || (unsigned __int8)sub_15F8BF0(v148) )
                          goto LABEL_296;
                        if ( !v275 )
                        {
                          ++v272;
                          goto LABEL_439;
                        }
                        LODWORD(v150) = (v275 - 1) & (((unsigned int)v149 >> 9) ^ ((unsigned int)v149 >> 4));
                        v151 = &v273[4 * (unsigned int)v150];
                        v152 = *v151;
                        if ( v149 != *v151 )
                        {
                          v212 = 1;
                          v213 = 0;
                          while ( v152 != -8 )
                          {
                            if ( v152 == -16 && !v213 )
                              v213 = v151;
                            v150 = (v275 - 1) & ((_DWORD)v150 + v212);
                            v151 = &v273[4 * v150];
                            v152 = *v151;
                            if ( v149 == *v151 )
                              goto LABEL_301;
                            ++v212;
                          }
                          if ( v213 )
                            v151 = v213;
                          ++v272;
                          v214 = v274 + 1;
                          if ( 4 * ((int)v274 + 1) < 3 * v275 )
                          {
                            if ( v275 - HIDWORD(v274) - v214 <= v275 >> 3 )
                            {
                              sub_186C7E0((__int64)&v272, v275);
                              if ( !v275 )
                              {
LABEL_478:
                                LODWORD(v274) = v274 + 1;
                                BUG();
                              }
                              v219 = 0;
                              v220 = (v275 - 1) & (((unsigned int)v149 >> 9) ^ ((unsigned int)v149 >> 4));
                              v221 = 1;
                              v214 = v274 + 1;
                              v151 = &v273[4 * v220];
                              v222 = *v151;
                              if ( v149 != *v151 )
                              {
                                while ( v222 != -8 )
                                {
                                  if ( v222 == -16 && !v219 )
                                    v219 = v151;
                                  v220 = (v275 - 1) & (v221 + v220);
                                  v151 = &v273[4 * v220];
                                  v222 = *v151;
                                  if ( v149 == *v151 )
                                    goto LABEL_431;
                                  ++v221;
                                }
                                if ( v219 )
                                  v151 = v219;
                              }
                            }
                            goto LABEL_431;
                          }
LABEL_439:
                          sub_186C7E0((__int64)&v272, 2 * v275);
                          if ( !v275 )
                            goto LABEL_478;
                          v215 = (v275 - 1) & (((unsigned int)v149 >> 9) ^ ((unsigned int)v149 >> 4));
                          v214 = v274 + 1;
                          v151 = &v273[4 * v215];
                          v216 = *v151;
                          if ( v149 != *v151 )
                          {
                            v217 = 1;
                            v218 = 0;
                            while ( v216 != -8 )
                            {
                              if ( v216 == -16 && !v218 )
                                v218 = v151;
                              v215 = (v275 - 1) & (v217 + v215);
                              v151 = &v273[4 * v215];
                              v216 = *v151;
                              if ( v149 == *v151 )
                                goto LABEL_431;
                              ++v217;
                            }
                            if ( v218 )
                              v151 = v218;
                          }
LABEL_431:
                          LODWORD(v274) = v214;
                          if ( *v151 != -8 )
                            --HIDWORD(v274);
                          *v151 = v149;
                          v153 = 0;
                          v151[1] = 0;
                          v151[2] = 0;
                          v151[3] = 0;
LABEL_434:
                          sub_186B0F0((__int64)(v151 + 1), v153, &v308);
                          v197 = v308.m128i_i64[0];
                          v198 = v323;
                          if ( v324 != v323 )
                            goto LABEL_435;
                          goto LABEL_393;
                        }
LABEL_301:
                        v153 = (_BYTE *)v151[1];
                        if ( (_BYTE *)v151[2] != v153 )
                          break;
LABEL_389:
                        if ( (_BYTE *)v151[3] == v153 )
                          goto LABEL_434;
                        v197 = v308.m128i_i64[0];
                        if ( v153 )
                        {
                          *(_QWORD *)v153 = v308.m128i_i64[0];
                          v153 = (_BYTE *)v151[2];
                        }
                        v151[2] = v153 + 8;
                        v198 = v323;
                        if ( v324 != v323 )
                          goto LABEL_435;
LABEL_393:
                        v199 = &v198[HIDWORD(v325)];
                        if ( v198 != v199 )
                        {
                          v200 = 0;
                          while ( v197 != *v198 )
                          {
                            if ( *v198 == -2 )
                              v200 = v198;
                            if ( v199 == ++v198 )
                            {
                              if ( !v200 )
                                goto LABEL_446;
                              *v200 = v197;
                              --v326;
                              ++v322;
                              goto LABEL_296;
                            }
                          }
                          goto LABEL_296;
                        }
LABEL_446:
                        if ( HIDWORD(v325) < (unsigned int)v325 )
                        {
                          ++HIDWORD(v325);
                          *v199 = v197;
                          ++v322;
                          goto LABEL_296;
                        }
LABEL_435:
                        sub_16CCBA0((__int64)&v322, v197);
LABEL_296:
                        v146 += 8;
                        if ( v147 == v146 )
                          goto LABEL_327;
                      }
                      v225 = v151;
                      v154 = (__int64 *)v151[1];
                      v155 = (__int64 *)v151[2];
                      while ( 1 )
                      {
                        v156 = *v154;
                        if ( *(_QWORD *)(*v154 + 40) == *(_QWORD *)(v308.m128i_i64[0] + 40) )
                          break;
LABEL_303:
                        if ( v155 == ++v154 )
                        {
                          v151 = v225;
                          v153 = (_BYTE *)v225[2];
                          goto LABEL_389;
                        }
                      }
                      v248 = *(_WORD *)(v308.m128i_i64[0] + 18);
                      v234 = *(_WORD *)(v156 + 18);
                      v157 = v323;
                      if ( v324 == v323 )
                      {
                        v162 = &v323[HIDWORD(v325)];
                        if ( v323 != v162 )
                        {
                          v163 = 0;
                          do
                          {
                            while ( 1 )
                            {
                              if ( v156 == *v157 )
                                goto LABEL_303;
                              if ( *v157 != -2 )
                                break;
                              if ( v162 == v157 + 1 )
                              {
                                v159 = *v154;
                                v163 = v157;
LABEL_318:
                                *v163 = v159;
                                --v326;
                                ++v322;
LABEL_319:
                                v164 = sub_161E8E0(v308.m128i_i64[0]);
                                if ( v164 )
                                {
                                  v167 = (__int64 *)sub_16498A0(v308.m128i_i64[0]);
                                  v168 = sub_1629050(v167, v164);
                                  if ( v168 )
                                  {
                                    for ( i = *(_QWORD *)(v168 + 8); i; i = *(_QWORD *)(i + 8) )
                                    {
                                      v205 = sub_1648700(i);
                                      v206 = v205;
                                      if ( *((_BYTE *)v205 + 16) == 78 )
                                      {
                                        v207 = *(v205 - 3);
                                        if ( !*(_BYTE *)(v207 + 16)
                                          && (*(_BYTE *)(v207 + 33) & 0x20) != 0
                                          && *(_DWORD *)(v207 + 36) == 36 )
                                        {
                                          v208 = *(_QWORD *)(v159 + 32);
                                          if ( v208 == *(_QWORD *)(v159 + 40) + 40LL || !v208 )
                                            v209 = 0;
                                          else
                                            v209 = v208 - 24;
                                          sub_15F22F0(v206, v209);
                                        }
                                      }
                                    }
                                  }
                                }
                                v249 = (unsigned int)(1 << v248) >> 1;
                                v169 = (unsigned int)(1 << v234) >> 1;
                                sub_164D160(
                                  v308.m128i_i64[0],
                                  v159,
                                  a3,
                                  *(double *)a4.m128i_i64,
                                  *(double *)a5.m128i_i64,
                                  *(double *)a6.m128i_i64,
                                  v165,
                                  v166,
                                  a9,
                                  a10);
                                v170 = v249;
                                if ( v249 != v169 )
                                {
                                  if ( v249 && v169 )
                                    goto LABEL_324;
                                  v210 = sub_1632FA0(*(_QWORD *)(v237 + 40));
                                  v211 = sub_15A9FE0(v210, *(_QWORD *)(v308.m128i_i64[0] + 56));
                                  v170 = v249;
                                  if ( v249 )
                                  {
                                    if ( !v169 )
                                      v169 = v211;
LABEL_324:
                                    if ( v170 > v169 )
                                      sub_15F8A20(v159, (unsigned int)(1 << *(_WORD *)(v308.m128i_i64[0] + 18)) >> 1);
                                  }
                                  else if ( v169 )
                                  {
                                    v170 = v211;
                                    goto LABEL_324;
                                  }
                                }
                                sub_15F20C0(v308.m128i_i64[0]);
                                *(_QWORD *)&v299[v146] = 0;
                                v146 += 8;
                                if ( v147 == v146 )
                                {
LABEL_327:
                                  v92 = v224;
                                  v83 = v229;
                                  v88 = v223;
                                  if ( v324 != v323 )
                                    _libc_free((unsigned __int64)v324);
                                  goto LABEL_143;
                                }
                                goto LABEL_297;
                              }
                              v163 = v157++;
                            }
                            ++v157;
                          }
                          while ( v162 != v157 );
                          if ( v163 )
                          {
                            v159 = *v154;
                            goto LABEL_318;
                          }
                        }
                        if ( HIDWORD(v325) < (unsigned int)v325 )
                        {
                          v159 = *v154;
                          ++HIDWORD(v325);
                          *v162 = v156;
                          ++v322;
                          goto LABEL_319;
                        }
                      }
                      sub_16CCBA0((__int64)&v322, v156);
                      if ( v158 )
                      {
                        v159 = v156;
                        goto LABEL_319;
                      }
                      goto LABEL_303;
                    }
                  }
                }
LABEL_143:
                if ( !(unsigned __int8)sub_186A5D0(v88, (__int64)v254) )
                {
                  v139 = sub_15E0530((__int64)v266[0]);
                  if ( sub_1602790(v139)
                    || (v192 = sub_15E0530((__int64)v266[0]),
                        v193 = sub_16033E0(v192),
                        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v193 + 48LL))(v193)) )
                  {
                    v140 = "AlwaysInline";
                    v233 = v264[0];
                    if ( v264[0] != 0x80000000 )
                      v140 = "Inlined";
                    v238 = (__int64)v140;
                    v141 = strlen(v140);
                    sub_15C9090((__int64)&v308, &v263);
                    sub_15CA330((__int64)&v322, (__int64)"inline", v238, v141, &v308, v92);
                    sub_15C9340((__int64)&v308, "Callee", 6u, v88);
                    v142 = sub_17C2270((__int64)&v322, (__int64)&v308);
                    sub_15CAB20(v142, " inlined into ", 0xEu);
                    if ( (__int64 *)v310.m128_u64[1] != &v312 )
                      j_j___libc_free_0(v310.m128_u64[1], v312 + 1);
                    if ( (__int64 *)v308.m128i_i64[0] != &v309 )
                      j_j___libc_free_0(v308.m128i_i64[0], v309 + 1);
                    sub_15C9340((__int64)&v308, "Caller", 6u, (__int64)v254);
                    sub_17C2270((__int64)&v322, (__int64)&v308);
                    if ( (__int64 *)v310.m128_u64[1] != &v312 )
                      j_j___libc_free_0(v310.m128_u64[1], v312 + 1);
                    if ( (__int64 *)v308.m128i_i64[0] != &v309 )
                      j_j___libc_free_0(v308.m128i_i64[0], v309 + 1);
                    if ( v233 == 0x80000000 )
                    {
                      sub_15CAB20((__int64)&v322, " with cost=always", 0x11u);
                    }
                    else
                    {
                      sub_15CAB20((__int64)&v322, " with cost=", 0xBu);
                      sub_15C9890((__int64)&v308, "Cost", 4, v264[0]);
                      sub_17C2270((__int64)&v322, (__int64)&v308);
                      if ( (__int64 *)v310.m128_u64[1] != &v312 )
                        j_j___libc_free_0(v310.m128_u64[1], v312 + 1);
                      if ( (__int64 *)v308.m128i_i64[0] != &v309 )
                        j_j___libc_free_0(v308.m128i_i64[0], v309 + 1);
                      sub_15CAB20((__int64)&v322, " (threshold=", 0xCu);
                      sub_15C9890((__int64)&v308, "Threshold", 9, v264[1]);
                      sub_17C2270((__int64)&v322, (__int64)&v308);
                      if ( (__int64 *)v310.m128_u64[1] != &v312 )
                        j_j___libc_free_0(v310.m128_u64[1], v312 + 1);
                      if ( (__int64 *)v308.m128i_i64[0] != &v309 )
                        j_j___libc_free_0(v308.m128i_i64[0], v309 + 1);
                      sub_15CAB20((__int64)&v322, ")", 1u);
                    }
                    sub_143AA50(v266, (__int64)&v322);
                    v143 = v328;
                    v322 = (char *)&unk_49ECF68;
                    v144 = &v328[11 * v329];
                    if ( v328 != v144 )
                    {
                      do
                      {
                        v144 -= 11;
                        v145 = (_QWORD *)v144[4];
                        if ( v145 != v144 + 6 )
                          j_j___libc_free_0(v145, v144[6] + 1LL);
                        if ( (_QWORD *)*v144 != v144 + 2 )
                          j_j___libc_free_0(*v144, v144[2] + 1LL);
                      }
                      while ( v143 != v144 );
                      v144 = v328;
                    }
                    if ( v144 != (_QWORD *)v330 )
                      _libc_free((unsigned __int64)v144);
                  }
                }
                if ( !(unsigned __int8)sub_1560180(v88 + 112, 3) )
                {
                  v96 = *(_QWORD *)(v88 + 80);
                  v94 = v88 + 72;
                  if ( v96 != v88 + 72 )
                  {
                    v97 = 0;
                    do
                    {
                      while ( 1 )
                      {
                        if ( !v96 )
                          BUG();
                        v98 = *(_QWORD *)(v96 + 24);
                        if ( v96 + 16 != v98 )
                          break;
                        v96 = *(_QWORD *)(v96 + 8);
                        if ( v94 == v96 )
                          goto LABEL_152;
                      }
                      v99 = 0;
                      do
                      {
                        v98 = *(_QWORD *)(v98 + 8);
                        ++v99;
                      }
                      while ( v96 + 16 != v98 );
                      v96 = *(_QWORD *)(v96 + 8);
                      v97 += v99;
                    }
                    while ( v94 != v96 );
LABEL_152:
                    v243 += v97;
                  }
                }
                if ( (_DWORD)v303 )
                {
                  v180 = (unsigned int)v291;
                  v181 = v291;
                  if ( (unsigned int)v291 >= HIDWORD(v291) )
                  {
                    sub_16CD150((__int64)&v290, v292, 0, 16, v94, v95);
                    v180 = (unsigned int)v291;
                  }
                  v182 = (unsigned __int64 *)&v290[16 * v180];
                  *v182 = v88;
                  v182[1] = v245;
                  v183 = v302;
                  LODWORD(v291) = v291 + 1;
                  v184 = &v302[24 * (unsigned int)v303];
                  if ( v302 != v184 )
                  {
                    v185 = (unsigned int)v294;
                    v186 = v181;
                    while ( 1 )
                    {
                      v190 = *((_QWORD *)v183 + 2);
                      v188 = 0;
                      v191 = *(_BYTE *)(v190 + 16);
                      if ( v191 <= 0x17u )
                        goto LABEL_368;
                      if ( v191 != 78 )
                        break;
                      v188 = v190 | 4;
                      if ( (unsigned int)v185 >= HIDWORD(v294) )
                      {
LABEL_373:
                        v255 = v186;
                        sub_16CD150((__int64)&v293, v295, 0, 16, v186, v95);
                        v185 = (unsigned int)v294;
                        v186 = v255;
                      }
LABEL_369:
                      v189 = (__int64 *)&v293[16 * v185];
                      v183 += 24;
                      *v189 = v188;
                      v189[1] = v186;
                      v185 = (unsigned int)(v294 + 1);
                      LODWORD(v294) = v294 + 1;
                      if ( v184 == v183 )
                        goto LABEL_154;
                    }
                    v187 = v190 & 0xFFFFFFFFFFFFFFFBLL;
                    v188 = 0;
                    if ( v191 == 29 )
                      v188 = v187;
LABEL_368:
                    if ( (unsigned int)v185 >= HIDWORD(v294) )
                      goto LABEL_373;
                    goto LABEL_369;
                  }
                }
LABEL_154:
                if ( v263 )
                  sub_161E7C0((__int64)&v263, v263);
                if ( *(_QWORD *)(v88 + 8) )
                  goto LABEL_157;
LABEL_226:
                if ( (*(_BYTE *)(v88 + 32) & 0xFu) - 7 <= 1 && !sub_186B010((__int64)&v284, v88) )
                {
                  v119 = (_QWORD *)v228[3];
                  v120 = v228 + 2;
                  if ( v119 )
                  {
                    v121 = v228 + 2;
                    v122 = (_QWORD *)v228[3];
                    do
                    {
                      while ( 1 )
                      {
                        v123 = v122[2];
                        v124 = v122[3];
                        if ( v122[4] >= v88 )
                          break;
                        v122 = (_QWORD *)v122[3];
                        if ( !v124 )
                          goto LABEL_233;
                      }
                      v121 = v122;
                      v122 = (_QWORD *)v122[2];
                    }
                    while ( v123 );
LABEL_233:
                    if ( v121 != v120 && v121[4] > v88 )
                      v121 = v228 + 2;
                    if ( *(_DWORD *)(v121[5] + 32LL) )
                      goto LABEL_157;
                    v125 = v228 + 2;
                    do
                    {
                      while ( 1 )
                      {
                        v126 = v119[2];
                        v127 = v119[3];
                        if ( v119[4] >= v88 )
                          break;
                        v119 = (_QWORD *)v119[3];
                        if ( !v127 )
                          goto LABEL_241;
                      }
                      v125 = v119;
                      v119 = (_QWORD *)v119[2];
                    }
                    while ( v126 );
LABEL_241:
                    if ( v125 != v120 && v125[4] <= v88 )
                      v120 = v125;
                  }
                  else if ( *(_DWORD *)(v228[7] + 32LL) )
                  {
                    goto LABEL_157;
                  }
                  v128 = (unsigned __int64 *)v120[5];
                  while ( 1 )
                  {
                    v129 = v128[2];
                    if ( v129 == v128[1] )
                      break;
                    while ( 1 )
                    {
                      --*(_DWORD *)(*(_QWORD *)(v129 - 8) + 32LL);
                      v130 = v128[2];
                      v131 = (_QWORD *)(v130 - 32);
                      v128[2] = v130 - 32;
                      v132 = *(_QWORD *)(v130 - 16);
                      if ( v132 == 0 || v132 == -8 || v132 == -16 )
                        break;
                      sub_1649B30(v131);
                      v129 = v128[2];
                      if ( v129 == v128[1] )
                        goto LABEL_249;
                    }
                  }
LABEL_249:
                  v133 = (_QWORD *)sub_13977A0(v228, v128);
                  v134 = (__int64)v133;
                  if ( v133 )
                  {
                    sub_15E3C20(v133);
                    sub_1648B90(v134);
                  }
                }
LABEL_157:
                v100 = v294;
                v101 = 16LL * (unsigned int)v294;
                v102 = &v293[v260];
                if ( *(_QWORD *)(a2 + 24) - *(_QWORD *)(a2 + 16) == 8 )
                {
                  v103 = (unsigned __int64)&v293[v101 - 16];
                  *(_QWORD *)v102 = *(_QWORD *)v103;
                  *((_DWORD *)v102 + 2) = *(_DWORD *)(v103 + 8);
                  LODWORD(v294) = v294 - 1;
                }
                else
                {
                  v136 = v102 + 16;
                  v137 = v101 - (v260 + 16);
                  v138 = v137 >> 4;
                  if ( v137 > 0 )
                  {
                    while ( 1 )
                    {
                      *(_QWORD *)v102 = *((_QWORD *)v102 + 2);
                      *((_DWORD *)v102 + 2) = *((_DWORD *)v102 + 6);
                      v102 = v136;
                      if ( !--v138 )
                        break;
                      v136 += 16;
                    }
                    v100 = v294;
                  }
                  LODWORD(v294) = v100 - 1;
                }
                v104 = v267;
                if ( v267 )
                {
                  sub_1368A00(v267);
                  j_j___libc_free_0(v104, 8);
                }
                v247 = 1;
                v235 = 1;
                goto LABEL_162;
              }
              while ( 1 )
              {
                v135 = &v290[16 * v89];
                if ( *(_QWORD *)v135 == v88 )
                  break;
                v89 = *((int *)v135 + 2);
                if ( (_DWORD)v89 == -1 )
                  goto LABEL_131;
              }
              ++v83;
LABEL_162:
              v84 = v83;
            }
            while ( (_DWORD)v294 != v83 );
            if ( !v247 )
              goto LABEL_164;
          }
        }
      }
      else
      {
        v74 = &v293[v78];
        v70 = (unsigned int)(v70 - 1);
        v75 = &v293[16 * v70];
        v76 = *(_QWORD *)v74;
        *(_QWORD *)v74 = *(_QWORD *)v75;
        v77 = *((_DWORD *)v75 + 2);
        *(_QWORD *)v75 = v76;
        LODWORD(v76) = *((_DWORD *)v74 + 2);
        *((_DWORD *)v74 + 2) = v77;
        *((_DWORD *)v75 + 2) = v76;
        if ( v71 >= (unsigned int)v70 )
          goto LABEL_122;
      }
    }
    v112 = &v285[HIDWORD(v287)];
    if ( v285 == v112 )
    {
      v73 = v285;
    }
    else
    {
      do
      {
        if ( *v72 == v82 )
          break;
        ++v72;
      }
      while ( v112 != v72 );
      v73 = &v285[HIDWORD(v287)];
    }
LABEL_203:
    while ( v112 != v72 )
    {
      if ( (unsigned __int64)*v72 < 0xFFFFFFFFFFFFFFFELL )
        break;
      ++v72;
    }
    goto LABEL_116;
  }
  v235 = 0;
  v203 = v290;
LABEL_186:
  if ( v203 != v292 )
    _libc_free((unsigned __int64)v203);
LABEL_188:
  if ( v293 != v295 )
    _libc_free((unsigned __int64)v293);
  if ( v286 != v285 )
    _libc_free((unsigned __int64)v286);
  if ( v270 )
    v270(v269, v269, 3);
  if ( v340 )
    sub_134CA00(v339);
  if ( v338 )
  {
    if ( v337 != v336 )
      _libc_free(v337);
    if ( v335 != v334 )
      _libc_free(v335);
    if ( (v332 & 1) == 0 )
      j___libc_free_0(v333);
  }
  return v235;
}
