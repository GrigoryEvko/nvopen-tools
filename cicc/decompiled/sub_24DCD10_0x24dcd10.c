// Function: sub_24DCD10
// Address: 0x24dcd10
//
__int64 __fastcall sub_24DCD10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  __int64 v7; // rcx
  _QWORD *v8; // r13
  _QWORD *v9; // r12
  __int64 v10; // rsi
  _QWORD *v11; // r14
  _QWORD *i; // rbx
  _QWORD *v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rax
  char v17; // al
  char v18; // si
  _QWORD *v19; // rax
  __int64 v20; // r14
  __int64 v21; // rdx
  _BYTE *v22; // r13
  unsigned __int64 v23; // rbx
  __int64 v24; // rdx
  unsigned int v25; // esi
  unsigned __int8 *v26; // r9
  __int64 (__fastcall *v27)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v28; // rax
  __int64 v29; // r11
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  unsigned __int8 *v34; // rax
  __int64 v35; // rbx
  __int64 v36; // r12
  __int64 v37; // rax
  unsigned __int64 *v38; // rax
  __int64 v39; // rdx
  _QWORD *v40; // rax
  unsigned __int64 v41; // rcx
  unsigned __int64 v42; // r8
  _QWORD *v43; // rsi
  __int64 v44; // rax
  char v45; // dl
  __int64 v46; // rax
  bool v47; // al
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 (__fastcall *v50)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v51; // rax
  unsigned __int8 *v52; // r10
  __int64 *v53; // r11
  _BYTE **v54; // rcx
  __int64 v55; // r14
  unsigned __int64 v56; // r14
  __int64 v57; // r10
  __int64 (__fastcall *v58)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v59; // rax
  __int64 v60; // r9
  __int64 v61; // r12
  __int64 v62; // rbx
  _QWORD *v63; // rax
  unsigned __int8 *v64; // r9
  _QWORD *v65; // r11
  __int64 v66; // rdx
  int v67; // ecx
  int v68; // eax
  _QWORD *v69; // rdi
  __int64 *v70; // rax
  __int64 v71; // rax
  __int64 v72; // r9
  __int64 v73; // r11
  __int64 v74; // r13
  unsigned __int64 v75; // r12
  _QWORD *v76; // r14
  _BYTE *v77; // rbx
  __int64 v78; // rdx
  unsigned int v79; // esi
  __int64 v80; // r11
  unsigned int v81; // ecx
  __int64 v82; // rdx
  _BYTE *v83; // r13
  unsigned __int64 v84; // rbx
  __int64 v85; // rdx
  unsigned int v86; // esi
  __int64 v87; // r9
  __int64 v88; // r14
  int v89; // eax
  unsigned int v90; // edx
  unsigned __int8 v91; // cl
  int v92; // r14d
  __int64 v93; // r14
  _BYTE *v94; // rsi
  _QWORD *v95; // r14
  __int64 v96; // r13
  _BYTE *v97; // rbx
  unsigned __int64 v98; // r12
  __int64 v99; // rdx
  unsigned int v100; // esi
  __int64 v101; // rdx
  int v102; // eax
  char v103; // al
  int v104; // edx
  __int64 v105; // rax
  __int64 v106; // rax
  __int64 *v107; // r14
  __int64 *v108; // rax
  unsigned __int64 v109; // r14
  __int16 v110; // ax
  __int64 v111; // r14
  int v112; // eax
  __int64 v113; // rax
  _BYTE *v114; // r14
  __int64 v115; // rax
  const char *v116; // rax
  __int64 v117; // rdx
  __int64 v118; // r8
  unsigned __int8 v119; // r8
  __int64 *v120; // r9
  __int64 v121; // rsi
  __int64 v122; // rdx
  __int64 v123; // rcx
  __int64 v124; // r8
  __int64 v125; // r9
  __int64 v126; // rdx
  _QWORD *v127; // r14
  __int64 v128; // rbx
  __int64 v129; // r13
  unsigned __int64 v130; // r12
  __int64 v131; // rsi
  char *v132; // r14
  char *v133; // rbx
  __int64 v134; // rsi
  char **v135; // r14
  char **v136; // r12
  _QWORD *v137; // r14
  char **v138; // rbx
  unsigned __int64 v139; // rdi
  char *v140; // rdi
  __int64 v141; // r14
  char *v142; // rbx
  __int64 v143; // rsi
  _QWORD *v144; // rdi
  __int64 v145; // r14
  __int64 v146; // rbx
  __int64 v147; // rsi
  __int64 *v148; // rdi
  __int64 v149; // r14
  unsigned __int64 v150; // rbx
  __int64 v151; // rsi
  __int64 v152; // rax
  __int64 v153; // r9
  _QWORD *v154; // rdi
  __int64 v155; // rax
  _QWORD *v156; // rdx
  __int64 v157; // rsi
  _QWORD *v158; // rax
  __int64 v159; // rdx
  __int64 v160; // r14
  __int64 v161; // rax
  bool v162; // al
  _QWORD *v163; // [rsp+8h] [rbp-3D8h]
  _QWORD *v164; // [rsp+10h] [rbp-3D0h]
  __int64 **v165; // [rsp+18h] [rbp-3C8h]
  __int64 v166; // [rsp+20h] [rbp-3C0h]
  __int64 v168; // [rsp+48h] [rbp-398h]
  unsigned int v169; // [rsp+48h] [rbp-398h]
  __int64 *v170; // [rsp+48h] [rbp-398h]
  __int64 v171; // [rsp+50h] [rbp-390h]
  __int64 v172; // [rsp+50h] [rbp-390h]
  unsigned int v173; // [rsp+50h] [rbp-390h]
  __int64 v174; // [rsp+60h] [rbp-380h]
  _QWORD *v175; // [rsp+60h] [rbp-380h]
  unsigned __int64 v176; // [rsp+60h] [rbp-380h]
  _QWORD *v177; // [rsp+60h] [rbp-380h]
  _QWORD *v178; // [rsp+60h] [rbp-380h]
  _QWORD *v179; // [rsp+60h] [rbp-380h]
  _QWORD *v180; // [rsp+60h] [rbp-380h]
  __int64 *v181; // [rsp+60h] [rbp-380h]
  __int64 v182; // [rsp+60h] [rbp-380h]
  __int64 v183; // [rsp+60h] [rbp-380h]
  _QWORD *v184; // [rsp+60h] [rbp-380h]
  _QWORD *v185; // [rsp+60h] [rbp-380h]
  _QWORD *v186; // [rsp+60h] [rbp-380h]
  _QWORD *v187; // [rsp+60h] [rbp-380h]
  _QWORD *v188; // [rsp+60h] [rbp-380h]
  __int64 v189; // [rsp+60h] [rbp-380h]
  __int64 v190; // [rsp+60h] [rbp-380h]
  __int64 v191; // [rsp+68h] [rbp-378h]
  _QWORD *v192; // [rsp+68h] [rbp-378h]
  unsigned __int8 *v193; // [rsp+68h] [rbp-378h]
  _QWORD *v194; // [rsp+68h] [rbp-378h]
  __int64 v195; // [rsp+68h] [rbp-378h]
  __int64 **v196; // [rsp+68h] [rbp-378h]
  unsigned __int8 *v197; // [rsp+68h] [rbp-378h]
  __int64 v198; // [rsp+68h] [rbp-378h]
  _QWORD *v199; // [rsp+68h] [rbp-378h]
  _QWORD *v200; // [rsp+68h] [rbp-378h]
  _QWORD *v201; // [rsp+68h] [rbp-378h]
  __int64 v202; // [rsp+68h] [rbp-378h]
  __int64 v203; // [rsp+68h] [rbp-378h]
  __int64 v204; // [rsp+68h] [rbp-378h]
  _QWORD *v205; // [rsp+68h] [rbp-378h]
  unsigned __int8 *v206; // [rsp+68h] [rbp-378h]
  __int64 v207; // [rsp+68h] [rbp-378h]
  _QWORD *v208; // [rsp+68h] [rbp-378h]
  __int64 v209; // [rsp+68h] [rbp-378h]
  __int64 v210; // [rsp+68h] [rbp-378h]
  _QWORD *v211; // [rsp+70h] [rbp-370h]
  __int64 v212; // [rsp+78h] [rbp-368h]
  unsigned int v213; // [rsp+90h] [rbp-350h]
  char v214; // [rsp+96h] [rbp-34Ah]
  bool v215; // [rsp+97h] [rbp-349h]
  _QWORD *v216; // [rsp+98h] [rbp-348h]
  _QWORD *v217; // [rsp+98h] [rbp-348h]
  _QWORD *v218; // [rsp+A0h] [rbp-340h]
  _QWORD *v219; // [rsp+A8h] [rbp-338h]
  _BYTE *v220; // [rsp+B8h] [rbp-328h] BYREF
  __int64 v221; // [rsp+C0h] [rbp-320h] BYREF
  __int64 v222; // [rsp+C8h] [rbp-318h]
  __int64 v223; // [rsp+D0h] [rbp-310h] BYREF
  __int64 v224; // [rsp+D8h] [rbp-308h]
  __int16 v225; // [rsp+F0h] [rbp-2F0h]
  _BYTE *v226; // [rsp+100h] [rbp-2E0h] BYREF
  __int64 v227; // [rsp+108h] [rbp-2D8h]
  _BYTE v228[32]; // [rsp+110h] [rbp-2D0h] BYREF
  __int64 v229; // [rsp+130h] [rbp-2B0h] BYREF
  __int64 *v230; // [rsp+138h] [rbp-2A8h]
  __int64 **v231; // [rsp+140h] [rbp-2A0h]
  unsigned __int8 *v232; // [rsp+150h] [rbp-290h]
  _BYTE *v233; // [rsp+158h] [rbp-288h] BYREF
  __int64 v234; // [rsp+160h] [rbp-280h]
  _BYTE v235[32]; // [rsp+168h] [rbp-278h] BYREF
  __int64 v236; // [rsp+188h] [rbp-258h]
  __int64 v237; // [rsp+190h] [rbp-250h]
  __int64 v238; // [rsp+198h] [rbp-248h]
  __int64 *v239; // [rsp+1A0h] [rbp-240h]
  void **v240; // [rsp+1A8h] [rbp-238h]
  void **v241; // [rsp+1B0h] [rbp-230h]
  __int64 v242; // [rsp+1B8h] [rbp-228h]
  int v243; // [rsp+1C0h] [rbp-220h]
  __int16 v244; // [rsp+1C4h] [rbp-21Ch]
  char v245; // [rsp+1C6h] [rbp-21Ah]
  __int64 v246; // [rsp+1C8h] [rbp-218h]
  __int64 v247; // [rsp+1D0h] [rbp-210h]
  void *v248; // [rsp+1D8h] [rbp-208h] BYREF
  void *v249; // [rsp+1E0h] [rbp-200h] BYREF
  __int64 v250; // [rsp+1E8h] [rbp-1F8h]
  unsigned __int64 v251; // [rsp+1F0h] [rbp-1F0h]
  __int64 v252; // [rsp+200h] [rbp-1E0h] BYREF
  __int64 v253; // [rsp+208h] [rbp-1D8h]
  unsigned __int64 v254; // [rsp+210h] [rbp-1D0h]
  __int64 v255; // [rsp+218h] [rbp-1C8h]
  _QWORD v256[2]; // [rsp+220h] [rbp-1C0h] BYREF
  const char *v257; // [rsp+230h] [rbp-1B0h] BYREF
  unsigned __int64 v258; // [rsp+238h] [rbp-1A8h]
  __int64 v259; // [rsp+240h] [rbp-1A0h]
  __int64 v260; // [rsp+248h] [rbp-198h] BYREF
  _QWORD v261[3]; // [rsp+250h] [rbp-190h] BYREF
  __int64 v262; // [rsp+268h] [rbp-178h]
  const char *v263; // [rsp+270h] [rbp-170h]
  _QWORD v264[4]; // [rsp+278h] [rbp-168h] BYREF
  unsigned __int64 v265; // [rsp+298h] [rbp-148h]
  const char *v266; // [rsp+2A0h] [rbp-140h]
  _QWORD v267[4]; // [rsp+2A8h] [rbp-138h] BYREF
  unsigned __int64 v268; // [rsp+2C8h] [rbp-118h]
  char v269; // [rsp+2D8h] [rbp-108h] BYREF
  char *v270; // [rsp+2F8h] [rbp-E8h]
  int v271; // [rsp+300h] [rbp-E0h]
  char v272; // [rsp+308h] [rbp-D8h] BYREF
  __int64 v273; // [rsp+330h] [rbp-B0h]
  unsigned int v274; // [rsp+340h] [rbp-A0h]
  char **v275; // [rsp+348h] [rbp-98h]
  unsigned int v276; // [rsp+350h] [rbp-90h]
  char *v277; // [rsp+358h] [rbp-88h] BYREF
  int v278; // [rsp+360h] [rbp-80h]
  char v279; // [rsp+368h] [rbp-78h] BYREF
  __int64 v280; // [rsp+398h] [rbp-48h]
  unsigned int v281; // [rsp+3A8h] [rbp-38h]

  v252 = (__int64)"llvm.coro.id";
  v254 = (unsigned __int64)"llvm.coro.id.retcon";
  v256[0] = "llvm.coro.id.retcon.once";
  v257 = "llvm.coro.id.async";
  v259 = (__int64)"llvm.coro.destroy";
  v261[0] = "llvm.coro.done";
  v261[2] = "llvm.coro.end";
  v263 = "llvm.coro.end.async";
  v264[1] = "llvm.coro.noop";
  v264[3] = "llvm.coro.free";
  v266 = "llvm.coro.promise";
  v267[1] = "llvm.coro.resume";
  v253 = 12;
  v255 = 19;
  v256[1] = 24;
  v258 = 18;
  v260 = 17;
  v261[1] = 14;
  v262 = 13;
  v264[0] = 19;
  v264[2] = 14;
  v265 = 14;
  v267[0] = 17;
  v267[2] = 16;
  v267[3] = "llvm.coro.suspend";
  v268 = 17;
  v214 = sub_24F32D0(a3, &v252, 13);
  if ( !v214 )
  {
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
    return a1;
  }
  sub_24F30B0(&v229, a3);
  v233 = v235;
  v234 = 0x200000000LL;
  v240 = &v248;
  v241 = &v249;
  v244 = 512;
  LOWORD(v238) = 0;
  v239 = v230;
  v245 = 7;
  v248 = &unk_49DA100;
  v242 = 0;
  v243 = 0;
  v246 = 0;
  v247 = 0;
  v236 = 0;
  v237 = 0;
  v249 = &unk_49DA0B0;
  v5 = sub_BCE3C0(v230, 0);
  v7 = a3 + 24;
  v251 = 0;
  v250 = v5;
  v211 = (_QWORD *)(a3 + 24);
  v218 = *(_QWORD **)(a3 + 32);
  if ( v218 != (_QWORD *)(a3 + 24) )
  {
    do
    {
      if ( !v218 )
      {
        v226 = v228;
        v227 = 0x400000000LL;
        BUG();
      }
      v8 = (_QWORD *)v218[3];
      v9 = v218 + 2;
      v212 = (__int64)(v218 - 7);
      v226 = v228;
      v10 = 0x400000000LL;
      v227 = 0x400000000LL;
      if ( v218 + 2 == v8 )
      {
        v11 = 0;
      }
      else
      {
        if ( !v8 )
          BUG();
        while ( 1 )
        {
          v11 = (_QWORD *)v8[4];
          if ( v11 != v8 + 3 )
            break;
          v8 = (_QWORD *)v8[1];
          if ( v9 == v8 )
            break;
          if ( !v8 )
            BUG();
        }
      }
      v215 = 0;
      v216 = 0;
      while ( v9 != v8 )
      {
        for ( i = (_QWORD *)v11[1]; ; i = (_QWORD *)v8[4] )
        {
          v13 = v8 - 3;
          if ( !v8 )
            v13 = 0;
          if ( i != v13 + 6 )
            break;
          v8 = (_QWORD *)v8[1];
          if ( v9 == v8 )
            break;
          if ( !v8 )
            BUG();
        }
        if ( (unsigned __int8)(*((_BYTE *)v11 - 24) - 34) > 0x33u )
          goto LABEL_36;
        v10 = 0x8000000000041LL;
        if ( !_bittest64(&v10, (unsigned int)*((unsigned __int8 *)v11 - 24) - 34) )
          goto LABEL_36;
        v219 = v11 - 3;
        switch ( (unsigned int)sub_B49240((__int64)(v11 - 3)) )
        {
          case ')':
            v10 = (__int64)(v11 - 3);
            v11 = i;
            sub_24DCC80((__int64)&v229, (__int64)v219, 1);
            continue;
          case '*':
            v191 = (__int64)v231;
            v174 = v219[-4 * (*((_DWORD *)v11 - 5) & 0x7FFFFFF)];
            sub_D5F1F0((__int64)&v233, (__int64)v219);
            v225 = 257;
            v16 = sub_AA4E30(v236);
            v17 = sub_AE5020(v16, v191);
            LOWORD(v256[0]) = 257;
            v18 = v17;
            v19 = sub_BD2C40(80, unk_3F10A14);
            v20 = (__int64)v19;
            if ( v19 )
              sub_B4D190((__int64)v19, v191, v174, (__int64)&v252, 0, v18, 0, 0);
            (*((void (__fastcall **)(void **, __int64, __int64 *, __int64, __int64))*v241 + 2))(
              v241,
              v20,
              &v223,
              v237,
              v238);
            v21 = 16LL * (unsigned int)v234;
            if ( v233 != &v233[v21] )
            {
              v192 = v8;
              v22 = &v233[v21];
              v175 = i;
              v23 = (unsigned __int64)v233;
              do
              {
                v24 = *(_QWORD *)(v23 + 8);
                v25 = *(_DWORD *)v23;
                v23 += 16LL;
                sub_B99FD0(v20, v25, v24);
              }
              while ( v22 != (_BYTE *)v23 );
              v8 = v192;
              i = v175;
            }
            v26 = v232;
            v225 = 257;
            v27 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, unsigned __int8 *))*((_QWORD *)*v240 + 7);
            if ( v27 == sub_928890 )
            {
              if ( *(_BYTE *)v20 > 0x15u || *v232 > 0x15u )
                goto LABEL_115;
              v193 = v232;
              v28 = sub_AAB310(0x20u, (unsigned __int8 *)v20, v232);
              v26 = v193;
              v29 = v28;
            }
            else
            {
              v206 = v232;
              v106 = v27((__int64)v240, 32u, (_BYTE *)v20, v232);
              v26 = v206;
              v29 = v106;
            }
            if ( v29 )
              goto LABEL_34;
LABEL_115:
            v197 = v26;
            LOWORD(v256[0]) = 257;
            v63 = sub_BD2C40(72, unk_3F10FD0);
            v64 = v197;
            v65 = v63;
            if ( v63 )
            {
              v66 = *(_QWORD *)(v20 + 8);
              v177 = v63;
              v67 = *(unsigned __int8 *)(v66 + 8);
              if ( (unsigned int)(v67 - 17) > 1 )
              {
                v71 = sub_BCB2A0(*(_QWORD **)v66);
                v73 = (__int64)v177;
                v72 = (__int64)v197;
              }
              else
              {
                v68 = *(_DWORD *)(v66 + 32);
                v69 = *(_QWORD **)v66;
                BYTE4(v222) = (_BYTE)v67 == 18;
                LODWORD(v222) = v68;
                v70 = (__int64 *)sub_BCB2A0(v69);
                v71 = sub_BCE1B0(v70, v222);
                v72 = (__int64)v197;
                v73 = (__int64)v177;
              }
              v198 = v73;
              sub_B523C0(v73, v71, 53, 32, v20, v72, (__int64)&v252, 0, 0, 0);
              v65 = (_QWORD *)v198;
            }
            v199 = v65;
            (*((void (__fastcall **)(void **, _QWORD *, __int64 *, __int64, __int64, unsigned __int8 *))*v241 + 2))(
              v241,
              v65,
              &v223,
              v237,
              v238,
              v64);
            v29 = (__int64)v199;
            if ( v233 != &v233[16 * (unsigned int)v234] )
            {
              v200 = v8;
              v74 = v29;
              v178 = v9;
              v75 = (unsigned __int64)v233;
              v76 = i;
              v77 = &v233[16 * (unsigned int)v234];
              do
              {
                v78 = *(_QWORD *)(v75 + 8);
                v79 = *(_DWORD *)v75;
                v75 += 16LL;
                sub_B99FD0(v74, v79, v78);
              }
              while ( v77 != (_BYTE *)v75 );
              v29 = v74;
              v9 = v178;
              i = v76;
              v8 = v200;
            }
LABEL_34:
            v10 = v29;
            goto LABEL_35;
          case '+':
          case ',':
            v10 = (__int64)(v11 - 3);
            if ( sub_AD7A80(
                   (_BYTE *)v219[4 * (1LL - (*((_DWORD *)v11 - 5) & 0x7FFFFFF))],
                   (__int64)v219,
                   *((_DWORD *)v11 - 5) & 0x7FFFFFF,
                   v7,
                   v14) )
            {
              goto LABEL_36;
            }
            goto LABEL_56;
          case '/':
            v32 = (unsigned int)v227;
            v7 = HIDWORD(v227);
            v33 = (unsigned int)v227 + 1LL;
            if ( v33 > HIDWORD(v227) )
            {
              v10 = (__int64)v228;
              sub_C8D5F0((__int64)&v226, v228, v33, 8u, v14, v15);
              v32 = (unsigned int)v227;
            }
            v6 = (unsigned __int64)v226;
            v11 = i;
            *(_QWORD *)&v226[8 * v32] = v219;
            LODWORD(v227) = v227 + 1;
            continue;
          case '0':
            v34 = sub_BD3990((unsigned __int8 *)v219[4 * (3LL - (*((_DWORD *)v11 - 5) & 0x7FFFFFF))], 0x8000000000041LL);
            if ( *v34 == 3 && **((_BYTE **)v34 - 4) != 10 )
              goto LABEL_36;
            if ( !*(v11 - 1) )
              goto LABEL_77;
            v194 = i;
            v35 = *(v11 - 1);
            v217 = v9;
            break;
          case '1':
          case '2':
          case '3':
            v10 = 49;
            v11 = i;
            sub_B2CD30(v212, 49);
            continue;
          case '4':
            if ( !v251 )
            {
              v107 = v239;
              v170 = v239;
              v166 = sub_B43CA0((__int64)v219);
              v252 = sub_BCE3C0(v239, 0);
              v108 = (__int64 *)sub_BCB120(v107);
              v109 = sub_BCF480(v108, &v252, 1, 0);
              v252 = sub_BCE3C0(v239, 0);
              v253 = v252;
              v165 = (__int64 **)sub_BD0EC0(&v252, 2, "NoopCoro.Frame", 0xEu, 0);
              v252 = (__int64)"__NoopCoro_ResumeDestroy";
              LOWORD(v256[0]) = 259;
              v207 = sub_BD2DA0(136);
              if ( v207 )
                sub_B2C3B0(v207, v109, 8, 0xFFFFFFFF, (__int64)&v252, v166);
              v110 = *(_WORD *)(v207 + 2) & 0xC00F;
              LOBYTE(v110) = v110 | 0x80;
              *(_WORD *)(v207 + 2) = v110;
              v181 = *(__int64 **)(v207 + 40);
              v111 = sub_BA8DC0((__int64)v181, (__int64)"llvm.dbg.cu", 11);
              v112 = 0;
              if ( v111 )
                v112 = sub_B91A00(v111);
              LODWORD(v253) = v112;
              v252 = v111;
              sub_BA95A0((__int64)&v252);
              v223 = v111;
              LODWORD(v224) = 0;
              sub_BA95A0((__int64)&v223);
              if ( (_DWORD)v224 != (_DWORD)v253 )
              {
                v113 = sub_BA8DC0((__int64)v181, (__int64)"llvm.dbg.cu", 11);
                LODWORD(v253) = 0;
                v252 = v113;
                sub_BA95A0((__int64)&v252);
                v114 = (_BYTE *)sub_BA9580((__int64)&v252);
                sub_AE0470((__int64)&v252, v181, 0, (__int64)v114);
                v223 = 0;
                v224 = 0;
                v115 = sub_ADD430((__int64)&v252, &v223, 2);
                v182 = sub_ADCD40((__int64)&v252, v115, 0, 0);
                v116 = sub_BD5D20(v207);
                v118 = (__int64)v114;
                if ( *v114 != 16 )
                {
                  v119 = *(v114 - 16);
                  if ( (v119 & 2) != 0 )
                    v120 = (__int64 *)*((_QWORD *)v114 - 4);
                  else
                    v120 = (__int64 *)&v114[-8 * ((v119 >> 2) & 0xF) - 16];
                  v118 = *v120;
                }
                v121 = sub_ADE3D0(
                         (__int64)&v252,
                         v114,
                         (__int64)v116,
                         v117,
                         (__int64)v116,
                         v117,
                         v118,
                         0,
                         v182,
                         0,
                         64,
                         8,
                         0,
                         0,
                         0,
                         0,
                         (__int64)byte_3F871B3,
                         0);
                sub_B994C0(v207, v121);
                sub_ADCDB0((__int64)&v252, v121, v122, v123, v124, v125);
                v126 = v281;
                if ( v281 )
                {
                  v164 = v8;
                  v163 = v9;
                  v183 = v280 + 56LL * v281;
                  v127 = i;
                  v128 = v280;
                  do
                  {
                    if ( *(_QWORD *)v128 != -8192 && *(_QWORD *)v128 != -4096 )
                    {
                      v129 = *(_QWORD *)(v128 + 8);
                      v130 = v129 + 8LL * *(unsigned int *)(v128 + 16);
                      if ( v129 != v130 )
                      {
                        do
                        {
                          v131 = *(_QWORD *)(v130 - 8);
                          v130 -= 8LL;
                          if ( v131 )
                            sub_B91220(v130, v131);
                        }
                        while ( v129 != v130 );
                        v130 = *(_QWORD *)(v128 + 8);
                      }
                      if ( v130 != v128 + 24 )
                        _libc_free(v130);
                    }
                    v128 += 56;
                  }
                  while ( v183 != v128 );
                  v8 = v164;
                  v9 = v163;
                  i = v127;
                  v126 = v281;
                }
                sub_C7D6A0(v280, 56 * v126, 8);
                v132 = &v277[8 * v278];
                if ( v277 != v132 )
                {
                  v184 = i;
                  v133 = v277;
                  do
                  {
                    v134 = *((_QWORD *)v132 - 1);
                    v132 -= 8;
                    if ( v134 )
                      sub_B91220((__int64)v132, v134);
                  }
                  while ( v133 != v132 );
                  i = v184;
                  v132 = v277;
                }
                if ( v132 != &v279 )
                  _libc_free((unsigned __int64)v132);
                v135 = &v275[7 * v276];
                if ( v275 != v135 )
                {
                  v185 = v9;
                  v136 = &v275[7 * v276];
                  v137 = i;
                  v138 = v275;
                  do
                  {
                    v136 -= 7;
                    v139 = (unsigned __int64)v136[5];
                    if ( (char **)v139 != v136 + 7 )
                      _libc_free(v139);
                    sub_C7D6A0((__int64)v136[2], 8LL * *((unsigned int *)v136 + 8), 8);
                  }
                  while ( v138 != v136 );
                  i = v137;
                  v9 = v185;
                  v135 = v275;
                }
                if ( v135 != &v277 )
                  _libc_free((unsigned __int64)v135);
                sub_C7D6A0(v273, 16LL * v274, 8);
                v140 = &v270[8 * v271];
                if ( v270 != v140 )
                {
                  v186 = i;
                  v141 = (__int64)&v270[8 * v271];
                  v142 = v270;
                  do
                  {
                    v143 = *(_QWORD *)(v141 - 8);
                    v141 -= 8;
                    if ( v143 )
                      sub_B91220(v141, v143);
                  }
                  while ( v142 != (char *)v141 );
                  i = v186;
                  v140 = v270;
                }
                if ( v140 != &v272 )
                  _libc_free((unsigned __int64)v140);
                if ( (char *)v268 != &v269 )
                  _libc_free(v268);
                if ( (_QWORD *)v265 != v267 )
                  _libc_free(v265);
                v144 = (_QWORD *)(v262 + 8LL * (unsigned int)v263);
                if ( (_QWORD *)v262 != v144 )
                {
                  v187 = i;
                  v145 = v262 + 8LL * (unsigned int)v263;
                  v146 = v262;
                  do
                  {
                    v147 = *(_QWORD *)(v145 - 8);
                    v145 -= 8;
                    if ( v147 )
                      sub_B91220(v145, v147);
                  }
                  while ( v146 != v145 );
                  i = v187;
                  v144 = (_QWORD *)v262;
                }
                if ( v144 != v264 )
                  _libc_free((unsigned __int64)v144);
                v148 = (__int64 *)(v258 + 8LL * (unsigned int)v259);
                if ( (__int64 *)v258 != v148 )
                {
                  v188 = i;
                  v149 = v258 + 8LL * (unsigned int)v259;
                  v150 = v258;
                  do
                  {
                    v151 = *(_QWORD *)(v149 - 8);
                    v149 -= 8;
                    if ( v151 )
                      sub_B91220(v149, v151);
                  }
                  while ( v150 != v149 );
                  i = v188;
                  v148 = (__int64 *)v258;
                }
                if ( v148 != &v260 )
                  _libc_free((unsigned __int64)v148);
              }
              v252 = (__int64)"entry";
              LOWORD(v256[0]) = 259;
              v152 = sub_22077B0(0x50u);
              v153 = v152;
              if ( v152 )
              {
                v189 = v152;
                sub_AA4D50(v152, (__int64)v170, (__int64)&v252, v207, 0);
                v153 = v189;
              }
              sub_B43C20((__int64)&v252, v153);
              v154 = sub_BD2C40(72, 0);
              if ( v154 )
                sub_B4BB80((__int64)v154, (__int64)v170, 0, 0, v252, v253);
              v223 = v207;
              v224 = v207;
              v155 = sub_AD24A0(v165, &v223, 2);
              BYTE4(v221) = 0;
              v156 = *(_QWORD **)(v155 + 8);
              v190 = v155;
              v252 = (__int64)"NoopCoro.Frame.Const";
              v208 = v156;
              v157 = unk_3F0FAE8;
              LOWORD(v256[0]) = 259;
              v158 = sub_BD2C40(88, unk_3F0FAE8);
              v160 = (__int64)v158;
              if ( v158 )
              {
                v157 = v166;
                sub_B30000((__int64)v158, v166, v208, 1, 8, v190, (__int64)&v252, 0, 0, v221, 0);
              }
              v251 = v160;
              sub_B31480(v160, v157, v159);
            }
            sub_D5F1F0((__int64)&v233, (__int64)v219);
            v56 = v251;
            v57 = (__int64)v231;
            v225 = 257;
            if ( v231 == *(__int64 ***)(v251 + 8) )
            {
              v60 = v251;
              goto LABEL_105;
            }
            v58 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v240 + 15);
            if ( v58 == sub_920130 )
            {
              if ( *(_BYTE *)v251 > 0x15u )
                goto LABEL_132;
              v196 = v231;
              if ( (unsigned __int8)sub_AC4810(0x31u) )
                v59 = sub_ADAB70(49, v56, v196, 0);
              else
                v59 = sub_AA93C0(0x31u, v56, (__int64)v196);
              v57 = (__int64)v196;
              v60 = v59;
            }
            else
            {
              v209 = (__int64)v231;
              v161 = v58((__int64)v240, 49u, (_BYTE *)v251, (__int64)v231);
              v57 = v209;
              v60 = v161;
            }
            if ( v60 )
              goto LABEL_105;
LABEL_132:
            LOWORD(v256[0]) = 257;
            v87 = sub_B51D30(49, v56, v57, (__int64)&v252, 0, 0);
            if ( *(_BYTE *)v87 > 0x1Cu )
            {
              switch ( *(_BYTE *)v87 )
              {
                case ')':
                case '+':
                case '-':
                case '/':
                case '2':
                case '5':
                case 'J':
                case 'K':
                case 'S':
                  goto LABEL_137;
                case 'T':
                case 'U':
                case 'V':
                  v88 = *(_QWORD *)(v87 + 8);
                  v89 = *(unsigned __int8 *)(v88 + 8);
                  v90 = v89 - 17;
                  v91 = *(_BYTE *)(v88 + 8);
                  if ( (unsigned int)(v89 - 17) <= 1 )
                    v91 = *(_BYTE *)(**(_QWORD **)(v88 + 16) + 8LL);
                  if ( v91 <= 3u || v91 == 5 || (v91 & 0xFD) == 4 )
                    goto LABEL_137;
                  if ( (_BYTE)v89 == 15 )
                  {
                    if ( (*(_BYTE *)(v88 + 9) & 4) == 0 )
                      break;
                    v210 = v87;
                    v162 = sub_BCB420(*(_QWORD *)(v87 + 8));
                    v87 = v210;
                    if ( !v162 )
                      break;
                    v88 = **(_QWORD **)(v88 + 16);
                    v89 = *(unsigned __int8 *)(v88 + 8);
                    v90 = v89 - 17;
                  }
                  else if ( (_BYTE)v89 == 16 )
                  {
                    do
                    {
                      v88 = *(_QWORD *)(v88 + 24);
                      LOBYTE(v89) = *(_BYTE *)(v88 + 8);
                    }
                    while ( (_BYTE)v89 == 16 );
                    v90 = (unsigned __int8)v89 - 17;
                  }
                  if ( v90 <= 1 )
                    LOBYTE(v89) = *(_BYTE *)(**(_QWORD **)(v88 + 16) + 8LL);
                  if ( (unsigned __int8)v89 <= 3u || (_BYTE)v89 == 5 || (v89 & 0xFD) == 4 )
                  {
LABEL_137:
                    v92 = v243;
                    if ( v242 )
                    {
                      v202 = v87;
                      sub_B99FD0(v87, 3u, v242);
                      v87 = v202;
                    }
                    v203 = v87;
                    sub_B45150(v87, v92);
                    v87 = v203;
                  }
                  break;
                default:
                  break;
              }
            }
            v204 = v87;
            (*((void (__fastcall **)(void **, __int64, __int64 *, __int64, __int64))*v241 + 2))(
              v241,
              v87,
              &v223,
              v237,
              v238);
            v60 = v204;
            v93 = 16LL * (unsigned int)v234;
            v94 = &v233[v93];
            if ( v233 != &v233[v93] )
            {
              v205 = v8;
              v95 = i;
              v96 = v60;
              v97 = v94;
              v180 = v9;
              v98 = (unsigned __int64)v233;
              do
              {
                v99 = *(_QWORD *)(v98 + 8);
                v100 = *(_DWORD *)v98;
                v98 += 16LL;
                sub_B99FD0(v96, v100, v99);
              }
              while ( v97 != (_BYTE *)v98 );
              v60 = v96;
              v9 = v180;
              i = v95;
              v8 = v205;
            }
LABEL_105:
            v10 = v60;
            goto LABEL_35;
          case '7':
            v195 = v219[-4 * (*((_DWORD *)v11 - 5) & 0x7FFFFFF)];
            v39 = v219[4 * (1LL - (*((_DWORD *)v11 - 5) & 0x7FFFFFF))];
            v40 = *(_QWORD **)(v39 + 24);
            if ( *(_DWORD *)(v39 + 32) > 0x40u )
              v40 = (_QWORD *)*v40;
            if ( v40 )
            {
              _BitScanReverse64(&v41, (unsigned __int64)v40);
              v42 = 0x8000000000000000LL >> ((unsigned __int8)v41 ^ 0x3Fu);
              v171 = -(__int64)v42;
            }
            else
            {
              v171 = -1;
              v42 = 1;
            }
            v168 = v42;
            v176 = sub_BCB2B0(v239);
            v254 = v176;
            v252 = v250;
            v253 = v250;
            v43 = sub_BD0B90(v230, &v252, 3, 0);
            v44 = sub_AE4AC0(v229 + 312, (__int64)v43);
            v45 = *(_BYTE *)(v44 + 64);
            v46 = *(_QWORD *)(v44 + 56);
            LOBYTE(v253) = v45;
            v252 = v46;
            v172 = v171 & (v168 + sub_CA1930(&v252) - 1);
            v47 = sub_AD7A80(
                    (_BYTE *)v219[4 * (2LL - (*((_DWORD *)v11 - 5) & 0x7FFFFFF))],
                    (__int64)v43,
                    v172,
                    *((_DWORD *)v11 - 5) & 0x7FFFFFF,
                    v168);
            LODWORD(v48) = v172;
            if ( v47 )
              v48 = -v172;
            v173 = v48;
            sub_D5F1F0((__int64)&v233, (__int64)v219);
            v225 = 257;
            v49 = sub_BCB2D0(v239);
            v220 = (_BYTE *)sub_ACD640(v49, v173, 0);
            v50 = (__int64 (__fastcall *)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))*((_QWORD *)*v240 + 8);
            if ( v50 == sub_920540 )
            {
              if ( sub_BCEA30(v176) )
                goto LABEL_125;
              if ( *(_BYTE *)v195 > 0x15u )
                goto LABEL_125;
              v51 = sub_24DCBC0(&v220, (__int64)&v221);
              if ( v54 != v51 )
                goto LABEL_125;
              LOBYTE(v256[0]) = 0;
              v55 = sub_AD9FD0(v176, v52, v53, 1, 3u, (__int64)&v252, 0);
              if ( LOBYTE(v256[0]) )
              {
                LOBYTE(v256[0]) = 0;
                if ( (unsigned int)v255 > 0x40 && v254 )
                  j_j___libc_free_0_0(v254);
                if ( (unsigned int)v253 > 0x40 && v252 )
                  j_j___libc_free_0_0(v252);
              }
            }
            else
            {
              v55 = v50((__int64)v240, v176, (_BYTE *)v195, &v220, 1, 3);
            }
            if ( v55 )
              goto LABEL_96;
LABEL_125:
            LOWORD(v256[0]) = 257;
            v55 = (__int64)sub_BD2C40(88, 2u);
            if ( !v55 )
              goto LABEL_128;
            v80 = *(_QWORD *)(v195 + 8);
            v81 = v213 & 0xE0000000 | 2;
            v213 = v81;
            if ( (unsigned int)*(unsigned __int8 *)(v80 + 8) - 17 <= 1 )
              goto LABEL_127;
            v101 = *((_QWORD *)v220 + 1);
            v102 = *(unsigned __int8 *)(v101 + 8);
            if ( v102 == 17 )
            {
              v103 = 0;
            }
            else
            {
              if ( v102 != 18 )
                goto LABEL_127;
              v103 = 1;
            }
            v104 = *(_DWORD *)(v101 + 32);
            BYTE4(v221) = v103;
            v169 = v81;
            LODWORD(v221) = v104;
            v105 = sub_BCE1B0((__int64 *)v80, v221);
            v81 = v169;
            v80 = v105;
LABEL_127:
            sub_B44260(v55, v80, 34, v81, 0, 0);
            *(_QWORD *)(v55 + 72) = v176;
            *(_QWORD *)(v55 + 80) = sub_B4DC50(v176, (__int64)&v220, 1);
            sub_B4D9A0(v55, v195, (__int64 *)&v220, 1, (__int64)&v252);
LABEL_128:
            sub_B4DDE0(v55, 3);
            (*((void (__fastcall **)(void **, __int64, __int64 *, __int64, __int64))*v241 + 2))(
              v241,
              v55,
              &v223,
              v237,
              v238);
            v82 = 16LL * (unsigned int)v234;
            if ( v233 != &v233[v82] )
            {
              v201 = v8;
              v83 = &v233[v82];
              v179 = i;
              v84 = (unsigned __int64)v233;
              do
              {
                v85 = *(_QWORD *)(v84 + 8);
                v86 = *(_DWORD *)v84;
                v84 += 16LL;
                sub_B99FD0(v55, v86, v85);
              }
              while ( v83 != (_BYTE *)v84 );
              v8 = v201;
              i = v179;
            }
LABEL_96:
            v10 = v55;
LABEL_35:
            sub_BD84D0((__int64)v219, v10);
            sub_B43D60(v219);
            goto LABEL_36;
          case '8':
            v10 = (__int64)(v11 - 3);
            v11 = i;
            sub_24DCC80((__int64)&v229, (__int64)v219, 0);
            continue;
          case '<':
            v215 = sub_AD7A80(
                     (_BYTE *)v219[4 * (1LL - (*((_DWORD *)v11 - 5) & 0x7FFFFFF))],
                     0x8000000000041LL,
                     *((_DWORD *)v11 - 5) & 0x7FFFFFF,
                     v7,
                     v14);
            if ( v215 )
            {
LABEL_56:
              v10 = sub_BD5C60((__int64)v219);
              v11[6] = sub_A7A090(v11 + 6, (__int64 *)v10, -1, 27);
              v11 = i;
            }
            else
            {
              v11 = i;
              v215 = v214;
            }
            continue;
          default:
            goto LABEL_36;
        }
        do
        {
          while ( 1 )
          {
            v36 = *(_QWORD *)(v35 + 24);
            if ( *(_BYTE *)v36 == 85 )
            {
              v37 = *(_QWORD *)(v36 - 32);
              if ( v37 )
              {
                if ( !*(_BYTE *)v37
                  && *(_QWORD *)(v37 + 24) == *(_QWORD *)(v36 + 80)
                  && (*(_BYTE *)(v37 + 33) & 0x20) != 0
                  && (unsigned int)(*(_DWORD *)(v37 + 36) - 39) <= 1 )
                {
                  break;
                }
              }
            }
            v35 = *(_QWORD *)(v35 + 8);
            if ( !v35 )
              goto LABEL_76;
          }
          v10 = sub_BD5C60(*(_QWORD *)(v35 + 24));
          *(_QWORD *)(v36 + 72) = sub_A7A090((__int64 *)(v36 + 72), (__int64 *)v10, -1, 27);
          v35 = *(_QWORD *)(v35 + 8);
        }
        while ( v35 );
LABEL_76:
        v9 = v217;
        i = v194;
LABEL_77:
        v6 = sub_B43CB0((__int64)v219);
        v7 = *((_DWORD *)v11 - 5) & 0x7FFFFFF;
        v38 = &v219[4 * (2 - v7)];
        if ( *v38 )
        {
          v10 = v38[2];
          v7 = v38[1];
          *(_QWORD *)v10 = v7;
          if ( v7 )
          {
            v10 = v38[2];
            *(_QWORD *)(v7 + 16) = v10;
          }
        }
        *v38 = v6;
        v216 = v11 - 3;
        if ( v6 )
        {
          v7 = *(_QWORD *)(v6 + 16);
          v10 = v6 + 16;
          v38[1] = v7;
          if ( v7 )
            *(_QWORD *)(v7 + 16) = v38 + 1;
          v38[2] = v10;
          v11 = i;
          *(_QWORD *)(v6 + 16) = v38;
          v216 = v219;
        }
        else
        {
LABEL_36:
          v11 = i;
        }
      }
      if ( v216 )
      {
        v7 = (__int64)v226;
        v10 = (__int64)&v226[8 * (unsigned int)v227];
        if ( v226 != (_BYTE *)v10 )
        {
          do
          {
            v30 = *(_QWORD *)v7 - 32LL * (*(_DWORD *)(*(_QWORD *)v7 + 4LL) & 0x7FFFFFF);
            if ( *(_QWORD *)v30 )
            {
              v31 = *(_QWORD *)(v30 + 8);
              **(_QWORD **)(v30 + 16) = v31;
              if ( v31 )
                *(_QWORD *)(v31 + 16) = *(_QWORD *)(v30 + 16);
            }
            *(_QWORD *)v30 = v216;
            v6 = v216[2];
            *(_QWORD *)(v30 + 8) = v6;
            if ( v6 )
              *(_QWORD *)(v6 + 16) = v30 + 8;
            v7 += 8;
            *(_QWORD *)(v30 + 16) = v216 + 2;
            v216[2] = v30;
          }
          while ( v7 != v10 );
        }
      }
      if ( v215 )
      {
        if ( (*((_BYTE *)v218 - 54) & 1) != 0 )
        {
          sub_B2C6D0(v212, v10, v6, v7);
          v61 = v218[5];
          v62 = v61 + 40LL * v218[6];
          if ( (*((_BYTE *)v218 - 54) & 1) != 0 )
          {
            sub_B2C6D0(v212, v10, v6, v7);
            v61 = v218[5];
          }
        }
        else
        {
          v61 = v218[5];
          v62 = v61 + 40LL * v218[6];
        }
        for ( ; v62 != v61; v61 += 40 )
        {
          if ( (unsigned __int8)sub_B2D700(v61) )
            sub_B2D5C0(v61, 22);
        }
      }
      if ( v226 != v228 )
        _libc_free((unsigned __int64)v226);
      v218 = (_QWORD *)v218[1];
    }
    while ( v211 != v218 );
  }
  v253 = (__int64)v256;
  v256[0] = &unk_4F82408;
  v254 = 0x100000002LL;
  LODWORD(v255) = 0;
  BYTE4(v255) = 1;
  v257 = 0;
  v258 = (unsigned __int64)v261;
  v259 = 2;
  LODWORD(v260) = 0;
  BYTE4(v260) = 1;
  v252 = 1;
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v256, (__int64)&v252);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v261, (__int64)&v257);
  if ( BYTE4(v260) )
  {
    if ( BYTE4(v255) )
      goto LABEL_52;
  }
  else
  {
    _libc_free(v258);
    if ( BYTE4(v255) )
      goto LABEL_52;
  }
  _libc_free(v253);
LABEL_52:
  nullsub_61();
  v248 = &unk_49DA100;
  nullsub_63();
  if ( v233 != v235 )
    _libc_free((unsigned __int64)v233);
  return a1;
}
