// Function: sub_2942CE0
// Address: 0x2942ce0
//
__int64 __fastcall sub_2942CE0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __m128i *v8; // rsi
  __int64 v9; // rax
  __int64 v10; // r15
  __m128i *v11; // r14
  unsigned int v12; // r12d
  char v14; // al
  __int64 v15; // rax
  char v16; // al
  __int64 v17; // rdx
  __m128i v18; // xmm4
  unsigned __int64 *v19; // rdx
  __m128i *v20; // rdx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rbx
  __int32 v24; // r13d
  __m128i *v25; // rdx
  __m128i *v26; // rax
  __m128i *m; // rdx
  unsigned int v28; // ebx
  __m128i *v29; // rdx
  char v30; // al
  unsigned int v31; // r15d
  __int64 (__fastcall *v32)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v33; // r14
  __int64 v34; // rdx
  __int64 v35; // r12
  __int64 v36; // rax
  unsigned __int8 *v37; // r13
  __m128i v38; // rax
  __int64 v39; // rdx
  __m128i v40; // xmm6
  unsigned __int64 *v41; // rdx
  __m128i *v42; // rdx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rbx
  __int32 v46; // r13d
  __m128i *v47; // rdx
  __m128i *v48; // rax
  __m128i *n; // rdx
  unsigned int v50; // ebx
  __m128i *v51; // rdx
  char v52; // al
  __int16 v53; // si
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // r15
  __int64 v57; // rax
  __m128i v58; // rax
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rbx
  _QWORD *v62; // rax
  _BYTE *v63; // rdx
  __int32 v64; // r12d
  _QWORD *ii; // rdx
  unsigned __int32 v66; // r12d
  __m128i v67; // rax
  char v68; // al
  __m128i *v69; // rdx
  __int64 v70; // r15
  __int64 v71; // rax
  __int64 v72; // r13
  __int64 v73; // rax
  unsigned __int64 v74; // rdx
  __int64 v75; // r8
  __int64 v76; // r9
  __int64 v77; // rbx
  __int32 v78; // r13d
  _BYTE *v79; // rdx
  _QWORD *v80; // rax
  _QWORD *j; // rdx
  unsigned int v82; // r13d
  __m128i v83; // rax
  char v84; // al
  __m128i *v85; // rdx
  char v86; // bl
  unsigned __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // r12
  __int64 v90; // r14
  char v91; // r15
  _QWORD *v92; // rax
  __int64 v93; // rbx
  __int64 v94; // r14
  __int64 v95; // r12
  __int64 v96; // rdx
  unsigned int v97; // esi
  __m128i *v98; // rcx
  __int64 v99; // r9
  __int64 v100; // rbx
  __int32 v101; // r8d
  _BYTE *v102; // rdx
  _QWORD *v103; // rax
  _QWORD *jj; // rdx
  __int64 v105; // r13
  unsigned __int32 v106; // r15d
  __int64 v107; // rax
  __int64 v108; // r14
  signed int v109; // esi
  __int64 v110; // rax
  unsigned __int64 v111; // rdi
  unsigned __int64 v112; // r13
  __int64 v113; // rbx
  __int64 v114; // rax
  unsigned int v115; // edi
  __int64 v116; // r12
  unsigned int v117; // ebx
  __int64 v118; // rdx
  __m128i v119; // xmm1
  unsigned __int64 *v120; // rdx
  __int64 v121; // r8
  __int64 v122; // r9
  __int64 v123; // rbx
  __int32 v124; // r13d
  _BYTE *v125; // rdx
  _QWORD *v126; // rax
  _QWORD *nn; // rdx
  unsigned int v128; // r15d
  __m128i v129; // rax
  char v130; // al
  __m128i *v131; // rdx
  __int64 v132; // r13
  _QWORD *v133; // rax
  __int64 v134; // rbx
  __int64 v135; // r12
  __int64 v136; // r13
  __int64 v137; // rdx
  unsigned int v138; // esi
  __int64 v139; // rdx
  unsigned __int64 v140; // rdi
  __int64 v141; // rdx
  __m128i v142; // xmm2
  unsigned __int64 *v143; // rdx
  __int64 v144; // r8
  __int64 v145; // r9
  __int64 v146; // rbx
  __int32 v147; // r13d
  _BYTE *v148; // rdx
  _QWORD *v149; // rax
  _QWORD *i; // rdx
  unsigned int v151; // ebx
  __m128i *v152; // rdx
  char v153; // al
  __int64 v154; // rdx
  __int64 v155; // rcx
  __int64 v156; // r8
  unsigned __int8 *v157; // r13
  __int64 (__fastcall *v158)(__int64, unsigned int, _BYTE *); // rax
  unsigned int v159; // r12d
  __int64 v160; // r15
  __int64 v161; // rdx
  __m128i v162; // rax
  unsigned __int64 v163; // rbx
  __int64 v164; // rax
  unsigned __int64 v165; // rdx
  __int64 v166; // r8
  __int64 v167; // r9
  __int64 v168; // rbx
  __int32 v169; // r13d
  __m128i *v170; // rdx
  __m128i *v171; // rax
  __m128i *k; // rdx
  unsigned int v173; // r13d
  __int64 v174; // r12
  __int64 v175; // r14
  char v176; // si
  __int64 v177; // rcx
  unsigned __int64 v178; // rax
  _QWORD *v179; // rax
  __int64 v180; // r9
  __int64 v181; // rbx
  unsigned __int64 v182; // r14
  _BYTE *v183; // r12
  __int64 v184; // rdx
  unsigned int v185; // esi
  unsigned __int64 v186; // rdi
  __m128i v187; // xmm7
  __m128i v188; // xmm5
  __m128i v189; // xmm3
  __m128i v190; // xmm3
  __int64 v191; // rbx
  int v192; // edx
  unsigned int v193; // ecx
  unsigned __int8 v194; // al
  __int64 *v195; // rax
  int v196; // r15d
  __int64 v197; // r15
  __int64 v198; // rbx
  __int64 v199; // rdx
  unsigned int v200; // esi
  __int64 v201; // rdx
  __int64 v202; // r13
  __int64 v203; // rax
  __int64 v204; // r13
  __int64 v205; // rbx
  unsigned int v206; // r13d
  int v207; // eax
  unsigned int v208; // ecx
  __int64 v209; // rax
  __int64 v210; // rcx
  __int64 v211; // rcx
  __int64 v212; // r15
  __int64 v213; // r14
  int v214; // eax
  __int64 v215; // rdx
  int v216; // ecx
  int v217; // eax
  _QWORD *v218; // rdi
  __int64 *v219; // rax
  __int64 v220; // rsi
  unsigned __int64 v221; // r13
  _BYTE *v222; // r12
  __int64 v223; // rdx
  unsigned int v224; // esi
  __int64 v225; // r13
  __int64 v226; // r12
  __int64 v227; // rdx
  unsigned int v228; // esi
  __m128i *v229; // rcx
  int v230; // r13d
  __m128i v231; // xmm5
  __m128i v232; // xmm5
  __m128i v233; // xmm7
  __m128i v234; // xmm6
  __m128i v235; // xmm1
  __int64 v236; // rax
  __m128i v237; // xmm3
  unsigned int kk; // ebx
  __m128i *v239; // rdx
  char v240; // al
  __int64 v241; // rax
  __int64 v242; // r8
  __int64 v243; // r9
  __int64 v244; // rax
  unsigned __int64 v245; // rdx
  __m128i v246; // rax
  __m128i *v247; // rdi
  __m128i *v248; // rsi
  __int64 mm; // rcx
  __int64 v250; // [rsp+8h] [rbp-4D8h]
  __int64 v251; // [rsp+18h] [rbp-4C8h]
  __int64 v252; // [rsp+18h] [rbp-4C8h]
  __int64 v253; // [rsp+20h] [rbp-4C0h]
  __int64 v254; // [rsp+38h] [rbp-4A8h]
  __int64 v255; // [rsp+40h] [rbp-4A0h]
  __int64 v256; // [rsp+48h] [rbp-498h]
  __int64 v257; // [rsp+50h] [rbp-490h]
  __int64 v258; // [rsp+60h] [rbp-480h]
  __int64 v259; // [rsp+68h] [rbp-478h]
  __int64 v260; // [rsp+68h] [rbp-478h]
  __int64 v261; // [rsp+68h] [rbp-478h]
  __int64 v262; // [rsp+68h] [rbp-478h]
  __int64 v263; // [rsp+70h] [rbp-470h]
  unsigned __int64 v264; // [rsp+80h] [rbp-460h]
  __int64 v265; // [rsp+88h] [rbp-458h]
  __int64 v266; // [rsp+90h] [rbp-450h]
  __int64 v267; // [rsp+A0h] [rbp-440h]
  __int64 v268; // [rsp+A0h] [rbp-440h]
  __int64 v269; // [rsp+A0h] [rbp-440h]
  unsigned int v270; // [rsp+A8h] [rbp-438h]
  __m128i *v271; // [rsp+A8h] [rbp-438h]
  __int64 v272; // [rsp+A8h] [rbp-438h]
  __int64 v273; // [rsp+A8h] [rbp-438h]
  __int64 v274; // [rsp+B0h] [rbp-430h]
  __int64 v275; // [rsp+B0h] [rbp-430h]
  __int32 v276; // [rsp+B0h] [rbp-430h]
  _BYTE *v277; // [rsp+B8h] [rbp-428h]
  __int64 v278; // [rsp+C0h] [rbp-420h]
  __m128i v280; // [rsp+D0h] [rbp-410h] BYREF
  __m128i v281; // [rsp+E0h] [rbp-400h] BYREF
  __int64 v282; // [rsp+F0h] [rbp-3F0h]
  __int64 v283; // [rsp+108h] [rbp-3D8h]
  __m128i v284; // [rsp+110h] [rbp-3D0h] BYREF
  __m128i v285; // [rsp+120h] [rbp-3C0h] BYREF
  __int64 v286; // [rsp+130h] [rbp-3B0h]
  __m128i v287; // [rsp+140h] [rbp-3A0h] BYREF
  __m128i v288; // [rsp+150h] [rbp-390h] BYREF
  __int64 v289; // [rsp+160h] [rbp-380h]
  __m128i v290; // [rsp+170h] [rbp-370h] BYREF
  __m128i v291; // [rsp+180h] [rbp-360h] BYREF
  __int64 v292; // [rsp+190h] [rbp-350h]
  __m128i v293; // [rsp+1A0h] [rbp-340h] BYREF
  __m128i v294; // [rsp+1B0h] [rbp-330h] BYREF
  __int64 v295; // [rsp+1C0h] [rbp-320h]
  __m128i v296; // [rsp+1D0h] [rbp-310h] BYREF
  __m128i v297; // [rsp+1E0h] [rbp-300h] BYREF
  __int64 v298; // [rsp+1F0h] [rbp-2F0h]
  __m128i v299; // [rsp+200h] [rbp-2E0h] BYREF
  __m128i v300; // [rsp+210h] [rbp-2D0h] BYREF
  __int64 v301; // [rsp+220h] [rbp-2C0h]
  __int64 v302; // [rsp+228h] [rbp-2B8h]
  char v303; // [rsp+230h] [rbp-2B0h]
  _BYTE *v304; // [rsp+240h] [rbp-2A0h] BYREF
  __int64 v305; // [rsp+248h] [rbp-298h]
  _BYTE v306[64]; // [rsp+250h] [rbp-290h] BYREF
  __m128i v307; // [rsp+290h] [rbp-250h] BYREF
  __m128i v308; // [rsp+2A0h] [rbp-240h] BYREF
  __int64 v309; // [rsp+2B0h] [rbp-230h]
  __int64 v310; // [rsp+2B8h] [rbp-228h]
  char v311; // [rsp+2C0h] [rbp-220h]
  _BYTE *v312; // [rsp+2E0h] [rbp-200h] BYREF
  __int64 v313; // [rsp+2E8h] [rbp-1F8h]
  _BYTE v314[40]; // [rsp+2F0h] [rbp-1F0h] BYREF
  __int64 v315; // [rsp+318h] [rbp-1C8h]
  __int64 v316; // [rsp+320h] [rbp-1C0h]
  __int64 v317; // [rsp+330h] [rbp-1B0h]
  __int64 v318; // [rsp+338h] [rbp-1A8h]
  void *v319; // [rsp+360h] [rbp-180h]
  __m128i v320; // [rsp+370h] [rbp-170h] BYREF
  _BYTE v321[28]; // [rsp+380h] [rbp-160h] BYREF
  unsigned int v322; // [rsp+39Ch] [rbp-144h]
  __int64 v323; // [rsp+3A8h] [rbp-138h]
  __int64 v324; // [rsp+3B0h] [rbp-130h]
  __int64 *v325; // [rsp+3C0h] [rbp-120h]
  __int64 v326; // [rsp+3C8h] [rbp-118h]
  __int64 v327; // [rsp+3D0h] [rbp-110h] BYREF
  unsigned int v328; // [rsp+3D8h] [rbp-108h]
  void *v329; // [rsp+3F0h] [rbp-F0h]
  __m128i v330[2]; // [rsp+410h] [rbp-D0h] BYREF
  __int16 v331; // [rsp+430h] [rbp-B0h]
  unsigned int v332; // [rsp+43Ch] [rbp-A4h]
  _BYTE *v333; // [rsp+460h] [rbp-80h]
  _BYTE v334[112]; // [rsp+470h] [rbp-70h] BYREF

  v2 = a1;
  *(_BYTE *)(a1 + 320) = 0;
  v3 = *(_QWORD *)(a2 + 80);
  if ( v3 )
    v3 -= 24;
  v304 = v306;
  v305 = 0x800000000LL;
  sub_2942980((__int64)&v304, v3);
  v8 = &v296;
  v264 = (unsigned __int64)v304;
  v277 = &v304[8 * (unsigned int)v305];
  if ( v304 != v277 )
  {
    while ( 1 )
    {
      v9 = *((_QWORD *)v277 - 1);
      v10 = *(_QWORD *)(v9 + 56);
      v278 = v9 + 48;
      if ( v9 + 48 != v10 )
        break;
LABEL_11:
      v277 -= 8;
      if ( (_BYTE *)v264 == v277 )
      {
        v2 = a1;
        goto LABEL_13;
      }
    }
    while ( 2 )
    {
      if ( !v10 )
        BUG();
      v11 = (__m128i *)(v10 - 24);
      switch ( *(_BYTE *)(v10 - 24) )
      {
        case 0x1E:
        case 0x1F:
        case 0x20:
        case 0x21:
        case 0x22:
        case 0x23:
        case 0x24:
        case 0x25:
        case 0x26:
        case 0x27:
        case 0x28:
        case 0x3C:
        case 0x40:
        case 0x41:
        case 0x42:
        case 0x50:
        case 0x51:
        case 0x57:
        case 0x58:
        case 0x59:
        case 0x5E:
        case 0x5F:
          goto LABEL_9;
        case 0x29:
          if ( *(_DWORD *)(a1 + 1152) )
          {
            v8 = (__m128i *)(v10 - 24);
            if ( !sub_293A020(a1, (unsigned __int8 *)(v10 - 24)) )
              goto LABEL_9;
          }
          v8 = (__m128i *)a1;
          sub_2939E80((__int64)&v287, a1, *(_QWORD *)(v10 - 16));
          if ( !(_BYTE)v289 )
            goto LABEL_9;
          v292 = 0;
          v290 = 0;
          v291 = 0;
          if ( (*(_BYTE *)(v10 - 17) & 0x40) == 0 )
          {
            v141 = *(_QWORD *)(v11[-2 * (*(_DWORD *)(v10 - 20) & 0x7FFFFFF)].m128i_i64[0] + 8);
            if ( v141 == *(_QWORD *)(v10 - 16) )
              goto LABEL_270;
LABEL_220:
            v8 = (__m128i *)a1;
            sub_2939E80((__int64)&v280, a1, v141);
            v142 = _mm_loadu_si128(&v281);
            v290 = _mm_loadu_si128(&v280);
            v292 = v282;
            v291 = v142;
            if ( (_BYTE)v282 && v287.m128i_i32[2] == v290.m128i_i32[2] )
              goto LABEL_222;
            goto LABEL_9;
          }
          v141 = *(_QWORD *)(**(_QWORD **)(v10 - 32) + 8LL);
          if ( v141 != *(_QWORD *)(v10 - 16) )
            goto LABEL_220;
LABEL_270:
          v189 = _mm_loadu_si128(&v288);
          v290 = _mm_loadu_si128(&v287);
          v292 = v289;
          v291 = v189;
LABEL_222:
          sub_23D0AB0((__int64)&v320, v10 - 24, 0, 0, 0);
          if ( (*(_BYTE *)(v10 - 17) & 0x40) != 0 )
            v143 = *(unsigned __int64 **)(v10 - 32);
          else
            v143 = (unsigned __int64 *)&v11[-2 * (*(_DWORD *)(v10 - 20) & 0x7FFFFFF)];
          sub_293CE40(v330, (_QWORD *)a1, v10 - 24, *v143, &v290);
          v146 = v287.m128i_u32[3];
          v147 = v287.m128i_i32[3];
          v312 = v314;
          v313 = 0x800000000LL;
          if ( !v287.m128i_i32[3] )
            goto LABEL_208;
          v148 = v314;
          v149 = v314;
          if ( v287.m128i_u32[3] > 8uLL )
          {
            sub_C8D5F0((__int64)&v312, v314, v287.m128i_u32[3], 8u, v144, v145);
            v148 = v312;
            v149 = &v312[8 * (unsigned int)v313];
          }
          for ( i = &v148[8 * v146]; i != v149; ++v149 )
          {
            if ( v149 )
              *v149 = 0;
          }
          LODWORD(v313) = v147;
          if ( !v287.m128i_i32[3] )
            goto LABEL_208;
          v272 = v10;
          v151 = 0;
          while ( 1 )
          {
            v296.m128i_i32[0] = v151;
            LOWORD(v298) = 265;
            v162.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v11);
            LOWORD(v295) = 773;
            v293 = v162;
            v294.m128i_i64[0] = (__int64)".i";
            v153 = v298;
            if ( (_BYTE)v298 )
            {
              if ( (_BYTE)v298 == 1 )
              {
                v233 = _mm_loadu_si128(&v294);
                v299 = _mm_loadu_si128(&v293);
                v301 = v295;
                v300 = v233;
              }
              else
              {
                if ( BYTE1(v298) == 1 )
                {
                  v254 = v296.m128i_i64[1];
                  v152 = (__m128i *)v296.m128i_i64[0];
                }
                else
                {
                  v152 = &v296;
                  v153 = 2;
                }
                v300.m128i_i64[0] = (__int64)v152;
                LOBYTE(v301) = 2;
                v299.m128i_i64[0] = (__int64)&v293;
                BYTE1(v301) = v153;
                v300.m128i_i64[1] = v254;
              }
            }
            else
            {
              LOWORD(v301) = 256;
            }
            v157 = (unsigned __int8 *)sub_293BC00((__int64)v330, v151);
            v158 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *))(*v325 + 48);
            v159 = *(unsigned __int8 *)(v272 - 24) - 29;
            if ( v158 == sub_9288C0 )
            {
              if ( *v157 > 0x15u )
                goto LABEL_324;
              v160 = sub_AAAFF0(v159, v157, v154, v155, v156);
            }
            else
            {
              v160 = ((__int64 (__fastcall *)(__int64 *, _QWORD, unsigned __int8 *, _QWORD))v158)(
                       v325,
                       v159,
                       v157,
                       v328);
            }
            if ( !v160 )
            {
LABEL_324:
              LOWORD(v309) = 257;
              v160 = sub_B50340(v159, (__int64)v157, (__int64)&v307, 0, 0);
              if ( (unsigned __int8)sub_920620(v160) )
              {
                v230 = v328;
                if ( v327 )
                  sub_B99FD0(v160, 3u, v327);
                sub_B45150(v160, v230);
              }
              (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)v326 + 16LL))(
                v326,
                v160,
                &v299,
                v323,
                v324);
              v225 = v320.m128i_i64[0] + 16LL * v320.m128i_u32[2];
              if ( v320.m128i_i64[0] != v225 )
              {
                v226 = v320.m128i_i64[0];
                do
                {
                  v227 = *(_QWORD *)(v226 + 8);
                  v228 = *(_DWORD *)v226;
                  v226 += 16;
                  sub_B99FD0(v160, v228, v227);
                }
                while ( v225 != v226 );
              }
            }
            v161 = v151++;
            *(_QWORD *)&v312[8 * v161] = v160;
            if ( v287.m128i_i32[3] <= v151 )
            {
              v10 = v272;
              goto LABEL_208;
            }
          }
        case 0x2A:
        case 0x2B:
        case 0x2C:
        case 0x2D:
        case 0x2E:
        case 0x2F:
        case 0x30:
        case 0x31:
        case 0x32:
        case 0x33:
        case 0x34:
        case 0x35:
        case 0x36:
        case 0x37:
        case 0x38:
        case 0x39:
        case 0x3A:
        case 0x3B:
          v8 = (__m128i *)(v10 - 24);
          v330[0].m128i_i64[0] = v10 - 24;
          v14 = sub_293D130(a1, v10 - 24, (unsigned __int8 **)v330);
          goto LABEL_17;
        case 0x3D:
          if ( *(_DWORD *)(a1 + 1152) )
          {
            v8 = (__m128i *)(v10 - 24);
            if ( !sub_293A020(a1, (unsigned __int8 *)(v10 - 24)) )
              goto LABEL_9;
          }
          if ( !*(_BYTE *)(a1 + 1129) )
            goto LABEL_9;
          if ( sub_B46500((unsigned __int8 *)(v10 - 24)) )
            goto LABEL_9;
          if ( (*(_BYTE *)(v10 - 22) & 1) != 0 )
            goto LABEL_9;
          v73 = sub_B43CC0(v10 - 24);
          v8 = (__m128i *)a1;
          _BitScanReverse64(&v74, 1LL << (*(_WORD *)(v10 - 22) >> 1));
          sub_293B690((__int64)&v307, a1, *(_QWORD *)(v10 - 16), 63 - (v74 ^ 0x3F), v73);
          if ( !v311 )
            goto LABEL_9;
          sub_23D0AB0((__int64)&v320, v10 - 24, 0, 0, 0);
          sub_293CE40(v330, (_QWORD *)a1, v10 - 24, *(_QWORD *)(v10 - 56), &v307);
          v77 = v307.m128i_u32[3];
          v78 = v307.m128i_i32[3];
          v312 = v314;
          v313 = 0x800000000LL;
          if ( v307.m128i_i32[3] )
          {
            v79 = v314;
            v80 = v314;
            if ( v307.m128i_u32[3] > 8uLL )
            {
              sub_C8D5F0((__int64)&v312, v314, v307.m128i_u32[3], 8u, v75, v76);
              v79 = v312;
              v80 = &v312[8 * (unsigned int)v313];
            }
            for ( j = &v79[8 * v77]; j != v80; ++v80 )
            {
              if ( v80 )
                *v80 = 0;
            }
            LODWORD(v313) = v78;
            if ( v307.m128i_i32[3] )
            {
              v82 = 0;
              v260 = v10 - 24;
              v251 = v10;
              do
              {
                v293.m128i_i32[0] = v82;
                LOWORD(v295) = 265;
                v83.m128i_i64[0] = (__int64)sub_BD5D20(v260);
                v290 = v83;
                LOWORD(v292) = 773;
                v291.m128i_i64[0] = (__int64)".i";
                v84 = v295;
                if ( (_BYTE)v295 )
                {
                  if ( (_BYTE)v295 == 1 )
                  {
                    v237 = _mm_loadu_si128(&v291);
                    v296 = _mm_loadu_si128(&v290);
                    v298 = v292;
                    v297 = v237;
                  }
                  else
                  {
                    if ( BYTE1(v295) == 1 )
                    {
                      v253 = v293.m128i_i64[1];
                      v85 = (__m128i *)v293.m128i_i64[0];
                    }
                    else
                    {
                      v85 = &v293;
                      v84 = 2;
                    }
                    v297.m128i_i64[0] = (__int64)v85;
                    LOBYTE(v298) = 2;
                    v296.m128i_i64[0] = (__int64)&v290;
                    BYTE1(v298) = v84;
                    v297.m128i_i64[1] = v253;
                  }
                }
                else
                {
                  LOWORD(v298) = 256;
                }
                v86 = -1;
                v269 = v82;
                v87 = ((v310 * v82) | (1LL << v309)) & -((v310 * v82) | (1LL << v309));
                if ( v87 )
                {
                  _BitScanReverse64(&v87, v87);
                  v86 = 63 - (v87 ^ 0x3F);
                }
                v88 = sub_293BC00((__int64)v330, v82);
                v89 = v308.m128i_i64[1];
                v90 = v88;
                if ( !v308.m128i_i64[1] || v307.m128i_i32[3] - 1 != v82 )
                  v89 = v308.m128i_i64[0];
                v91 = v86;
                LOWORD(v301) = 257;
                v92 = sub_BD2C40(80, 1u);
                v93 = (__int64)v92;
                if ( v92 )
                  sub_B4D190((__int64)v92, v89, v90, (__int64)&v299, 0, v91, 0, 0);
                (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)v326 + 16LL))(
                  v326,
                  v93,
                  &v296,
                  v323,
                  v324);
                v94 = v320.m128i_i64[0];
                v95 = v320.m128i_i64[0] + 16LL * v320.m128i_u32[2];
                if ( v320.m128i_i64[0] != v95 )
                {
                  do
                  {
                    v96 = *(_QWORD *)(v94 + 8);
                    v97 = *(_DWORD *)v94;
                    v94 += 16;
                    sub_B99FD0(v93, v97, v96);
                  }
                  while ( v95 != v94 );
                }
                ++v82;
                *(_QWORD *)&v312[8 * v269] = v93;
              }
              while ( v307.m128i_i32[3] > v82 );
              v11 = (__m128i *)v260;
              v10 = v251;
            }
          }
          v98 = &v307;
          goto LABEL_209;
        case 0x3E:
          if ( !*(_BYTE *)(a1 + 1129) )
            goto LABEL_9;
          if ( sub_B46500((unsigned __int8 *)(v10 - 24)) )
            goto LABEL_9;
          if ( (*(_BYTE *)(v10 - 22) & 1) != 0 )
            goto LABEL_9;
          v163 = *(_QWORD *)(v10 - 88);
          v164 = sub_B43CC0(v10 - 24);
          v8 = (__m128i *)a1;
          _BitScanReverse64(&v165, 1LL << (*(_WORD *)(v10 - 22) >> 1));
          sub_293B690((__int64)&v299, a1, *(_QWORD *)(v163 + 8), 63 - (v165 ^ 0x3F), v164);
          if ( !v303 )
            goto LABEL_9;
          sub_23D0AB0((__int64)&v312, v10 - 24, 0, 0, 0);
          sub_293CE40(&v320, (_QWORD *)a1, v10 - 24, *(_QWORD *)(v10 - 56), &v299);
          sub_293CE40(v330, (_QWORD *)a1, v10 - 24, v163, &v299);
          v168 = v299.m128i_u32[3];
          v169 = v299.m128i_i32[3];
          v307.m128i_i64[0] = (__int64)&v308;
          v307.m128i_i64[1] = 0x800000000LL;
          if ( v299.m128i_i32[3] )
          {
            v170 = &v308;
            v171 = &v308;
            if ( v299.m128i_u32[3] > 8uLL )
            {
              sub_C8D5F0((__int64)&v307, &v308, v299.m128i_u32[3], 8u, v166, v167);
              v170 = (__m128i *)v307.m128i_i64[0];
              v171 = (__m128i *)(v307.m128i_i64[0] + 8LL * v307.m128i_u32[2]);
            }
            for ( k = (__m128i *)((char *)v170 + 8 * v168); k != v171; v171 = (__m128i *)((char *)v171 + 8) )
            {
              if ( v171 )
                v171->m128i_i64[0] = 0;
            }
            v307.m128i_i32[2] = v169;
            if ( v299.m128i_i32[3] )
            {
              v173 = 0;
              do
              {
                v174 = sub_293BC00((__int64)v330, v173);
                v175 = sub_293BC00((__int64)&v320, v173);
                v274 = v173;
                v176 = -1;
                v177 = (v302 * v173) | (1LL << v301);
                if ( (v177 & -v177) != 0 )
                {
                  _BitScanReverse64(&v178, v177 & -v177);
                  v176 = 63 - (v178 ^ 0x3F);
                }
                LOWORD(v298) = 257;
                v179 = sub_BD2C40(80, unk_3F10A10);
                v181 = (__int64)v179;
                if ( v179 )
                  sub_B4D3C0((__int64)v179, v174, v175, 0, v176, v180, 0, 0);
                (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)v318 + 16LL))(
                  v318,
                  v181,
                  &v296,
                  v315,
                  v316);
                v182 = (unsigned __int64)v312;
                v183 = &v312[16 * (unsigned int)v313];
                if ( v312 != v183 )
                {
                  do
                  {
                    v184 = *(_QWORD *)(v182 + 8);
                    v185 = *(_DWORD *)v182;
                    v182 += 16LL;
                    sub_B99FD0(v181, v185, v184);
                  }
                  while ( v183 != (_BYTE *)v182 );
                }
                ++v173;
                *(_QWORD *)(v307.m128i_i64[0] + 8 * v274) = v181;
              }
              while ( v299.m128i_i32[3] > v173 );
              v11 = (__m128i *)(v10 - 24);
            }
          }
          v8 = &v307;
          sub_293A860((__int64)v11, (__int64)&v307);
          v186 = v307.m128i_i64[0];
          if ( (__m128i *)v307.m128i_i64[0] == &v308 )
            goto LABEL_335;
          goto LABEL_334;
        case 0x3F:
          v8 = (__m128i *)(v10 - 24);
          v14 = sub_293D7E0(a1, v10 - 24);
          goto LABEL_17;
        case 0x43:
        case 0x44:
        case 0x45:
        case 0x46:
        case 0x47:
        case 0x48:
        case 0x49:
        case 0x4A:
        case 0x4B:
        case 0x4C:
        case 0x4D:
        case 0x4F:
          v8 = (__m128i *)(v10 - 24);
          v16 = sub_293E340(a1, (unsigned __int8 *)(v10 - 24));
          v4 = *(_QWORD *)(v10 + 8);
          if ( !v16 )
            goto LABEL_21;
          goto LABEL_18;
        case 0x4E:
          v8 = (__m128i *)(v10 - 24);
          v14 = sub_293E8A0(a1, v10 - 24);
          goto LABEL_17;
        case 0x52:
          if ( *(_DWORD *)(a1 + 1152) )
          {
            v8 = (__m128i *)(v10 - 24);
            if ( !sub_293A020(a1, (unsigned __int8 *)(v10 - 24)) )
              goto LABEL_9;
          }
          v8 = (__m128i *)a1;
          sub_2939E80((__int64)&v284, a1, *(_QWORD *)(v10 - 16));
          if ( !(_BYTE)v286 )
            goto LABEL_9;
          v289 = 0;
          v287 = 0;
          v288 = 0;
          if ( (*(_BYTE *)(v10 - 17) & 0x40) != 0 )
          {
            v17 = *(_QWORD *)(**(_QWORD **)(v10 - 32) + 8LL);
            if ( v17 == *(_QWORD *)(v10 - 16) )
            {
LABEL_268:
              v188 = _mm_loadu_si128(&v285);
              v287 = _mm_loadu_si128(&v284);
              v289 = v286;
              v288 = v188;
LABEL_28:
              sub_23D0AB0((__int64)&v312, v10 - 24, 0, 0, 0);
              if ( (*(_BYTE *)(v10 - 17) & 0x40) != 0 )
                v19 = *(unsigned __int64 **)(v10 - 32);
              else
                v19 = (unsigned __int64 *)&v11[-2 * (*(_DWORD *)(v10 - 20) & 0x7FFFFFF)];
              sub_293CE40(&v320, (_QWORD *)a1, v10 - 24, *v19, &v287);
              if ( (*(_BYTE *)(v10 - 17) & 0x40) != 0 )
                v20 = *(__m128i **)(v10 - 32);
              else
                v20 = &v11[-2 * (*(_DWORD *)(v10 - 20) & 0x7FFFFFF)];
              sub_293CE40(v330, (_QWORD *)a1, v10 - 24, v20[2].m128i_u64[0], &v287);
              v23 = v284.m128i_u32[3];
              v24 = v284.m128i_i32[3];
              v307.m128i_i64[0] = (__int64)&v308;
              v307.m128i_i64[1] = 0x800000000LL;
              if ( !v284.m128i_i32[3] )
                goto LABEL_332;
              v25 = &v308;
              v26 = &v308;
              if ( v284.m128i_u32[3] > 8uLL )
              {
                sub_C8D5F0((__int64)&v307, &v308, v284.m128i_u32[3], 8u, v21, v22);
                v25 = (__m128i *)v307.m128i_i64[0];
                v26 = (__m128i *)(v307.m128i_i64[0] + 8LL * v307.m128i_u32[2]);
              }
              for ( m = (__m128i *)((char *)v25 + 8 * v23); m != v26; v26 = (__m128i *)((char *)v26 + 8) )
              {
                if ( v26 )
                  v26->m128i_i64[0] = 0;
              }
              v307.m128i_i32[2] = v24;
              if ( !v284.m128i_i32[3] )
                goto LABEL_332;
              v267 = v10;
              v28 = 0;
              v265 = v10 - 24;
              while ( 2 )
              {
                v35 = sub_293BC00((__int64)&v320, v28);
                v36 = sub_293BC00((__int64)v330, v28);
                v293.m128i_i32[0] = v28;
                LOWORD(v295) = 265;
                v37 = (unsigned __int8 *)v36;
                v38.m128i_i64[0] = (__int64)sub_BD5D20(v265);
                v290 = v38;
                LOWORD(v292) = 773;
                v291.m128i_i64[0] = (__int64)".i";
                v30 = v295;
                if ( (_BYTE)v295 )
                {
                  if ( (_BYTE)v295 == 1 )
                  {
                    v234 = _mm_loadu_si128(&v291);
                    v296 = _mm_loadu_si128(&v290);
                    v298 = v292;
                    v297 = v234;
                  }
                  else
                  {
                    if ( BYTE1(v295) == 1 )
                    {
                      v255 = v293.m128i_i64[1];
                      v29 = (__m128i *)v293.m128i_i64[0];
                    }
                    else
                    {
                      v29 = &v293;
                      v30 = 2;
                    }
                    v297.m128i_i64[0] = (__int64)v29;
                    LOBYTE(v298) = 2;
                    v296.m128i_i64[0] = (__int64)&v290;
                    BYTE1(v298) = v30;
                    v297.m128i_i64[1] = v255;
                  }
                }
                else
                {
                  LOWORD(v298) = 256;
                }
                v31 = *(_WORD *)(v267 - 22) & 0x3F;
                v32 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v317
                                                                                                  + 56LL);
                if ( v32 == sub_928890 )
                {
                  if ( *(_BYTE *)v35 <= 0x15u && *v37 <= 0x15u )
                  {
                    v33 = sub_AAB310(v31, (unsigned __int8 *)v35, v37);
                    goto LABEL_49;
                  }
                  goto LABEL_317;
                }
                v33 = v32(v317, v31, (_BYTE *)v35, v37);
LABEL_49:
                if ( !v33 )
                {
LABEL_317:
                  LOWORD(v301) = 257;
                  v33 = (__int64)sub_BD2C40(72, unk_3F10FD0);
                  if ( v33 )
                  {
                    v215 = *(_QWORD *)(v35 + 8);
                    v216 = *(unsigned __int8 *)(v215 + 8);
                    if ( (unsigned int)(v216 - 17) > 1 )
                    {
                      v220 = sub_BCB2A0(*(_QWORD **)v215);
                    }
                    else
                    {
                      v217 = *(_DWORD *)(v215 + 32);
                      v218 = *(_QWORD **)v215;
                      BYTE4(v283) = (_BYTE)v216 == 18;
                      LODWORD(v283) = v217;
                      v219 = (__int64 *)sub_BCB2A0(v218);
                      v220 = sub_BCE1B0(v219, v283);
                    }
                    sub_B523C0(v33, v220, 53, v31, v35, (__int64)v37, (__int64)&v299, 0, 0, 0);
                  }
                  (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)v318 + 16LL))(
                    v318,
                    v33,
                    &v296,
                    v315,
                    v316);
                  v221 = (unsigned __int64)v312;
                  v222 = &v312[16 * (unsigned int)v313];
                  if ( v312 != v222 )
                  {
                    do
                    {
                      v223 = *(_QWORD *)(v221 + 8);
                      v224 = *(_DWORD *)v221;
                      v221 += 16LL;
                      sub_B99FD0(v33, v224, v223);
                    }
                    while ( v222 != (_BYTE *)v221 );
                  }
                }
                v34 = v28++;
                *(_QWORD *)(v307.m128i_i64[0] + 8 * v34) = v33;
                if ( v284.m128i_i32[3] <= v28 )
                {
                  v10 = v267;
                  v11 = (__m128i *)v265;
LABEL_332:
                  v229 = &v284;
                  goto LABEL_333;
                }
                continue;
              }
            }
          }
          else
          {
            v17 = *(_QWORD *)(v11[-2 * (*(_DWORD *)(v10 - 20) & 0x7FFFFFF)].m128i_i64[0] + 8);
            if ( v17 == *(_QWORD *)(v10 - 16) )
              goto LABEL_268;
          }
          v8 = (__m128i *)a1;
          sub_2939E80((__int64)&v280, a1, v17);
          v18 = _mm_loadu_si128(&v281);
          v287 = _mm_loadu_si128(&v280);
          v289 = v282;
          v288 = v18;
          if ( (_BYTE)v282 && v284.m128i_i32[2] == v287.m128i_i32[2] )
            goto LABEL_28;
LABEL_9:
          v10 = *(_QWORD *)(v10 + 8);
LABEL_10:
          if ( v278 == v10 )
            goto LABEL_11;
          continue;
        case 0x53:
          if ( *(_DWORD *)(a1 + 1152) )
          {
            v8 = (__m128i *)(v10 - 24);
            if ( !sub_293A020(a1, (unsigned __int8 *)(v10 - 24)) )
              goto LABEL_9;
          }
          v8 = (__m128i *)a1;
          sub_2939E80((__int64)&v287, a1, *(_QWORD *)(v10 - 16));
          if ( !(_BYTE)v289 )
            goto LABEL_9;
          v292 = 0;
          v290 = 0;
          v291 = 0;
          if ( (*(_BYTE *)(v10 - 17) & 0x40) != 0 )
          {
            v39 = *(_QWORD *)(**(_QWORD **)(v10 - 32) + 8LL);
            if ( v39 == *(_QWORD *)(v10 - 16) )
            {
LABEL_266:
              v187 = _mm_loadu_si128(&v288);
              v290 = _mm_loadu_si128(&v287);
              v292 = v289;
              v291 = v187;
              goto LABEL_59;
            }
          }
          else
          {
            v39 = *(_QWORD *)(v11[-2 * (*(_DWORD *)(v10 - 20) & 0x7FFFFFF)].m128i_i64[0] + 8);
            if ( v39 == *(_QWORD *)(v10 - 16) )
              goto LABEL_266;
          }
          v8 = (__m128i *)a1;
          sub_2939E80((__int64)&v280, a1, v39);
          v40 = _mm_loadu_si128(&v281);
          v290 = _mm_loadu_si128(&v280);
          v292 = v282;
          v291 = v40;
          if ( !(_BYTE)v282 || v287.m128i_i32[2] != v290.m128i_i32[2] )
            goto LABEL_9;
LABEL_59:
          sub_23D0AB0((__int64)&v312, v10 - 24, 0, 0, 0);
          if ( (*(_BYTE *)(v10 - 17) & 0x40) != 0 )
            v41 = *(unsigned __int64 **)(v10 - 32);
          else
            v41 = (unsigned __int64 *)&v11[-2 * (*(_DWORD *)(v10 - 20) & 0x7FFFFFF)];
          sub_293CE40(&v320, (_QWORD *)a1, v10 - 24, *v41, &v290);
          if ( (*(_BYTE *)(v10 - 17) & 0x40) != 0 )
            v42 = *(__m128i **)(v10 - 32);
          else
            v42 = &v11[-2 * (*(_DWORD *)(v10 - 20) & 0x7FFFFFF)];
          sub_293CE40(v330, (_QWORD *)a1, v10 - 24, v42[2].m128i_u64[0], &v290);
          v45 = v287.m128i_u32[3];
          v46 = v287.m128i_i32[3];
          v307.m128i_i64[0] = (__int64)&v308;
          v307.m128i_i64[1] = 0x800000000LL;
          if ( v287.m128i_i32[3] )
          {
            v47 = &v308;
            v48 = &v308;
            if ( v287.m128i_u32[3] > 8uLL )
            {
              sub_C8D5F0((__int64)&v307, &v308, v287.m128i_u32[3], 8u, v43, v44);
              v47 = (__m128i *)v307.m128i_i64[0];
              v48 = (__m128i *)(v307.m128i_i64[0] + 8LL * v307.m128i_u32[2]);
            }
            for ( n = (__m128i *)((char *)v47 + 8 * v45); n != v48; v48 = (__m128i *)((char *)v48 + 8) )
            {
              if ( v48 )
                v48->m128i_i64[0] = 0;
            }
            v307.m128i_i32[2] = v46;
            if ( v287.m128i_i32[3] )
            {
              v263 = v10;
              v50 = 0;
              do
              {
                v56 = sub_293BC00((__int64)&v320, v50);
                v57 = sub_293BC00((__int64)v330, v50);
                v296.m128i_i32[0] = v50;
                v268 = v57;
                LOWORD(v298) = 265;
                v58.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v11);
                v293 = v58;
                LOWORD(v295) = 773;
                v294.m128i_i64[0] = (__int64)".i";
                v52 = v298;
                if ( (_BYTE)v298 )
                {
                  if ( (_BYTE)v298 == 1 )
                  {
                    v235 = _mm_loadu_si128(&v294);
                    v299 = _mm_loadu_si128(&v293);
                    v301 = v295;
                    v300 = v235;
                  }
                  else
                  {
                    if ( BYTE1(v298) == 1 )
                    {
                      v257 = v296.m128i_i64[1];
                      v51 = (__m128i *)v296.m128i_i64[0];
                    }
                    else
                    {
                      v51 = &v296;
                      v52 = 2;
                    }
                    v300.m128i_i64[0] = (__int64)v51;
                    LOBYTE(v301) = 2;
                    v299.m128i_i64[0] = (__int64)&v293;
                    v300.m128i_i64[1] = v257;
                    BYTE1(v301) = v52;
                  }
                }
                else
                {
                  LOWORD(v301) = 256;
                }
                v53 = *(_WORD *)(v263 - 22);
                v284.m128i_i32[1] = 0;
                v54 = sub_B35C90((__int64)&v312, v53 & 0x3F, v56, v268, (__int64)&v299, 0, v284.m128i_u32[0], 0);
                v55 = v50++;
                *(_QWORD *)(v307.m128i_i64[0] + 8 * v55) = v54;
              }
              while ( v287.m128i_i32[3] > v50 );
              v10 = v263;
            }
          }
          v229 = &v287;
LABEL_333:
          v8 = v11;
          sub_293CAB0(a1, (unsigned __int64)v11, (__int64)&v307, (__int64)v229);
          v186 = v307.m128i_i64[0];
          if ( (__m128i *)v307.m128i_i64[0] != &v308 )
LABEL_334:
            _libc_free(v186);
LABEL_335:
          if ( v333 != v334 )
            _libc_free((unsigned __int64)v333);
          if ( v325 != &v327 )
            _libc_free((unsigned __int64)v325);
          nullsub_61();
          v319 = &unk_49DA100;
          nullsub_63();
          v111 = (unsigned __int64)v312;
          if ( v312 != v314 )
            goto LABEL_214;
LABEL_215:
          v4 = *(_QWORD *)(v10 + 8);
LABEL_18:
          v15 = *(_QWORD *)(v10 - 16);
          v10 = v4;
          if ( *(_BYTE *)(v15 + 8) == 7 )
            sub_B43D60(v11);
          goto LABEL_10;
        case 0x54:
          v8 = (__m128i *)a1;
          sub_2939E80((__int64)&v293, a1, *(_QWORD *)(v10 - 16));
          if ( !(_BYTE)v295 )
            goto LABEL_9;
          sub_23D0AB0((__int64)&v320, v10 - 24, 0, 0, 0);
          v61 = v293.m128i_u32[3];
          v62 = v314;
          v63 = v314;
          v64 = v293.m128i_i32[3];
          v312 = v314;
          v313 = 0x800000000LL;
          if ( v293.m128i_i32[3] )
          {
            if ( v293.m128i_u32[3] > 8uLL )
            {
              sub_C8D5F0((__int64)&v312, v314, v293.m128i_u32[3], 8u, v59, v60);
              v63 = v312;
              v62 = &v312[8 * (unsigned int)v313];
            }
            for ( ii = &v63[8 * v61]; ii != v62; ++v62 )
            {
              if ( v62 )
                *v62 = 0;
            }
            LODWORD(v313) = v64;
            v270 = *(_DWORD *)(v10 - 20) & 0x7FFFFFF;
            if ( v293.m128i_i32[3] )
            {
              v259 = v10;
              v66 = 0;
              do
              {
                v299.m128i_i32[0] = v66;
                LOWORD(v301) = 265;
                v67.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v11);
                v296 = v67;
                LOWORD(v298) = 773;
                v297.m128i_i64[0] = (__int64)".i";
                v68 = v301;
                if ( (_BYTE)v301 )
                {
                  if ( (_BYTE)v301 == 1 )
                  {
                    v231 = _mm_loadu_si128(&v297);
                    v307 = _mm_loadu_si128(&v296);
                    v309 = v298;
                    v308 = v231;
                  }
                  else
                  {
                    if ( BYTE1(v301) == 1 )
                    {
                      v258 = v299.m128i_i64[1];
                      v69 = (__m128i *)v299.m128i_i64[0];
                    }
                    else
                    {
                      v69 = &v299;
                      v68 = 2;
                    }
                    v308.m128i_i64[0] = (__int64)v69;
                    LOBYTE(v309) = 2;
                    v307.m128i_i64[0] = (__int64)&v296;
                    BYTE1(v309) = v68;
                    v308.m128i_i64[1] = v258;
                  }
                }
                else
                {
                  LOWORD(v309) = 256;
                }
                v70 = v294.m128i_i64[1];
                if ( !v294.m128i_i64[1] || v293.m128i_i32[3] - 1 != v66 )
                  v70 = v294.m128i_i64[0];
                v331 = 257;
                v71 = sub_BD2DA0(80);
                v72 = v71;
                if ( v71 )
                {
                  sub_B44260(v71, v70, 55, 0x8000000u, 0, 0);
                  *(_DWORD *)(v72 + 72) = v270;
                  sub_BD6B50((unsigned __int8 *)v72, (const char **)v330);
                  sub_BD2A10(v72, *(_DWORD *)(v72 + 72), 1);
                }
                if ( *(_BYTE *)v72 > 0x1Cu )
                {
                  switch ( *(_BYTE *)v72 )
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
                      goto LABEL_287;
                    case 'T':
                    case 'U':
                    case 'V':
                      v191 = *(_QWORD *)(v72 + 8);
                      v192 = *(unsigned __int8 *)(v191 + 8);
                      v193 = v192 - 17;
                      v194 = *(_BYTE *)(v191 + 8);
                      if ( (unsigned int)(v192 - 17) <= 1 )
                        v194 = *(_BYTE *)(**(_QWORD **)(v191 + 16) + 8LL);
                      if ( v194 <= 3u || v194 == 5 || (v194 & 0xFD) == 4 )
                        goto LABEL_287;
                      if ( (_BYTE)v192 == 15 )
                      {
                        if ( (*(_BYTE *)(v191 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(v72 + 8)) )
                          break;
                        v195 = *(__int64 **)(v191 + 16);
                        v191 = *v195;
                        v192 = *(unsigned __int8 *)(*v195 + 8);
                        v193 = v192 - 17;
                      }
                      else if ( (_BYTE)v192 == 16 )
                      {
                        do
                        {
                          v191 = *(_QWORD *)(v191 + 24);
                          LOBYTE(v192) = *(_BYTE *)(v191 + 8);
                        }
                        while ( (_BYTE)v192 == 16 );
                        v193 = (unsigned __int8)v192 - 17;
                      }
                      if ( v193 <= 1 )
                        LOBYTE(v192) = *(_BYTE *)(**(_QWORD **)(v191 + 16) + 8LL);
                      if ( (unsigned __int8)v192 <= 3u || (_BYTE)v192 == 5 || (v192 & 0xFD) == 4 )
                      {
LABEL_287:
                        v196 = v328;
                        if ( v327 )
                          sub_B99FD0(v72, 3u, v327);
                        sub_B45150(v72, v196);
                      }
                      break;
                    default:
                      break;
                  }
                }
                (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)v326 + 16LL))(
                  v326,
                  v72,
                  &v307,
                  v323,
                  v324);
                v197 = v320.m128i_i64[0];
                v198 = v320.m128i_i64[0] + 16LL * v320.m128i_u32[2];
                if ( v320.m128i_i64[0] != v198 )
                {
                  do
                  {
                    v199 = *(_QWORD *)(v197 + 8);
                    v200 = *(_DWORD *)v197;
                    v197 += 16;
                    sub_B99FD0(v72, v200, v199);
                  }
                  while ( v198 != v197 );
                }
                v201 = v66++;
                *(_QWORD *)&v312[8 * v201] = v72;
              }
              while ( v293.m128i_i32[3] > v66 );
              v10 = v259;
            }
          }
          else
          {
            v270 = *(_DWORD *)(v10 - 20) & 0x7FFFFFF;
          }
          if ( v270 )
          {
            v202 = v270;
            v266 = (__int64)v11;
            v273 = v10;
            v203 = 8 * v202;
            v204 = 0;
            v261 = v203;
            do
            {
              sub_293CE40(v330, (_QWORD *)a1, v266, *(_QWORD *)(*(_QWORD *)(v273 - 32) + 4 * v204), &v293);
              v205 = *(_QWORD *)(*(_QWORD *)(v273 - 32) + 32LL * *(unsigned int *)(v273 + 48) + v204);
              if ( v293.m128i_i32[3] )
              {
                v275 = v204;
                v206 = 0;
                do
                {
                  v212 = *(_QWORD *)&v312[8 * v206];
                  v213 = sub_293BC00((__int64)v330, v206);
                  v214 = *(_DWORD *)(v212 + 4) & 0x7FFFFFF;
                  if ( v214 == *(_DWORD *)(v212 + 72) )
                  {
                    sub_B48D90(v212);
                    v214 = *(_DWORD *)(v212 + 4) & 0x7FFFFFF;
                  }
                  v207 = (v214 + 1) & 0x7FFFFFF;
                  v208 = v207 | *(_DWORD *)(v212 + 4) & 0xF8000000;
                  v209 = *(_QWORD *)(v212 - 8) + 32LL * (unsigned int)(v207 - 1);
                  *(_DWORD *)(v212 + 4) = v208;
                  if ( *(_QWORD *)v209 )
                  {
                    v210 = *(_QWORD *)(v209 + 8);
                    **(_QWORD **)(v209 + 16) = v210;
                    if ( v210 )
                      *(_QWORD *)(v210 + 16) = *(_QWORD *)(v209 + 16);
                  }
                  *(_QWORD *)v209 = v213;
                  if ( v213 )
                  {
                    v211 = *(_QWORD *)(v213 + 16);
                    *(_QWORD *)(v209 + 8) = v211;
                    if ( v211 )
                      *(_QWORD *)(v211 + 16) = v209 + 8;
                    *(_QWORD *)(v209 + 16) = v213 + 16;
                    *(_QWORD *)(v213 + 16) = v209;
                  }
                  ++v206;
                  *(_QWORD *)(*(_QWORD *)(v212 - 8)
                            + 32LL * *(unsigned int *)(v212 + 72)
                            + 8LL * ((*(_DWORD *)(v212 + 4) & 0x7FFFFFFu) - 1)) = v205;
                }
                while ( v293.m128i_i32[3] > v206 );
                v204 = v275;
              }
              if ( v333 != v334 )
                _libc_free((unsigned __int64)v333);
              v204 += 8;
            }
            while ( v204 != v261 );
            v10 = v273;
            v11 = (__m128i *)v266;
          }
          v8 = v11;
          sub_293CAB0(a1, (unsigned __int64)v11, (__int64)&v312, (__int64)&v293);
          v140 = (unsigned __int64)v312;
          if ( v312 == v314 )
            goto LABEL_213;
          goto LABEL_212;
        case 0x55:
          v8 = (__m128i *)(v10 - 24);
          v14 = sub_293F430(a1, (unsigned __int8 *)(v10 - 24));
          goto LABEL_17;
        case 0x56:
          v8 = (__m128i *)(v10 - 24);
          v14 = sub_2940540(a1, v10 - 24);
          goto LABEL_17;
        case 0x5A:
          v8 = (__m128i *)(v10 - 24);
          v14 = sub_2940C10(a1, (unsigned __int8 *)(v10 - 24));
          goto LABEL_17;
        case 0x5B:
          v8 = (__m128i *)(v10 - 24);
          v14 = sub_2941410(a1, v10 - 24);
LABEL_17:
          v4 = *(_QWORD *)(v10 + 8);
          if ( v14 )
            goto LABEL_18;
LABEL_21:
          v10 = v4;
          goto LABEL_10;
        case 0x5C:
          if ( *(_DWORD *)(a1 + 1152) )
          {
            v8 = (__m128i *)(v10 - 24);
            if ( !sub_293A020(a1, (unsigned __int8 *)(v10 - 24)) )
              goto LABEL_9;
          }
          sub_2939E80((__int64)&v299, a1, *(_QWORD *)(v10 - 16));
          v8 = (__m128i *)a1;
          sub_2939E80((__int64)&v307, a1, *(_QWORD *)(*(_QWORD *)(v10 - 88) + 8LL));
          if ( !(_BYTE)v301 || !(_BYTE)v309 || v299.m128i_i32[2] > 1u || v307.m128i_i32[2] > 1u )
            goto LABEL_9;
          sub_293CE40(&v320, (_QWORD *)a1, v10 - 24, *(_QWORD *)(v10 - 88), &v307);
          sub_293CE40(v330, (_QWORD *)a1, v10 - 24, *(_QWORD *)(v10 - 56), &v307);
          v100 = v299.m128i_u32[3];
          v101 = v299.m128i_i32[3];
          v312 = v314;
          v313 = 0x800000000LL;
          if ( !v299.m128i_i32[3] )
            goto LABEL_158;
          v102 = v314;
          v103 = v314;
          if ( v299.m128i_u32[3] > 8uLL )
          {
            v276 = v299.m128i_i32[3];
            sub_C8D5F0((__int64)&v312, v314, v299.m128i_u32[3], 8u, v299.m128i_u32[3], v99);
            v102 = v312;
            v101 = v276;
            v103 = &v312[8 * (unsigned int)v313];
          }
          for ( jj = &v102[8 * v100]; jj != v103; ++v103 )
          {
            if ( v103 )
              *v103 = 0;
          }
          LODWORD(v313) = v101;
          if ( !v299.m128i_i32[3] )
            goto LABEL_158;
          v105 = v10;
          v106 = 0;
          v271 = v11;
          while ( 2 )
          {
            while ( 1 )
            {
              v108 = v106;
              v109 = *(_DWORD *)(*(_QWORD *)(v105 + 48) + 4LL * v106);
              if ( v109 < 0 )
                break;
              if ( v322 > v109 )
              {
                v107 = sub_293BC00((__int64)&v320, v109);
                *(_QWORD *)&v312[8 * v106] = v107;
                goto LABEL_153;
              }
              ++v106;
              v110 = sub_293BC00((__int64)v330, v109 - v322);
              *(_QWORD *)&v312[8 * v108] = v110;
              if ( v299.m128i_i32[3] <= v106 )
              {
LABEL_157:
                v10 = v105;
                v11 = v271;
LABEL_158:
                v8 = v11;
                sub_293CAB0(a1, (unsigned __int64)v11, (__int64)&v312, (__int64)&v299);
                if ( v312 != v314 )
                  _libc_free((unsigned __int64)v312);
                if ( v333 != v334 )
                  _libc_free((unsigned __int64)v333);
                v111 = (unsigned __int64)v325;
                if ( v325 == &v327 )
                  goto LABEL_215;
LABEL_214:
                _libc_free(v111);
                goto LABEL_215;
              }
            }
            v236 = sub_ACADE0(*(__int64 ***)(v299.m128i_i64[0] + 24));
            *(_QWORD *)&v312[8 * v106] = v236;
LABEL_153:
            if ( v299.m128i_i32[3] <= ++v106 )
              goto LABEL_157;
            continue;
          }
        case 0x5D:
          v112 = *(_QWORD *)(v10 - 56);
          v113 = *(_QWORD *)(v112 + 8);
          v312 = v314;
          v313 = 0x800000000LL;
          if ( *(_BYTE *)(v113 + 8) != 15 )
            goto LABEL_9;
          if ( !(unsigned __int8)sub_2939FC0(v113) )
            goto LABEL_9;
          if ( *(_BYTE *)v112 != 85 )
            goto LABEL_9;
          v114 = *(_QWORD *)(v112 - 32);
          if ( !v114 )
            goto LABEL_9;
          if ( *(_BYTE *)v114 )
            goto LABEL_9;
          if ( *(_QWORD *)(v114 + 24) != *(_QWORD *)(v112 + 80) )
            goto LABEL_9;
          v115 = *(_DWORD *)(v114 + 36);
          if ( !v115 )
            goto LABEL_9;
          v8 = *(__m128i **)(a1 + 1120);
          if ( !(unsigned __int8)sub_9B74D0(v115, (__int64)v8) )
            goto LABEL_178;
          v8 = (__m128i *)a1;
          sub_2939E80((__int64)&v293, a1, **(_QWORD **)(v113 + 16));
          if ( !(_BYTE)v295 )
            goto LABEL_178;
          if ( *(_DWORD *)(v113 + 12) <= 1u )
            goto LABEL_385;
          v116 = v113;
          v117 = 1;
          while ( 1 )
          {
            v8 = (__m128i *)a1;
            sub_2939E80((__int64)v330, a1, *(_QWORD *)(*(_QWORD *)(v116 + 16) + 8LL * v117));
            if ( !(_BYTE)v331 || v330[0].m128i_i32[2] != v293.m128i_i32[2] )
              break;
            if ( *(_DWORD *)(v116 + 12) <= ++v117 )
            {
              v11 = (__m128i *)(v10 - 24);
LABEL_385:
              sub_23D0AB0((__int64)&v320, (__int64)v11, 0, 0, 0);
              sub_293CE40(v330, (_QWORD *)a1, (__int64)v11, v112, &v293);
              v290.m128i_i32[0] = **(_DWORD **)(v10 + 48);
              if ( v332 )
              {
                for ( kk = 0; kk < v332; ++kk )
                {
                  LOWORD(v301) = 265;
                  v299.m128i_i32[0] = v290.m128i_i32[0];
                  v246.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v11);
                  v296 = v246;
                  v297.m128i_i64[0] = (__int64)".elem";
                  v240 = v301;
                  LOWORD(v298) = 773;
                  if ( (_BYTE)v301 )
                  {
                    if ( (_BYTE)v301 == 1 )
                    {
                      v247 = &v307;
                      v248 = &v296;
                      for ( mm = 10; mm; --mm )
                      {
                        v247->m128i_i32[0] = v248->m128i_i32[0];
                        v248 = (__m128i *)((char *)v248 + 4);
                        v247 = (__m128i *)((char *)v247 + 4);
                      }
                    }
                    else
                    {
                      if ( BYTE1(v301) == 1 )
                      {
                        v250 = v299.m128i_i64[1];
                        v239 = (__m128i *)v299.m128i_i64[0];
                      }
                      else
                      {
                        v239 = &v299;
                        v240 = 2;
                      }
                      v308.m128i_i64[0] = (__int64)v239;
                      LOBYTE(v309) = 2;
                      v307.m128i_i64[0] = (__int64)&v296;
                      v308.m128i_i64[1] = v250;
                      BYTE1(v309) = v240;
                    }
                  }
                  else
                  {
                    LOWORD(v309) = 256;
                  }
                  v241 = sub_293BC00((__int64)v330, kk);
                  v242 = sub_94D3D0((unsigned int **)&v320, v241, (__int64)&v290, 1, (__int64)&v307);
                  v244 = (unsigned int)v313;
                  v245 = (unsigned int)v313 + 1LL;
                  if ( v245 > HIDWORD(v313) )
                  {
                    v262 = v242;
                    sub_C8D5F0((__int64)&v312, v314, v245, 8u, v242, v243);
                    v244 = (unsigned int)v313;
                    v242 = v262;
                  }
                  *(_QWORD *)&v312[8 * v244] = v242;
                  LODWORD(v313) = v313 + 1;
                }
              }
              v8 = v11;
              sub_293CAB0(a1, (unsigned __int64)v11, (__int64)&v312, (__int64)&v293);
              if ( v333 != v334 )
                _libc_free((unsigned __int64)v333);
              nullsub_61();
              v329 = &unk_49DA100;
              nullsub_63();
              if ( (_BYTE *)v320.m128i_i64[0] != v321 )
                _libc_free(v320.m128i_u64[0]);
              v111 = (unsigned __int64)v312;
              if ( v312 != v314 )
                goto LABEL_214;
              goto LABEL_215;
            }
          }
LABEL_178:
          if ( v312 == v314 )
            goto LABEL_9;
          _libc_free((unsigned __int64)v312);
          v10 = *(_QWORD *)(v10 + 8);
          goto LABEL_10;
        case 0x60:
          if ( *(_DWORD *)(a1 + 1152) )
          {
            v8 = (__m128i *)(v10 - 24);
            if ( !sub_293A020(a1, (unsigned __int8 *)(v10 - 24)) )
              goto LABEL_9;
          }
          v8 = (__m128i *)a1;
          sub_2939E80((__int64)&v287, a1, *(_QWORD *)(v10 - 16));
          if ( !(_BYTE)v289 )
            goto LABEL_9;
          v292 = 0;
          v290 = 0;
          v291 = 0;
          if ( (*(_BYTE *)(v10 - 17) & 0x40) != 0 )
          {
            v118 = *(_QWORD *)(**(_QWORD **)(v10 - 32) + 8LL);
            if ( v118 != *(_QWORD *)(v10 - 16) )
            {
LABEL_184:
              v8 = (__m128i *)a1;
              sub_2939E80((__int64)&v280, a1, v118);
              v119 = _mm_loadu_si128(&v281);
              v290 = _mm_loadu_si128(&v280);
              v292 = v282;
              v291 = v119;
              if ( !(_BYTE)v282 || v287.m128i_i32[2] != v290.m128i_i32[2] )
                goto LABEL_9;
LABEL_186:
              sub_23D0AB0((__int64)&v320, v10 - 24, 0, 0, 0);
              if ( (*(_BYTE *)(v10 - 17) & 0x40) != 0 )
                v120 = *(unsigned __int64 **)(v10 - 32);
              else
                v120 = (unsigned __int64 *)&v11[-2 * (*(_DWORD *)(v10 - 20) & 0x7FFFFFF)];
              sub_293CE40(v330, (_QWORD *)a1, v10 - 24, *v120, &v290);
              v123 = v287.m128i_u32[3];
              v124 = v287.m128i_i32[3];
              v312 = v314;
              v313 = 0x800000000LL;
              if ( v287.m128i_i32[3] )
              {
                v125 = v314;
                v126 = v314;
                if ( v287.m128i_u32[3] > 8uLL )
                {
                  sub_C8D5F0((__int64)&v312, v314, v287.m128i_u32[3], 8u, v121, v122);
                  v125 = v312;
                  v126 = &v312[8 * (unsigned int)v313];
                }
                for ( nn = &v125[8 * v123]; nn != v126; ++v126 )
                {
                  if ( v126 )
                    *v126 = 0;
                }
                LODWORD(v313) = v124;
                if ( v287.m128i_i32[3] )
                {
                  v252 = v10;
                  v128 = 0;
                  do
                  {
                    v296.m128i_i32[0] = v128;
                    LOWORD(v298) = 265;
                    v129.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v11);
                    v299 = v129;
                    LOWORD(v301) = 773;
                    v300.m128i_i64[0] = (__int64)".i";
                    v130 = v298;
                    if ( (_BYTE)v298 )
                    {
                      if ( (_BYTE)v298 == 1 )
                      {
                        v232 = _mm_loadu_si128(&v300);
                        v293 = _mm_loadu_si128(&v299);
                        v295 = v301;
                        v294 = v232;
                      }
                      else
                      {
                        if ( BYTE1(v298) == 1 )
                        {
                          v256 = v296.m128i_i64[1];
                          v131 = (__m128i *)v296.m128i_i64[0];
                        }
                        else
                        {
                          v131 = &v296;
                          v130 = 2;
                        }
                        v294.m128i_i64[0] = (__int64)v131;
                        LOBYTE(v295) = 2;
                        v293.m128i_i64[0] = (__int64)&v299;
                        v294.m128i_i64[1] = v256;
                        BYTE1(v295) = v130;
                      }
                    }
                    else
                    {
                      LOWORD(v295) = 256;
                    }
                    v132 = sub_293BC00((__int64)v330, v128);
                    LOWORD(v309) = 257;
                    v133 = sub_BD2C40(72, 1u);
                    v134 = (__int64)v133;
                    if ( v133 )
                      sub_B549F0((__int64)v133, v132, (__int64)&v307, 0, 0);
                    (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)v326 + 16LL))(
                      v326,
                      v134,
                      &v293,
                      v323,
                      v324);
                    v135 = v320.m128i_i64[0];
                    v136 = v320.m128i_i64[0] + 16LL * v320.m128i_u32[2];
                    if ( v320.m128i_i64[0] != v136 )
                    {
                      do
                      {
                        v137 = *(_QWORD *)(v135 + 8);
                        v138 = *(_DWORD *)v135;
                        v135 += 16;
                        sub_B99FD0(v134, v138, v137);
                      }
                      while ( v136 != v135 );
                    }
                    v139 = v128++;
                    *(_QWORD *)&v312[8 * v139] = v134;
                  }
                  while ( v287.m128i_i32[3] > v128 );
                  v10 = v252;
                }
              }
LABEL_208:
              v98 = &v287;
LABEL_209:
              v8 = v11;
              sub_293CAB0(a1, (unsigned __int64)v11, (__int64)&v312, (__int64)v98);
              if ( v312 != v314 )
                _libc_free((unsigned __int64)v312);
              v140 = (unsigned __int64)v333;
              if ( v333 != v334 )
LABEL_212:
                _libc_free(v140);
LABEL_213:
              nullsub_61();
              v329 = &unk_49DA100;
              nullsub_63();
              v111 = v320.m128i_i64[0];
              if ( (_BYTE *)v320.m128i_i64[0] == v321 )
                goto LABEL_215;
              goto LABEL_214;
            }
          }
          else
          {
            v118 = *(_QWORD *)(v11[-2 * (*(_DWORD *)(v10 - 20) & 0x7FFFFFF)].m128i_i64[0] + 8);
            if ( v118 != *(_QWORD *)(v10 - 16) )
              goto LABEL_184;
          }
          v190 = _mm_loadu_si128(&v288);
          v290 = _mm_loadu_si128(&v287);
          v292 = v289;
          v291 = v190;
          goto LABEL_186;
        default:
          BUG();
      }
    }
  }
LABEL_13:
  v12 = sub_2941D40(v2, (__int64)v8, v4, v5, v6, v7);
  if ( v304 != v306 )
    _libc_free((unsigned __int64)v304);
  return v12;
}
