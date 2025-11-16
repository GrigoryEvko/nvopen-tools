// Function: sub_FA80E0
// Address: 0xfa80e0
//
__int64 __fastcall sub_FA80E0(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rax
  unsigned int v4; // eax
  unsigned __int64 v5; // rsi
  __int64 v6; // r8
  __int64 v7; // r9
  __m128i *v9; // rax
  unsigned __int64 v10; // rdx
  int v11; // r12d
  __int64 v12; // rcx
  _QWORD *v13; // rax
  _QWORD *v14; // r15
  unsigned __int64 v15; // r14
  int v16; // ebx
  unsigned __int64 v17; // r13
  char v18; // al
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __m128i v23; // xmm7
  __m128i v24; // xmm7
  __m128i v25; // xmm1
  __m128i v26; // xmm3
  __int64 v27; // r15
  __int64 v28; // rbx
  __m128i v29; // xmm5
  unsigned __int8 *v30; // rdi
  __int64 v31; // rsi
  unsigned __int8 *v32; // rbx
  char v33; // al
  unsigned __int64 v34; // rax
  __int64 v35; // rax
  unsigned __int8 **v36; // rbx
  unsigned __int8 **v37; // r13
  unsigned __int8 *v38; // r12
  unsigned int v39; // edi
  __int64 v40; // rax
  unsigned __int8 *v41; // r10
  char v42; // dl
  __m128i *v43; // r14
  __int64 *v44; // r13
  __int64 *v45; // rbx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 *v48; // rbx
  __int64 v49; // rax
  __int64 v50; // rdx
  unsigned __int64 v51; // r15
  __int64 v52; // rcx
  int v53; // edx
  __int64 v54; // rax
  __int64 v55; // rdi
  unsigned __int64 v56; // r13
  __int64 v57; // rax
  unsigned __int8 *v58; // r11
  __int64 v59; // rax
  int v60; // edx
  bool v61; // zf
  int v62; // edx
  __int64 v63; // r8
  unsigned __int8 *v64; // r11
  unsigned __int8 v65; // al
  __int64 v66; // rax
  unsigned __int64 v67; // r14
  unsigned __int64 v68; // rdx
  __m128i *v69; // rdi
  unsigned __int64 v70; // rax
  unsigned __int8 **v71; // r14
  __int64 v72; // rax
  __int64 v73; // r13
  int v74; // edx
  int v75; // r14d
  __int64 v76; // rax
  __int64 v77; // r8
  unsigned __int64 v78; // rdx
  __m128i *v79; // rdi
  unsigned __int64 v80; // rax
  unsigned __int8 **v81; // r10
  __int64 v82; // rax
  int v83; // edx
  __int64 v84; // rcx
  __int64 v85; // rax
  __int64 v86; // r14
  __int64 v87; // rcx
  __int64 v88; // rdx
  __int64 v89; // r12
  __int64 v90; // rsi
  __int64 v91; // rdi
  __int64 v92; // rax
  __int64 v93; // r13
  __int64 v94; // rdx
  __int64 v95; // r12
  __int64 v96; // r14
  unsigned int v97; // esi
  __int64 v98; // rax
  __int64 v99; // rcx
  char *v100; // rax
  unsigned int v101; // eax
  __int64 *v102; // rcx
  unsigned int v103; // edi
  _DWORD *v104; // rcx
  __int64 v105; // rax
  __int64 v106; // r12
  __int64 **v107; // r10
  unsigned __int64 v108; // rax
  unsigned __int64 v109; // r12
  __int64 v110; // r12
  unsigned __int64 v111; // rdx
  __int64 v112; // rbx
  __m128i *v113; // rcx
  __int32 v114; // edi
  unsigned __int8 **v115; // r11
  unsigned __int64 v116; // rsi
  __int64 v117; // rax
  __int64 v118; // r12
  int v119; // ebx
  bool v120; // al
  __int64 v121; // rax
  unsigned __int64 v122; // rdx
  int v123; // edi
  __int64 v124; // rsi
  int v125; // r10d
  __int64 *v126; // rax
  int v127; // edi
  __int64 v128; // rsi
  int v129; // r10d
  __int64 *v130; // r13
  __int64 v131; // rax
  __int64 v132; // rax
  __int64 v133; // rsi
  bool v134; // of
  unsigned __int8 *v135; // r12
  unsigned __int64 v136; // rdx
  unsigned __int64 v137; // rax
  unsigned __int8 v138; // cl
  unsigned __int8 *v139; // rax
  __int64 *v140; // r13
  __int64 *v141; // r13
  __int64 *v142; // r14
  __int64 *v143; // r12
  __int64 v144; // rdx
  __int64 v145; // rcx
  __int64 v146; // rax
  __int64 v147; // rsi
  unsigned __int8 *v148; // rsi
  __int64 *v149; // r13
  __int64 v150; // rdi
  __int64 v151; // rax
  __int64 v152; // rdx
  __int64 v153; // r12
  __int64 v154; // rbx
  __int64 v155; // rsi
  __int64 v156; // rax
  __int64 v157; // rdx
  __int64 v158; // r14
  __int64 v159; // rcx
  int v160; // eax
  __int64 v161; // r13
  __int64 v162; // rdx
  __int64 v163; // r13
  __int64 v164; // r12
  __int64 v165; // r12
  __int64 v166; // r10
  __int64 v167; // rdx
  _QWORD **v168; // rbx
  _QWORD **v169; // r12
  _QWORD *v170; // rdi
  __int64 v171; // rax
  __int64 v172; // rax
  __int64 v173; // rax
  __int64 v174; // rax
  __int64 *v175; // r15
  __int64 v176; // r14
  __int64 v177; // rax
  unsigned __int64 v178; // rdx
  __int64 v179; // rax
  char v180; // al
  __int16 v181; // dx
  bool v182; // cc
  unsigned __int64 v183; // rax
  __int64 v184; // [rsp-10h] [rbp-4B0h]
  unsigned __int64 v185; // [rsp-10h] [rbp-4B0h]
  unsigned __int8 *v186; // [rsp-8h] [rbp-4A8h]
  __int64 v187; // [rsp+8h] [rbp-498h]
  unsigned __int8 *v188; // [rsp+20h] [rbp-480h]
  unsigned __int64 v189; // [rsp+30h] [rbp-470h]
  __int64 v190; // [rsp+30h] [rbp-470h]
  unsigned __int8 *v191; // [rsp+30h] [rbp-470h]
  _QWORD *v192; // [rsp+38h] [rbp-468h]
  char v193; // [rsp+38h] [rbp-468h]
  __int64 **v194; // [rsp+38h] [rbp-468h]
  __int64 v195; // [rsp+40h] [rbp-460h]
  __int64 v196; // [rsp+48h] [rbp-458h]
  __int64 v197; // [rsp+48h] [rbp-458h]
  __int64 v198; // [rsp+50h] [rbp-450h]
  _BYTE *v199; // [rsp+68h] [rbp-438h]
  unsigned __int8 *v200; // [rsp+70h] [rbp-430h]
  int v201; // [rsp+70h] [rbp-430h]
  unsigned __int8 *v202; // [rsp+78h] [rbp-428h]
  unsigned int v203; // [rsp+78h] [rbp-428h]
  int v205; // [rsp+88h] [rbp-418h]
  __int64 v206; // [rsp+90h] [rbp-410h]
  __int64 v207; // [rsp+98h] [rbp-408h]
  char v208; // [rsp+A0h] [rbp-400h]
  __int64 *v209; // [rsp+A0h] [rbp-400h]
  __int64 v211; // [rsp+B0h] [rbp-3F0h]
  unsigned __int8 *v212; // [rsp+B8h] [rbp-3E8h]
  int v213; // [rsp+B8h] [rbp-3E8h]
  __int64 v215; // [rsp+C0h] [rbp-3E0h]
  unsigned __int8 **v216; // [rsp+C8h] [rbp-3D8h]
  int v217; // [rsp+C8h] [rbp-3D8h]
  int v218; // [rsp+C8h] [rbp-3D8h]
  __int64 v219; // [rsp+C8h] [rbp-3D8h]
  __int64 v220; // [rsp+C8h] [rbp-3D8h]
  _QWORD *v221; // [rsp+D0h] [rbp-3D0h]
  unsigned __int8 *v222; // [rsp+D0h] [rbp-3D0h]
  __int64 v223; // [rsp+D0h] [rbp-3D0h]
  __int64 v224; // [rsp+D0h] [rbp-3D0h]
  __int64 v225; // [rsp+D0h] [rbp-3D0h]
  unsigned int v226; // [rsp+D8h] [rbp-3C8h]
  unsigned __int8 v227; // [rsp+DFh] [rbp-3C1h]
  _BYTE *v228; // [rsp+E0h] [rbp-3C0h] BYREF
  __int64 v229; // [rsp+E8h] [rbp-3B8h]
  _BYTE v230[16]; // [rsp+F0h] [rbp-3B0h] BYREF
  _BYTE *v231; // [rsp+100h] [rbp-3A0h] BYREF
  __int64 v232; // [rsp+108h] [rbp-398h]
  _BYTE v233[32]; // [rsp+110h] [rbp-390h] BYREF
  __m128i v234; // [rsp+130h] [rbp-370h] BYREF
  __m128i v235; // [rsp+140h] [rbp-360h] BYREF
  _BYTE v236[16]; // [rsp+150h] [rbp-350h] BYREF
  void (__fastcall *v237)(_BYTE *, _BYTE *, __int64); // [rsp+160h] [rbp-340h]
  unsigned __int8 (__fastcall *v238)(_BYTE *); // [rsp+168h] [rbp-338h]
  __m128i v239; // [rsp+170h] [rbp-330h] BYREF
  __m128i v240; // [rsp+180h] [rbp-320h] BYREF
  _BYTE v241[16]; // [rsp+190h] [rbp-310h] BYREF
  void (__fastcall *v242)(_BYTE *, _BYTE *, __int64); // [rsp+1A0h] [rbp-300h]
  _OWORD v243[2]; // [rsp+1B0h] [rbp-2F0h] BYREF
  _BYTE v244[16]; // [rsp+1D0h] [rbp-2D0h] BYREF
  void (__fastcall *v245)(_BYTE *, _BYTE *, __int64); // [rsp+1E0h] [rbp-2C0h]
  __int64 v246; // [rsp+1F0h] [rbp-2B0h] BYREF
  __int64 v247; // [rsp+1F8h] [rbp-2A8h]
  __m128i *v248; // [rsp+200h] [rbp-2A0h] BYREF
  unsigned int v249; // [rsp+208h] [rbp-298h]
  __m128i v250; // [rsp+240h] [rbp-260h] BYREF
  __m128i v251; // [rsp+250h] [rbp-250h] BYREF
  char v252; // [rsp+260h] [rbp-240h] BYREF
  char v253; // [rsp+261h] [rbp-23Fh]
  void (__fastcall *v254)(char *, char *, __int64); // [rsp+270h] [rbp-230h]
  unsigned __int8 (__fastcall *v255)(char *, unsigned __int64); // [rsp+278h] [rbp-228h]
  _BYTE v256[16]; // [rsp+2A0h] [rbp-200h] BYREF
  void (__fastcall *v257)(_BYTE *, _BYTE *, __int64); // [rsp+2B0h] [rbp-1F0h]
  __m128i v258; // [rsp+2C0h] [rbp-1E0h] BYREF
  __m128i v259; // [rsp+2D0h] [rbp-1D0h] BYREF
  _BYTE v260[16]; // [rsp+2E0h] [rbp-1C0h] BYREF
  void (__fastcall *v261)(_BYTE *, _BYTE *, __int64); // [rsp+2F0h] [rbp-1B0h]
  __int64 v262; // [rsp+2F8h] [rbp-1A8h]
  __m128i v263; // [rsp+300h] [rbp-1A0h] BYREF
  __m128i v264; // [rsp+310h] [rbp-190h] BYREF
  __int64 v265; // [rsp+320h] [rbp-180h] BYREF
  int v266; // [rsp+328h] [rbp-178h]
  __int16 v267; // [rsp+32Ch] [rbp-174h]
  char v268; // [rsp+32Eh] [rbp-172h]
  void (__fastcall *v269)(__int64 *, __int64 *, __int64); // [rsp+330h] [rbp-170h]
  __int64 v270; // [rsp+338h] [rbp-168h]
  void *v271; // [rsp+340h] [rbp-160h] BYREF
  void *v272; // [rsp+348h] [rbp-158h] BYREF
  __int64 v273; // [rsp+350h] [rbp-150h] BYREF
  char *v274; // [rsp+358h] [rbp-148h]
  __int64 v275; // [rsp+360h] [rbp-140h]
  int v276; // [rsp+368h] [rbp-138h]
  char v277; // [rsp+36Ch] [rbp-134h]
  char v278; // [rsp+370h] [rbp-130h] BYREF

  v199 = *(_BYTE **)(a2 - 96);
  if ( *v199 == 83 )
    return 0;
  v207 = *(_QWORD *)(a2 + 40);
  v3 = sub_986580(a3);
  v195 = sub_B46EC0(v3, 0);
  v226 = qword_4F8D348;
  v198 = *(_QWORD *)(a2 - 32);
  v4 = 256;
  LOBYTE(v4) = a3 != v198;
  v5 = v4;
  v227 = sub_F8F5B0(a2, v4, (_QWORD *)*a1);
  if ( !v227 )
    return 0;
  v9 = (__m128i *)&v248;
  v10 = (unsigned __int64)&v250;
  v246 = 0;
  v247 = 1;
  do
  {
    v9->m128i_i64[0] = -4096;
    ++v9;
  }
  while ( v9 != &v250 );
  v231 = v233;
  v232 = 0x400000000LL;
  v208 = qword_4F8D0A8;
  if ( (_BYTE)qword_4F8D0A8 )
    v208 = *(_BYTE *)(a1[6] + 9);
  v11 = 0;
  v273 = 0;
  v228 = v230;
  v229 = 0x200000000LL;
  v274 = &v278;
  v275 = 32;
  v12 = *(_QWORD *)(a3 + 48);
  v13 = *(_QWORD **)(a3 + 56);
  v276 = 0;
  v277 = 1;
  v221 = v13;
  v211 = 0;
  v14 = (_QWORD *)(v12 & 0xFFFFFFFFFFFFFFF8LL);
  v212 = 0;
  if ( v13 != (_QWORD *)(v12 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v15 = *v14 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v15 )
LABEL_405:
          BUG();
        v16 = *(unsigned __int8 *)(v15 - 24);
        v17 = v15 - 24;
        if ( (_BYTE)v16 == 85 )
          break;
        if ( (unsigned __int8)sub_B46970((unsigned __int8 *)(v15 - 24)) || (unsigned int)(v16 - 30) <= 0xA )
        {
LABEL_14:
          if ( v208 )
          {
            v5 = *a1;
            if ( sub_F8F4A0((unsigned __int8 *)(v15 - 24), *a1) )
            {
              v105 = (unsigned int)v229;
              if ( (unsigned int)v229 < (unsigned int)qword_4F8CFC8 )
              {
                if ( (unsigned __int64)(unsigned int)v229 + 1 > HIDWORD(v229) )
                {
                  v5 = (unsigned __int64)v230;
                  sub_C8D5F0((__int64)&v228, v230, (unsigned int)v229 + 1LL, 8u, v6, v7);
                  v105 = (unsigned int)v229;
                }
                v10 = (unsigned __int64)v228;
                v216 = (unsigned __int8 **)(v15 - 24);
                *(_QWORD *)&v228[8 * v105] = v17;
                LODWORD(v229) = v229 + 1;
                goto LABEL_66;
              }
            }
          }
          if ( v11 == 1 )
            goto LABEL_56;
          v5 = a2;
          v18 = sub_991A70((unsigned __int8 *)(v15 - 24), a2, *(_QWORD *)(a1[6] + 16), 0, 0, 1u, 0);
          v6 = v184;
          v7 = (__int64)v186;
          if ( !v18 )
          {
            if ( byte_4F8CD28 != 1 )
              goto LABEL_56;
            if ( v212 )
              goto LABEL_56;
            if ( *(_BYTE *)(v15 - 24) != 62 )
              goto LABEL_56;
            v216 = (unsigned __int8 **)(v15 - 24);
            if ( sub_B46500((unsigned __int8 *)(v15 - 24)) || (*(_BYTE *)(v15 - 22) & 1) != 0 )
              goto LABEL_56;
            v202 = *(unsigned __int8 **)(v15 - 56);
            v187 = *(_QWORD *)(*(_QWORD *)(v15 - 88) + 8LL);
            sub_AA72C0(&v250, v207, 1);
            sub_F94A80(&v258, &v250, v19, v20, v21, v22);
            if ( v257 )
              v257(v256, v256, 3);
            if ( v254 )
              v254(&v252, &v252, 3);
            v23 = _mm_loadu_si128(&v259);
            v234 = _mm_loadu_si128(&v258);
            v235 = v23;
            sub_F99F40((__int64)v236, (__int64)v260);
            v24 = _mm_loadu_si128(&v264);
            v239 = _mm_loadu_si128(&v263);
            v240 = v24;
            sub_F99F40((__int64)v241, (__int64)&v265);
            v192 = v14;
            v213 = 10;
LABEL_27:
            v25 = _mm_loadu_si128(&v240);
            v250 = _mm_loadu_si128(&v239);
            v251 = v25;
            sub_F99F40((__int64)&v252, (__int64)v241);
            v5 = (unsigned __int64)v236;
            v26 = _mm_loadu_si128(&v235);
            v243[0] = _mm_loadu_si128(&v234);
            v243[1] = v26;
            sub_F99F40((__int64)v244, (__int64)v236);
            v27 = *(_QWORD *)&v243[0];
            v28 = v250.m128i_i64[0];
            if ( v245 )
            {
              v5 = (unsigned __int64)v244;
              v245(v244, v244, 3);
            }
            if ( v254 )
            {
              v5 = (unsigned __int64)&v252;
              v254(&v252, &v252, 3);
            }
            if ( v27 == v28 )
              goto LABEL_55;
            v29 = _mm_loadu_si128(&v235);
            v250 = _mm_loadu_si128(&v234);
            v251 = v29;
            sub_F99F40((__int64)&v252, (__int64)v236);
            do
            {
              v30 = 0;
              v31 = *(_QWORD *)v250.m128i_i64[0];
              v250.m128i_i16[4] = 0;
              v5 = v31 & 0xFFFFFFFFFFFFFFF8LL;
              v250.m128i_i64[0] = v5;
              if ( v5 )
                v5 -= 24LL;
              if ( !v254 )
                goto LABEL_278;
            }
            while ( !v255(&v252, v5) );
            v32 = (unsigned __int8 *)v250.m128i_i64[0];
            if ( v250.m128i_i64[0] )
              v32 = (unsigned __int8 *)(v250.m128i_i64[0] - 24);
            if ( v254 )
            {
              v5 = (unsigned __int64)&v252;
              v254(&v252, &v252, 3);
            }
            if ( !--v213 )
            {
LABEL_55:
              sub_A17130((__int64)v241);
              sub_A17130((__int64)v236);
              sub_A17130((__int64)&v265);
              sub_A17130((__int64)v260);
LABEL_56:
              v227 = 0;
              goto LABEL_57;
            }
            v30 = v32;
            v33 = sub_B46490((__int64)v32);
            v10 = *v32;
            if ( v33 )
            {
              v14 = v192;
              if ( (_BYTE)v10 == 62 )
                goto LABEL_44;
LABEL_288:
              v212 = 0;
            }
            else if ( (_BYTE)v10 == 62 )
            {
              v14 = v192;
LABEL_44:
              v212 = 0;
              if ( v202 == *((unsigned __int8 **)v32 - 4) )
              {
                v135 = (unsigned __int8 *)*((_QWORD *)v32 - 8);
                if ( v187 != *((_QWORD *)v135 + 1) )
                  goto LABEL_288;
                if ( sub_B46500(v32) )
                  goto LABEL_288;
                v5 = *((unsigned __int16 *)v32 + 1);
                if ( (v5 & 1) != 0 )
                  goto LABEL_288;
                _BitScanReverse64(&v136, 1LL << (*(_WORD *)(v15 - 22) >> 1));
                v10 = v136 ^ 0x3F;
                _BitScanReverse64(&v137, 1LL << ((unsigned __int16)v5 >> 1));
                v5 = (unsigned int)(63 - v10);
                v138 = 63 - (v137 ^ 0x3F);
                v139 = 0;
                if ( (unsigned __int8)(63 - v10) <= v138 )
                  v139 = v135;
                v212 = v139;
              }
            }
            else
            {
              if ( (_BYTE)v10 != 61 )
                goto LABEL_277;
              if ( v202 != *((unsigned __int8 **)v32 - 4) )
                goto LABEL_277;
              if ( v187 != *((_QWORD *)v32 + 1) )
                goto LABEL_277;
              v30 = v32;
              if ( sub_B46500(v32) )
                goto LABEL_277;
              LOWORD(v177) = *((_WORD *)v32 + 1);
              if ( (v177 & 1) != 0 )
                goto LABEL_277;
              _BitScanReverse64(&v178, 1LL << (*(_WORD *)(v15 - 22) >> 1));
              _BitScanReverse64((unsigned __int64 *)&v177, 1LL << ((unsigned __int16)v177 >> 1));
              v10 = v178 ^ 0x3F;
              if ( (unsigned __int8)(63 - v10) > (unsigned __int8)(63 - (v177 ^ 0x3F)) )
                goto LABEL_277;
              v30 = sub_98ACB0(v202, 6u);
              if ( !(unsigned __int8)sub_CF7600(v30, &v250)
                || (v5 = 0, (unsigned __int8)sub_D13FA0((__int64)v30, 0, 0))
                || v250.m128i_i8[0]
                && (v179 = sub_B43CC0((__int64)v32),
                    v186 = v30,
                    v5 = v187,
                    v30 = v202,
                    v180 = sub_D30730((__int64)v202, v187, v179, 0, 0, 0, 0),
                    v10 = v185,
                    !v180) )
              {
LABEL_277:
                while ( 1 )
                {
                  v133 = *(_QWORD *)v234.m128i_i64[0];
                  v234.m128i_i16[4] = 0;
                  v5 = v133 & 0xFFFFFFFFFFFFFFF8LL;
                  v234.m128i_i64[0] = v5;
                  if ( v5 )
                    v5 -= 24LL;
                  if ( !v237 )
                    break;
                  v30 = v236;
                  if ( v238(v236) )
                    goto LABEL_27;
                }
LABEL_278:
                sub_4263D6(v30, v5, v10);
              }
              v212 = v32;
              v14 = v192;
            }
            if ( v242 )
            {
              v5 = (unsigned __int64)v241;
              v242(v241, v241, 3);
            }
            if ( v237 )
            {
              v5 = (unsigned __int64)v236;
              v237(v236, v236, 3);
            }
            if ( v269 )
            {
              v5 = (unsigned __int64)&v265;
              v269(&v265, &v265, 3);
            }
            if ( v261 )
            {
              v5 = (unsigned __int64)v260;
              v261(v260, v260, 3);
            }
            if ( !v212 )
              goto LABEL_56;
            v11 = 1;
LABEL_66:
            v34 = v211;
            if ( !v211 )
            {
              if ( v212 )
                v34 = v15 - 24;
              v211 = v34;
            }
            goto LABEL_70;
          }
          v216 = (unsigned __int8 **)(v15 - 24);
          v11 = 1;
          if ( v212 )
            goto LABEL_66;
          v203 = qword_4F8D348;
          v106 = 32LL * (*(_DWORD *)(v15 - 20) & 0x7FFFFFF);
          v107 = (__int64 **)*a1;
          if ( (*(_BYTE *)(v15 - 17) & 0x40) != 0 )
          {
            v108 = *(_QWORD *)(v15 - 32);
            v109 = v108 + v106;
          }
          else
          {
            v108 = v17 - v106;
            v109 = v15 - 24;
          }
          v110 = v109 - v108;
          v258.m128i_i64[0] = (__int64)&v259;
          v111 = v110 >> 5;
          v258.m128i_i64[1] = 0x400000000LL;
          v112 = v110 >> 5;
          if ( (unsigned __int64)v110 > 0x80 )
          {
            v189 = v108;
            v194 = v107;
            sub_C8D5F0((__int64)&v258, &v259, v111, 8u, v184, (__int64)&v259);
            v115 = (unsigned __int8 **)v258.m128i_i64[0];
            v114 = v258.m128i_i32[2];
            v111 = v110 >> 5;
            v107 = v194;
            v108 = v189;
            v113 = (__m128i *)(v258.m128i_i64[0] + 8LL * v258.m128i_u32[2]);
          }
          else
          {
            v113 = &v259;
            v114 = 0;
            v115 = (unsigned __int8 **)&v259;
          }
          if ( v110 > 0 )
          {
            v116 = 0;
            do
            {
              v113->m128i_i64[v116 / 8] = *(_QWORD *)(v108 + 4 * v116);
              v116 += 8LL;
              --v112;
            }
            while ( v112 );
            v115 = (unsigned __int8 **)v258.m128i_i64[0];
            v114 = v258.m128i_i32[2];
          }
          v5 = v15 - 24;
          v258.m128i_i32[2] = v114 + v111;
          v117 = sub_DFCEF0(v107, (unsigned __int8 *)(v15 - 24), v115, (unsigned int)(v114 + v111), 3);
          v7 = (__int64)&v259;
          v118 = v117;
          v119 = v10;
          if ( (__m128i *)v258.m128i_i64[0] != &v259 )
            _libc_free(v258.m128i_i64[0], v5);
          if ( v119 )
            v120 = v119 > 0;
          else
            v120 = v118 > v203;
          if ( v120 )
            goto LABEL_56;
          v11 = 1;
LABEL_70:
          v35 = 4LL * (*(_DWORD *)(v15 - 20) & 0x7FFFFFF);
          v36 = (unsigned __int8 **)(v17 - v35 * 8);
          if ( (*(_BYTE *)(v15 - 17) & 0x40) != 0 )
          {
            v36 = *(unsigned __int8 ***)(v15 - 32);
            v216 = &v36[v35];
          }
          v37 = v216;
          if ( v36 != v216 )
          {
            v217 = v11;
            while ( 1 )
            {
              while ( 1 )
              {
                v38 = *v36;
                if ( **v36 > 0x1Cu && v207 == *((_QWORD *)v38 + 5) && !(unsigned __int8)sub_B46970(*v36) )
                  break;
                v36 += 4;
                if ( v37 == v36 )
                  goto LABEL_83;
              }
              v6 = v247 & 1;
              if ( (v247 & 1) != 0 )
              {
                v7 = (__int64)&v248;
                v5 = 3;
              }
              else
              {
                v97 = v249;
                v7 = (__int64)v248;
                if ( !v249 )
                {
                  v101 = v247;
                  ++v246;
                  v102 = 0;
                  v103 = ((unsigned int)v247 >> 1) + 1;
                  goto LABEL_179;
                }
                v5 = v249 - 1;
              }
              v39 = v5 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
              v40 = v7 + 16LL * v39;
              v41 = *(unsigned __int8 **)v40;
              if ( v38 != *(unsigned __int8 **)v40 )
                break;
LABEL_81:
              ++*(_DWORD *)(v40 + 8);
LABEL_82:
              v36 += 4;
              if ( v37 == v36 )
              {
LABEL_83:
                v11 = v217;
                goto LABEL_84;
              }
            }
            v10 = 1;
            v102 = 0;
            while ( v41 != (unsigned __int8 *)-4096LL )
            {
              if ( !v102 && v41 == (unsigned __int8 *)-8192LL )
                v102 = (__int64 *)v40;
              v39 = v5 & (v10 + v39);
              v40 = v7 + 16LL * v39;
              v41 = *(unsigned __int8 **)v40;
              if ( v38 == *(unsigned __int8 **)v40 )
                goto LABEL_81;
              v10 = (unsigned int)(v10 + 1);
            }
            v7 = 12;
            v97 = 4;
            if ( !v102 )
              v102 = (__int64 *)v40;
            v101 = v247;
            ++v246;
            v103 = ((unsigned int)v247 >> 1) + 1;
            if ( !(_BYTE)v6 )
            {
              v97 = v249;
LABEL_179:
              v7 = 3 * v97;
            }
            if ( (unsigned int)v7 <= 4 * v103 )
            {
              sub_FA7CC0((__int64)&v246, 2 * v97);
              if ( (v247 & 1) != 0 )
              {
                v6 = (__int64)&v248;
                v123 = 3;
              }
              else
              {
                v6 = (__int64)v248;
                if ( !v249 )
                  goto LABEL_406;
                v123 = v249 - 1;
              }
              v101 = v247;
              LODWORD(v124) = v123 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
              v102 = (__int64 *)(v6 + 16LL * (unsigned int)v124);
              v7 = *v102;
              if ( v38 != (unsigned __int8 *)*v102 )
              {
                v125 = 1;
                v126 = 0;
                while ( v7 != -4096 )
                {
                  if ( v7 == -8192 && !v126 )
                    v126 = v102;
                  v10 = (unsigned int)(v125 + 1);
                  v124 = v123 & (unsigned int)(v124 + v125);
                  v102 = (__int64 *)(v6 + 16 * v124);
                  v7 = *v102;
                  if ( v38 == (unsigned __int8 *)*v102 )
                    goto LABEL_235;
                  ++v125;
                }
LABEL_233:
                if ( v126 )
                  v102 = v126;
LABEL_235:
                v101 = v247;
              }
            }
            else
            {
              v6 = v97 - HIDWORD(v247) - v103;
              if ( (unsigned int)v6 <= v97 >> 3 )
              {
                sub_FA7CC0((__int64)&v246, v97);
                if ( (v247 & 1) != 0 )
                {
                  v6 = (__int64)&v248;
                  v127 = 3;
                }
                else
                {
                  v6 = (__int64)v248;
                  if ( !v249 )
                  {
LABEL_406:
                    LODWORD(v247) = (2 * ((unsigned int)v247 >> 1) + 2) | v247 & 1;
                    BUG();
                  }
                  v127 = v249 - 1;
                }
                v101 = v247;
                LODWORD(v128) = v127 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
                v102 = (__int64 *)(v6 + 16LL * (unsigned int)v128);
                v7 = *v102;
                if ( v38 != (unsigned __int8 *)*v102 )
                {
                  v129 = 1;
                  v126 = 0;
                  while ( v7 != -4096 )
                  {
                    if ( v7 == -8192 && !v126 )
                      v126 = v102;
                    v10 = (unsigned int)(v129 + 1);
                    v128 = v127 & (unsigned int)(v128 + v129);
                    v102 = (__int64 *)(v6 + 16 * v128);
                    v7 = *v102;
                    if ( v38 == (unsigned __int8 *)*v102 )
                      goto LABEL_235;
                    ++v129;
                  }
                  goto LABEL_233;
                }
              }
            }
            v5 = 2 * (v101 >> 1) + 2;
            LODWORD(v247) = v5 | v101 & 1;
            if ( *v102 != -4096 )
              --HIDWORD(v247);
            *v102 = (__int64)v38;
            v104 = v102 + 1;
            *v104 = 0;
            *v104 = 1;
            goto LABEL_82;
          }
LABEL_84:
          v14 = (_QWORD *)(*v14 & 0xFFFFFFFFFFFFFFF8LL);
          if ( v14 == v221 )
            goto LABEL_85;
        }
        else
        {
LABEL_171:
          v5 = 0;
          if ( !sub_F90050(*(_QWORD *)(v15 - 8), 0, (__int64)&v273) )
            goto LABEL_14;
LABEL_172:
          if ( !v277 )
            goto LABEL_206;
          v100 = v274;
          v99 = HIDWORD(v275);
          v10 = (unsigned __int64)&v274[8 * HIDWORD(v275)];
          if ( v274 != (char *)v10 )
          {
            while ( v17 != *(_QWORD *)v100 )
            {
              v100 += 8;
              if ( (char *)v10 == v100 )
                goto LABEL_215;
            }
            goto LABEL_84;
          }
LABEL_215:
          if ( HIDWORD(v275) < (unsigned int)v275 )
          {
            ++HIDWORD(v275);
            *(_QWORD *)v10 = v17;
            ++v273;
            v14 = (_QWORD *)(*v14 & 0xFFFFFFFFFFFFFFF8LL);
            if ( v14 == v221 )
              goto LABEL_85;
          }
          else
          {
LABEL_206:
            v5 = v15 - 24;
            sub_C8CC70((__int64)&v273, v15 - 24, v10, v99, v6, v7);
            v14 = (_QWORD *)(*v14 & 0xFFFFFFFFFFFFFFF8LL);
            if ( v14 == v221 )
              goto LABEL_85;
          }
        }
      }
      v98 = *(_QWORD *)(v15 - 56);
      if ( !v98 )
        goto LABEL_170;
      if ( *(_BYTE *)v98
        || *(_QWORD *)(v98 + 24) != *(_QWORD *)(v15 + 56)
        || (*(_BYTE *)(v98 + 33) & 0x20) == 0
        || (v10 = (unsigned int)(*(_DWORD *)(v98 + 36) - 68), (unsigned int)v10 > 3) )
      {
        if ( *(_BYTE *)v98
          || *(_QWORD *)(v98 + 24) != *(_QWORD *)(v15 + 56)
          || (*(_BYTE *)(v98 + 33) & 0x20) == 0
          || *(_DWORD *)(v98 + 36) != 291 )
        {
          break;
        }
      }
      v121 = (unsigned int)v232;
      v122 = (unsigned int)v232 + 1LL;
      if ( v122 > HIDWORD(v232) )
      {
        v5 = (unsigned __int64)v233;
        sub_C8D5F0((__int64)&v231, v233, v122, 8u, v6, v7);
        v121 = (unsigned int)v232;
      }
      v10 = (unsigned __int64)v231;
      *(_QWORD *)&v231[8 * v121] = v17;
      LODWORD(v232) = v232 + 1;
      v14 = (_QWORD *)(*v14 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v14 == v221 )
        goto LABEL_85;
    }
    if ( !*(_BYTE *)v98 )
    {
      v99 = *(_QWORD *)(v15 + 56);
      if ( *(_QWORD *)(v98 + 24) == v99 && (*(_BYTE *)(v98 + 33) & 0x20) != 0 && *(_DWORD *)(v98 + 36) == 11 )
        goto LABEL_172;
    }
LABEL_170:
    if ( (unsigned __int8)sub_B46970((unsigned __int8 *)(v15 - 24)) )
      goto LABEL_14;
    goto LABEL_171;
  }
LABEL_85:
  v42 = v247 & 1;
  if ( (unsigned int)v247 >> 1 )
  {
    if ( v42 )
    {
      v43 = &v250;
      v44 = (__int64 *)&v248;
    }
    else
    {
      v46 = v249;
      v45 = (__int64 *)v248;
      v44 = (__int64 *)v248;
      v43 = &v248[v249];
      if ( v43 == v248 )
        goto LABEL_93;
    }
    do
    {
      if ( *v44 != -8192 && *v44 != -4096 )
        break;
      v44 += 2;
    }
    while ( v44 != (__int64 *)v43 );
  }
  else
  {
    if ( v42 )
    {
      v130 = (__int64 *)&v248;
      v131 = 8;
    }
    else
    {
      v130 = (__int64 *)v248;
      v131 = 2LL * v249;
    }
    v44 = &v130[v131];
    v43 = (__m128i *)v44;
  }
  if ( v42 )
  {
    v45 = (__int64 *)&v248;
    v47 = 8;
    goto LABEL_94;
  }
  v45 = (__int64 *)v248;
  v46 = v249;
LABEL_93:
  v47 = 2 * v46;
LABEL_94:
  v48 = &v45[v47];
  while ( v44 != v48 )
  {
    v5 = *((unsigned int *)v44 + 2);
    if ( (unsigned __int8)sub_BD3610(*v44, v5) )
    {
      if ( v11 == 1 )
        goto LABEL_56;
      v11 = 1;
    }
    for ( v44 += 2; v43 != (__m128i *)v44; v44 += 2 )
    {
      if ( *v44 != -4096 && *v44 != -8192 )
        break;
    }
  }
  v193 = v227;
  if ( !v211 )
    v193 = (_DWORD)v229 != 0;
  v209 = (__int64 *)*a1;
  sub_B2D610(*(_QWORD *)(v207 + 72), 18);
  v49 = sub_AA5930(v195);
  v5 = 0;
  v218 = 0;
  v206 = v50;
  v51 = v49;
  v215 = 0;
  if ( v49 == v50 )
  {
    if ( v193 )
      goto LABEL_151;
    goto LABEL_56;
  }
  v205 = v11;
  do
  {
    v52 = *(_QWORD *)(v51 - 8);
    v53 = *(_DWORD *)(v51 + 4) & 0x7FFFFFF;
    if ( !v53 )
      goto LABEL_260;
    v54 = 0;
    v55 = v52 + 32LL * *(unsigned int *)(v51 + 72);
    do
    {
      if ( v207 == *(_QWORD *)(v55 + 8 * v54) )
      {
        v56 = *(_QWORD *)(v52 + 32 * v54);
        goto LABEL_112;
      }
      ++v54;
    }
    while ( v53 != (_DWORD)v54 );
    v56 = *(_QWORD *)(v52 + 0x1FFFFFFFE0LL);
LABEL_112:
    v57 = 0;
    do
    {
      if ( a3 == *(_QWORD *)(v55 + 8 * v57) )
      {
        v58 = *(unsigned __int8 **)(v52 + 32 * v57);
        goto LABEL_116;
      }
      ++v57;
    }
    while ( v53 != (_DWORD)v57 );
    v58 = *(unsigned __int8 **)(v52 + 0x1FFFFFFFE0LL);
LABEL_116:
    v222 = v58;
    if ( (unsigned __int8 *)v56 == v58 )
      goto LABEL_260;
    v59 = sub_DFD2D0(v209, 57, *(_QWORD *)(v51 + 8));
    v61 = v60 == 1;
    v62 = 1;
    if ( !v61 )
      v62 = v218;
    v218 = v62;
    if ( __OFADD__(v59, v215) )
    {
      v182 = v59 <= 0;
      v183 = 0x8000000000000000LL;
      if ( !v182 )
        v183 = 0x7FFFFFFFFFFFFFFFLL;
      v215 = v183;
    }
    else
    {
      v215 += v59;
    }
    v5 = v51;
    if ( (unsigned __int8)sub_F91900((unsigned __int8 *)v56, (unsigned __int8 *)v51, 0) )
      goto LABEL_148;
    v5 = v51;
    if ( (unsigned __int8)sub_F91900(v222, (unsigned __int8 *)v51, 0) )
      goto LABEL_148;
    v64 = v222;
    v65 = *v222;
    if ( *(_BYTE *)v56 != 5 )
    {
      if ( v65 != 5 )
      {
        v5 = v227;
        goto LABEL_260;
      }
      v75 = 0;
      v73 = 0;
      goto LABEL_137;
    }
    if ( v65 != 5 )
      v64 = 0;
    v66 = 32LL * (*(_DWORD *)(v56 + 4) & 0x7FFFFFF);
    v67 = v56 - v66;
    if ( (*(_BYTE *)(v56 + 7) & 0x40) != 0 )
      v67 = *(_QWORD *)(v56 - 8);
    v68 = v66 >> 5;
    v258.m128i_i64[0] = (__int64)&v259;
    v258.m128i_i64[1] = 0x400000000LL;
    if ( (unsigned __int64)v66 > 0x80 )
    {
      v191 = v64;
      v197 = v66 >> 5;
      sub_C8D5F0((__int64)&v258, &v259, v68, 8u, v63, (__int64)&v259);
      v68 = v197;
      v64 = v191;
      v69 = (__m128i *)(v258.m128i_i64[0] + 8LL * v258.m128i_u32[2]);
    }
    else
    {
      if ( !v66 )
      {
        v71 = (unsigned __int8 **)&v259;
        goto LABEL_134;
      }
      v69 = &v259;
    }
    v70 = 0;
    do
    {
      v69->m128i_i64[v70 / 8] = *(_QWORD *)(v67 + 4 * v70);
      v70 += 8LL;
    }
    while ( 8 * v68 != v70 );
    v71 = (unsigned __int8 **)v258.m128i_i64[0];
    LODWORD(v66) = v258.m128i_i32[2];
LABEL_134:
    v5 = v56;
    v200 = v64;
    v258.m128i_i32[2] = v68 + v66;
    v72 = sub_DFCEF0((__int64 **)v209, (unsigned __int8 *)v56, v71, (unsigned int)(v68 + v66), 3);
    v64 = v200;
    v73 = v72;
    v75 = v74;
    if ( (__m128i *)v258.m128i_i64[0] != &v259 )
    {
      _libc_free(v258.m128i_i64[0], v5);
      v64 = v200;
    }
    if ( !v64 )
    {
      v85 = (unsigned int)(2 * qword_4F8D348);
LABEL_256:
      if ( v75 )
        goto LABEL_282;
      if ( v73 > v85 )
        goto LABEL_148;
      goto LABEL_258;
    }
LABEL_137:
    v76 = 32LL * (*((_DWORD *)v64 + 1) & 0x7FFFFFF);
    v77 = (__int64)&v64[-v76];
    if ( (v64[7] & 0x40) != 0 )
      v77 = *((_QWORD *)v64 - 1);
    v258.m128i_i64[0] = (__int64)&v259;
    v258.m128i_i64[1] = 0x400000000LL;
    v78 = v76 >> 5;
    if ( (unsigned __int64)v76 > 0x80 )
    {
      v188 = v64;
      v190 = v77;
      v196 = v76 >> 5;
      sub_C8D5F0((__int64)&v258, &v259, v78, 8u, v77, (__int64)&v259);
      v78 = v196;
      v77 = v190;
      v64 = v188;
      v79 = (__m128i *)(v258.m128i_i64[0] + 8LL * v258.m128i_u32[2]);
    }
    else
    {
      if ( !v76 )
      {
        v81 = (unsigned __int8 **)&v259;
        goto LABEL_145;
      }
      v79 = &v259;
    }
    v80 = 0;
    do
    {
      v79->m128i_i64[v80 / 8] = *(_QWORD *)(v77 + 4 * v80);
      v80 += 8LL;
    }
    while ( v80 != 8 * v78 );
    v81 = (unsigned __int8 **)v258.m128i_i64[0];
    LODWORD(v76) = v258.m128i_i32[2];
LABEL_145:
    v5 = (unsigned __int64)v64;
    v258.m128i_i32[2] = v78 + v76;
    v82 = sub_DFCEF0((__int64 **)v209, v64, v81, (unsigned int)(v78 + v76), 3);
    v84 = v82;
    if ( (__m128i *)v258.m128i_i64[0] != &v259 )
    {
      v201 = v83;
      v223 = v82;
      _libc_free(v258.m128i_i64[0], v5);
      v83 = v201;
      v84 = v223;
    }
    v85 = (unsigned int)(2 * qword_4F8D348);
    if ( v83 == 1 )
      goto LABEL_148;
    v134 = __OFADD__(v84, v73);
    v73 += v84;
    if ( !v134 )
      goto LABEL_256;
    if ( v84 > 0 )
    {
      if ( !v75 )
        goto LABEL_148;
LABEL_282:
      if ( v75 > 0 )
        goto LABEL_148;
      goto LABEL_258;
    }
    if ( v75 )
      goto LABEL_282;
LABEL_258:
    if ( v205 == 1 )
      goto LABEL_148;
    v205 = 1;
    v5 = v227;
LABEL_260:
    v132 = *(_QWORD *)(v51 + 32);
    if ( !v132 )
      goto LABEL_405;
    v51 = 0;
    if ( *(_BYTE *)(v132 - 24) == 84 )
      v51 = v132 - 24;
  }
  while ( v206 != v51 );
  v193 |= v5;
LABEL_148:
  if ( !v193 || v218 || v226 < v215 )
    goto LABEL_56;
LABEL_151:
  if ( v212 )
  {
    v263.m128i_i64[1] = sub_BD5C60(a2);
    v264.m128i_i64[0] = (__int64)&v271;
    v264.m128i_i64[1] = (__int64)&v272;
    v271 = &unk_49DA1B0;
    v258.m128i_i64[0] = (__int64)&v259;
    v258.m128i_i64[1] = 0x200000000LL;
    v272 = &unk_49DA0B0;
    v265 = 0;
    v266 = 0;
    v267 = 512;
    v268 = 7;
    v269 = 0;
    v270 = 0;
    v261 = 0;
    v262 = 0;
    v263.m128i_i16[0] = 0;
    sub_D5F1F0((__int64)&v258, a2);
    v86 = *(_QWORD *)(v211 - 64);
    if ( a3 == v198 )
    {
      v87 = (__int64)v212;
      v88 = *(_QWORD *)(v211 - 64);
    }
    else
    {
      v87 = *(_QWORD *)(v211 - 64);
      v88 = (__int64)v212;
    }
    v253 = 1;
    v250.m128i_i64[0] = (__int64)"spec.store.select";
    v252 = 3;
    v224 = sub_B36550((unsigned int **)&v258, (__int64)v199, v88, v87, (__int64)&v250, a2);
    sub_AC2B30(v211 - 64, v224);
    v89 = sub_B10CD0(v211 + 48);
    v90 = sub_B10CD0(a2 + 48);
    sub_AE8F10(v211, v90, v89);
    if ( (*(_BYTE *)(v211 + 7) & 0x20) != 0 )
    {
      v90 = 38;
      v91 = sub_B91C10(v211, 38);
      if ( v91 )
      {
        v92 = sub_AE94B0(v91);
        *(_QWORD *)&v243[0] = v86;
        v93 = v92;
        v95 = v94;
        *((_QWORD *)&v243[0] + 1) = v224;
        if ( v94 != v92 )
        {
          v219 = v86;
          do
          {
            v96 = *(_QWORD *)(v93 + 24);
            sub_B58E30(&v250, v96);
            v90 = (__int64)v243;
            if ( sub_BD2E50(v250.m128i_i64, (__int64 *)v243) )
            {
              v90 = *(_QWORD *)&v243[0];
              sub_B59720(v96, *(__int64 *)&v243[0], *((unsigned __int8 **)&v243[0] + 1));
            }
            v93 = *(_QWORD *)(v93 + 8);
          }
          while ( v95 != v93 );
          v86 = v219;
        }
      }
      if ( (*(_BYTE *)(v211 + 7) & 0x20) != 0 )
      {
        v90 = 38;
        v173 = sub_B91C10(v211, 38);
        if ( v173 )
        {
          v174 = *(_QWORD *)(v173 + 8);
          v90 = v174 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v174 & 4) == 0 )
            v90 = 0;
          sub_B967C0(&v250, (__m128i *)v90);
          v175 = (__int64 *)v250.m128i_i64[0];
          v239.m128i_i64[0] = v86;
          v140 = (__int64 *)(v250.m128i_i64[0] + 8LL * v250.m128i_u32[2]);
          v239.m128i_i64[1] = v224;
          if ( (__int64 *)v250.m128i_i64[0] != v140 )
          {
            do
            {
              v176 = *v175;
              sub_B129C0(v243, *v175);
              v90 = (__int64)&v239;
              if ( sub_F9EA20((__int64 *)v243, v239.m128i_i64) )
              {
                v90 = v239.m128i_i64[0];
                sub_B13360(v176, (unsigned __int8 *)v239.m128i_i64[0], (unsigned __int8 *)v239.m128i_i64[1], 0);
              }
              ++v175;
            }
            while ( v140 != v175 );
            v140 = (__int64 *)v250.m128i_i64[0];
          }
          if ( v140 != (__int64 *)&v251 )
            _libc_free(v140, v90);
        }
      }
    }
    nullsub_61();
    v271 = &unk_49DA1B0;
    nullsub_63();
    if ( (__m128i *)v258.m128i_i64[0] != &v259 )
      _libc_free(v258.m128i_i64[0], v90);
  }
  v141 = *(__int64 **)(a3 + 56);
  v142 = (__int64 *)(a3 + 48);
  if ( v141 != (__int64 *)(a3 + 48) )
  {
    do
    {
      v143 = v141;
      v141 = (__int64 *)v141[1];
      if ( (v143 - 3 != (__int64 *)v211 || !v212)
        && (*((_BYTE *)v143 - 24) != 85
         || (v172 = *(v143 - 7)) == 0
         || *(_BYTE *)v172
         || *(_QWORD *)(v172 + 24) != v143[7]
         || (*(_BYTE *)(v172 + 33) & 0x20) == 0
         || *(_DWORD *)(v172 + 36) != 68) )
      {
        v258.m128i_i64[0] = 0;
        if ( v143 + 3 != (__int64 *)&v258 )
        {
          v147 = v143[3];
          if ( v147 )
          {
            sub_B91220((__int64)(v143 + 3), v147);
            v148 = (unsigned __int8 *)v258.m128i_i64[0];
            v143[3] = v258.m128i_i64[0];
            if ( v148 )
              sub_B976B0((__int64)&v258, v148, (__int64)(v143 + 3));
          }
        }
      }
      sub_B44E20((unsigned __int8 *)v143 - 24);
      if ( (unsigned __int8)sub_B19060((__int64)&v273, (__int64)(v143 - 3), v144, v145) )
      {
        v146 = sub_ACADE0((__int64 **)*(v143 - 2));
        sub_BD84D0((__int64)(v143 - 3), v146);
        sub_B43D60(v143 - 3);
      }
    }
    while ( v141 != v142 );
    v149 = *(__int64 **)(a3 + 56);
    if ( v149 != v142 )
    {
      do
      {
        if ( !v149 )
          BUG();
        v150 = v149[5];
        if ( v150 )
        {
          v151 = sub_B14240(v150);
          v153 = v152;
          v154 = v151;
          while ( v154 != v153 )
          {
            v155 = v154;
            v154 = *(_QWORD *)(v154 + 8);
            if ( *(_BYTE *)(v155 + 32) || *(_BYTE *)(v155 + 64) != 2 )
              sub_B44590((__int64)(v149 - 3), (_QWORD *)v155);
          }
        }
        v149 = (__int64 *)v149[1];
      }
      while ( v149 != v142 );
      v142 = *(__int64 **)(a3 + 56);
    }
  }
  sub_AA80F0(
    v207,
    (unsigned __int64 *)(a2 + 24),
    0,
    a3,
    v142,
    1,
    (__int64 *)(*(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL),
    0);
  if ( (_DWORD)v229 )
  {
    LOBYTE(v181) = a3 != v198;
    HIBYTE(v181) = 1;
    sub_F956F0(a2, (__int64)&v228, v181);
  }
  v5 = a2;
  v263.m128i_i64[1] = sub_BD5C60(a2);
  v264.m128i_i64[0] = (__int64)&v271;
  v264.m128i_i64[1] = (__int64)&v272;
  v267 = 512;
  v271 = &unk_49DA1B0;
  v258.m128i_i64[0] = (__int64)&v259;
  v258.m128i_i64[1] = 0x200000000LL;
  v263.m128i_i16[0] = 0;
  v272 = &unk_49DA0B0;
  v265 = 0;
  v266 = 0;
  v268 = 7;
  v269 = 0;
  v270 = 0;
  v261 = 0;
  v262 = 0;
  sub_D5F1F0((__int64)&v258, a2);
  v156 = sub_AA5930(v195);
  v220 = v157;
  v158 = v156;
  for ( *(_QWORD *)&v243[0] = v156; v220 != *(_QWORD *)&v243[0]; v158 = *(_QWORD *)&v243[0] )
  {
    v159 = *(_QWORD *)(v158 - 8);
    v160 = *(_DWORD *)(v158 + 4) & 0x7FFFFFF;
    if ( v160 )
    {
      v161 = 0;
      v162 = v159 + 32LL * *(unsigned int *)(v158 + 72);
      do
      {
        if ( v207 == *(_QWORD *)(v162 + 8 * v161) )
        {
          v163 = 32 * v161;
          goto LABEL_339;
        }
        ++v161;
      }
      while ( v160 != (_DWORD)v161 );
      v163 = 0x1FFFFFFFE0LL;
LABEL_339:
      v164 = 0;
      do
      {
        if ( a3 == *(_QWORD *)(v162 + 8 * v164) )
        {
          v165 = 32 * v164;
          goto LABEL_343;
        }
        ++v164;
      }
      while ( v160 != (_DWORD)v164 );
      v165 = 0x1FFFFFFFE0LL;
    }
    else
    {
      v165 = 0x1FFFFFFFE0LL;
      v163 = 0x1FFFFFFFE0LL;
    }
LABEL_343:
    v166 = *(_QWORD *)(v159 + v163);
    v167 = *(_QWORD *)(v159 + v165);
    if ( v166 != v167 )
    {
      if ( a3 != v198 )
      {
        v167 = *(_QWORD *)(v159 + v163);
        v166 = *(_QWORD *)(v159 + v165);
      }
      v253 = 1;
      v250.m128i_i64[0] = (__int64)"spec.select";
      v252 = 3;
      v225 = sub_B36550((unsigned int **)&v258, (__int64)v199, v167, v166, (__int64)&v250, a2);
      sub_AC2B30(v163 + *(_QWORD *)(v158 - 8), v225);
      v5 = v225;
      sub_AC2B30(v165 + *(_QWORD *)(v158 - 8), v225);
    }
    sub_F8F2F0((__int64)v243);
  }
  v168 = (_QWORD **)v231;
  v169 = (_QWORD **)&v231[8 * (unsigned int)v232];
  if ( v231 != (_BYTE *)v169 )
  {
    do
    {
      v170 = *v168;
      if ( *(_BYTE *)*v168 != 85
        || (v171 = *(v170 - 4)) == 0
        || *(_BYTE *)v171
        || *(_QWORD *)(v171 + 24) != v170[10]
        || (*(_BYTE *)(v171 + 33) & 0x20) == 0
        || *(_DWORD *)(v171 + 36) != 68 )
      {
        sub_B43D60(v170);
      }
      ++v168;
    }
    while ( v169 != v168 );
  }
  nullsub_61();
  v271 = &unk_49DA1B0;
  nullsub_63();
  if ( (__m128i *)v258.m128i_i64[0] != &v259 )
    _libc_free(v258.m128i_i64[0], v5);
LABEL_57:
  if ( !v277 )
    _libc_free(v274, v5);
  if ( v228 != v230 )
    _libc_free(v228, v5);
  if ( v231 != v233 )
    _libc_free(v231, v5);
  if ( (v247 & 1) == 0 )
    sub_C7D6A0((__int64)v248, 16LL * v249, 8);
  return v227;
}
