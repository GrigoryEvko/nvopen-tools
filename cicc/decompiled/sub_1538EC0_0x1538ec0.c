// Function: sub_1538EC0
// Address: 0x1538ec0
//
__int64 __fastcall sub_1538EC0(__int64 *a1, _QWORD *a2, unsigned __int8 a3, __int64 a4, char a5, __m128i *a6)
{
  _BYTE *v9; // rsi
  unsigned int *v10; // rax
  __int64 v11; // rsi
  int *v12; // r12
  __m128i *v13; // r15
  int *v14; // r13
  __int64 *v15; // r14
  __int64 v16; // rax
  _QWORD *v17; // rcx
  _QWORD *v18; // rbx
  unsigned __int64 **v19; // rax
  _QWORD *v20; // r15
  unsigned __int64 **v21; // r12
  unsigned __int64 *v22; // rdx
  unsigned __int64 v23; // rax
  int *v24; // rdi
  int v25; // edx
  int *v26; // rsi
  __int64 v27; // r8
  __int64 v28; // rcx
  __int64 v29; // rax
  _QWORD *v30; // r12
  __int64 v31; // rax
  unsigned int v32; // ebx
  __int64 v33; // rax
  __int64 v34; // rdi
  volatile signed __int32 *v35; // r8
  __int64 v36; // rax
  unsigned int v37; // r13d
  unsigned int v38; // eax
  _QWORD *v39; // r12
  _QWORD *v40; // rax
  _QWORD *v41; // rsi
  _QWORD *v42; // rbx
  __int64 v43; // r13
  __int64 v44; // rbx
  volatile signed __int32 *v45; // r12
  signed __int32 v46; // edx
  signed __int32 v47; // edx
  _QWORD *v48; // rdi
  _QWORD *v49; // rdi
  _QWORD *v50; // rdi
  _QWORD *v51; // rdi
  __int64 v52; // r12
  _QWORD *v53; // rdi
  _QWORD *v54; // rdi
  __int64 v55; // r12
  _QWORD *v56; // rdi
  _QWORD *v57; // rdi
  __int64 v58; // r12
  _QWORD *v59; // rdi
  _QWORD *v60; // rdi
  _QWORD *v61; // rdi
  __int64 v62; // r12
  _QWORD *v63; // rdi
  _QWORD *v64; // rdi
  _QWORD *v65; // rdi
  _QWORD *v66; // rdi
  __int64 v67; // rdx
  __int64 v68; // rax
  _QWORD *v69; // rdi
  size_t v70; // rdx
  __int64 v71; // rax
  __int64 v72; // rdx
  unsigned int v73; // esi
  unsigned __int64 v74; // rdi
  __int64 v75; // rax
  int v76; // esi
  unsigned int *v77; // rbx
  _QWORD *v78; // rdi
  unsigned int v79; // edx
  _QWORD *v80; // rax
  __int64 v81; // r8
  __int64 v82; // rbx
  __int64 k; // r14
  __int64 v84; // rax
  __int64 v85; // r12
  int v86; // ebx
  __int64 v87; // rax
  __int64 v88; // rax
  int v89; // ebx
  int v90; // ebx
  __int64 v91; // rax
  __int64 v92; // rsi
  unsigned int v93; // ecx
  __int64 *v94; // rdx
  __int64 v95; // r9
  __int64 v96; // rsi
  __int64 v97; // rax
  int v98; // ebx
  unsigned int v99; // ecx
  __int64 *v100; // rdx
  __int64 v101; // r9
  _DWORD *v102; // rbx
  unsigned int v103; // r15d
  int v104; // ecx
  int v105; // r10d
  int v106; // eax
  unsigned int v107; // ecx
  unsigned int v108; // r9d
  int v109; // eax
  unsigned int v110; // ecx
  unsigned int v111; // eax
  __int64 v112; // rdi
  __int64 v113; // rdx
  int v114; // edx
  unsigned int v115; // eax
  unsigned int v116; // ecx
  __int64 v117; // rdi
  __int64 v118; // rdx
  int v119; // edx
  unsigned int m; // r10d
  int v121; // edx
  unsigned int v122; // r8d
  unsigned int v123; // r8d
  unsigned int v124; // edx
  __int64 v125; // rdi
  __int64 v126; // rdx
  unsigned int v127; // edx
  unsigned int v128; // ecx
  int v129; // eax
  __int64 v130; // r11
  __int64 v131; // r10
  unsigned int v132; // r15d
  unsigned int v133; // ecx
  unsigned int n; // eax
  int v135; // edx
  unsigned int v136; // r8d
  unsigned int v137; // r8d
  unsigned int v138; // edx
  __int64 v139; // rdi
  __int64 v140; // rdx
  unsigned int v141; // edx
  unsigned int v142; // ecx
  int v143; // r15d
  __int64 **v144; // rbx
  int v145; // edx
  __int64 *v146; // rax
  __int64 v147; // rcx
  __int64 *v148; // rcx
  __int64 v149; // rsi
  __int64 *v150; // r12
  __int64 *v151; // r14
  char *v152; // rbx
  unsigned int v153; // r15d
  __int64 v154; // r10
  __int64 v155; // rax
  __int64 v156; // rax
  unsigned int v157; // r11d
  char *v158; // r12
  __int64 v159; // r14
  __int64 *v160; // rdx
  __int64 v161; // rax
  __int64 *v162; // rax
  __int64 v163; // rdx
  __int64 v164; // rdi
  __int64 v165; // rdx
  unsigned int v166; // eax
  int v167; // edx
  unsigned int v168; // r10d
  __int64 v169; // rax
  __int64 v170; // rdx
  int v171; // edx
  __int64 v172; // rdi
  __int64 v173; // rdx
  int v174; // edx
  unsigned int v175; // eax
  int v176; // edx
  int v177; // edx
  int v178; // r8d
  unsigned __int64 *v179; // rsi
  unsigned int v180; // r8d
  char v181; // r11
  __int64 v182; // rax
  unsigned int v183; // r14d
  _QWORD *v184; // r13
  unsigned int v185; // r8d
  _QWORD *v186; // rax
  __int64 v187; // rdi
  __int64 v188; // rax
  _DWORD *v189; // r12
  _QWORD *v190; // rbx
  __int64 v191; // rcx
  int v192; // edx
  __int64 v193; // r11
  int v194; // edi
  _QWORD *v195; // rsi
  int *v196; // rbx
  _QWORD *v197; // rdi
  int v199; // r8d
  __int64 *v200; // r13
  __int64 *v201; // rbx
  unsigned __int32 v202; // eax
  __int64 v203; // r14
  __int64 v204; // r12
  unsigned __int64 v205; // rdx
  __int64 v206; // r15
  __int64 v207; // rax
  __int64 v208; // rcx
  __int64 j; // rax
  __int64 *v210; // r13
  __int64 *v211; // rbx
  unsigned __int32 v212; // eax
  __int64 v213; // r14
  __int64 v214; // r12
  unsigned __int64 v215; // rdx
  __int64 v216; // r15
  __int64 v217; // rax
  __int64 v218; // rcx
  __int64 i; // rax
  bool v220; // zf
  __int64 v221; // rsi
  unsigned int *v222; // rax
  int ii; // edx
  int v224; // ecx
  unsigned int v225; // esi
  _DWORD *v226; // rbx
  unsigned __int64 v227; // rsi
  __m128i *v228; // rax
  int v229; // ecx
  _QWORD *v230; // rdx
  _QWORD *v231; // rcx
  __int64 v232; // r12
  int v233; // esi
  __int64 v234; // rdi
  int v235; // edi
  _QWORD *v236; // rcx
  int v237; // edx
  __int64 v238; // rcx
  _QWORD *v239; // [rsp+10h] [rbp-740h]
  unsigned int v240; // [rsp+1Ch] [rbp-734h]
  __int64 v241; // [rsp+28h] [rbp-728h]
  _QWORD *v242; // [rsp+30h] [rbp-720h]
  unsigned int v243; // [rsp+38h] [rbp-718h]
  __int64 v244; // [rsp+38h] [rbp-718h]
  __int64 v245; // [rsp+38h] [rbp-718h]
  unsigned int v246; // [rsp+40h] [rbp-710h]
  unsigned int v247; // [rsp+40h] [rbp-710h]
  int v248; // [rsp+40h] [rbp-710h]
  unsigned int v249; // [rsp+40h] [rbp-710h]
  __int64 v250; // [rsp+40h] [rbp-710h]
  char v251; // [rsp+4Bh] [rbp-705h]
  int v252; // [rsp+4Ch] [rbp-704h]
  int v253; // [rsp+4Ch] [rbp-704h]
  unsigned int v254; // [rsp+4Ch] [rbp-704h]
  unsigned int v255; // [rsp+4Ch] [rbp-704h]
  unsigned int v256; // [rsp+4Ch] [rbp-704h]
  char v257; // [rsp+4Ch] [rbp-704h]
  unsigned int v258; // [rsp+4Ch] [rbp-704h]
  __int64 v260; // [rsp+50h] [rbp-700h]
  __int64 v262; // [rsp+60h] [rbp-6F0h]
  unsigned int v263; // [rsp+60h] [rbp-6F0h]
  unsigned int v264; // [rsp+60h] [rbp-6F0h]
  unsigned int *v265; // [rsp+68h] [rbp-6E8h]
  __int64 v266; // [rsp+68h] [rbp-6E8h]
  __int64 *v267; // [rsp+68h] [rbp-6E8h]
  __int64 v268; // [rsp+68h] [rbp-6E8h]
  __int64 v269; // [rsp+70h] [rbp-6E0h]
  __m128i *v270; // [rsp+70h] [rbp-6E0h]
  _QWORD *v271; // [rsp+78h] [rbp-6D8h]
  __int64 *v272; // [rsp+78h] [rbp-6D8h]
  __int64 v273; // [rsp+80h] [rbp-6D0h]
  _QWORD *v274; // [rsp+80h] [rbp-6D0h]
  __int64 v275; // [rsp+80h] [rbp-6D0h]
  __int64 *v276; // [rsp+A0h] [rbp-6B0h]
  _QWORD *v277; // [rsp+A8h] [rbp-6A8h]
  _QWORD *v278; // [rsp+A8h] [rbp-6A8h]
  __int64 v279; // [rsp+B8h] [rbp-698h] BYREF
  __int64 v280; // [rsp+C0h] [rbp-690h] BYREF
  __int64 v281; // [rsp+C8h] [rbp-688h]
  __int64 v282; // [rsp+D0h] [rbp-680h]
  unsigned int v283; // [rsp+D8h] [rbp-678h]
  unsigned __int128 v284; // [rsp+E0h] [rbp-670h] BYREF
  _BYTE v285[256]; // [rsp+F0h] [rbp-660h] BYREF
  unsigned __int128 v286; // [rsp+1F0h] [rbp-560h] BYREF
  __int32 v287; // [rsp+200h] [rbp-550h] BYREF
  char v288; // [rsp+204h] [rbp-54Ch] BYREF
  _QWORD *v289[2]; // [rsp+400h] [rbp-350h] BYREF
  _QWORD *v290; // [rsp+410h] [rbp-340h]
  __int64 v291[14]; // [rsp+418h] [rbp-338h] BYREF
  __int64 v292; // [rsp+488h] [rbp-2C8h]
  __int64 v293; // [rsp+490h] [rbp-2C0h]
  __int64 v294; // [rsp+4E8h] [rbp-268h]
  __int64 v295; // [rsp+4F0h] [rbp-260h]
  __int64 v296; // [rsp+520h] [rbp-230h]
  int v297; // [rsp+530h] [rbp-220h]
  char v298; // [rsp+558h] [rbp-1F8h]
  __int64 v299; // [rsp+618h] [rbp-138h]
  __int64 v300; // [rsp+620h] [rbp-130h]
  unsigned int v301; // [rsp+634h] [rbp-11Ch]
  unsigned int v302; // [rsp+638h] [rbp-118h]
  unsigned int v303; // [rsp+63Ch] [rbp-114h]
  unsigned int v304; // [rsp+640h] [rbp-110h]
  __int64 v305; // [rsp+648h] [rbp-108h]
  __int64 v306; // [rsp+650h] [rbp-100h] BYREF
  int v307; // [rsp+658h] [rbp-F8h] BYREF
  int *v308; // [rsp+660h] [rbp-F0h]
  int *v309; // [rsp+668h] [rbp-E8h]
  int *v310; // [rsp+670h] [rbp-E0h]
  __int64 v311; // [rsp+678h] [rbp-D8h]
  int v312; // [rsp+680h] [rbp-D0h]
  unsigned __int64 v313; // [rsp+688h] [rbp-C8h]
  __int64 v314; // [rsp+690h] [rbp-C0h]
  char v315; // [rsp+698h] [rbp-B8h]
  __m128i *v316; // [rsp+6A0h] [rbp-B0h]
  _BYTE v317[112]; // [rsp+6A8h] [rbp-A8h] BYREF
  __int64 v318; // [rsp+718h] [rbp-38h]

  v289[0] = a2;
  v9 = (_BYTE *)a1[24];
  if ( v9 == (_BYTE *)a1[25] )
  {
    sub_12DDA10((__int64)(a1 + 23), v9, v289);
  }
  else
  {
    if ( v9 )
    {
      *(_QWORD *)v9 = a2;
      v9 = (_BYTE *)a1[24];
    }
    a1[24] = (__int64)(v9 + 8);
  }
  v10 = (unsigned int *)a1[1];
  v11 = *a1;
  v289[1] = a1 + 2;
  v12 = &v307;
  v265 = v10;
  v289[0] = v10;
  v269 = v11;
  v290 = a2;
  sub_15467B0(v291, a2, a3);
  v305 = a4;
  v309 = &v307;
  v307 = 0;
  v308 = 0;
  v310 = &v307;
  v311 = 0;
  v313 = 0;
  v312 = (v293 - v292) >> 4;
  if ( !a4 || (v262 = a4 + 8, v273 = *(_QWORD *)(a4 + 24), v273 == a4 + 8) )
  {
    v13 = (__m128i *)&v286;
  }
  else
  {
    v13 = (__m128i *)&v286;
    do
    {
      v14 = v12;
      v15 = *(__int64 **)(v273 + 56);
      v276 = *(__int64 **)(v273 + 64);
      if ( v15 != v276 )
      {
        do
        {
          v16 = *v15;
          if ( *(_DWORD *)(*v15 + 8) == 1 )
          {
            v17 = *(_QWORD **)(v16 + 72);
            v18 = *(_QWORD **)(v16 + 80);
            if ( v17 != v18 )
            {
              v19 = (unsigned __int64 **)v13;
              v20 = v17;
              v21 = v19;
              do
              {
                while ( 1 )
                {
                  v22 = (unsigned __int64 *)(*v20 & 0xFFFFFFFFFFFFFFF8LL);
                  if ( (*v20 & 4) == 0 || !v22[1] )
                    break;
                  v20 += 2;
                  if ( v18 == v20 )
                    goto LABEL_24;
                }
                v23 = *v22;
                v24 = v308;
                v25 = v312 + 1;
                *(_QWORD *)&v284 = v23;
                v26 = v14;
                ++v312;
                if ( !v308 )
                  goto LABEL_22;
                do
                {
                  while ( 1 )
                  {
                    v27 = *((_QWORD *)v24 + 2);
                    v28 = *((_QWORD *)v24 + 3);
                    if ( v23 <= *((_QWORD *)v24 + 4) )
                      break;
                    v24 = (int *)*((_QWORD *)v24 + 3);
                    if ( !v28 )
                      goto LABEL_20;
                  }
                  v26 = v24;
                  v24 = (int *)*((_QWORD *)v24 + 2);
                }
                while ( v27 );
LABEL_20:
                if ( v26 == v14 || v23 < *((_QWORD *)v26 + 4) )
                {
LABEL_22:
                  *(_QWORD *)&v286 = &v284;
                  v29 = sub_1536920(&v306, v26, v21);
                  v25 = v312;
                  v26 = (int *)v29;
                }
                v20 += 2;
                v26[10] = v25;
              }
              while ( v18 != v20 );
LABEL_24:
              v13 = (__m128i *)v21;
            }
          }
          ++v15;
        }
        while ( v276 != v15 );
        v12 = v14;
      }
      v273 = sub_220EF30(v273);
    }
    while ( v262 != v273 );
  }
  v314 = v269;
  v315 = a5;
  v316 = a6;
  sub_16C9F80(v317);
  v30 = v289[0];
  v318 = v265[2] + 8LL * *(unsigned int *)(*(_QWORD *)v265 + 8LL);
  sub_1526BE0(v289[0], 0xDu, 5u);
  sub_1531130(&v284);
  BYTE8(v286) |= 1u;
  *(_QWORD *)&v286 = 1;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 0;
  BYTE8(v286) = 6;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 0;
  BYTE8(v286) = 8;
  sub_1525B40(v284, v13);
  v31 = *((_QWORD *)&v284 + 1);
  *(_QWORD *)&v286 = v284;
  v284 = 0u;
  *((_QWORD *)&v286 + 1) = v31;
  v32 = sub_15271D0(v30, v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  sub_1528330(v30, 1u, "LLVM7.0.1", 9, v32);
  sub_1531130(v13);
  v33 = *((_QWORD *)&v286 + 1);
  v34 = v286;
  v286 = 0u;
  v35 = (volatile signed __int32 *)*((_QWORD *)&v284 + 1);
  v284 = __PAIR128__(v33, v34);
  if ( v35 )
  {
    sub_A191D0(v35);
    if ( *((_QWORD *)&v286 + 1) )
      sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
    v34 = v284;
  }
  BYTE8(v286) |= 1u;
  *(_QWORD *)&v286 = 2;
  sub_1525B40(v34, v13);
  *(_QWORD *)&v286 = 6;
  BYTE8(v286) = 4;
  sub_1525B40(v284, v13);
  v36 = *((_QWORD *)&v284 + 1);
  *(_QWORD *)&v286 = v284;
  v284 = 0u;
  *((_QWORD *)&v286 + 1) = v36;
  v37 = sub_15271D0(v30, v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  v287 = 0;
  *(_QWORD *)&v286 = &v287;
  *((_QWORD *)&v286 + 1) = 0x100000001LL;
  if ( v37 )
  {
    v280 = 0x100000002LL;
    sub_1527BB0((__int64)v30, v37, (__int64)&v287, 1, 0, 0, (__int64)&v280);
  }
  else
  {
    sub_1524D80(v30, 3u, *((_DWORD *)v30 + 4));
    sub_1524E40(v30, 2u, 6);
    sub_1524E40(v30, 1u, 6);
    sub_1525280(v30, *(unsigned int *)v286, 6);
  }
  sub_15263C0((__int64 **)v30);
  if ( (__int32 *)v286 != &v287 )
    _libc_free(v286);
  if ( *((_QWORD *)&v284 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v284 + 1));
  sub_1526BE0(v289[0], 8u, 3u);
  v38 = *(_DWORD *)(v314 + 8);
  *(_QWORD *)&v286 = &v284;
  *(_QWORD *)&v284 = 2;
  v240 = v38;
  *((_QWORD *)&v286 + 1) = 1;
  sub_152A900((_DWORD *)v289[0], 1u, v13->m128i_i64, 0);
  v39 = v289[0];
  sub_1526BE0(v289[0], 0, 2u);
  v40 = (_QWORD *)v39[9];
  v41 = (_QWORD *)v39[10];
  *((_DWORD *)v39 + 5) = -1;
  v277 = v40;
  if ( v40 != v41 )
  {
    v274 = v39;
    v42 = v40;
    do
    {
      v43 = v42[1];
      if ( v42[2] != v43 )
      {
        v271 = v42;
        v44 = v42[2];
        do
        {
          while ( 1 )
          {
            v45 = *(volatile signed __int32 **)(v43 + 8);
            if ( v45 )
            {
              if ( &_pthread_key_create )
              {
                v46 = _InterlockedExchangeAdd(v45 + 2, 0xFFFFFFFF);
              }
              else
              {
                v46 = *((_DWORD *)v45 + 2);
                *((_DWORD *)v45 + 2) = v46 - 1;
              }
              if ( v46 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v45 + 16LL))(v45);
                if ( &_pthread_key_create )
                {
                  v47 = _InterlockedExchangeAdd(v45 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v47 = *((_DWORD *)v45 + 3);
                  *((_DWORD *)v45 + 3) = v47 - 1;
                }
                if ( v47 == 1 )
                  break;
              }
            }
            v43 += 16;
            if ( v44 == v43 )
              goto LABEL_55;
          }
          v43 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v45 + 24LL))(v45);
        }
        while ( v44 != v43 );
LABEL_55:
        v42 = v271;
        v43 = v271[1];
      }
      if ( v43 )
        j_j___libc_free_0(v43, v42[3] - v43);
      v42 += 4;
    }
    while ( v41 != v42 );
    v274[10] = v277;
  }
  sub_1531130(&v284);
  *(_QWORD *)&v286 = 3;
  BYTE8(v286) = 2;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 8;
  BYTE8(v286) = 4;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 0;
  BYTE8(v286) = 6;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 8;
  BYTE8(v286) = 2;
  sub_1525B40(v284, v13);
  v48 = v289[0];
  v286 = v284;
  if ( *((_QWORD *)&v284 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v284 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v284 + 1) + 8LL);
  }
  sub_1527040((__int64)v48, 0xEu, v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  if ( *((_QWORD *)&v284 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v284 + 1));
  sub_1531130(&v284);
  BYTE8(v286) |= 1u;
  *(_QWORD *)&v286 = 1;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 8;
  BYTE8(v286) = 4;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 0;
  BYTE8(v286) = 6;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 7;
  BYTE8(v286) = 2;
  sub_1525B40(v284, v13);
  v49 = v289[0];
  v286 = v284;
  if ( *((_QWORD *)&v284 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v284 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v284 + 1) + 8LL);
  }
  sub_1527040((__int64)v49, 0xEu, v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  if ( *((_QWORD *)&v284 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v284 + 1));
  sub_1531130(&v284);
  BYTE8(v286) |= 1u;
  *(_QWORD *)&v286 = 1;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 8;
  BYTE8(v286) = 4;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 0;
  BYTE8(v286) = 6;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 0;
  BYTE8(v286) = 8;
  sub_1525B40(v284, v13);
  v50 = v289[0];
  v286 = v284;
  if ( *((_QWORD *)&v284 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v284 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v284 + 1) + 8LL);
  }
  sub_1527040((__int64)v50, 0xEu, v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  if ( *((_QWORD *)&v284 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v284 + 1));
  sub_1531130(&v284);
  BYTE8(v286) |= 1u;
  *(_QWORD *)&v286 = 2;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 8;
  BYTE8(v286) = 4;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 0;
  BYTE8(v286) = 6;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 0;
  BYTE8(v286) = 8;
  sub_1525B40(v284, v13);
  v51 = v289[0];
  v286 = v284;
  if ( *((_QWORD *)&v284 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v284 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v284 + 1) + 8LL);
  }
  sub_1527040((__int64)v51, 0xEu, v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  if ( *((_QWORD *)&v284 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v284 + 1));
  sub_1531130(&v284);
  BYTE8(v286) |= 1u;
  *(_QWORD *)&v286 = 1;
  sub_1525B40(v284, v13);
  v52 = v284;
  *(_QWORD *)&v286 = sub_153EE90(v291);
  BYTE8(v286) = BYTE8(v286) & 0xF0 | 2;
  sub_1525B40(v52, v13);
  v53 = v289[0];
  v286 = v284;
  if ( *((_QWORD *)&v284 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v284 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v284 + 1) + 8LL);
  }
  sub_1527040((__int64)v53, 0xBu, v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  if ( *((_QWORD *)&v284 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v284 + 1));
  sub_1531130(&v284);
  BYTE8(v286) |= 1u;
  *(_QWORD *)&v286 = 4;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 8;
  BYTE8(v286) = 4;
  sub_1525B40(v284, v13);
  v54 = v289[0];
  v286 = v284;
  if ( *((_QWORD *)&v284 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v284 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v284 + 1) + 8LL);
  }
  sub_1527040((__int64)v54, 0xBu, v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  if ( *((_QWORD *)&v284 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v284 + 1));
  sub_1531130(&v284);
  BYTE8(v286) |= 1u;
  *(_QWORD *)&v286 = 11;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 4;
  BYTE8(v286) = 2;
  sub_1525B40(v284, v13);
  v55 = v284;
  *(_QWORD *)&v286 = sub_153EE90(v291);
  BYTE8(v286) = BYTE8(v286) & 0xF0 | 2;
  sub_1525B40(v55, v13);
  *(_QWORD *)&v286 = 8;
  BYTE8(v286) = 4;
  sub_1525B40(v284, v13);
  v56 = v289[0];
  v286 = v284;
  if ( *((_QWORD *)&v284 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v284 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v284 + 1) + 8LL);
  }
  sub_1527040((__int64)v56, 0xBu, v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  if ( *((_QWORD *)&v284 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v284 + 1));
  sub_1531130(&v284);
  BYTE8(v286) |= 1u;
  *(_QWORD *)&v286 = 2;
  sub_1525B40(v284, v13);
  v57 = v289[0];
  v286 = v284;
  if ( *((_QWORD *)&v284 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v284 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v284 + 1) + 8LL);
  }
  sub_1527040((__int64)v57, 0xBu, v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  if ( *((_QWORD *)&v284 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v284 + 1));
  sub_1531130(&v284);
  BYTE8(v286) |= 1u;
  *(_QWORD *)&v286 = 20;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 6;
  BYTE8(v286) = 4;
  sub_1525B40(v284, v13);
  v58 = v284;
  *(_QWORD *)&v286 = sub_153EE90(v291);
  BYTE8(v286) = BYTE8(v286) & 0xF0 | 2;
  sub_1525B40(v58, v13);
  *(_QWORD *)&v286 = 4;
  BYTE8(v286) = 4;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 1;
  BYTE8(v286) = 2;
  sub_1525B40(v284, v13);
  v59 = v289[0];
  v286 = v284;
  if ( *((_QWORD *)&v284 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v284 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v284 + 1) + 8LL);
  }
  sub_1527040((__int64)v59, 0xCu, v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  if ( *((_QWORD *)&v284 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v284 + 1));
  sub_1531130(&v284);
  BYTE8(v286) |= 1u;
  *(_QWORD *)&v286 = 2;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 6;
  BYTE8(v286) = 4;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 6;
  BYTE8(v286) = 4;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 4;
  BYTE8(v286) = 2;
  sub_1525B40(v284, v13);
  v60 = v289[0];
  v286 = v284;
  if ( *((_QWORD *)&v284 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v284 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v284 + 1) + 8LL);
  }
  sub_1527040((__int64)v60, 0xCu, v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  if ( *((_QWORD *)&v284 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v284 + 1));
  sub_1531130(&v284);
  BYTE8(v286) |= 1u;
  *(_QWORD *)&v286 = 2;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 6;
  BYTE8(v286) = 4;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 6;
  BYTE8(v286) = 4;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 4;
  BYTE8(v286) = 2;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 8;
  BYTE8(v286) = 2;
  sub_1525B40(v284, v13);
  v61 = v289[0];
  v286 = v284;
  if ( *((_QWORD *)&v284 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v284 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v284 + 1) + 8LL);
  }
  sub_1527040((__int64)v61, 0xCu, v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  if ( *((_QWORD *)&v284 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v284 + 1));
  sub_1531130(&v284);
  BYTE8(v286) |= 1u;
  *(_QWORD *)&v286 = 3;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 6;
  BYTE8(v286) = 4;
  sub_1525B40(v284, v13);
  v62 = v284;
  *(_QWORD *)&v286 = sub_153EE90(v291);
  BYTE8(v286) = BYTE8(v286) & 0xF0 | 2;
  sub_1525B40(v62, v13);
  *(_QWORD *)&v286 = 4;
  BYTE8(v286) = 2;
  sub_1525B40(v284, v13);
  v63 = v289[0];
  v286 = v284;
  if ( *((_QWORD *)&v284 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v284 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v284 + 1) + 8LL);
  }
  sub_1527040((__int64)v63, 0xCu, v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  if ( *((_QWORD *)&v284 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v284 + 1));
  sub_1531130(&v284);
  BYTE8(v286) |= 1u;
  *(_QWORD *)&v286 = 10;
  sub_1525B40(v284, v13);
  v64 = v289[0];
  v286 = v284;
  if ( *((_QWORD *)&v284 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v284 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v284 + 1) + 8LL);
  }
  sub_1527040((__int64)v64, 0xCu, v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  if ( *((_QWORD *)&v284 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v284 + 1));
  sub_1531130(&v284);
  BYTE8(v286) |= 1u;
  *(_QWORD *)&v286 = 10;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 6;
  BYTE8(v286) = 4;
  sub_1525B40(v284, v13);
  v65 = v289[0];
  v286 = v284;
  if ( *((_QWORD *)&v284 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v284 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v284 + 1) + 8LL);
  }
  sub_1527040((__int64)v65, 0xCu, v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  if ( *((_QWORD *)&v284 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v284 + 1));
  sub_1531130(&v284);
  BYTE8(v286) |= 1u;
  *(_QWORD *)&v286 = 15;
  sub_1525B40(v284, v13);
  v66 = v289[0];
  v286 = v284;
  if ( *((_QWORD *)&v284 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v284 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v284 + 1) + 8LL);
  }
  sub_1527040((__int64)v66, 0xCu, v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  if ( *((_QWORD *)&v284 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v284 + 1));
  sub_1531130(&v284);
  BYTE8(v286) |= 1u;
  *(_QWORD *)&v286 = 43;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 1;
  BYTE8(v286) = 2;
  sub_1525B40(v284, v13);
  v67 = 0;
  v68 = (v291[8] - v291[7]) >> 3;
  if ( (_DWORD)v68 )
  {
    _BitScanReverse((unsigned int *)&v68, v68);
    v67 = 32 - ((unsigned int)v68 ^ 0x1F);
  }
  *(_QWORD *)&v286 = v67;
  BYTE8(v286) = BYTE8(v286) & 0xF0 | 2;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 0;
  BYTE8(v286) = 6;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 6;
  BYTE8(v286) = 4;
  sub_1525B40(v284, v13);
  v69 = v289[0];
  v286 = v284;
  if ( *((_QWORD *)&v284 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v284 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v284 + 1) + 8LL);
  }
  sub_1527040((__int64)v69, 0xCu, v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  if ( *((_QWORD *)&v284 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v284 + 1));
  sub_15263C0((__int64 **)v289[0]);
  sub_152F610((_DWORD **)v289);
  sub_152FD80((_DWORD **)v289);
  sub_15311A0((__int64 ***)v289);
  sub_1524F60(v289);
  sub_1536CD0((__int64 *)v289, 12, v70);
  v71 = v292;
  v72 = (v293 - v292) >> 4;
  if ( (_DWORD)v72 )
  {
    v73 = 0;
    while ( *(_BYTE *)(*(_QWORD *)v71 + 16LL) <= 3u )
    {
      ++v73;
      v71 += 16;
      if ( (_DWORD)v72 == v73 )
        goto LABEL_179;
    }
    sub_1531F90((__int64 ***)v289, v73, v72, 1);
  }
LABEL_179:
  sub_1530020((__int64 ***)v289);
  sub_1533CF0((__int64 *)v289);
  if ( v298 )
    sub_1530BF0((__int64)v289, 0);
  *(_QWORD *)&v284 = v285;
  *((_QWORD *)&v284 + 1) = 0x800000000LL;
  sub_1632070(v290, &v284);
  if ( !DWORD2(v284) )
  {
    v74 = v284;
    if ( (_BYTE *)v284 == v285 )
      goto LABEL_184;
    goto LABEL_183;
  }
  sub_1526BE0(v289[0], 0x15u, 3u);
  v210 = (__int64 *)v284;
  *(_QWORD *)&v286 = &v287;
  v211 = (__int64 *)(v284 + 16LL * DWORD2(v284));
  *((_QWORD *)&v286 + 1) = 0x4000000000LL;
  if ( (__int64 *)v284 != v211 )
  {
    v212 = 64;
    v213 = (__int64)v13;
    while ( 1 )
    {
      v214 = v210[1];
      v215 = v212;
      v216 = *v210;
      v217 = 0;
      if ( v214 > v215 )
      {
        sub_16CD150(v213, &v287, v210[1], 8);
        v217 = DWORD2(v286);
      }
      v218 = v286 + 8 * v217;
      if ( v214 > 0 )
      {
        for ( i = 0; i != v214; ++i )
          *(_QWORD *)(v218 + 8 * i) = *(char *)(v216 + i);
        LODWORD(v217) = DWORD2(v286);
      }
      v210 += 2;
      DWORD2(v286) = v217 + v214;
      sub_152F3D0((_DWORD *)v289[0], 1u, v213, 0);
      DWORD2(v286) = 0;
      if ( v211 == v210 )
        break;
      v212 = HIDWORD(v286);
    }
    v13 = (__m128i *)v213;
  }
  sub_15263C0((__int64 **)v289[0]);
  if ( (__int32 *)v286 != &v287 )
    _libc_free(v286);
  v74 = v284;
  if ( (_BYTE *)v284 != v285 )
LABEL_183:
    _libc_free(v74);
LABEL_184:
  *(_QWORD *)&v284 = v285;
  *((_QWORD *)&v284 + 1) = 0x800000000LL;
  sub_16032E0(*v290, &v284);
  if ( DWORD2(v284) )
  {
    sub_1526BE0(v289[0], 0x1Au, 2u);
    v200 = (__int64 *)v284;
    *(_QWORD *)&v286 = &v287;
    v201 = (__int64 *)(v284 + 16LL * DWORD2(v284));
    *((_QWORD *)&v286 + 1) = 0x4000000000LL;
    if ( (__int64 *)v284 != v201 )
    {
      v202 = 64;
      v203 = (__int64)v13;
      while ( 1 )
      {
        v204 = v200[1];
        v205 = v202;
        v206 = *v200;
        v207 = 0;
        if ( v204 > v205 )
        {
          sub_16CD150(v203, &v287, v200[1], 8);
          v207 = DWORD2(v286);
        }
        v208 = v286 + 8 * v207;
        if ( v204 > 0 )
        {
          for ( j = 0; j != v204; ++j )
            *(_QWORD *)(v208 + 8 * j) = *(char *)(v206 + j);
          LODWORD(v207) = DWORD2(v286);
        }
        v200 += 2;
        DWORD2(v286) = v207 + v204;
        sub_152F3D0((_DWORD *)v289[0], 1u, v203, 0);
        DWORD2(v286) = 0;
        if ( v201 == v200 )
          break;
        v202 = HIDWORD(v286);
      }
      v13 = (__m128i *)v203;
    }
    sub_15263C0((__int64 **)v289[0]);
    if ( (__int32 *)v286 != &v287 )
      _libc_free(v286);
  }
  if ( (_BYTE *)v284 != v285 )
    _libc_free(v284);
  v280 = 0;
  v281 = 0;
  v282 = 0;
  v283 = 0;
  v239 = v290 + 3;
  v242 = (_QWORD *)v290[4];
  if ( v242 != v290 + 3 )
  {
    v270 = v13;
    while ( 1 )
    {
      v75 = 0;
      if ( v242 )
        v75 = (__int64)(v242 - 7);
      v275 = v75;
      if ( !(unsigned __int8)sub_15E4F60(v75) )
        break;
LABEL_189:
      v242 = (_QWORD *)v242[1];
      if ( v239 == v242 )
      {
        v13 = v270;
        goto LABEL_325;
      }
    }
    v76 = v283;
    v77 = (unsigned int *)v289[0];
    *(_QWORD *)&v284 = v275;
    if ( v283 )
    {
      v78 = v289[0];
      v79 = (v283 - 1) & (((unsigned int)v275 >> 9) ^ ((unsigned int)v275 >> 4));
      v80 = (_QWORD *)(v281 + 16LL * v79);
      v81 = *v80;
      if ( v275 == *v80 )
      {
LABEL_195:
        v80[1] = v77[2] + 8LL * *(unsigned int *)(*(_QWORD *)v77 + 8LL);
        sub_1526BE0(v78, 0xCu, 4u);
        sub_1547D80(v291, v275);
        *((_QWORD *)&v284 + 1) = 0x4000000000LL;
        *(_QWORD *)&v284 = v285;
        LODWORD(v286) = (v300 - v299) >> 3;
        sub_1525B90((__int64)&v284, v270);
        sub_1528260((_DWORD *)v289[0], 1u, (__int64)&v284, 0);
        DWORD2(v284) = 0;
        v263 = v304;
        sub_1531F90((__int64 ***)v289, v303, v304, 0);
        if ( v301 < (unsigned __int64)((v295 - v294) >> 3) )
        {
          sub_1526BE0(v289[0], 0xFu, 3u);
          *((_QWORD *)&v286 + 1) = 0x4000000000LL;
          *(_QWORD *)&v286 = &v287;
          sub_152AB40((__int64 *)v289, (_QWORD *)(v294 + 8LL * v301), v302, (__int64)v270);
          sub_15334D0(
            v289,
            (__int64 *)(v294 + 8 * (v302 + (unsigned __int64)v301)),
            ((v295 - v294) >> 3) - (v302 + (unsigned __int64)v301),
            (__int64)v270,
            0,
            0);
          sub_15263C0((__int64 **)v289[0]);
          if ( (__int32 *)v286 != &v287 )
            _libc_free(v286);
        }
        v251 = (*(_DWORD *)(v275 + 32) & 0x100000) != 0;
        v266 = *(_QWORD *)(v275 + 80);
        if ( v266 != v275 + 72 )
        {
          v82 = 0;
          do
          {
            if ( !v266 )
              BUG();
            for ( k = *(_QWORD *)(v266 + 24); v266 + 16 != k; k = *(_QWORD *)(k + 8) )
            {
              if ( !k )
              {
                sub_1528720((_DWORD **)v289, 0, v263, (__int64)&v284);
                BUG();
              }
              sub_1528720((_DWORD **)v289, k - 24, v263, (__int64)&v284);
              v263 -= (*(_BYTE *)(*(_QWORD *)(k - 24) + 8LL) == 0) - 1;
              v251 |= *(__int16 *)(k - 6) < 0;
              v84 = sub_15C70A0(k + 24);
              v85 = v84;
              if ( v84 )
              {
                if ( v84 == v82 )
                {
                  sub_1528260((_DWORD *)v289[0], 0x21u, (__int64)&v284, 0);
                }
                else
                {
                  v86 = *(_DWORD *)(v84 + 4);
                  v87 = DWORD2(v284);
                  if ( DWORD2(v284) >= HIDWORD(v284) )
                  {
                    sub_16CD150(&v284, v285, 0, 4);
                    v87 = DWORD2(v284);
                  }
                  *(_DWORD *)(v284 + 4 * v87) = v86;
                  v88 = (unsigned int)(DWORD2(v284) + 1);
                  DWORD2(v284) = v88;
                  v89 = *(unsigned __int16 *)(v85 + 2);
                  if ( HIDWORD(v284) <= (unsigned int)v88 )
                  {
                    sub_16CD150(&v284, v285, 0, 4);
                    v88 = DWORD2(v284);
                  }
                  *(_DWORD *)(v284 + 4 * v88) = v89;
                  v90 = v297;
                  v91 = (unsigned int)++DWORD2(v284);
                  if ( v297 )
                  {
                    v92 = *(_QWORD *)(v85 - 8LL * *(unsigned int *)(v85 + 8));
                    v93 = (v297 - 1) & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
                    v94 = (__int64 *)(v296 + 16LL * v93);
                    v95 = *v94;
                    if ( v92 == *v94 )
                    {
LABEL_211:
                      v90 = *((_DWORD *)v94 + 3);
                    }
                    else
                    {
                      v177 = 1;
                      while ( v95 != -4 )
                      {
                        v199 = v177 + 1;
                        v93 = (v297 - 1) & (v177 + v93);
                        v94 = (__int64 *)(v296 + 16LL * v93);
                        v95 = *v94;
                        if ( v92 == *v94 )
                          goto LABEL_211;
                        v177 = v199;
                      }
                      v90 = 0;
                    }
                  }
                  if ( HIDWORD(v284) <= (unsigned int)v91 )
                  {
                    sub_16CD150(&v284, v285, 0, 4);
                    v91 = DWORD2(v284);
                  }
                  v96 = 0;
                  *(_DWORD *)(v284 + 4 * v91) = v90;
                  v97 = (unsigned int)++DWORD2(v284);
                  if ( *(_DWORD *)(v85 + 8) == 2 )
                    v96 = *(_QWORD *)(v85 - 8);
                  v98 = v297;
                  if ( v297 )
                  {
                    v99 = (v297 - 1) & (((unsigned int)v96 >> 9) ^ ((unsigned int)v96 >> 4));
                    v100 = (__int64 *)(v296 + 16LL * v99);
                    v101 = *v100;
                    if ( v96 == *v100 )
                    {
LABEL_218:
                      v98 = *((_DWORD *)v100 + 3);
                    }
                    else
                    {
                      v176 = 1;
                      while ( v101 != -4 )
                      {
                        v178 = v176 + 1;
                        v99 = (v297 - 1) & (v176 + v99);
                        v100 = (__int64 *)(v296 + 16LL * v99);
                        v101 = *v100;
                        if ( v96 == *v100 )
                          goto LABEL_218;
                        v176 = v178;
                      }
                      v98 = 0;
                    }
                  }
                  if ( (unsigned int)v97 >= HIDWORD(v284) )
                  {
                    sub_16CD150(&v284, v285, 0, 4);
                    v97 = DWORD2(v284);
                  }
                  *(_DWORD *)(v284 + 4 * v97) = v98;
                  v102 = v289[0];
                  v103 = ++DWORD2(v284);
                  v104 = *((_DWORD *)v289[0] + 2);
                  v105 = *((_DWORD *)v289[0] + 4);
                  v106 = *((_DWORD *)v289[0] + 3) | (3 << v104);
                  v107 = v105 + v104;
                  *((_DWORD *)v289[0] + 3) = v106;
                  v108 = v106;
                  if ( v107 > 0x1F )
                  {
                    v169 = *(_QWORD *)v102;
                    v170 = *(unsigned int *)(*(_QWORD *)v102 + 8LL);
                    if ( (unsigned __int64)*(unsigned int *)(*(_QWORD *)v102 + 12LL) - v170 <= 3 )
                    {
                      v249 = v108;
                      v257 = v105;
                      v260 = *(_QWORD *)v102;
                      sub_16CD150(*(_QWORD *)v102, v169 + 16, v170 + 4, 1);
                      v169 = v260;
                      v108 = v249;
                      LOBYTE(v105) = v257;
                      v170 = *(unsigned int *)(v260 + 8);
                    }
                    *(_DWORD *)(*(_QWORD *)v169 + v170) = v108;
                    *(_DWORD *)(v169 + 8) += 4;
                    v171 = v102[2];
                    v108 = 3u >> (32 - v171);
                    if ( !v171 )
                      v108 = 0;
                    v107 = ((_BYTE)v105 + (_BYTE)v171) & 0x1F;
                    v102[2] = v107;
                  }
                  else
                  {
                    v102[2] = v107;
                  }
                  v109 = 35 << v107;
                  v110 = v107 + 6;
                  v111 = v108 | v109;
                  v102[3] = v111;
                  if ( v110 > 0x1F )
                  {
                    v112 = *(_QWORD *)v102;
                    v113 = *(unsigned int *)(*(_QWORD *)v102 + 8LL);
                    if ( (unsigned __int64)*(unsigned int *)(*(_QWORD *)v102 + 12LL) - v113 <= 3 )
                    {
                      v254 = v111;
                      sub_16CD150(v112, v112 + 16, v113 + 4, 1);
                      v111 = v254;
                      v113 = *(unsigned int *)(v112 + 8);
                    }
                    *(_DWORD *)(*(_QWORD *)v112 + v113) = v111;
                    v111 = 0;
                    *(_DWORD *)(v112 + 8) += 4;
                    v114 = v102[2];
                    if ( v114 )
                      v111 = 0x23u >> (32 - v114);
                    v110 = ((_BYTE)v114 + 6) & 0x1F;
                  }
                  v115 = v111 | (1 << v110);
                  v102[2] = v110;
                  v116 = v110 + 6;
                  v102[3] = v115;
                  if ( v116 > 0x1F )
                  {
                    v117 = *(_QWORD *)v102;
                    v118 = *(unsigned int *)(*(_QWORD *)v102 + 8LL);
                    if ( (unsigned __int64)*(unsigned int *)(*(_QWORD *)v102 + 12LL) - v118 <= 3 )
                    {
                      v256 = v115;
                      sub_16CD150(v117, v117 + 16, v118 + 4, 1);
                      v115 = v256;
                      v118 = *(unsigned int *)(v117 + 8);
                    }
                    *(_DWORD *)(*(_QWORD *)v117 + v118) = v115;
                    v115 = 0;
                    *(_DWORD *)(v117 + 8) += 4;
                    v119 = v102[2];
                    if ( v119 )
                      v115 = 1u >> (32 - v119);
                    v102[3] = v115;
                    v116 = ((_BYTE)v119 + 6) & 0x1F;
                  }
                  v102[2] = v116;
                  for ( m = v103; m > 0x1F; v102[2] = v116 )
                  {
                    v123 = m & 0x1F | 0x20;
                    v124 = v123 << v116;
                    v116 += 6;
                    v115 |= v124;
                    v102[3] = v115;
                    if ( v116 > 0x1F )
                    {
                      v125 = *(_QWORD *)v102;
                      v126 = *(unsigned int *)(*(_QWORD *)v102 + 8LL);
                      if ( (unsigned __int64)*(unsigned int *)(*(_QWORD *)v102 + 12LL) - v126 <= 3 )
                      {
                        v243 = v115;
                        v246 = m;
                        v252 = m & 0x1F | 0x20;
                        sub_16CD150(v125, v125 + 16, v126 + 4, 1);
                        v115 = v243;
                        m = v246;
                        v123 = v252;
                        v126 = *(unsigned int *)(v125 + 8);
                      }
                      *(_DWORD *)(*(_QWORD *)v125 + v126) = v115;
                      v115 = 0;
                      *(_DWORD *)(v125 + 8) += 4;
                      v121 = v102[2];
                      v122 = v123 >> (32 - v121);
                      if ( v121 )
                        v115 = v122;
                      v116 = ((_BYTE)v121 + 6) & 0x1F;
                      v102[3] = v115;
                    }
                    m >>= 5;
                  }
                  v127 = m << v116;
                  v128 = v116 + 6;
                  v129 = v127 | v115;
                  v102[3] = v129;
                  if ( v128 > 0x1F )
                  {
                    v164 = *(_QWORD *)v102;
                    v165 = *(unsigned int *)(*(_QWORD *)v102 + 8LL);
                    if ( (unsigned __int64)*(unsigned int *)(*(_QWORD *)v102 + 12LL) - v165 <= 3 )
                    {
                      v248 = v129;
                      v255 = m;
                      sub_16CD150(v164, v164 + 16, v165 + 4, 1);
                      v129 = v248;
                      m = v255;
                      v165 = *(unsigned int *)(v164 + 8);
                    }
                    *(_DWORD *)(*(_QWORD *)v164 + v165) = v129;
                    v166 = 0;
                    *(_DWORD *)(v164 + 8) += 4;
                    v167 = v102[2];
                    v168 = m >> (32 - v167);
                    if ( v167 )
                      v166 = v168;
                    v102[3] = v166;
                    v102[2] = ((_BYTE)v167 + 6) & 0x1F;
                  }
                  else
                  {
                    v102[2] = v128;
                  }
                  v130 = 0;
                  v131 = 4LL * v103;
                  if ( v103 )
                  {
                    do
                    {
                      v132 = v102[3];
                      v133 = v102[2];
                      for ( n = *(_DWORD *)(v284 + v130); n > 0x1F; v102[2] = v133 )
                      {
                        v137 = n & 0x1F | 0x20;
                        v138 = v137 << v133;
                        v133 += 6;
                        v132 |= v138;
                        v102[3] = v132;
                        if ( v133 > 0x1F )
                        {
                          v139 = *(_QWORD *)v102;
                          v140 = *(unsigned int *)(*(_QWORD *)v102 + 8LL);
                          if ( (unsigned __int64)*(unsigned int *)(*(_QWORD *)v102 + 12LL) - v140 <= 3 )
                          {
                            v241 = v130;
                            v244 = v131;
                            v247 = n;
                            v253 = n & 0x1F | 0x20;
                            sub_16CD150(v139, v139 + 16, v140 + 4, 1);
                            v130 = v241;
                            v131 = v244;
                            n = v247;
                            v140 = *(unsigned int *)(v139 + 8);
                            v137 = v253;
                          }
                          *(_DWORD *)(*(_QWORD *)v139 + v140) = v132;
                          v132 = 0;
                          *(_DWORD *)(v139 + 8) += 4;
                          v135 = v102[2];
                          v136 = v137 >> (32 - v135);
                          if ( v135 )
                            v132 = v136;
                          v133 = ((_BYTE)v135 + 6) & 0x1F;
                          v102[3] = v132;
                        }
                        n >>= 5;
                      }
                      v141 = n << v133;
                      v142 = v133 + 6;
                      v143 = v141 | v132;
                      v102[3] = v143;
                      if ( v142 > 0x1F )
                      {
                        v172 = *(_QWORD *)v102;
                        v173 = *(unsigned int *)(*(_QWORD *)v102 + 8LL);
                        if ( (unsigned __int64)*(unsigned int *)(*(_QWORD *)v102 + 12LL) - v173 <= 3 )
                        {
                          v245 = v130;
                          v250 = v131;
                          v258 = n;
                          sub_16CD150(v172, v172 + 16, v173 + 4, 1);
                          v130 = v245;
                          v131 = v250;
                          n = v258;
                          v173 = *(unsigned int *)(v172 + 8);
                        }
                        *(_DWORD *)(*(_QWORD *)v172 + v173) = v143;
                        *(_DWORD *)(v172 + 8) += 4;
                        v174 = v102[2];
                        v175 = n >> (32 - v174);
                        if ( !v174 )
                          v175 = 0;
                        v102[3] = v175;
                        v102[2] = ((_BYTE)v174 + 6) & 0x1F;
                      }
                      else
                      {
                        v102[2] = v142;
                      }
                      v130 += 4;
                    }
                    while ( v131 != v130 );
                  }
                  DWORD2(v284) = 0;
                  v82 = v85;
                }
              }
            }
            v266 = *(_QWORD *)(v266 + 8);
          }
          while ( v275 + 72 != v266 );
        }
        v144 = *(__int64 ***)(v275 + 104);
        if ( v144 && *((_DWORD *)v144 + 3) )
        {
          sub_1526BE0(v289[0], 0xEu, 4u);
          *(_QWORD *)&v286 = &v287;
          *((_QWORD *)&v286 + 1) = 0x4000000000LL;
          v145 = *((_DWORD *)v144 + 2);
          if ( v145 )
          {
            v146 = *v144;
            v147 = **v144;
            if ( v147 != -8 && v147 )
            {
              v150 = *v144;
            }
            else
            {
              v148 = v146 + 1;
              do
              {
                do
                {
                  v149 = *v148;
                  v150 = v148++;
                }
                while ( v149 == -8 );
              }
              while ( !v149 );
            }
            v272 = &v146[v145];
            if ( v272 != v150 )
            {
              while ( 1 )
              {
                v151 = (__int64 *)*v150;
                v152 = (char *)(*v150 + 16);
                v153 = sub_1524040(v152, *(_QWORD *)*v150);
                v154 = (unsigned int)sub_153E840(v291);
                v155 = DWORD2(v286);
                if ( DWORD2(v286) >= HIDWORD(v286) )
                {
                  v268 = v154;
                  sub_16CD150(v270, &v287, 0, 8);
                  v155 = DWORD2(v286);
                  v154 = v268;
                }
                *(_QWORD *)(v286 + 8 * v155) = v154;
                v156 = (unsigned int)++DWORD2(v286);
                if ( *(_BYTE *)(v151[1] + 16) == 18 )
                  break;
                if ( !v153 )
                {
                  v153 = 1;
                  v157 = 6;
                  goto LABEL_280;
                }
                v157 = 5;
                if ( v153 != 1 )
                {
                  v153 = 1;
LABEL_279:
                  v157 = 4;
                }
LABEL_280:
                if ( v152 != &v152[*v151] )
                {
                  v267 = v150;
                  v158 = &v152[*v151];
                  do
                  {
                    v159 = (unsigned __int8)*v152;
                    if ( (unsigned int)v156 >= HIDWORD(v286) )
                    {
                      v264 = v157;
                      sub_16CD150(v270, &v287, 0, 8);
                      v156 = DWORD2(v286);
                      v157 = v264;
                    }
                    ++v152;
                    *(_QWORD *)(v286 + 8 * v156) = v159;
                    v156 = (unsigned int)++DWORD2(v286);
                  }
                  while ( v158 != v152 );
                  v150 = v267;
                }
                sub_152F3D0((_DWORD *)v289[0], v153, (__int64)v270, v157);
                v160 = v150 + 1;
                DWORD2(v286) = 0;
                v161 = v150[1];
                if ( v161 != -8 && v161 )
                {
                  ++v150;
                  if ( v272 == v160 )
                    goto LABEL_292;
                }
                else
                {
                  v162 = v150 + 2;
                  do
                  {
                    do
                    {
                      v163 = *v162;
                      v150 = v162++;
                    }
                    while ( v163 == -8 );
                  }
                  while ( !v163 );
                  if ( v272 == v150 )
                    goto LABEL_292;
                }
              }
              v220 = v153 == 0;
              v157 = 7;
              v153 = 2;
              if ( v220 )
                goto LABEL_280;
              goto LABEL_279;
            }
          }
LABEL_292:
          sub_15263C0((__int64 **)v289[0]);
          if ( (__int32 *)v286 != &v287 )
            _libc_free(v286);
        }
        if ( v251 )
          sub_1530240((__int64 ***)v289, v275);
        if ( v298 )
          sub_1530BF0((__int64)v289, v275);
        sub_1540000(v291);
        sub_15263C0((__int64 **)v289[0]);
        if ( (_BYTE *)v284 != v285 )
          _libc_free(v284);
        goto LABEL_189;
      }
      v235 = 1;
      v236 = 0;
      while ( v81 != -8 )
      {
        if ( !v236 && v81 == -16 )
          v236 = v80;
        v79 = (v283 - 1) & (v235 + v79);
        v80 = (_QWORD *)(v281 + 16LL * v79);
        v81 = *v80;
        if ( v275 == *v80 )
        {
          v78 = v289[0];
          goto LABEL_195;
        }
        ++v235;
      }
      if ( v236 )
        v80 = v236;
      ++v280;
      v237 = v282 + 1;
      if ( 4 * ((int)v282 + 1) < 3 * v283 )
      {
        v238 = v275;
        if ( v283 - HIDWORD(v282) - v237 <= v283 >> 3 )
          goto LABEL_444;
LABEL_439:
        LODWORD(v282) = v237;
        if ( *v80 != -8 )
          --HIDWORD(v282);
        *v80 = v238;
        v80[1] = 0;
        v78 = v289[0];
        goto LABEL_195;
      }
    }
    else
    {
      ++v280;
    }
    v76 = 2 * v283;
LABEL_444:
    sub_1538D00((__int64)&v280, v76);
    sub_1538C50((__int64)&v280, (__int64 *)&v284, v270);
    v80 = (_QWORD *)v286;
    v238 = v284;
    v237 = v282 + 1;
    goto LABEL_439;
  }
LABEL_325:
  if ( v305 )
    sub_1535340((__int64 ***)v289);
  v179 = (unsigned __int64 *)(**(_QWORD **)v289[0] + (unsigned int)(v313 >> 3));
  v180 = ((*((unsigned int *)v289[0] + 2) - v318 + 8 * (unsigned __int64)*(unsigned int *)(*v289[0] + 8LL)) >> 5) + 1;
  if ( (v313 & 7) != 0 )
  {
    v181 = v313 & 7;
    *v179 = (unsigned int)*v179 & ~(-1 << v181)
          | ((v180 & ~(-1 << (32 - v181))) << v181)
          | ((~(-1 << v181) & (v180 >> (32 - v181)) | HIDWORD(*v179) & (-1 << v181)) << 32);
  }
  else
  {
    *(_DWORD *)v179 = v180;
  }
  sub_1526BE0(v289[0], 0xEu, 4u);
  sub_1531130(&v284);
  BYTE8(v286) |= 1u;
  *(_QWORD *)&v286 = 3;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 8;
  BYTE8(v286) = 4;
  sub_1525B40(v284, v13);
  *(_QWORD *)&v286 = 8;
  BYTE8(v286) = 4;
  sub_1525B40(v284, v13);
  v182 = *((_QWORD *)&v284 + 1);
  *(_QWORD *)&v286 = v284;
  v284 = 0u;
  *((_QWORD *)&v286 + 1) = v182;
  v183 = sub_15271D0(v289[0], v13->m128i_i64);
  if ( *((_QWORD *)&v286 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v286 + 1));
  v184 = (_QWORD *)v290[4];
  v278 = v290 + 3;
  if ( v184 != v290 + 3 )
  {
    while ( 1 )
    {
      v190 = v184 - 7;
      if ( !v184 )
        v190 = 0;
      if ( (unsigned __int8)sub_15E4F60(v190) )
        goto LABEL_337;
      *(_QWORD *)&v286 = (unsigned int)sub_153E840(v291);
      if ( !v283 )
        break;
      v185 = (v283 - 1) & (((unsigned int)v190 >> 9) ^ ((unsigned int)v190 >> 4));
      v186 = (_QWORD *)(v281 + 16LL * v185);
      v187 = *v186;
      if ( (_QWORD *)*v186 != v190 )
      {
        v229 = 1;
        v230 = 0;
        while ( v187 != -8 )
        {
          if ( !v230 && v187 == -16 )
            v230 = v186;
          v185 = (v283 - 1) & (v229 + v185);
          v186 = (_QWORD *)(v281 + 16LL * v185);
          v187 = *v186;
          if ( v190 == (_QWORD *)*v186 )
            goto LABEL_334;
          ++v229;
        }
        if ( v230 )
          v186 = v230;
        ++v280;
        v192 = v282 + 1;
        if ( 4 * ((int)v282 + 1) < 3 * v283 )
        {
          if ( v283 - HIDWORD(v282) - v192 <= v283 >> 3 )
          {
            sub_1538D00((__int64)&v280, v283);
            if ( !v283 )
            {
LABEL_467:
              LODWORD(v282) = v282 + 1;
              BUG();
            }
            v231 = 0;
            LODWORD(v232) = (v283 - 1) & (((unsigned int)v190 >> 9) ^ ((unsigned int)v190 >> 4));
            v192 = v282 + 1;
            v233 = 1;
            v186 = (_QWORD *)(v281 + 16LL * (unsigned int)v232);
            v234 = *v186;
            if ( (_QWORD *)*v186 != v190 )
            {
              while ( v234 != -8 )
              {
                if ( !v231 && v234 == -16 )
                  v231 = v186;
                v232 = (v283 - 1) & ((_DWORD)v232 + v233);
                v186 = (_QWORD *)(v281 + 16 * v232);
                v234 = *v186;
                if ( v190 == (_QWORD *)*v186 )
                  goto LABEL_405;
                ++v233;
              }
              if ( v231 )
                v186 = v231;
            }
          }
          goto LABEL_405;
        }
LABEL_343:
        sub_1538D00((__int64)&v280, 2 * v283);
        if ( !v283 )
          goto LABEL_467;
        LODWORD(v191) = (v283 - 1) & (((unsigned int)v190 >> 9) ^ ((unsigned int)v190 >> 4));
        v192 = v282 + 1;
        v186 = (_QWORD *)(v281 + 16LL * (unsigned int)v191);
        v193 = *v186;
        if ( v190 != (_QWORD *)*v186 )
        {
          v194 = 1;
          v195 = 0;
          while ( v193 != -8 )
          {
            if ( v193 == -16 && !v195 )
              v195 = v186;
            v191 = (v283 - 1) & ((_DWORD)v191 + v194);
            v186 = (_QWORD *)(v281 + 16 * v191);
            v193 = *v186;
            if ( v190 == (_QWORD *)*v186 )
              goto LABEL_405;
            ++v194;
          }
          if ( v195 )
            v186 = v195;
        }
LABEL_405:
        LODWORD(v282) = v192;
        if ( *v186 != -8 )
          --HIDWORD(v282);
        *v186 = v190;
        v186[1] = 0;
        v188 = 0;
        goto LABEL_335;
      }
LABEL_334:
      v188 = v186[1];
LABEL_335:
      v189 = v289[0];
      *((_QWORD *)&v286 + 1) = ((unsigned __int64)(v188 - v318) >> 5) + 1;
      if ( v183 )
      {
        v279 = 0x100000003LL;
        sub_152A250((__int64)v289[0], v183, (__int64)v13, 2, 0, 0, (__int64)&v279);
LABEL_337:
        v184 = (_QWORD *)v184[1];
        if ( v278 == v184 )
          goto LABEL_351;
      }
      else
      {
        sub_1524D80((_DWORD *)v289[0], 3u, *((_DWORD *)v289[0] + 4));
        sub_1524E40(v189, 3u, 6);
        sub_1524E40(v189, 2u, 6);
        sub_1525280(v189, v286, 6);
        sub_1525280(v189, *((unsigned __int64 *)&v286 + 1), 6);
        v184 = (_QWORD *)v184[1];
        if ( v278 == v184 )
          goto LABEL_351;
      }
    }
    ++v280;
    goto LABEL_343;
  }
LABEL_351:
  sub_15263C0((__int64 **)v289[0]);
  if ( *((_QWORD *)&v284 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v284 + 1));
  if ( v315 )
  {
    v221 = *(_QWORD *)v314 + v240;
    sub_16CB0E0(v317);
    v222 = (unsigned int *)sub_16CB220(v317, v221);
    for ( ii = 0; ii != 20; ii += 4 )
    {
      v224 = ii;
      v225 = *v222++;
      *((_DWORD *)&v286 + (v224 >> 2)) = _byteswap_ulong(v225);
    }
    v226 = v289[0];
    sub_1524D80((_DWORD *)v289[0], 3u, *((_DWORD *)v289[0] + 4));
    sub_1524E40(v226, 0x11u, 6);
    sub_1524E40(v226, 5u, 6);
    do
    {
      v227 = v13->m128i_u32[0];
      v13 = (__m128i *)((char *)v13 + 4);
      sub_1525280(v226, v227, 6);
    }
    while ( v13 != (__m128i *)&v288 );
    v228 = v316;
    if ( v316 )
    {
      *v316 = _mm_loadu_si128((const __m128i *)&v286);
      v228[1].m128i_i32[0] = v287;
    }
  }
  sub_15263C0((__int64 **)v289[0]);
  j___libc_free_0(v281);
  v196 = v308;
  if ( v308 )
  {
    do
    {
      sub_1524320(v196[3]);
      v197 = v196;
      v196 = (_QWORD *)v196[2];
      j_j___libc_free_0(v197, 48);
    }
    while ( v196 );
  }
  return sub_1525790(v291);
}
