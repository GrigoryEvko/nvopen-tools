// Function: sub_A38520
// Address: 0xa38520
//
__int64 __fastcall sub_A38520(__int64 *a1, __int64 a2, unsigned __int8 a3, __int64 a4, char a5, __m128i *a6)
{
  _BYTE *v9; // rsi
  __int64 v10; // rbx
  _QWORD *v11; // r12
  __int64 v12; // r13
  __int64 v13; // r12
  __int64 v14; // rax
  unsigned int v15; // ebx
  __int64 v16; // rax
  __int64 v17; // rdi
  volatile signed __int32 *v18; // r8
  __int64 v19; // rax
  unsigned int v20; // r14d
  __int64 v21; // r12
  int v22; // edx
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rbx
  _QWORD *v26; // rax
  _QWORD *v27; // rcx
  _QWORD *v28; // r13
  __int64 v29; // r14
  __int64 v30; // r12
  volatile signed __int32 *v31; // r15
  signed __int32 v32; // eax
  signed __int32 v33; // eax
  __int64 v34; // rdi
  int v35; // ebx
  __int64 v36; // rdi
  int v37; // ebx
  __int64 v38; // rdi
  int v39; // ebx
  __int64 v40; // rdi
  int v41; // ebx
  __int64 v42; // r12
  __int64 v43; // rax
  __int64 v44; // rdi
  int v45; // ebx
  __int64 v46; // rdi
  int v47; // ebx
  __int64 v48; // r12
  __int64 v49; // rax
  __int64 v50; // rdi
  int v51; // ebx
  __int64 v52; // rdi
  int v53; // ebx
  __int64 v54; // r12
  __int64 v55; // rax
  __int64 v56; // rdi
  int v57; // ebx
  __int64 v58; // rdi
  int v59; // ebx
  __int64 v60; // rdi
  int v61; // ebx
  __int64 v62; // rdi
  int v63; // ebx
  __int64 v64; // rdi
  int v65; // ebx
  __int64 v66; // r12
  __int64 v67; // rax
  __int64 v68; // rdi
  int v69; // ebx
  __int64 v70; // r12
  __int64 v71; // rax
  __int64 v72; // rdi
  int v73; // ebx
  __int64 v74; // rdi
  int v75; // ebx
  __int64 v76; // rdi
  int v77; // ebx
  __int64 v78; // rdi
  int v79; // ebx
  __int64 v80; // rsi
  __int64 v81; // rax
  __int64 v82; // rdi
  int v83; // ebx
  __int64 v84; // rdi
  int v85; // ebx
  __int64 v86; // rsi
  __int64 v87; // rbx
  unsigned int v88; // r15d
  int v89; // eax
  int v90; // r12d
  __int64 v91; // rax
  int v92; // r10d
  unsigned int j; // edx
  __int64 v94; // rdi
  unsigned int v95; // edx
  __int64 v96; // rax
  __int64 v97; // r8
  unsigned __int64 v98; // rdx
  _QWORD *v99; // r14
  __int64 v100; // rbx
  __int64 v101; // rax
  __int64 v102; // r15
  __int64 v103; // rdx
  __int64 v104; // r12
  __int64 v105; // r13
  __int64 v106; // rax
  int v107; // r12d
  __int64 v108; // rax
  unsigned __int64 v109; // rdx
  __int64 v110; // rax
  int v111; // edx
  int v112; // r12d
  unsigned __int64 v113; // rdx
  __int64 v114; // rax
  unsigned int v115; // edx
  unsigned int v116; // ebx
  __int64 v117; // r15
  __int64 v118; // rbx
  __int64 v119; // r12
  __int32 v120; // r13d
  __int64 v121; // rsi
  __int64 v122; // r12
  _BYTE **v123; // rax
  __int64 v124; // rdx
  unsigned int v125; // esi
  __int64 v126; // rsi
  __int64 v127; // rdi
  __int64 v128; // rsi
  __int64 v129; // rsi
  _QWORD *v130; // rbx
  _QWORD *v131; // r14
  __int64 v132; // rsi
  __int64 v133; // rbx
  _QWORD *v134; // r14
  __int64 v135; // r12
  __int64 v136; // rax
  __int64 v137; // r15
  unsigned __int64 v138; // rbx
  unsigned __int64 v139; // r12
  __int64 v140; // rax
  __int64 v141; // rax
  unsigned int v142; // r15d
  _QWORD *v143; // r12
  _QWORD *v144; // r13
  int v145; // r11d
  _QWORD *v146; // rax
  unsigned int v147; // edi
  _QWORD *v148; // rdx
  __int64 v149; // r8
  __int64 v150; // rax
  __int64 v151; // rbx
  _QWORD *v152; // rbx
  unsigned int v153; // ecx
  int v154; // edx
  __int64 v155; // r8
  int v156; // edi
  _QWORD *v157; // rsi
  __int64 v158; // rdi
  __int64 v159; // rax
  __int64 v160; // rbx
  __int64 v161; // rdi
  __int64 v163; // rbx
  __int64 v164; // rax
  __int64 v165; // r12
  __int64 *v166; // rdx
  __int64 v167; // rax
  __int64 v168; // r15
  __int64 v169; // r14
  __int64 v170; // rcx
  __int64 k; // rax
  __int64 v172; // rbx
  __int64 *v173; // r14
  __int64 *v174; // r15
  __int64 v175; // rbx
  __int64 v176; // r12
  __int64 v177; // rax
  __int64 v178; // rcx
  __int64 n; // rax
  __int64 v180; // rbx
  __int64 *v181; // r14
  __int64 *v182; // r15
  __int64 v183; // rbx
  __int64 v184; // r12
  __int64 v185; // rax
  __int64 v186; // rcx
  __int64 m; // rax
  unsigned int *v188; // rax
  int ii; // edx
  int v190; // ecx
  unsigned int v191; // esi
  __int64 v192; // r12
  __m128i *v193; // rbx
  unsigned __int64 v194; // rsi
  __m128i *v195; // rax
  int v196; // edi
  unsigned int v197; // ecx
  __int64 v198; // r8
  __int64 v199; // [rsp+18h] [rbp-748h]
  _QWORD *v200; // [rsp+30h] [rbp-730h]
  __int64 i; // [rsp+30h] [rbp-730h]
  _QWORD *v202; // [rsp+30h] [rbp-730h]
  __int64 v203; // [rsp+30h] [rbp-730h]
  _QWORD *v205; // [rsp+38h] [rbp-728h]
  __int128 v206; // [rsp+40h] [rbp-720h] BYREF
  __int128 v207; // [rsp+50h] [rbp-710h] BYREF
  __int128 v208; // [rsp+60h] [rbp-700h] BYREF
  __int128 v209; // [rsp+70h] [rbp-6F0h] BYREF
  __int128 v210; // [rsp+80h] [rbp-6E0h] BYREF
  __int128 v211; // [rsp+90h] [rbp-6D0h] BYREF
  __int128 v212; // [rsp+A0h] [rbp-6C0h] BYREF
  __int128 v213; // [rsp+B0h] [rbp-6B0h] BYREF
  __int128 v214; // [rsp+C0h] [rbp-6A0h] BYREF
  __int128 v215; // [rsp+D0h] [rbp-690h] BYREF
  __int128 v216; // [rsp+E0h] [rbp-680h] BYREF
  __int128 v217; // [rsp+F0h] [rbp-670h] BYREF
  __int128 v218; // [rsp+100h] [rbp-660h] BYREF
  __int128 v219; // [rsp+110h] [rbp-650h] BYREF
  __int128 v220; // [rsp+120h] [rbp-640h] BYREF
  __int128 v221; // [rsp+130h] [rbp-630h] BYREF
  __int128 v222; // [rsp+140h] [rbp-620h] BYREF
  __int128 v223; // [rsp+150h] [rbp-610h] BYREF
  __m128i v224; // [rsp+160h] [rbp-600h] BYREF
  __int32 v225; // [rsp+170h] [rbp-5F0h]
  char v226; // [rsp+174h] [rbp-5ECh] BYREF
  unsigned __int128 v227; // [rsp+180h] [rbp-5E0h] BYREF
  _BYTE v228[128]; // [rsp+190h] [rbp-5D0h] BYREF
  __m128i v229; // [rsp+210h] [rbp-550h] BYREF
  __int64 v230; // [rsp+220h] [rbp-540h] BYREF
  unsigned int v231; // [rsp+228h] [rbp-538h]
  __int64 v232[2]; // [rsp+420h] [rbp-340h] BYREF
  _QWORD *v233; // [rsp+430h] [rbp-330h]
  __int64 v234[14]; // [rsp+438h] [rbp-328h] BYREF
  _BYTE **v235; // [rsp+4A8h] [rbp-2B8h]
  __int64 v236; // [rsp+4B0h] [rbp-2B0h]
  _QWORD *v237; // [rsp+4F0h] [rbp-270h]
  _QWORD *v238; // [rsp+4F8h] [rbp-268h]
  char v239; // [rsp+578h] [rbp-1E8h]
  __int64 v240; // [rsp+588h] [rbp-1D8h]
  unsigned int v241; // [rsp+598h] [rbp-1C8h]
  __int64 v242; // [rsp+5D8h] [rbp-188h]
  __int64 v243; // [rsp+5E0h] [rbp-180h]
  __int64 v244; // [rsp+668h] [rbp-F8h]
  __int64 v245; // [rsp+680h] [rbp-E0h]
  unsigned __int64 v246; // [rsp+6A8h] [rbp-B8h]
  char v247; // [rsp+6B0h] [rbp-B0h]
  __m128i *v248; // [rsp+6B8h] [rbp-A8h]
  _BYTE v249[96]; // [rsp+6C0h] [rbp-A0h] BYREF
  __int64 v250; // [rsp+720h] [rbp-40h]

  v232[0] = a2;
  v9 = (_BYTE *)a1[21];
  if ( v9 == (_BYTE *)a1[22] )
  {
    v203 = a4;
    sub_A28060((__int64)(a1 + 20), v9, v232);
    a4 = v203;
  }
  else
  {
    if ( v9 )
    {
      *(_QWORD *)v9 = a2;
      v9 = (_BYTE *)a1[21];
    }
    a1[21] = (__int64)(v9 + 8);
  }
  v10 = *a1;
  sub_A28440((__int64)v232, a2, (__int64)(a1 + 1), *a1, a3, a4);
  v247 = a5;
  v248 = a6;
  sub_C89F10(v249);
  v11 = *(_QWORD **)(v10 + 32);
  v12 = *(_QWORD *)(*(_QWORD *)(v10 + 24) + 8LL);
  if ( v11 && (unsigned __int8)sub_CB7440(*(_QWORD *)(v10 + 32)) )
  {
    if ( !(unsigned __int8)sub_CB7440(v11) )
      BUG();
    v12 += v11[4] - v11[2] + (*(__int64 (__fastcall **)(_QWORD *))(*v11 + 80LL))(v11);
  }
  v13 = v232[0];
  v250 = *(unsigned int *)(v10 + 48) + 8 * v12;
  sub_A19830(v232[0], 0xDu, 5u);
  sub_A23770(&v227);
  sub_A186C0(v227, 1, 1);
  sub_A186C0(v227, 0, 6);
  sub_A186C0(v227, 0, 8);
  v14 = *((_QWORD *)&v227 + 1);
  v229.m128i_i64[0] = v227;
  v227 = 0u;
  v229.m128i_i64[1] = v14;
  v15 = sub_A1AB30(v13, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  sub_A215B0(v13, 1u, "LLVM20.0.0", 10, v15);
  sub_A23770(&v229);
  v16 = v229.m128i_i64[1];
  v17 = v229.m128i_i64[0];
  v229 = 0u;
  v18 = (volatile signed __int32 *)*((_QWORD *)&v227 + 1);
  v227 = __PAIR128__(v16, v17);
  if ( v18 )
  {
    sub_A191D0(v18);
    if ( v229.m128i_i64[1] )
      sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
    v17 = v227;
  }
  sub_A186C0(v17, 2, 1);
  sub_A186C0(v227, 6, 4);
  v19 = *((_QWORD *)&v227 + 1);
  v229.m128i_i64[0] = v227;
  v227 = 0u;
  v229.m128i_i64[1] = v19;
  v20 = sub_A1AB30(v13, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  v229.m128i_i32[0] = 0;
  if ( v20 )
  {
    sub_A20C50(v13, v20, (__int64)&v229, 1, 0, 0, 2u, 1);
  }
  else
  {
    sub_A17B10(v13, 3u, *(_DWORD *)(v13 + 56));
    sub_A17CC0(v13, 2u, 6);
    sub_A17CC0(v13, 1u, 6);
    sub_A17DE0(v13, v229.m128i_u32[0], 6);
  }
  sub_A192A0(v13);
  if ( *((_QWORD *)&v227 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v227 + 1));
  sub_A19830(v232[0], 8u, 3u);
  v21 = v232[0];
  v22 = *(_DWORD *)(v232[0] + 56);
  v23 = v232[0];
  v24 = *(_QWORD *)(*(_QWORD *)(v232[0] + 24) + 8LL);
  *(_BYTE *)(v232[0] + 96) = 1;
  *(_QWORD *)(v23 + 88) = v24;
  sub_A17B10(v23, 3u, v22);
  sub_A17CC0(v21, 1u, 6);
  sub_A17CC0(v21, 1u, 6);
  sub_A17CC0(v21, 2u, 6);
  v25 = v232[0];
  sub_A19830(v232[0], 0, 2u);
  v26 = *(_QWORD **)(v25 + 128);
  v27 = *(_QWORD **)(v25 + 136);
  *(_DWORD *)(v25 + 60) = -1;
  v200 = v26;
  v205 = v27;
  if ( v26 != v27 )
  {
    v28 = v26;
    do
    {
      v29 = v28[2];
      v30 = v28[1];
      if ( v29 != v30 )
      {
        do
        {
          while ( 1 )
          {
            v31 = *(volatile signed __int32 **)(v30 + 8);
            if ( v31 )
            {
              if ( &_pthread_key_create )
              {
                v32 = _InterlockedExchangeAdd(v31 + 2, 0xFFFFFFFF);
              }
              else
              {
                v32 = *((_DWORD *)v31 + 2);
                *((_DWORD *)v31 + 2) = v32 - 1;
              }
              if ( v32 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v31 + 16LL))(v31);
                if ( &_pthread_key_create )
                {
                  v33 = _InterlockedExchangeAdd(v31 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v33 = *((_DWORD *)v31 + 3);
                  *((_DWORD *)v31 + 3) = v33 - 1;
                }
                if ( v33 == 1 )
                  break;
              }
            }
            v30 += 16;
            if ( v29 == v30 )
              goto LABEL_34;
          }
          v30 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v31 + 24LL))(v31);
        }
        while ( v29 != v30 );
LABEL_34:
        v30 = v28[1];
      }
      if ( v30 )
        j_j___libc_free_0(v30, v28[3] - v30);
      v28 += 4;
    }
    while ( v205 != v28 );
    *(_QWORD *)(v25 + 136) = v200;
  }
  sub_A23770(&v206);
  sub_A186C0(v206, 3, 2);
  sub_A186C0(v206, 8, 4);
  sub_A186C0(v206, 0, 6);
  sub_A186C0(v206, 8, 2);
  v34 = v232[0];
  v229 = (__m128i)v206;
  if ( *((_QWORD *)&v206 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v206 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v206 + 1) + 8LL);
  }
  v35 = sub_A1A630(v34, 14, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v35 != 4 )
    goto LABEL_395;
  if ( *((_QWORD *)&v206 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v206 + 1));
  sub_A23770(&v207);
  sub_A186C0(v207, 1, 1);
  sub_A186C0(v207, 8, 4);
  sub_A186C0(v207, 0, 6);
  sub_A186C0(v207, 7, 2);
  v36 = v232[0];
  v229 = (__m128i)v207;
  if ( *((_QWORD *)&v207 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v207 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v207 + 1) + 8LL);
  }
  v37 = sub_A1A630(v36, 14, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v37 != 5 )
    goto LABEL_395;
  if ( *((_QWORD *)&v207 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v207 + 1));
  sub_A23770(&v208);
  sub_A186C0(v208, 1, 1);
  sub_A186C0(v208, 8, 4);
  sub_A186C0(v208, 0, 6);
  sub_A186C0(v208, 0, 8);
  v38 = v232[0];
  v229 = (__m128i)v208;
  if ( *((_QWORD *)&v208 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v208 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v208 + 1) + 8LL);
  }
  v39 = sub_A1A630(v38, 14, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v39 != 6 )
    goto LABEL_395;
  if ( *((_QWORD *)&v208 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v208 + 1));
  sub_A23770(&v209);
  sub_A186C0(v209, 2, 1);
  sub_A186C0(v209, 8, 4);
  sub_A186C0(v209, 0, 6);
  sub_A186C0(v209, 0, 8);
  v40 = v232[0];
  v229 = (__m128i)v209;
  if ( *((_QWORD *)&v209 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v209 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v209 + 1) + 8LL);
  }
  v41 = sub_A1A630(v40, 14, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v41 != 7 )
    goto LABEL_395;
  if ( *((_QWORD *)&v209 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v209 + 1));
  sub_A23770(&v210);
  sub_A186C0(v210, 1, 1);
  v42 = v210;
  v43 = sub_A3FA40(v234);
  sub_A186C0(v42, v43, 2);
  v44 = v232[0];
  v229 = (__m128i)v210;
  if ( *((_QWORD *)&v210 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v210 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v210 + 1) + 8LL);
  }
  v45 = sub_A1A630(v44, 11, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v45 != 4 )
    goto LABEL_395;
  if ( *((_QWORD *)&v210 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v210 + 1));
  sub_A23770(&v211);
  sub_A186C0(v211, 4, 1);
  sub_A186C0(v211, 8, 4);
  v46 = v232[0];
  v229 = (__m128i)v211;
  if ( *((_QWORD *)&v211 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v211 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v211 + 1) + 8LL);
  }
  v47 = sub_A1A630(v46, 11, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v47 != 5 )
    goto LABEL_395;
  if ( *((_QWORD *)&v211 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v211 + 1));
  sub_A23770(&v212);
  sub_A186C0(v212, 11, 1);
  sub_A186C0(v212, 4, 2);
  v48 = v212;
  v49 = sub_A3FA40(v234);
  sub_A186C0(v48, v49, 2);
  sub_A186C0(v212, 8, 4);
  v50 = v232[0];
  v229 = (__m128i)v212;
  if ( *((_QWORD *)&v212 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v212 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v212 + 1) + 8LL);
  }
  v51 = sub_A1A630(v50, 11, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v51 != 6 )
    goto LABEL_395;
  if ( *((_QWORD *)&v212 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v212 + 1));
  sub_A23770(&v213);
  sub_A186C0(v213, 2, 1);
  v52 = v232[0];
  v229 = (__m128i)v213;
  if ( *((_QWORD *)&v213 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v213 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v213 + 1) + 8LL);
  }
  v53 = sub_A1A630(v52, 11, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v53 != 7 )
    goto LABEL_395;
  if ( *((_QWORD *)&v213 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v213 + 1));
  sub_A23770(&v214);
  sub_A186C0(v214, 20, 1);
  sub_A186C0(v214, 6, 4);
  v54 = v214;
  v55 = sub_A3FA40(v234);
  sub_A186C0(v54, v55, 2);
  sub_A186C0(v214, 4, 4);
  sub_A186C0(v214, 1, 2);
  v56 = v232[0];
  v229 = (__m128i)v214;
  if ( *((_QWORD *)&v214 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v214 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v214 + 1) + 8LL);
  }
  v57 = sub_A1A630(v56, 12, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v57 != 4 )
    goto LABEL_395;
  if ( *((_QWORD *)&v214 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v214 + 1));
  sub_A23770(&v215);
  sub_A186C0(v215, 56, 1);
  sub_A186C0(v215, 6, 4);
  sub_A186C0(v215, 4, 2);
  v58 = v232[0];
  v229 = (__m128i)v215;
  if ( *((_QWORD *)&v215 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v215 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v215 + 1) + 8LL);
  }
  v59 = sub_A1A630(v58, 12, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v59 != 5 )
    goto LABEL_395;
  if ( *((_QWORD *)&v215 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v215 + 1));
  sub_A23770(&v216);
  sub_A186C0(v216, 56, 1);
  sub_A186C0(v216, 6, 4);
  sub_A186C0(v216, 4, 2);
  sub_A186C0(v216, 8, 2);
  v60 = v232[0];
  v229 = (__m128i)v216;
  if ( *((_QWORD *)&v216 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v216 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v216 + 1) + 8LL);
  }
  v61 = sub_A1A630(v60, 12, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v61 != 6 )
    goto LABEL_395;
  if ( *((_QWORD *)&v216 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v216 + 1));
  sub_A23770(&v217);
  sub_A186C0(v217, 2, 1);
  sub_A186C0(v217, 6, 4);
  sub_A186C0(v217, 6, 4);
  sub_A186C0(v217, 4, 2);
  v62 = v232[0];
  v229 = (__m128i)v217;
  if ( *((_QWORD *)&v217 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v217 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v217 + 1) + 8LL);
  }
  v63 = sub_A1A630(v62, 12, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v63 != 7 )
    goto LABEL_395;
  if ( *((_QWORD *)&v217 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v217 + 1));
  sub_A23770(&v218);
  sub_A186C0(v218, 2, 1);
  sub_A186C0(v218, 6, 4);
  sub_A186C0(v218, 6, 4);
  sub_A186C0(v218, 4, 2);
  sub_A186C0(v218, 8, 2);
  v64 = v232[0];
  v229 = (__m128i)v218;
  if ( *((_QWORD *)&v218 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v218 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v218 + 1) + 8LL);
  }
  v65 = sub_A1A630(v64, 12, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v65 != 8 )
    goto LABEL_395;
  if ( *((_QWORD *)&v218 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v218 + 1));
  sub_A23770(&v219);
  sub_A186C0(v219, 3, 1);
  sub_A186C0(v219, 6, 4);
  v66 = v219;
  v67 = sub_A3FA40(v234);
  sub_A186C0(v66, v67, 2);
  sub_A186C0(v219, 4, 2);
  v68 = v232[0];
  v229 = (__m128i)v219;
  if ( *((_QWORD *)&v219 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v219 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v219 + 1) + 8LL);
  }
  v69 = sub_A1A630(v68, 12, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v69 != 9 )
    goto LABEL_395;
  if ( *((_QWORD *)&v219 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v219 + 1));
  sub_A23770(&v220);
  sub_A186C0(v220, 3, 1);
  sub_A186C0(v220, 6, 4);
  v70 = v220;
  v71 = sub_A3FA40(v234);
  sub_A186C0(v70, v71, 2);
  sub_A186C0(v220, 4, 2);
  sub_A186C0(v220, 8, 2);
  v72 = v232[0];
  v229 = (__m128i)v220;
  if ( *((_QWORD *)&v220 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v220 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v220 + 1) + 8LL);
  }
  v73 = sub_A1A630(v72, 12, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v73 != 10 )
    goto LABEL_395;
  if ( *((_QWORD *)&v220 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v220 + 1));
  sub_A23770(&v221);
  sub_A186C0(v221, 10, 1);
  v74 = v232[0];
  v229 = (__m128i)v221;
  if ( *((_QWORD *)&v221 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v221 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v221 + 1) + 8LL);
  }
  v75 = sub_A1A630(v74, 12, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v75 != 11 )
    goto LABEL_395;
  if ( *((_QWORD *)&v221 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v221 + 1));
  sub_A23770(&v222);
  sub_A186C0(v222, 10, 1);
  sub_A186C0(v222, 6, 4);
  v76 = v232[0];
  v229 = (__m128i)v222;
  if ( *((_QWORD *)&v222 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v222 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v222 + 1) + 8LL);
  }
  v77 = sub_A1A630(v76, 12, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v77 != 12 )
    goto LABEL_395;
  if ( *((_QWORD *)&v222 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v222 + 1));
  sub_A23770(&v223);
  sub_A186C0(v223, 15, 1);
  v78 = v232[0];
  v229 = (__m128i)v223;
  if ( *((_QWORD *)&v223 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v223 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v223 + 1) + 8LL);
  }
  v79 = sub_A1A630(v78, 12, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v79 != 13 )
    goto LABEL_395;
  if ( *((_QWORD *)&v223 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v223 + 1));
  sub_A23770(&v224);
  sub_A186C0(v224.m128i_i64[0], 43, 1);
  sub_A186C0(v224.m128i_i64[0], 3, 2);
  v80 = 0;
  v81 = (v234[8] - v234[7]) >> 3;
  if ( (_DWORD)v81 )
  {
    _BitScanReverse((unsigned int *)&v81, v81);
    v80 = (int)(32 - (v81 ^ 0x1F));
  }
  sub_A186C0(v224.m128i_i64[0], v80, 2);
  sub_A186C0(v224.m128i_i64[0], 0, 6);
  sub_A186C0(v224.m128i_i64[0], 6, 4);
  v82 = v232[0];
  v229 = v224;
  if ( v224.m128i_i64[1] )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(v224.m128i_i64[1] + 8), 1u);
    else
      ++*(_DWORD *)(v224.m128i_i64[1] + 8);
  }
  v83 = sub_A1A630(v82, 12, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v83 != 14 )
    goto LABEL_395;
  if ( v224.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v224.m128i_i64[1]);
  sub_A23770(&v227);
  sub_A186C0(v227, 64, 1);
  sub_A186C0(v227, 7, 4);
  sub_A186C0(v227, 7, 4);
  sub_A186C0(v227, 7, 4);
  sub_A186C0(v227, 6, 4);
  v84 = v232[0];
  v229 = (__m128i)v227;
  if ( *((_QWORD *)&v227 + 1) )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(*((_QWORD *)&v227 + 1) + 8LL), 1u);
    else
      ++*(_DWORD *)(*((_QWORD *)&v227 + 1) + 8LL);
  }
  v85 = sub_A1A630(v84, 12, v229.m128i_i64);
  if ( v229.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v229.m128i_i64[1]);
  if ( v85 != 15 )
LABEL_395:
    BUG();
  if ( *((_QWORD *)&v227 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v227 + 1));
  sub_A192A0(v232[0]);
  sub_A25160(v232);
  sub_A1FDA0((__int64)v232);
  if ( v243 != v242 )
  {
    v86 = 9;
    sub_A19830(v232[0], 9u, 3u);
    v87 = v242;
    v229.m128i_i64[0] = (__int64)&v230;
    v229.m128i_i64[1] = 0x4000000000LL;
    for ( i = v243; i != v87; v229.m128i_i32[2] = 0 )
    {
      v88 = -1;
      v89 = sub_A74480(v87);
      v90 = v89 - 1;
      if ( v89 )
      {
        do
        {
          v91 = sub_A74490(v87, v88);
          if ( v91 )
          {
            if ( v241 )
            {
              v92 = 1;
              for ( j = (v241 - 1)
                      & (((0xBF58476D1CE4E5B9LL
                         * (((unsigned __int64)(37 * v88) << 32) | ((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4))) >> 31)
                       ^ (484763065 * (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4)))); ; j = (v241 - 1) & v95 )
              {
                v94 = v240 + 24LL * j;
                if ( *(_DWORD *)v94 == v88 && v91 == *(_QWORD *)(v94 + 8) )
                  break;
                if ( *(_DWORD *)v94 == -1 && *(_QWORD *)(v94 + 8) == -4 )
                  goto LABEL_213;
                v95 = v92 + j;
                ++v92;
              }
            }
            else
            {
LABEL_213:
              v94 = v240 + 24LL * v241;
            }
            v96 = v229.m128i_u32[2];
            v97 = *(unsigned int *)(v94 + 16);
            v98 = v229.m128i_u32[2] + 1LL;
            if ( v98 > v229.m128i_u32[3] )
            {
              v199 = *(unsigned int *)(v94 + 16);
              sub_C8D5F0(&v229, &v230, v98, 8);
              v96 = v229.m128i_u32[2];
              v97 = v199;
            }
            *(_QWORD *)(v229.m128i_i64[0] + 8 * v96) = v97;
            ++v229.m128i_i32[2];
          }
          ++v88;
        }
        while ( v90 != v88 );
      }
      v87 += 8;
      v86 = 2;
      sub_A1FB70(v232[0], 2u, (__int64)&v229, 0);
    }
    sub_A192A0(v232[0]);
    if ( (__int64 *)v229.m128i_i64[0] != &v230 )
      _libc_free(v229.m128i_i64[0], v86);
  }
  v229.m128i_i64[0] = (__int64)&v230;
  v229.m128i_i64[1] = 0x4000000000LL;
  v202 = v238;
  if ( v237 != v238 )
  {
    v99 = v237;
    do
    {
      v100 = *v99;
      v101 = sub_AA8810(*v99);
      v102 = v101;
      v104 = v103;
      if ( v247 )
        sub_C8B160(v249, v101);
      v105 = v232[1];
      v106 = sub_C94890(v102, v104);
      v107 = sub_C0CA60(v105, v102, (v106 << 32) | (unsigned int)v104);
      v108 = v229.m128i_u32[2];
      v109 = v229.m128i_u32[2] + 1LL;
      if ( v109 > v229.m128i_u32[3] )
      {
        sub_C8D5F0(&v229, &v230, v109, 4);
        v108 = v229.m128i_u32[2];
      }
      *(_DWORD *)(v229.m128i_i64[0] + 4 * v108) = v107;
      ++v229.m128i_i32[2];
      sub_AA8810(v100);
      v110 = v229.m128i_u32[2];
      v112 = v111;
      v113 = v229.m128i_u32[2] + 1LL;
      if ( v113 > v229.m128i_u32[3] )
      {
        sub_C8D5F0(&v229, &v230, v113, 4);
        v110 = v229.m128i_u32[2];
      }
      *(_DWORD *)(v229.m128i_i64[0] + 4 * v110) = v112;
      v114 = (unsigned int)++v229.m128i_i32[2];
      v115 = *(_DWORD *)(v100 + 8);
      if ( v115 > 4 )
        BUG();
      v116 = v115 + 1;
      if ( v114 + 1 > (unsigned __int64)v229.m128i_u32[3] )
      {
        sub_C8D5F0(&v229, &v230, v114 + 1, 4);
        v114 = v229.m128i_u32[2];
      }
      v117 = 0;
      *(_DWORD *)(v229.m128i_i64[0] + 4 * v114) = v116;
      v118 = v232[0];
      ++v229.m128i_i32[2];
      v119 = v229.m128i_u32[2];
      v120 = v229.m128i_i32[2];
      sub_A17B10(v232[0], 3u, *(_DWORD *)(v232[0] + 56));
      sub_A17B10(v118, 0xCu, 6);
      v121 = (unsigned int)v119;
      sub_A17CC0(v118, v119, 6);
      v122 = 4 * v119;
      if ( v120 )
      {
        do
        {
          v121 = *(unsigned int *)(v229.m128i_i64[0] + v117);
          v117 += 4;
          sub_A17CC0(v118, v121, 6);
        }
        while ( v122 != v117 );
      }
      v229.m128i_i32[2] = 0;
      ++v99;
    }
    while ( v202 != v99 );
    if ( (__int64 *)v229.m128i_i64[0] != &v230 )
      _libc_free(v229.m128i_i64[0], v121);
  }
  sub_A28DF0(v232);
  v123 = v235;
  v124 = (v236 - (__int64)v235) >> 4;
  if ( (_DWORD)v124 )
  {
    v125 = 0;
    while ( **v123 <= 3u )
    {
      ++v125;
      v123 += 2;
      if ( (_DWORD)v124 == v125 )
        goto LABEL_242;
    }
    sub_A25DF0(v232, v125, v124, 1);
  }
LABEL_242:
  v126 = (__int64)&v227;
  *(_QWORD *)&v227 = v228;
  v229.m128i_i64[0] = (__int64)&v230;
  v229.m128i_i64[1] = 0x4000000000LL;
  *((_QWORD *)&v227 + 1) = 0x800000000LL;
  sub_BA8BF0(v233, &v227);
  if ( !DWORD2(v227) )
  {
    v127 = v227;
    if ( (_BYTE *)v227 == v228 )
      goto LABEL_245;
    goto LABEL_244;
  }
  v126 = 22;
  sub_A19830(v232[0], 0x16u, 3u);
  v163 = DWORD2(v227);
  if ( DWORD2(v227) )
  {
    v164 = v229.m128i_u32[2];
    v165 = 0;
    do
    {
      if ( v164 + 1 > (unsigned __int64)v229.m128i_u32[3] )
      {
        sub_C8D5F0(&v229, &v230, v164 + 1, 8);
        v164 = v229.m128i_u32[2];
      }
      *(_QWORD *)(v229.m128i_i64[0] + 8 * v164) = v165;
      v166 = (__int64 *)(v227 + 16 * v165);
      ++v229.m128i_i32[2];
      v167 = v229.m128i_u32[2];
      v168 = v166[1];
      v169 = *v166;
      if ( v168 + (unsigned __int64)v229.m128i_u32[2] > v229.m128i_u32[3] )
      {
        sub_C8D5F0(&v229, &v230, v168 + v229.m128i_u32[2], 8);
        v167 = v229.m128i_u32[2];
      }
      v170 = v229.m128i_i64[0] + 8 * v167;
      if ( v168 > 0 )
      {
        for ( k = 0; k != v168; ++k )
          *(_QWORD *)(v170 + 8 * k) = *(char *)(v169 + k);
        LODWORD(v167) = v229.m128i_i32[2];
      }
      v126 = 6;
      ++v165;
      v229.m128i_i32[2] = v168 + v167;
      sub_A1FB70(v232[0], 6u, (__int64)&v229, 0);
      v164 = 0;
      v229.m128i_i32[2] = 0;
    }
    while ( v163 != v165 );
  }
  sub_A192A0(v232[0]);
  v127 = v227;
  if ( (_BYTE *)v227 != v228 )
LABEL_244:
    _libc_free(v127, v126);
LABEL_245:
  if ( (__int64 *)v229.m128i_i64[0] != &v230 )
    _libc_free(v229.m128i_i64[0], v126);
  sub_A27070((__int64)v232);
  if ( v239 )
    sub_A207F0(v232, 0);
  *(_QWORD *)&v227 = v228;
  *((_QWORD *)&v227 + 1) = 0x800000000LL;
  sub_BA8C00(v233, &v227);
  v128 = DWORD2(v227);
  if ( DWORD2(v227) )
  {
    v128 = 21;
    sub_A19830(v232[0], 0x15u, 3u);
    v180 = 16LL * DWORD2(v227);
    v229.m128i_i64[0] = (__int64)&v230;
    v181 = (__int64 *)(v227 + v180);
    v229.m128i_i64[1] = 0x4000000000LL;
    if ( (_QWORD)v227 != (_QWORD)v227 + v180 )
    {
      v182 = (__int64 *)v227;
      v183 = *(_QWORD *)(v227 + 8);
      v184 = *(_QWORD *)v227;
      v185 = 0;
      if ( (unsigned __int64)v183 > 0x40 )
        goto LABEL_334;
      while ( 1 )
      {
        v186 = v229.m128i_i64[0] + 8 * v185;
        if ( v183 > 0 )
        {
          for ( m = 0; m != v183; ++m )
            *(_QWORD *)(v186 + 8 * m) = *(char *)(v184 + m);
          LODWORD(v185) = v229.m128i_i32[2];
        }
        v128 = 1;
        v182 += 2;
        v229.m128i_i32[2] = v183 + v185;
        sub_A1FB70(v232[0], 1u, (__int64)&v229, 0);
        v229.m128i_i32[2] = 0;
        if ( v181 == v182 )
          break;
        v183 = v182[1];
        v184 = *v182;
        v185 = 0;
        if ( v183 > (unsigned __int64)v229.m128i_u32[3] )
        {
LABEL_334:
          sub_C8D5F0(&v229, &v230, v183, 8);
          v185 = v229.m128i_u32[2];
        }
      }
    }
    sub_A192A0(v232[0]);
    if ( (__int64 *)v229.m128i_i64[0] != &v230 )
      _libc_free(v229.m128i_i64[0], v128);
  }
  if ( (_BYTE *)v227 != v228 )
    _libc_free(v227, v128);
  *(_QWORD *)&v227 = v228;
  v129 = (__int64)&v227;
  *((_QWORD *)&v227 + 1) = 0x800000000LL;
  sub_B6F820(*v233, &v227);
  if ( DWORD2(v227) )
  {
    v129 = 26;
    sub_A19830(v232[0], 0x1Au, 2u);
    v172 = 16LL * DWORD2(v227);
    v229.m128i_i64[0] = (__int64)&v230;
    v173 = (__int64 *)(v227 + v172);
    v229.m128i_i64[1] = 0x4000000000LL;
    if ( (_QWORD)v227 != (_QWORD)v227 + v172 )
    {
      v174 = (__int64 *)v227;
      v175 = *(_QWORD *)(v227 + 8);
      v176 = *(_QWORD *)v227;
      v177 = 0;
      if ( (unsigned __int64)v175 > 0x40 )
        goto LABEL_323;
      while ( 1 )
      {
        v178 = v229.m128i_i64[0] + 8 * v177;
        if ( v175 > 0 )
        {
          for ( n = 0; n != v175; ++n )
            *(_QWORD *)(v178 + 8 * n) = *(char *)(v176 + n);
          LODWORD(v177) = v229.m128i_i32[2];
        }
        v129 = 1;
        v174 += 2;
        v229.m128i_i32[2] = v175 + v177;
        sub_A1FB70(v232[0], 1u, (__int64)&v229, 0);
        v229.m128i_i32[2] = 0;
        if ( v173 == v174 )
          break;
        v175 = v174[1];
        v176 = *v174;
        v177 = 0;
        if ( v175 > (unsigned __int64)v229.m128i_u32[3] )
        {
LABEL_323:
          sub_C8D5F0(&v229, &v230, v175, 8);
          v177 = v229.m128i_u32[2];
        }
      }
    }
    sub_A192A0(v232[0]);
    if ( (__int64 *)v229.m128i_i64[0] != &v230 )
      _libc_free(v229.m128i_i64[0], v129);
  }
  if ( (_BYTE *)v227 != v228 )
    _libc_free(v227, v129);
  v229 = 0u;
  v230 = 0;
  v130 = v233 + 3;
  v231 = 0;
  v131 = (_QWORD *)v233[4];
  if ( v131 != v233 + 3 )
  {
    do
    {
      while ( 1 )
      {
        v132 = (__int64)(v131 - 7);
        if ( !v131 )
          v132 = 0;
        if ( !(unsigned __int8)sub_B2FC80(v132) )
          break;
        v131 = (_QWORD *)v131[1];
        if ( v130 == v131 )
          goto LABEL_262;
      }
      sub_A360E0((__int64)v232, v132, (__int64)&v229);
      v131 = (_QWORD *)v131[1];
    }
    while ( v130 != v131 );
  }
LABEL_262:
  if ( v244 )
    sub_A2D2B0(v232);
  v133 = v232[0];
  v134 = *(_QWORD **)(v232[0] + 32);
  v135 = *(_QWORD *)(*(_QWORD *)(v232[0] + 24) + 8LL);
  if ( v134 )
  {
    if ( (unsigned __int8)sub_CB7440(*(_QWORD *)(v232[0] + 32)) )
    {
      if ( !(unsigned __int8)sub_CB7440(v134) )
        BUG();
      v136 = (*(__int64 (__fastcall **)(_QWORD *))(*v134 + 80LL))(v134);
      v137 = v232[0];
      v135 += v136 + v134[4] - v134[2];
    }
    else
    {
      v137 = v232[0];
    }
  }
  else
  {
    v137 = v232[0];
  }
  v138 = *(unsigned int *)(v133 + 48) - v250 + 8 * v135;
  v139 = v246;
  LODWORD(v138) = (v138 >> 5) + 1;
  sub_A177B0(v137, v246, (unsigned __int8)v138);
  sub_A177B0(v137, v139 + 8, BYTE1(v138));
  LODWORD(v138) = WORD1(v138);
  sub_A177B0(v137, v139 + 16, (unsigned __int8)v138);
  sub_A177B0(v137, v139 + 24, (unsigned int)v138 >> 8);
  sub_A19830(v232[0], 0xEu, 4u);
  sub_A23770(&v224);
  sub_A186C0(v224.m128i_i64[0], 3, 1);
  sub_A186C0(v224.m128i_i64[0], 8, 4);
  sub_A186C0(v224.m128i_i64[0], 8, 4);
  v140 = v224.m128i_i64[0];
  v224.m128i_i64[0] = 0;
  *(_QWORD *)&v227 = v140;
  v141 = v224.m128i_i64[1];
  v224.m128i_i64[1] = 0;
  *((_QWORD *)&v227 + 1) = v141;
  v142 = sub_A1AB30(v232[0], (__int64 *)&v227);
  if ( *((_QWORD *)&v227 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v227 + 1));
  v143 = (_QWORD *)v233[4];
  if ( v143 != v233 + 3 )
  {
    v144 = v233 + 3;
    while ( 1 )
    {
      v152 = v143 - 7;
      if ( !v143 )
        v152 = 0;
      if ( (unsigned __int8)sub_B2FC80(v152) )
        goto LABEL_276;
      *(_QWORD *)&v227 = (unsigned int)sub_A3F3B0(v234);
      if ( !v231 )
        break;
      v145 = 1;
      v146 = 0;
      v147 = (v231 - 1) & (((unsigned int)v152 >> 9) ^ ((unsigned int)v152 >> 4));
      v148 = (_QWORD *)(v229.m128i_i64[1] + 16LL * v147);
      v149 = *v148;
      if ( (_QWORD *)*v148 != v152 )
      {
        while ( v149 != -4096 )
        {
          if ( !v146 && v149 == -8192 )
            v146 = v148;
          v147 = (v231 - 1) & (v145 + v147);
          v148 = (_QWORD *)(v229.m128i_i64[1] + 16LL * v147);
          v149 = *v148;
          if ( v152 == (_QWORD *)*v148 )
            goto LABEL_273;
          ++v145;
        }
        if ( !v146 )
          v146 = v148;
        ++v229.m128i_i64[0];
        v154 = v230 + 1;
        if ( 4 * ((int)v230 + 1) < 3 * v231 )
        {
          if ( v231 - HIDWORD(v230) - v154 <= v231 >> 3 )
          {
            sub_A2B080((__int64)&v229, v231);
            if ( !v231 )
            {
LABEL_393:
              LODWORD(v230) = v230 + 1;
              BUG();
            }
            v196 = 1;
            v197 = (v231 - 1) & (((unsigned int)v152 >> 9) ^ ((unsigned int)v152 >> 4));
            v154 = v230 + 1;
            v157 = 0;
            v146 = (_QWORD *)(v229.m128i_i64[1] + 16LL * v197);
            v198 = *v146;
            if ( (_QWORD *)*v146 != v152 )
            {
              while ( v198 != -4096 )
              {
                if ( !v157 && v198 == -8192 )
                  v157 = v146;
                v197 = (v231 - 1) & (v196 + v197);
                v146 = (_QWORD *)(v229.m128i_i64[1] + 16LL * v197);
                v198 = *v146;
                if ( v152 == (_QWORD *)*v146 )
                  goto LABEL_356;
                ++v196;
              }
LABEL_384:
              if ( v157 )
                v146 = v157;
            }
          }
LABEL_356:
          LODWORD(v230) = v154;
          if ( *v146 != -4096 )
            --HIDWORD(v230);
          *v146 = v152;
          v146[1] = 0;
          v150 = 0;
          goto LABEL_274;
        }
LABEL_282:
        sub_A2B080((__int64)&v229, 2 * v231);
        if ( !v231 )
          goto LABEL_393;
        v153 = (v231 - 1) & (((unsigned int)v152 >> 9) ^ ((unsigned int)v152 >> 4));
        v154 = v230 + 1;
        v146 = (_QWORD *)(v229.m128i_i64[1] + 16LL * v153);
        v155 = *v146;
        if ( (_QWORD *)*v146 != v152 )
        {
          v156 = 1;
          v157 = 0;
          while ( v155 != -4096 )
          {
            if ( !v157 && v155 == -8192 )
              v157 = v146;
            v153 = (v231 - 1) & (v156 + v153);
            v146 = (_QWORD *)(v229.m128i_i64[1] + 16LL * v153);
            v155 = *v146;
            if ( v152 == (_QWORD *)*v146 )
              goto LABEL_356;
            ++v156;
          }
          goto LABEL_384;
        }
        goto LABEL_356;
      }
LABEL_273:
      v150 = v148[1];
LABEL_274:
      v151 = v232[0];
      *((_QWORD *)&v227 + 1) = ((unsigned __int64)(v150 - v250) >> 5) + 1;
      if ( v142 )
      {
        sub_A1B020(v232[0], v142, (__int64)&v227, 2, 0, 0, 3u, 1);
LABEL_276:
        v143 = (_QWORD *)v143[1];
        if ( v144 == v143 )
          goto LABEL_294;
      }
      else
      {
        sub_A17B10(v232[0], 3u, *(_DWORD *)(v232[0] + 56));
        sub_A17CC0(v151, 3u, 6);
        sub_A17CC0(v151, 2u, 6);
        sub_A17DE0(v151, v227, 6);
        sub_A17DE0(v151, *((unsigned __int64 *)&v227 + 1), 6);
        v143 = (_QWORD *)v143[1];
        if ( v144 == v143 )
          goto LABEL_294;
      }
    }
    ++v229.m128i_i64[0];
    goto LABEL_282;
  }
LABEL_294:
  sub_A192A0(v232[0]);
  if ( v224.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v224.m128i_i64[1]);
  v158 = v232[0];
  v159 = *(_QWORD *)(v232[0] + 88);
  if ( *(_BYTE *)(v232[0] + 96) )
    *(_BYTE *)(v232[0] + 96) = 0;
  if ( v247 )
  {
    sub_C8B060(v249, v159 + **(_QWORD **)(v158 + 24), *(_QWORD *)(*(_QWORD *)(v158 + 24) + 8LL) - v159);
    sub_C8B2A0(&v227, v249);
    v188 = (unsigned int *)&v227;
    for ( ii = 0; ii != 20; ii += 4 )
    {
      v190 = ii;
      v191 = *v188++;
      v224.m128i_i32[v190 >> 2] = _byteswap_ulong(v191);
    }
    v192 = v232[0];
    sub_A17B10(v232[0], 3u, *(_DWORD *)(v232[0] + 56));
    sub_A17CC0(v192, 0x11u, 6);
    sub_A17CC0(v192, 5u, 6);
    v193 = &v224;
    do
    {
      v194 = v193->m128i_u32[0];
      v193 = (__m128i *)((char *)v193 + 4);
      sub_A17DE0(v192, v194, 6);
    }
    while ( v193 != (__m128i *)&v226 );
    v195 = v248;
    if ( v248 )
    {
      *v248 = _mm_loadu_si128(&v224);
      v195[1].m128i_i32[0] = v225;
    }
    v158 = v232[0];
  }
  sub_A192A0(v158);
  sub_C7D6A0(v229.m128i_i64[1], 16LL * v231, 8);
  v160 = v245;
  while ( v160 )
  {
    sub_A167C0(*(_QWORD *)(v160 + 24));
    v161 = v160;
    v160 = *(_QWORD *)(v160 + 16);
    j_j___libc_free_0(v161, 48);
  }
  return sub_A17F40(v234);
}
