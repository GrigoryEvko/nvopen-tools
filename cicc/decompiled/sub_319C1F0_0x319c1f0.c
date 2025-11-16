// Function: sub_319C1F0
// Address: 0x319c1f0
//
void __fastcall sub_319C1F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        unsigned int a6,
        char a7,
        char a8,
        char a9,
        _QWORD *a10,
        __int64 a11)
{
  __int64 v11; // r13
  _QWORD *v13; // r14
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // rsi
  int v20; // eax
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // r14
  unsigned __int8 *v24; // rax
  unsigned __int8 *v25; // r14
  _BYTE *v26; // rdx
  __int64 v27; // rax
  unsigned __int64 v28; // rax
  char v29; // r12
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // rax
  __int64 v32; // rbx
  __int64 v33; // r13
  int v34; // eax
  int v35; // eax
  unsigned int v36; // edx
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // r13
  char v42; // r14
  __int64 v43; // r9
  _QWORD *v44; // r12
  unsigned int *v45; // r14
  unsigned int *v46; // r13
  __int64 v47; // rdx
  unsigned int v48; // esi
  _QWORD *v49; // rax
  __int64 v50; // r9
  __int64 v51; // r13
  __int64 v52; // r14
  unsigned int *v53; // r14
  unsigned int *v54; // rbx
  __int64 v55; // rdx
  unsigned int v56; // esi
  __int16 v57; // ax
  __int16 v58; // ax
  _BYTE *v59; // rax
  __int64 v60; // r13
  int v61; // eax
  int v62; // eax
  unsigned int v63; // edx
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // rdx
  unsigned __int64 v69; // rax
  __int64 v70; // rdx
  unsigned __int64 v71; // rax
  __int64 v72; // r12
  __int64 v73; // rax
  __int64 v74; // r14
  __int64 v75; // rax
  __int64 v76; // r12
  _BYTE *v77; // rax
  __int64 v78; // rax
  __int64 v79; // rbx
  _QWORD *v80; // rax
  __int64 v81; // r14
  __int64 v82; // rbx
  unsigned int *v83; // rbx
  unsigned int *v84; // r12
  __int64 v85; // rdx
  unsigned int v86; // esi
  unsigned __int64 v87; // rax
  int v88; // edx
  _QWORD *v89; // rdi
  _QWORD *v90; // rax
  _QWORD *v91; // rax
  __int64 v92; // r9
  __int64 v93; // rbx
  unsigned int *v94; // r14
  unsigned int *v95; // r13
  __int64 v96; // rdx
  unsigned int v97; // esi
  __int64 v98; // r14
  _QWORD *v99; // rax
  __int64 v100; // r9
  __int64 v101; // r13
  _BYTE *v102; // r14
  _BYTE *v103; // rbx
  __int64 v104; // rdx
  unsigned int v105; // esi
  __int64 v106; // rax
  __int64 v107; // rbx
  int v108; // eax
  int v109; // eax
  unsigned int v110; // edx
  __int64 v111; // rax
  __int64 v112; // rdx
  __int64 v113; // rdx
  __int64 v114; // r13
  _QWORD *v115; // r12
  unsigned int *v116; // r14
  unsigned int *v117; // r13
  __int64 v118; // rdx
  unsigned int v119; // esi
  __int64 v120; // r14
  __int64 v121; // r9
  _QWORD *v122; // r13
  __int64 v123; // r14
  unsigned int *v124; // r14
  unsigned int *v125; // rbx
  __int64 v126; // rdx
  unsigned int v127; // esi
  __int16 v128; // ax
  __int16 v129; // ax
  _BYTE *v130; // rax
  __int64 v131; // rdx
  int v132; // eax
  int v133; // eax
  unsigned int v134; // ecx
  __int64 v135; // rax
  __int64 v136; // rcx
  __int64 v137; // rcx
  __int64 v138; // rax
  __int64 v139; // r12
  _QWORD *v140; // rax
  __int64 v141; // r9
  __int64 v142; // rbx
  unsigned int *v143; // r13
  unsigned int *v144; // r12
  __int64 v145; // rdx
  unsigned int v146; // esi
  __int64 v147; // rax
  __int64 v148; // rax
  _BYTE *v149; // rax
  __int64 v150; // r14
  _QWORD *v151; // rax
  __int64 v152; // r9
  __int64 v153; // r12
  unsigned int *v154; // r14
  unsigned int *v155; // rbx
  __int64 v156; // rdx
  unsigned int v157; // esi
  unsigned __int64 v158; // rax
  int v159; // edx
  _QWORD *v160; // rdi
  _QWORD *v161; // rax
  __int64 v162; // rax
  __int64 v163; // r12
  _QWORD *v164; // rax
  __int64 v165; // r9
  __int64 v166; // rbx
  unsigned int *v167; // r13
  unsigned int *v168; // r12
  __int64 v169; // rdx
  unsigned int v170; // esi
  __int64 v171; // rax
  __int64 v172; // rax
  __int64 v173; // [rsp-10h] [rbp-3F0h]
  __int64 v174; // [rsp-8h] [rbp-3E8h]
  char v175; // [rsp+4h] [rbp-3DCh]
  char v176; // [rsp+8h] [rbp-3D8h]
  _QWORD *v177; // [rsp+10h] [rbp-3D0h]
  __int64 v178; // [rsp+18h] [rbp-3C8h]
  __int64 v179; // [rsp+20h] [rbp-3C0h]
  __int64 v180; // [rsp+40h] [rbp-3A0h]
  __int64 v181; // [rsp+40h] [rbp-3A0h]
  unsigned int *v182; // [rsp+48h] [rbp-398h]
  __int64 v183; // [rsp+50h] [rbp-390h]
  char v184; // [rsp+50h] [rbp-390h]
  char v185; // [rsp+58h] [rbp-388h]
  char v186; // [rsp+58h] [rbp-388h]
  _BYTE *v187; // [rsp+60h] [rbp-380h]
  unsigned int v188; // [rsp+68h] [rbp-378h]
  __int64 v189; // [rsp+68h] [rbp-378h]
  __int64 v190; // [rsp+70h] [rbp-370h]
  _BYTE *v193; // [rsp+90h] [rbp-350h]
  __int64 v194; // [rsp+98h] [rbp-348h]
  __int64 v195; // [rsp+98h] [rbp-348h]
  __int64 v196; // [rsp+A8h] [rbp-338h]
  __int64 v197; // [rsp+B0h] [rbp-330h]
  __int64 v198; // [rsp+B8h] [rbp-328h]
  _QWORD *v199; // [rsp+D8h] [rbp-308h]
  __int64 v200; // [rsp+D8h] [rbp-308h]
  __int64 v201; // [rsp+E0h] [rbp-300h]
  __int64 *v202; // [rsp+F0h] [rbp-2F0h]
  __int64 v204; // [rsp+F8h] [rbp-2E8h]
  __int64 v205; // [rsp+F8h] [rbp-2E8h]
  __int64 *v206; // [rsp+100h] [rbp-2E0h] BYREF
  _BYTE *v207; // [rsp+108h] [rbp-2D8h] BYREF
  _BYTE v208[32]; // [rsp+110h] [rbp-2D0h] BYREF
  __int16 v209; // [rsp+130h] [rbp-2B0h]
  __int64 v210[4]; // [rsp+140h] [rbp-2A0h] BYREF
  __int16 v211; // [rsp+160h] [rbp-280h]
  unsigned int *v212; // [rsp+170h] [rbp-270h] BYREF
  unsigned int v213; // [rsp+178h] [rbp-268h]
  char v214; // [rsp+180h] [rbp-260h] BYREF
  __int64 v215; // [rsp+1A8h] [rbp-238h]
  __int64 v216; // [rsp+1B0h] [rbp-230h]
  __int64 v217; // [rsp+1C8h] [rbp-218h]
  void *v218; // [rsp+1F0h] [rbp-1F0h]
  unsigned int *v219; // [rsp+200h] [rbp-1E0h] BYREF
  __int64 v220; // [rsp+208h] [rbp-1D8h]
  _BYTE v221[32]; // [rsp+210h] [rbp-1D0h] BYREF
  __int64 v222; // [rsp+230h] [rbp-1B0h]
  __int64 v223; // [rsp+238h] [rbp-1A8h]
  __int64 v224; // [rsp+240h] [rbp-1A0h]
  __int64 v225; // [rsp+248h] [rbp-198h]
  void **v226; // [rsp+250h] [rbp-190h]
  void **v227; // [rsp+258h] [rbp-188h]
  __int64 v228; // [rsp+260h] [rbp-180h]
  int v229; // [rsp+268h] [rbp-178h]
  __int16 v230; // [rsp+26Ch] [rbp-174h]
  char v231; // [rsp+26Eh] [rbp-172h]
  __int64 v232; // [rsp+270h] [rbp-170h]
  __int64 v233; // [rsp+278h] [rbp-168h]
  void *v234; // [rsp+280h] [rbp-160h] BYREF
  void *v235; // [rsp+288h] [rbp-158h] BYREF
  _BYTE *v236; // [rsp+290h] [rbp-150h] BYREF
  __int64 v237; // [rsp+298h] [rbp-148h]
  _BYTE v238[16]; // [rsp+2A0h] [rbp-140h] BYREF
  __int16 v239; // [rsp+2B0h] [rbp-130h]
  __int64 v240; // [rsp+2C0h] [rbp-120h]
  __int64 v241; // [rsp+2C8h] [rbp-118h]
  __int64 v242; // [rsp+2D0h] [rbp-110h]
  __int64 v243; // [rsp+2D8h] [rbp-108h]
  void **v244; // [rsp+2E0h] [rbp-100h]
  void **v245; // [rsp+2E8h] [rbp-F8h]
  __int64 v246; // [rsp+2F0h] [rbp-F0h]
  int v247; // [rsp+2F8h] [rbp-E8h]
  __int16 v248; // [rsp+2FCh] [rbp-E4h]
  char v249; // [rsp+2FEh] [rbp-E2h]
  __int64 v250; // [rsp+300h] [rbp-E0h]
  __int64 v251; // [rsp+308h] [rbp-D8h]
  void *v252; // [rsp+310h] [rbp-D0h] BYREF
  void *v253; // [rsp+318h] [rbp-C8h] BYREF
  char *v254; // [rsp+320h] [rbp-C0h] BYREF
  __int64 v255; // [rsp+328h] [rbp-B8h]
  _BYTE v256[16]; // [rsp+330h] [rbp-B0h] BYREF
  __int16 v257; // [rsp+340h] [rbp-A0h]
  __int64 v258; // [rsp+350h] [rbp-90h]
  __int64 v259; // [rsp+358h] [rbp-88h]
  __int64 v260; // [rsp+360h] [rbp-80h]
  __int64 v261; // [rsp+368h] [rbp-78h]
  void **v262; // [rsp+370h] [rbp-70h]
  void **v263; // [rsp+378h] [rbp-68h]
  __int64 v264; // [rsp+380h] [rbp-60h]
  int v265; // [rsp+388h] [rbp-58h]
  __int16 v266; // [rsp+38Ch] [rbp-54h]
  char v267; // [rsp+38Eh] [rbp-52h]
  __int64 v268; // [rsp+390h] [rbp-50h]
  __int64 v269; // [rsp+398h] [rbp-48h]
  void *v270; // [rsp+3A0h] [rbp-40h] BYREF
  void *v271; // [rsp+3A8h] [rbp-38h] BYREF

  v11 = a6;
  v13 = *(_QWORD **)(a1 + 40);
  v257 = 259;
  v254 = "post-loop-memcpy-expansion";
  v14 = sub_AA8550(v13, (__int64 *)(a1 + 24), 0, (__int64)&v254, 0);
  v15 = v13[9];
  v197 = v14;
  v199 = v13;
  v183 = sub_B2BEC0(v15);
  v202 = (__int64 *)sub_AA48A0((__int64)v13);
  v206 = v202;
  v16 = sub_B8CD90(&v206, (__int64)"MemCopyDomain", 13, 0);
  v182 = (unsigned int *)sub_B8CD90(&v206, (__int64)"MemCopyAliasScope", 17, v16);
  v194 = sub_DFDDE0(
           a10,
           v202,
           a4,
           *(_DWORD *)(*(_QWORD *)(a2 + 8) + 8LL) >> 8,
           *(_DWORD *)(*(_QWORD *)(a3 + 8) + 8LL) >> 8,
           a5,
           v11,
           a11);
  v17 = sub_9208B0(v183, v194);
  v255 = v18;
  v254 = (char *)((unsigned __int64)(v17 + 7) >> 3);
  v188 = sub_CA1930(&v254);
  v177 = v13 + 6;
  v19 = v13[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v13 + 6 == (_QWORD *)v19 )
  {
    v21 = 0;
  }
  else
  {
    if ( !v19 )
      BUG();
    v20 = *(unsigned __int8 *)(v19 - 24);
    v21 = v19 - 24;
    if ( (unsigned int)(v20 - 30) >= 0xB )
      v21 = 0;
  }
  sub_23D0AB0((__int64)&v212, v21, 0, 0, 0);
  v22 = 0;
  v196 = *(_QWORD *)(a4 + 8);
  if ( *(_BYTE *)(v196 + 8) == 12 )
    v22 = *(_QWORD *)(a4 + 8);
  v23 = v22;
  v178 = v22;
  v201 = sub_BCB2B0(v202);
  v24 = (unsigned __int8 *)sub_ACD640(v23, v188, 0);
  v187 = 0;
  v25 = v24;
  if ( v194 != v201 )
  {
    v187 = (_BYTE *)sub_319A760((__int64 *)&v212, a4, v24, v188);
    v26 = v187;
    if ( !v187 )
      v26 = (_BYTE *)sub_319A760((__int64 *)&v212, a4, v25, v188);
    v257 = 257;
    a4 = sub_929DE0(&v212, (_BYTE *)a4, v26, (__int64)&v254, 0, 0);
  }
  v254 = "loop-memcpy-expansion";
  v257 = 259;
  v27 = sub_22077B0(0x50u);
  v198 = v27;
  if ( v27 )
    sub_AA4D50(v27, (__int64)v202, (__int64)&v254, v15, v197);
  v225 = sub_AA48A0(v198);
  v226 = &v234;
  v227 = &v235;
  v219 = (unsigned int *)v221;
  v234 = &unk_49DA100;
  v220 = 0x200000000LL;
  LOWORD(v224) = 0;
  v235 = &unk_49DA0B0;
  v223 = v198 + 48;
  v230 = 512;
  v228 = 0;
  v229 = 0;
  v231 = 7;
  v232 = 0;
  v233 = 0;
  v222 = v198;
  v28 = (v188 | (unsigned __int64)(1LL << a5)) & -(__int64)(v188 | (unsigned __int64)(1LL << a5));
  if ( v28 )
  {
    _BitScanReverse64(&v28, v28);
    v176 = 63 - (v28 ^ 0x3F);
    v29 = v176;
  }
  else
  {
    v176 = -1;
    v29 = -1;
  }
  v30 = v188 | (unsigned __int64)(1LL << v11);
  if ( (v30 & -(__int64)v30) != 0 )
  {
    _BitScanReverse64(&v31, v30 & -(__int64)(v188 | (unsigned __int64)(1LL << v11)));
    v175 = 63 - (v31 ^ 0x3F);
    v185 = v175;
  }
  else
  {
    v185 = -1;
    v175 = -1;
  }
  v254 = "loop-index";
  v257 = 259;
  v32 = sub_D5C860((__int64 *)&v219, v196, 2, (__int64)&v254);
  v33 = sub_AD64C0(v196, 0, 0);
  v34 = *(_DWORD *)(v32 + 4) & 0x7FFFFFF;
  if ( v34 == *(_DWORD *)(v32 + 72) )
  {
    sub_B48D90(v32);
    v34 = *(_DWORD *)(v32 + 4) & 0x7FFFFFF;
  }
  v35 = (v34 + 1) & 0x7FFFFFF;
  v36 = v35 | *(_DWORD *)(v32 + 4) & 0xF8000000;
  v37 = *(_QWORD *)(v32 - 8) + 32LL * (unsigned int)(v35 - 1);
  *(_DWORD *)(v32 + 4) = v36;
  if ( *(_QWORD *)v37 )
  {
    v38 = *(_QWORD *)(v37 + 8);
    **(_QWORD **)(v37 + 16) = v38;
    if ( v38 )
      *(_QWORD *)(v38 + 16) = *(_QWORD *)(v37 + 16);
  }
  *(_QWORD *)v37 = v33;
  if ( v33 )
  {
    v39 = *(_QWORD *)(v33 + 16);
    *(_QWORD *)(v37 + 8) = v39;
    if ( v39 )
      *(_QWORD *)(v39 + 16) = v37 + 8;
    *(_QWORD *)(v37 + 16) = v33 + 16;
    *(_QWORD *)(v33 + 16) = v37;
  }
  *(_QWORD *)(*(_QWORD *)(v32 - 8) + 32LL * *(unsigned int *)(v32 + 72)
                                   + 8LL * ((*(_DWORD *)(v32 + 4) & 0x7FFFFFFu) - 1)) = v199;
  v257 = 257;
  v236 = (_BYTE *)v32;
  v40 = sub_921130(&v219, v201, a2, &v236, 1, (__int64)&v254, 3u);
  v239 = 257;
  v41 = v40;
  v42 = v29;
  v257 = 257;
  v44 = sub_BD2C40(80, 1u);
  if ( v44 )
  {
    sub_B4D190((__int64)v44, v194, v41, (__int64)&v254, a7, v42, 0, 0);
    v43 = v173;
  }
  (*((void (__fastcall **)(void **, _QWORD *, _BYTE **, __int64, __int64, __int64))*v227 + 2))(
    v227,
    v44,
    &v236,
    v223,
    v224,
    v43);
  v45 = v219;
  v46 = &v219[4 * (unsigned int)v220];
  if ( v219 != v46 )
  {
    do
    {
      v47 = *((_QWORD *)v45 + 1);
      v48 = *v45;
      v45 += 4;
      sub_B99FD0((__int64)v44, v48, v47);
    }
    while ( v46 != v45 );
  }
  if ( !a9 )
  {
    v254 = (char *)v182;
    v147 = sub_B9C770(v202, (__int64 *)&v254, (__int64 *)1, 0, 1);
    sub_B99FD0((__int64)v44, 7u, v147);
  }
  v257 = 257;
  v236 = (_BYTE *)v32;
  v179 = sub_921130(&v219, v201, a3, &v236, 1, (__int64)&v254, 3u);
  v257 = 257;
  v49 = sub_BD2C40(80, unk_3F10A10);
  v51 = (__int64)v49;
  if ( v49 )
    sub_B4D3C0((__int64)v49, (__int64)v44, v179, a8, v185, v50, 0, 0);
  (*((void (__fastcall **)(void **, __int64, char **, __int64, __int64))*v227 + 2))(v227, v51, &v254, v223, v224);
  v52 = 4LL * (unsigned int)v220;
  if ( v219 != &v219[v52] )
  {
    v180 = v32;
    v53 = &v219[v52];
    v54 = v219;
    do
    {
      v55 = *((_QWORD *)v54 + 1);
      v56 = *v54;
      v54 += 4;
      sub_B99FD0(v51, v56, v55);
    }
    while ( v53 != v54 );
    v32 = v180;
  }
  if ( !a9 )
  {
    v254 = (char *)v182;
    v148 = sub_B9C770(v202, (__int64 *)&v254, (__int64 *)1, 0, 1);
    sub_B99FD0(v51, 8u, v148);
  }
  if ( BYTE4(a11) )
  {
    v57 = *((_WORD *)v44 + 1);
    *((_BYTE *)v44 + 72) = 1;
    v57 &= 0xFC7Fu;
    LOBYTE(v57) = v57 | 0x80;
    *((_WORD *)v44 + 1) = v57;
    v58 = *(_WORD *)(v51 + 2);
    *(_BYTE *)(v51 + 72) = 1;
    v58 &= 0xFC7Fu;
    LOBYTE(v58) = v58 | 0x80;
    *(_WORD *)(v51 + 2) = v58;
  }
  v257 = 257;
  v59 = (_BYTE *)sub_AD64C0(v196, v188, 0);
  v60 = sub_929C50(&v219, (_BYTE *)v32, v59, (__int64)&v254, 0, 0);
  v61 = *(_DWORD *)(v32 + 4) & 0x7FFFFFF;
  if ( v61 == *(_DWORD *)(v32 + 72) )
  {
    sub_B48D90(v32);
    v61 = *(_DWORD *)(v32 + 4) & 0x7FFFFFF;
  }
  v62 = (v61 + 1) & 0x7FFFFFF;
  v63 = v62 | *(_DWORD *)(v32 + 4) & 0xF8000000;
  v64 = *(_QWORD *)(v32 - 8) + 32LL * (unsigned int)(v62 - 1);
  *(_DWORD *)(v32 + 4) = v63;
  if ( *(_QWORD *)v64 )
  {
    v65 = *(_QWORD *)(v64 + 8);
    **(_QWORD **)(v64 + 16) = v65;
    if ( v65 )
      *(_QWORD *)(v65 + 16) = *(_QWORD *)(v64 + 16);
  }
  *(_QWORD *)v64 = v60;
  if ( v60 )
  {
    v66 = *(_QWORD *)(v60 + 16);
    *(_QWORD *)(v64 + 8) = v66;
    if ( v66 )
      *(_QWORD *)(v66 + 16) = v64 + 8;
    *(_QWORD *)(v64 + 16) = v60 + 16;
    *(_QWORD *)(v60 + 16) = v64;
  }
  *(_QWORD *)(*(_QWORD *)(v32 - 8) + 32LL * *(unsigned int *)(v32 + 72)
                                   + 8LL * ((*(_DWORD *)(v32 + 4) & 0x7FFFFFFu) - 1)) = v198;
  if ( v194 != v201 )
  {
    if ( !BYTE4(a11) )
    {
      v181 = v201;
      v67 = sub_9208B0(v183, v201);
      goto LABEL_55;
    }
    if ( v188 != (_DWORD)a11 )
    {
      v181 = sub_BCD140(v202, 8 * (int)a11);
      v67 = sub_9208B0(v183, v181);
LABEL_55:
      v255 = v68;
      v254 = (char *)((unsigned __int64)(v67 + 7) >> 3);
      v184 = -1;
      v189 = (unsigned int)sub_CA1930(&v254);
      v69 = (v189 | (1LL << v176)) & -(v189 | (1LL << v176));
      if ( v69 )
      {
        _BitScanReverse64(&v69, v69);
        v184 = 63 - (v69 ^ 0x3F);
      }
      v186 = -1;
      v70 = v189 | (1LL << v175);
      if ( (v70 & -v70) != 0 )
      {
        _BitScanReverse64(&v71, v70 & -(v189 | (1LL << v175)));
        v186 = 63 - (v71 ^ 0x3F);
      }
      v72 = v199[9];
      v254 = "loop-memcpy-residual";
      v257 = 259;
      v73 = sub_22077B0(0x50u);
      v195 = v73;
      if ( v73 )
        sub_AA4D50(v73, (__int64)v202, (__int64)&v254, v72, v197);
      v74 = v199[9];
      v254 = "loop-memcpy-residual-header";
      v257 = 259;
      v75 = sub_22077B0(0x50u);
      v76 = v75;
      if ( v75 )
        sub_AA4D50(v75, (__int64)v202, (__int64)&v254, v74, 0);
      v77 = (_BYTE *)sub_ACD640(v178, 0, 0);
      v239 = 257;
      v193 = v77;
      v78 = sub_92B530(&v212, 0x21u, a4, v77, (__int64)&v236);
      v257 = 257;
      v79 = v78;
      v80 = sub_BD2C40(72, 3u);
      v81 = (__int64)v80;
      if ( v80 )
        sub_B4C9A0((__int64)v80, v198, v76, v79, 3u, 0, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, char **, __int64, __int64))(*(_QWORD *)v217 + 16LL))(
        v217,
        v81,
        &v254,
        v215,
        v216);
      v82 = 4LL * v213;
      if ( v212 != &v212[v82] )
      {
        v190 = v76;
        v83 = &v212[v82];
        v84 = v212;
        do
        {
          v85 = *((_QWORD *)v84 + 1);
          v86 = *v84;
          v84 += 4;
          sub_B99FD0(v81, v86, v85);
        }
        while ( v83 != v84 );
        v76 = v190;
      }
      v87 = v199[6] & 0xFFFFFFFFFFFFFFF8LL;
      if ( v177 == (_QWORD *)v87 )
      {
        v89 = 0;
      }
      else
      {
        if ( !v87 )
          BUG();
        v88 = *(unsigned __int8 *)(v87 - 24);
        v89 = 0;
        v90 = (_QWORD *)(v87 - 24);
        if ( (unsigned int)(v88 - 30) < 0xB )
          v89 = v90;
      }
      sub_B43D60(v89);
      v239 = 257;
      v200 = sub_92B530(&v219, 0x24u, v60, (_BYTE *)a4, (__int64)&v236);
      v257 = 257;
      v91 = sub_BD2C40(72, 3u);
      v93 = (__int64)v91;
      if ( v91 )
        sub_B4C9A0((__int64)v91, v198, v76, v200, 3u, v92, 0, 0);
      (*((void (__fastcall **)(void **, __int64, char **, __int64, __int64))*v227 + 2))(v227, v93, &v254, v223, v224);
      v94 = v219;
      v95 = &v219[4 * (unsigned int)v220];
      if ( v219 != v95 )
      {
        do
        {
          v96 = *((_QWORD *)v94 + 1);
          v97 = *v94;
          v94 += 4;
          sub_B99FD0(v93, v97, v96);
        }
        while ( v95 != v94 );
      }
      v243 = sub_AA48A0(v76);
      v244 = &v252;
      v245 = &v253;
      v236 = v238;
      v252 = &unk_49DA100;
      v237 = 0x200000000LL;
      LOWORD(v242) = 0;
      v253 = &unk_49DA0B0;
      v241 = v76 + 48;
      v248 = 512;
      v211 = 257;
      v246 = 0;
      v247 = 0;
      v249 = 7;
      v250 = 0;
      v251 = 0;
      v240 = v76;
      v98 = sub_92B530((unsigned int **)&v236, 0x21u, (__int64)v187, v193, (__int64)v210);
      v257 = 257;
      v99 = sub_BD2C40(72, 3u);
      v101 = (__int64)v99;
      if ( v99 )
        sub_B4C9A0((__int64)v99, v195, v197, v98, 3u, v100, 0, 0);
      (*((void (__fastcall **)(void **, __int64, char **, __int64, __int64))*v245 + 2))(v245, v101, &v254, v241, v242);
      v102 = v236;
      v103 = &v236[16 * (unsigned int)v237];
      if ( v236 != v103 )
      {
        do
        {
          v104 = *((_QWORD *)v102 + 1);
          v105 = *(_DWORD *)v102;
          v102 += 16;
          sub_B99FD0(v101, v105, v104);
        }
        while ( v103 != v102 );
      }
      v106 = sub_AA48A0(v195);
      v267 = 7;
      v261 = v106;
      v262 = &v270;
      v263 = &v271;
      v254 = v256;
      v270 = &unk_49DA100;
      v255 = 0x200000000LL;
      v266 = 512;
      v271 = &unk_49DA0B0;
      LOWORD(v260) = 0;
      v259 = v195 + 48;
      v258 = v195;
      v264 = 0;
      v265 = 0;
      v268 = 0;
      v269 = 0;
      v210[0] = (__int64)"residual-loop-index";
      v211 = 259;
      v107 = sub_D5C860((__int64 *)&v254, v196, 2, (__int64)v210);
      v108 = *(_DWORD *)(v107 + 4) & 0x7FFFFFF;
      if ( v108 == *(_DWORD *)(v107 + 72) )
      {
        sub_B48D90(v107);
        v108 = *(_DWORD *)(v107 + 4) & 0x7FFFFFF;
      }
      v109 = (v108 + 1) & 0x7FFFFFF;
      v110 = v109 | *(_DWORD *)(v107 + 4) & 0xF8000000;
      v111 = *(_QWORD *)(v107 - 8) + 32LL * (unsigned int)(v109 - 1);
      *(_DWORD *)(v107 + 4) = v110;
      if ( *(_QWORD *)v111 )
      {
        v112 = *(_QWORD *)(v111 + 8);
        **(_QWORD **)(v111 + 16) = v112;
        if ( v112 )
          *(_QWORD *)(v112 + 16) = *(_QWORD *)(v111 + 16);
      }
      *(_QWORD *)v111 = v193;
      if ( v193 )
      {
        v113 = *((_QWORD *)v193 + 2);
        *(_QWORD *)(v111 + 8) = v113;
        if ( v113 )
          *(_QWORD *)(v113 + 16) = v111 + 8;
        *(_QWORD *)(v111 + 16) = v193 + 16;
        *((_QWORD *)v193 + 2) = v111;
      }
      *(_QWORD *)(*(_QWORD *)(v107 - 8)
                + 32LL * *(unsigned int *)(v107 + 72)
                + 8LL * ((*(_DWORD *)(v107 + 4) & 0x7FFFFFFu) - 1)) = v76;
      v211 = 257;
      v207 = (_BYTE *)sub_929C50((unsigned int **)&v254, (_BYTE *)a4, (_BYTE *)v107, (__int64)v210, 0, 0);
      v211 = 257;
      v114 = sub_921130((unsigned int **)&v254, v201, a2, &v207, 1, (__int64)v210, 3u);
      v209 = 257;
      v211 = 257;
      v115 = sub_BD2C40(80, 1u);
      if ( v115 )
        sub_B4D190((__int64)v115, v181, v114, (__int64)v210, a7, v184, 0, 0);
      (*((void (__fastcall **)(void **, _QWORD *, _BYTE *, __int64, __int64))*v263 + 2))(v263, v115, v208, v259, v260);
      v116 = (unsigned int *)v254;
      v117 = (unsigned int *)&v254[16 * (unsigned int)v255];
      if ( v254 != (char *)v117 )
      {
        do
        {
          v118 = *((_QWORD *)v116 + 1);
          v119 = *v116;
          v116 += 4;
          sub_B99FD0((__int64)v115, v119, v118);
        }
        while ( v117 != v116 );
      }
      if ( !a9 )
      {
        v210[0] = (__int64)v182;
        v172 = sub_B9C770(v202, v210, (__int64 *)1, 0, 1);
        sub_B99FD0((__int64)v115, 7u, v172);
      }
      v211 = 257;
      v120 = sub_921130((unsigned int **)&v254, v201, a3, &v207, 1, (__int64)v210, 3u);
      v211 = 257;
      v122 = sub_BD2C40(80, unk_3F10A10);
      if ( v122 )
      {
        sub_B4D3C0((__int64)v122, (__int64)v115, v120, a8, v186, v121, 0, 0);
        v121 = v174;
      }
      (*((void (__fastcall **)(void **, _QWORD *, __int64 *, __int64, __int64, __int64))*v263 + 2))(
        v263,
        v122,
        v210,
        v259,
        v260,
        v121);
      v123 = 16LL * (unsigned int)v255;
      if ( v254 != &v254[v123] )
      {
        v204 = v107;
        v124 = (unsigned int *)&v254[v123];
        v125 = (unsigned int *)v254;
        do
        {
          v126 = *((_QWORD *)v125 + 1);
          v127 = *v125;
          v125 += 4;
          sub_B99FD0((__int64)v122, v127, v126);
        }
        while ( v124 != v125 );
        v107 = v204;
      }
      if ( !a9 )
      {
        v210[0] = (__int64)v182;
        v171 = sub_B9C770(v202, v210, (__int64 *)1, 0, 1);
        sub_B99FD0((__int64)v122, 8u, v171);
      }
      if ( BYTE4(a11) )
      {
        v128 = *((_WORD *)v115 + 1);
        *((_BYTE *)v115 + 72) = 1;
        v128 &= 0xFC7Fu;
        LOBYTE(v128) = v128 | 0x80;
        *((_WORD *)v115 + 1) = v128;
        v129 = *((_WORD *)v122 + 1);
        *((_BYTE *)v122 + 72) = 1;
        v129 &= 0xFC7Fu;
        LOBYTE(v129) = v129 | 0x80;
        *((_WORD *)v122 + 1) = v129;
      }
      v211 = 257;
      v130 = (_BYTE *)sub_AD64C0(v196, v189, 0);
      v131 = sub_929C50((unsigned int **)&v254, (_BYTE *)v107, v130, (__int64)v210, 0, 0);
      v132 = *(_DWORD *)(v107 + 4) & 0x7FFFFFF;
      if ( v132 == *(_DWORD *)(v107 + 72) )
      {
        v205 = v131;
        sub_B48D90(v107);
        v131 = v205;
        v132 = *(_DWORD *)(v107 + 4) & 0x7FFFFFF;
      }
      v133 = (v132 + 1) & 0x7FFFFFF;
      v134 = v133 | *(_DWORD *)(v107 + 4) & 0xF8000000;
      v135 = *(_QWORD *)(v107 - 8) + 32LL * (unsigned int)(v133 - 1);
      *(_DWORD *)(v107 + 4) = v134;
      if ( *(_QWORD *)v135 )
      {
        v136 = *(_QWORD *)(v135 + 8);
        **(_QWORD **)(v135 + 16) = v136;
        if ( v136 )
          *(_QWORD *)(v136 + 16) = *(_QWORD *)(v135 + 16);
      }
      *(_QWORD *)v135 = v131;
      if ( v131 )
      {
        v137 = *(_QWORD *)(v131 + 16);
        *(_QWORD *)(v135 + 8) = v137;
        if ( v137 )
          *(_QWORD *)(v137 + 16) = v135 + 8;
        *(_QWORD *)(v135 + 16) = v131 + 16;
        *(_QWORD *)(v131 + 16) = v135;
      }
      *(_QWORD *)(*(_QWORD *)(v107 - 8)
                + 32LL * *(unsigned int *)(v107 + 72)
                + 8LL * ((*(_DWORD *)(v107 + 4) & 0x7FFFFFFu) - 1)) = v195;
      v209 = 257;
      v138 = sub_92B530((unsigned int **)&v254, 0x24u, v131, v187, (__int64)v208);
      v211 = 257;
      v139 = v138;
      v140 = sub_BD2C40(72, 3u);
      v142 = (__int64)v140;
      if ( v140 )
        sub_B4C9A0((__int64)v140, v195, v197, v139, 3u, v141, 0, 0);
      (*((void (__fastcall **)(void **, __int64, __int64 *, __int64, __int64))*v263 + 2))(v263, v142, v210, v259, v260);
      v143 = (unsigned int *)v254;
      v144 = (unsigned int *)&v254[16 * (unsigned int)v255];
      if ( v254 != (char *)v144 )
      {
        do
        {
          v145 = *((_QWORD *)v143 + 1);
          v146 = *v143;
          v143 += 4;
          sub_B99FD0(v142, v146, v145);
        }
        while ( v144 != v143 );
      }
      nullsub_61();
      v270 = &unk_49DA100;
      nullsub_63();
      if ( v254 != v256 )
        _libc_free((unsigned __int64)v254);
      nullsub_61();
      v252 = &unk_49DA100;
      nullsub_63();
      if ( v236 != v238 )
        _libc_free((unsigned __int64)v236);
      goto LABEL_123;
    }
  }
  v149 = (_BYTE *)sub_ACD640(v178, 0, 0);
  v239 = 257;
  v150 = sub_92B530(&v212, 0x21u, a4, v149, (__int64)&v236);
  v257 = 257;
  v151 = sub_BD2C40(72, 3u);
  v153 = (__int64)v151;
  if ( v151 )
    sub_B4C9A0((__int64)v151, v198, v197, v150, 3u, v152, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, char **, __int64, __int64))(*(_QWORD *)v217 + 16LL))(
    v217,
    v153,
    &v254,
    v215,
    v216);
  v154 = v212;
  v155 = &v212[4 * v213];
  if ( v212 != v155 )
  {
    do
    {
      v156 = *((_QWORD *)v154 + 1);
      v157 = *v154;
      v154 += 4;
      sub_B99FD0(v153, v157, v156);
    }
    while ( v155 != v154 );
  }
  v158 = v199[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v177 == (_QWORD *)v158 )
  {
    v160 = 0;
  }
  else
  {
    if ( !v158 )
      BUG();
    v159 = *(unsigned __int8 *)(v158 - 24);
    v160 = 0;
    v161 = (_QWORD *)(v158 - 24);
    if ( (unsigned int)(v159 - 30) < 0xB )
      v160 = v161;
  }
  sub_B43D60(v160);
  v239 = 257;
  v162 = sub_92B530(&v219, 0x24u, v60, (_BYTE *)a4, (__int64)&v236);
  v257 = 257;
  v163 = v162;
  v164 = sub_BD2C40(72, 3u);
  v166 = (__int64)v164;
  if ( v164 )
    sub_B4C9A0((__int64)v164, v198, v197, v163, 3u, v165, 0, 0);
  (*((void (__fastcall **)(void **, __int64, char **, __int64, __int64))*v227 + 2))(v227, v166, &v254, v223, v224);
  v167 = v219;
  v168 = &v219[4 * (unsigned int)v220];
  if ( v219 != v168 )
  {
    do
    {
      v169 = *((_QWORD *)v167 + 1);
      v170 = *v167;
      v167 += 4;
      sub_B99FD0(v166, v170, v169);
    }
    while ( v168 != v167 );
  }
LABEL_123:
  nullsub_61();
  v234 = &unk_49DA100;
  nullsub_63();
  if ( v219 != (unsigned int *)v221 )
    _libc_free((unsigned __int64)v219);
  nullsub_61();
  v218 = &unk_49DA100;
  nullsub_63();
  if ( v212 != (unsigned int *)&v214 )
    _libc_free((unsigned __int64)v212);
}
