// Function: sub_12E54A0
// Address: 0x12e54a0
//
char *__fastcall sub_12E54A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  __int64 v6; // rax
  __int64 v7; // rax
  int *v8; // rax
  int v9; // eax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rax
  char **v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  char v22; // al
  unsigned __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // rbx
  __int64 v29; // rdx
  __int64 *v30; // rax
  __int64 v31; // rcx
  __int64 v32; // r13
  __int64 (__fastcall *v33)(char **, __int64, __int64, __int64); // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  int *v36; // rax
  int v37; // eax
  char v38; // al
  _DWORD *v39; // rdi
  char *result; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  int *v43; // rax
  int v44; // eax
  __int64 v45; // rbx
  __int64 v46; // r12
  __int64 v47; // rdi
  __int64 v48; // rax
  _QWORD *v49; // rbx
  _QWORD *v50; // r12
  _BYTE *v51; // rsi
  __int64 (*v52)(); // rax
  _QWORD *v53; // rbx
  _QWORD *v54; // rax
  int v55; // r13d
  _QWORD *v56; // r14
  _QWORD *v57; // r15
  bool v58; // zf
  __int64 v59; // rdx
  __m128i *v60; // rax
  __m128i si128; // xmm0
  _DWORD *v62; // r12
  _DWORD *v63; // rax
  _DWORD *v64; // rbx
  __int64 v65; // rdi
  __int64 v66; // rax
  char v67; // al
  char v68; // al
  __int64 v69; // rax
  char v70; // al
  __int64 v71; // rax
  char *v72; // rdi
  char *v73; // rsi
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // rax
  __int64 v79; // rsi
  __int64 v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rcx
  __int64 v83; // r8
  __int64 v84; // r9
  char *v85; // rsi
  char *v86; // rdi
  __int64 v87; // rsi
  __int64 v88; // rax
  __int64 v89; // rax
  char *v90; // rdi
  char *v91; // rsi
  __int64 v92; // rdx
  __int64 v93; // rsi
  __int64 v94; // rdx
  __int64 v95; // rcx
  __int64 v96; // r8
  __int64 v97; // r9
  __int64 v98; // rax
  __int64 v99; // rsi
  __int64 v100; // rdx
  __int64 v101; // rax
  __int64 v102; // rsi
  __int64 v103; // rax
  __int64 v104; // rsi
  __int64 v105; // rdx
  __int64 v106; // rsi
  __int64 v107; // rdx
  __int64 v108; // rsi
  __int64 v109; // rdx
  char *v110; // rdi
  char *v111; // rsi
  __int64 v112; // rdx
  __int64 v113; // rcx
  __int64 v114; // r8
  __int64 v115; // r9
  __int64 v116; // rax
  _DWORD *v117; // rax
  __int64 v118; // rax
  __int64 v119; // rsi
  __int64 v120; // rdx
  __int64 v121; // rcx
  __int64 v122; // r8
  __int64 v123; // r9
  __int64 v124; // rax
  char *v125; // rdi
  char *v126; // rsi
  __int64 v127; // rdx
  __int64 v128; // rcx
  __int64 v129; // rsi
  __int64 v130; // rax
  __int64 v131; // rdx
  char *v132; // rsi
  char *v133; // rdi
  __int64 v134; // rsi
  __int64 v135; // rax
  __int64 v136; // rax
  __int64 v137; // rax
  __int64 v138; // rax
  __int64 v139; // rax
  __int64 v140; // rax
  __int64 v141; // rax
  __int64 v142; // rax
  __int64 v143; // rax
  __int64 v144; // rax
  __int64 v145; // rax
  __int64 v146; // rax
  __int64 v147; // rax
  __int64 v148; // rax
  __int64 v149; // rax
  __int64 v150; // rax
  __int64 v151; // rax
  __int64 v152; // rax
  __int64 v153; // rax
  __int64 v154; // rax
  __int64 v155; // rsi
  __int64 v156; // rax
  __int64 v157; // rax
  __int64 v158; // rax
  __int64 v159; // rax
  __int64 v160; // rax
  __int64 v161; // rax
  __int64 v162; // rax
  __int64 v163; // rsi
  __int64 v164; // rsi
  __int64 v165; // rdx
  __int64 v166; // rax
  __int64 v167; // rax
  __int64 v168; // rax
  __int64 v169; // rsi
  __int64 v170; // rdx
  __int64 v171; // rcx
  __int64 v172; // r8
  __int64 v173; // r9
  __int64 v174; // rax
  char *v175; // rdi
  char *v176; // rsi
  __int64 v177; // rdx
  __int64 v178; // rcx
  __int64 v179; // r8
  __int64 v180; // r9
  __int64 v181; // rax
  __int64 v182; // rsi
  __int64 v183; // rax
  __int64 v184; // rsi
  __int64 v185; // rdx
  __int64 v186; // rsi
  __int64 v187; // rdx
  __int64 v188; // rcx
  __int64 v189; // r8
  __int64 v190; // r9
  char *v191; // rdi
  char *v192; // rsi
  __int64 v193; // rdx
  __int64 v194; // rcx
  __int64 v195; // r8
  __int64 v196; // r9
  __int64 v197; // rax
  __int64 v198; // rax
  __int64 v199; // rax
  __int64 v200; // rax
  __int64 v201; // rax
  __int64 v202; // rsi
  __int64 v203; // rax
  __int64 v204; // rax
  __int64 v205; // rsi
  __int64 v206; // rax
  __int64 v207; // rax
  __int64 v208; // rax
  __int64 v209; // rax
  __int64 v210; // rax
  __int64 v211; // rsi
  __int64 v212; // rsi
  __int64 v213; // rsi
  __int64 v214; // rax
  __int64 v215; // rax
  __int64 v216; // rsi
  __int64 v217; // rsi
  __int64 v218; // rsi
  __int64 v219; // rax
  __int64 v220; // rax
  __int64 v221; // rax
  __int64 v222; // rax
  __int64 v223; // [rsp-10h] [rbp-7F0h]
  char *v224; // [rsp-10h] [rbp-7F0h]
  __int64 v225; // [rsp-10h] [rbp-7F0h]
  __int64 v226; // [rsp-10h] [rbp-7F0h]
  __int64 v227; // [rsp-10h] [rbp-7F0h]
  char *v228; // [rsp-10h] [rbp-7F0h]
  __int64 v229; // [rsp-10h] [rbp-7F0h]
  __int64 v230; // [rsp-8h] [rbp-7E8h]
  char *v231; // [rsp-8h] [rbp-7E8h]
  __int64 v232; // [rsp-8h] [rbp-7E8h]
  __int64 v233; // [rsp-8h] [rbp-7E8h]
  char *v234; // [rsp-8h] [rbp-7E8h]
  __int64 v235; // [rsp-8h] [rbp-7E8h]
  _QWORD *v236; // [rsp+8h] [rbp-7D8h]
  __int64 v237; // [rsp+18h] [rbp-7C8h]
  bool v238; // [rsp+20h] [rbp-7C0h]
  __int64 v239; // [rsp+20h] [rbp-7C0h]
  unsigned int v240; // [rsp+38h] [rbp-7A8h]
  unsigned int v241; // [rsp+38h] [rbp-7A8h]
  unsigned int v242; // [rsp+38h] [rbp-7A8h]
  unsigned __int8 v243; // [rsp+38h] [rbp-7A8h]
  char v244; // [rsp+40h] [rbp-7A0h]
  __int64 v245; // [rsp+48h] [rbp-798h]
  __int64 v247; // [rsp+58h] [rbp-788h]
  int v248; // [rsp+58h] [rbp-788h]
  __int64 (__fastcall *v249)(__int64, char *, __int64, __int64, _QWORD *, __int64, _QWORD *, _QWORD *, _QWORD *, __int64, _QWORD); // [rsp+58h] [rbp-788h]
  _QWORD *v251; // [rsp+60h] [rbp-780h]
  _BYTE v253[32]; // [rsp+70h] [rbp-770h] BYREF
  _QWORD *v254; // [rsp+90h] [rbp-750h] BYREF
  __int64 v255; // [rsp+98h] [rbp-748h]
  _QWORD v256[2]; // [rsp+A0h] [rbp-740h] BYREF
  _QWORD v257[2]; // [rsp+B0h] [rbp-730h] BYREF
  _QWORD v258[2]; // [rsp+C0h] [rbp-720h] BYREF
  _QWORD *v259; // [rsp+D0h] [rbp-710h]
  __int64 v260; // [rsp+D8h] [rbp-708h]
  _QWORD v261[2]; // [rsp+E0h] [rbp-700h] BYREF
  _QWORD v262[4]; // [rsp+F0h] [rbp-6F0h] BYREF
  int v263; // [rsp+110h] [rbp-6D0h]
  char *v264; // [rsp+118h] [rbp-6C8h]
  _QWORD v265[8]; // [rsp+120h] [rbp-6C0h] BYREF
  _QWORD *v266; // [rsp+160h] [rbp-680h] BYREF
  _QWORD v267[2]; // [rsp+170h] [rbp-670h] BYREF
  int v268; // [rsp+180h] [rbp-660h]
  int v269; // [rsp+18Ch] [rbp-654h]
  char *v270; // [rsp+1A0h] [rbp-640h] BYREF
  __int64 v271; // [rsp+1A8h] [rbp-638h]
  _WORD v272[32]; // [rsp+1B0h] [rbp-630h] BYREF
  __int64 v273; // [rsp+1F0h] [rbp-5F0h]
  __int64 v274; // [rsp+1F8h] [rbp-5E8h]
  __int64 v275; // [rsp+200h] [rbp-5E0h]
  int v276; // [rsp+208h] [rbp-5D8h]
  _QWORD v277[6]; // [rsp+210h] [rbp-5D0h] BYREF
  char v278[8]; // [rsp+240h] [rbp-5A0h] BYREF
  __int64 *v279; // [rsp+248h] [rbp-598h]
  __int64 v280; // [rsp+258h] [rbp-588h] BYREF
  __int64 *v281; // [rsp+268h] [rbp-578h]
  __int64 v282; // [rsp+278h] [rbp-568h] BYREF
  _QWORD *v283; // [rsp+288h] [rbp-558h]
  _QWORD *v284; // [rsp+290h] [rbp-550h]
  __int64 v285; // [rsp+298h] [rbp-548h]
  _QWORD v286[2]; // [rsp+2A0h] [rbp-540h] BYREF
  __int64 v287; // [rsp+2B0h] [rbp-530h]
  __int64 v288; // [rsp+2B8h] [rbp-528h]
  int v289; // [rsp+2C0h] [rbp-520h]
  __int16 v290; // [rsp+2C4h] [rbp-51Ch]
  __int64 v291; // [rsp+318h] [rbp-4C8h]
  unsigned int v292; // [rsp+328h] [rbp-4B8h]
  __int64 v293; // [rsp+338h] [rbp-4A8h]
  __int64 v294; // [rsp+348h] [rbp-498h]
  __int64 v295; // [rsp+350h] [rbp-490h]
  __int64 v296; // [rsp+360h] [rbp-480h]
  char s[16]; // [rsp+370h] [rbp-470h] BYREF
  _OWORD v298[65]; // [rsp+380h] [rbp-460h] BYREF
  _DWORD *v299; // [rsp+790h] [rbp-50h]
  int v300; // [rsp+798h] [rbp-48h]
  int v301; // [rsp+7A0h] [rbp-40h]

  if ( !*(_BYTE *)(a4 + 4384) )
  {
    LOWORD(v298[0]) = 260;
    *(_QWORD *)s = a1 + 30;
    sub_16E1010(&v266);
    v255 = 0;
    v254 = v256;
    LOBYTE(v256[0]) = 0;
    v6 = sub_1632FA0(a1);
    if ( 8 * (unsigned int)sub_15A9520(v6, 0) == 64 )
      sub_2241130(&v254, 0, v255, "nvptx64", 7);
    else
      sub_2241130(&v254, 0, v255, "nvptx", 5);
    v257[1] = 0;
    v257[0] = v258;
    LOBYTE(v258[0]) = 0;
    v7 = sub_16D3AC0(&v254, v257);
    if ( !v7 )
    {
      v286[0] = 30;
      *(_QWORD *)s = v298;
      v60 = (__m128i *)sub_22409D0(s, v286, 0);
      si128 = _mm_load_si128((const __m128i *)&xmmword_4281B20);
      *(_QWORD *)s = v60;
      *(_QWORD *)&v298[0] = v286[0];
      qmemcpy(&v60[1], " nvptx target\n", 14);
      *v60 = si128;
      *(_QWORD *)&s[8] = v286[0];
      *(_BYTE *)(*(_QWORD *)s + v286[0]) = 0;
      sub_1C3EFD0(s, 1);
      if ( *(_OWORD **)s != v298 )
        j_j___libc_free_0(*(_QWORD *)s, *(_QWORD *)&v298[0] + 1LL);
LABEL_121:
      if ( (_QWORD *)v257[0] != v258 )
        j_j___libc_free_0(v257[0], v258[0] + 1LL);
      if ( v254 != v256 )
        j_j___libc_free_0(v254, v256[0] + 1LL);
      result = (char *)v267;
      if ( v266 != v267 )
        return (char *)j_j___libc_free_0(v266, v267[0] + 1LL);
      return result;
    }
    v247 = v7;
    v259 = v261;
    v260 = 0;
    LOBYTE(v261[0]) = 0;
    v277[0] = 0;
    v277[1] = 1;
    v277[2] = 8;
    v277[3] = 1;
    v277[4] = 1;
    v277[5] = 0;
    sub_167F890(v278);
    if ( v268
      && (v239 = v247,
          v59 = a1[31],
          v236 = v259,
          v237 = v260,
          v270 = (char *)a1[30],
          v271 = v59,
          (v249 = *(__int64 (__fastcall **)(__int64, char *, __int64, __int64, _QWORD *, __int64, _QWORD *, _QWORD *, _QWORD *, __int64, _QWORD))(v247 + 88)) != 0) )
    {
      v265[0] = 0x100000000LL;
      LOWORD(v287) = 261;
      BYTE4(v262[0]) = 0;
      v286[0] = &v270;
      sub_16E1010(s);
      v245 = v249(v239, s, a2, a3, v236, v237, v277, v262, v265, 3, 0);
      if ( *(_OWORD **)s != v298 )
        j_j___libc_free_0(*(_QWORD *)s, *(_QWORD *)&v298[0] + 1LL);
    }
    else
    {
      v245 = 0;
    }
    v272[0] = 260;
    v270 = (char *)(a1 + 30);
    sub_16E1010(s);
    sub_14A04B0(v286, s);
    if ( *(_OWORD **)s != v298 )
      j_j___libc_free_0(*(_QWORD *)s, *(_QWORD *)&v298[0] + 1LL);
    sub_149CBC0(v286);
    sub_1BFB9A0(v265, a2, a3, v269 == 23);
    v8 = (int *)sub_16D40F0(qword_4FBB430);
    if ( v8 )
      v9 = *v8;
    else
      v9 = qword_4FBB430[2];
    HIDWORD(v265[0]) = v9;
    sub_1611EE0(v253);
    v270 = (char *)v272;
    v271 = 0x800000000LL;
    v273 = 0;
    v274 = 0;
    v275 = 0;
    v276 = 0;
    if ( v245 )
      sub_1700880(s);
    else
      sub_14A3CD0(s);
    v10 = sub_14A4230(s);
    sub_1619140(v253, v10, 0);
    if ( *(_QWORD *)&v298[0] )
      (*(void (__fastcall **)(char *, char *, __int64))&v298[0])(s, s, 3);
    v11 = sub_22077B0(368);
    v12 = v11;
    if ( v11 )
      sub_149CCE0(v11, v286);
    sub_12DE0B0((__int64)&v270, v12, 0, 0);
    v13 = sub_22077B0(208);
    v14 = v13;
    if ( v13 )
      sub_1BFB520(v13, v265);
    sub_12DE0B0((__int64)&v270, v14, 1u, 0);
    v15 = sub_14A7550();
    sub_12DE0B0((__int64)&v270, v15, 0, 0);
    v16 = sub_1361950();
    sub_12DE0B0((__int64)&v270, v16, 0, 0);
    v17 = &v270;
    v18 = sub_1CB0F50();
    sub_12DE0B0((__int64)&v270, v18, 1u, 0);
    v22 = *(_BYTE *)(a4 + 4224);
    v244 = v22;
    if ( v22 || *(_BYTE *)(a4 + 3528) || *(_BYTE *)(a4 + 3568) || (v19 = *(unsigned __int8 *)(a4 + 3608), v238 = v19) )
    {
      v23 = *(_QWORD *)(a4 + 3648);
      if ( *(_QWORD *)(a4 + 3656) != 3 )
      {
        v19 = *(unsigned __int8 *)(a4 + 4304);
        v238 = v19;
        if ( !(_BYTE)v19 )
        {
          v244 = 1;
          goto LABEL_24;
        }
        v20 = (unsigned __int8)v19;
        if ( v22 )
        {
          v238 = 0;
          goto LABEL_24;
        }
        goto LABEL_188;
      }
      if ( *(_WORD *)v23 == 24941 && *(_BYTE *)(v23 + 2) == 120 )
      {
        v19 = *(unsigned __int8 *)(a4 + 4304);
        v238 = v19;
        if ( !(_BYTE)v19 )
        {
          v244 = 1;
          v238 = 1;
          goto LABEL_191;
        }
        v20 = 0;
        goto LABEL_196;
      }
      if ( *(_WORD *)v23 == 26989 && *(_BYTE *)(v23 + 2) == 100 )
      {
        v238 = 1;
        goto LABEL_297;
      }
      v19 = 1;
    }
    else
    {
      if ( *(_QWORD *)(a4 + 3656) != 3 )
      {
LABEL_159:
        v17 = &v270;
        v18 = sub_1C8A4D0(0);
        sub_12DE0B0((__int64)&v270, v18, 0, 0);
        LOBYTE(v20) = *(_BYTE *)(a4 + 4224);
        v244 = *(_BYTE *)(a4 + 4304);
        if ( !v244 )
          goto LABEL_38;
        if ( (_BYTE)v20 )
        {
          v244 = 0;
          v27 = *(_QWORD *)(a4 + 4488);
          if ( *(_QWORD *)(a4 + 4496) == v27 )
            goto LABEL_162;
LABEL_39:
          v28 = 0;
          do
          {
            v29 = 16 * v28;
            v30 = (__int64 *)(16 * v28 + v27);
            v31 = *((unsigned int *)v30 + 2);
            v32 = *v30;
            if ( (_BYTE)v20 && (unsigned int)v31 > *(_DWORD *)(a4 + 4228) )
            {
              v18 = a4;
              v17 = &v270;
              v240 = *((_DWORD *)v30 + 2);
              sub_12DE330(&v270, (_BYTE *)a4);
              *(_BYTE *)(a4 + 4224) = 0;
              v31 = v240;
            }
            if ( *(_BYTE *)(a4 + 3528) && (unsigned int)v31 > *(_DWORD *)(a4 + 3532) )
            {
              v18 = 1;
              v17 = &v270;
              v241 = v31;
              sub_12DE8F0((__int64)&v270, 1, a4);
              *(_BYTE *)(a4 + 3528) = 0;
              v31 = v241;
            }
            if ( *(_BYTE *)(a4 + 3568) && (unsigned int)v31 > *(_DWORD *)(a4 + 3572) )
            {
              v18 = 2;
              v17 = &v270;
              v242 = v31;
              sub_12DE8F0((__int64)&v270, 2, a4);
              *(_BYTE *)(a4 + 3568) = 0;
              v31 = v242;
            }
            if ( *(_BYTE *)(a4 + 3608) && (unsigned int)v31 > *(_DWORD *)(a4 + 3612) )
            {
              v18 = 3;
              v17 = &v270;
              sub_12DE8F0((__int64)&v270, 3, a4);
              *(_BYTE *)(a4 + 3608) = 0;
            }
            v33 = *(__int64 (__fastcall **)(char **, __int64, __int64, __int64))(v32 + 72);
            if ( v33 )
            {
              v18 = v33(v17, v18, v29, v31);
              if ( v18 )
              {
                v17 = &v270;
                sub_12DE0B0((__int64)&v270, v18, 1u, 0);
              }
            }
            if ( *(_BYTE *)(a4 + 3904) )
            {
              *(_QWORD *)s = v298;
              sub_12D3E60((__int64 *)s, byte_3F871B3, (__int64)byte_3F871B3);
              v35 = sub_16E8CB0(s, byte_3F871B3, v34);
              v18 = sub_15E9F00(v35, s, 0);
              sub_12DE0B0((__int64)&v270, v18, 0, 0);
              v17 = *(char ***)s;
              if ( *(_OWORD **)s != v298 )
              {
                v18 = *(_QWORD *)&v298[0] + 1LL;
                j_j___libc_free_0(*(_QWORD *)s, *(_QWORD *)&v298[0] + 1LL);
              }
            }
            v27 = *(_QWORD *)(a4 + 4488);
            ++v28;
            LOBYTE(v20) = *(_BYTE *)(a4 + 4224);
            v23 = (*(_QWORD *)(a4 + 4496) - v27) >> 4;
          }
          while ( v23 > v28 );
LABEL_78:
          if ( !(_BYTE)v20 )
          {
            if ( *(_BYTE *)(a4 + 3528) )
            {
LABEL_163:
              sub_12DE8F0((__int64)&v270, 1, a4);
              goto LABEL_84;
            }
LABEL_80:
            if ( *(_BYTE *)(a4 + 3568) )
            {
              sub_12DE8F0((__int64)&v270, 2, a4);
              goto LABEL_84;
            }
            if ( *(_BYTE *)(a4 + 3608) )
            {
              sub_12DE8F0((__int64)&v270, 3, a4);
              goto LABEL_84;
            }
            if ( !v238 )
            {
              if ( *(_BYTE *)(a4 + 2680) )
                goto LABEL_84;
              goto LABEL_294;
            }
            v71 = *(_QWORD *)(a4 + 3648);
            if ( *(_WORD *)v71 == 24941 && *(_BYTE *)(v71 + 2) == 120 )
            {
              v118 = ((__int64 (*)(void))sub_1CEF8F0)();
              sub_12DE0B0((__int64)&v270, v118, 1u, 0);
              v119 = sub_215D9D0();
              sub_12DE0B0((__int64)&v270, v119, 1u, 0);
              if ( !*(_BYTE *)(a4 + 880) )
              {
                v124 = sub_1857160(&v270, v119, v120, v121, v122, v123);
                sub_12DE0B0((__int64)&v270, v124, 1u, 0);
              }
              *(_QWORD *)&v298[0] = 0;
              v125 = (char *)&v270;
              v126 = (char *)sub_1A62BF0(1, 0, 0, 1, 0, 0, 1, (__int64)s);
              sub_12DE0B0((__int64)&v270, (__int64)v126, 0, 0);
              if ( *(_QWORD *)&v298[0] )
              {
                v126 = s;
                v125 = s;
                (*(void (__fastcall **)(char *, char *, __int64, __int64, __int64, __int64))&v298[0])(
                  s,
                  s,
                  3,
                  v128,
                  v227,
                  v233);
              }
              if ( !*(_BYTE *)(a4 + 2040) )
              {
                v221 = sub_1B26330(v125, v126, v127);
                sub_12DE0B0((__int64)&v270, v221, 1u, 0);
              }
              v129 = sub_17060B0(0, 0);
              sub_12DE0B0((__int64)&v270, v129, 0, 0);
              if ( !*(_BYTE *)(a4 + 280) )
              {
                v220 = sub_18DEFF0();
                sub_12DE0B0((__int64)&v270, v220, 0, 0);
              }
              *(_QWORD *)&v298[0] = 0;
              v130 = sub_1A62BF0(1, 0, 0, 1, 0, 0, 1, (__int64)s);
              sub_12DE0B0((__int64)&v270, v130, 0, 0);
              v132 = v228;
              v133 = v234;
              if ( *(_QWORD *)&v298[0] )
              {
                v132 = s;
                v133 = s;
                (*(void (__fastcall **)(char *, char *, __int64))&v298[0])(s, s, 3);
              }
              if ( !*(_BYTE *)(a4 + 2640) )
              {
                v222 = sub_18B1DE0(v133, v132, v131);
                sub_12DE0B0((__int64)&v270, v222, 1u, 0);
              }
              if ( *(_BYTE *)(a4 + 1760) )
                goto LABEL_84;
LABEL_294:
              v134 = sub_1C8E680(0);
              sub_12DE0B0((__int64)&v270, v134, 1u, 0);
LABEL_84:
              if ( !v244 && *(_BYTE *)(a4 + 3488) )
              {
                v41 = sub_1C98160(*(_BYTE *)(a4 + 2920) != 0);
                sub_12DE0B0((__int64)&v270, v41, 1u, 0);
              }
              v42 = sub_1CEBD10();
              sub_12DE0B0((__int64)&v270, v42, 1u, 0);
              if ( !*(_BYTE *)(a4 + 2800) && !*(_BYTE *)(a4 + 4464) )
              {
                v66 = sub_1654860(1);
                sub_12DE0B0((__int64)&v270, v66, 1u, 0);
              }
              sub_12DFE00((__int64)&v270, (__int64)v253, a4);
              v43 = (int *)sub_16D40F0(qword_4FBB3B0);
              if ( v43 )
                v44 = *v43;
              else
                v44 = qword_4FBB3B0[2];
              if ( v44 == 2 && (**(_BYTE **)(a4 + 4480) & 4) != 0 )
              {
                v51 = 0;
                v263 = 1;
                *(_QWORD *)s = v298;
                *(_QWORD *)&s[8] = 0x10000000000LL;
                v264 = s;
                memset(&v262[1], 0, 24);
                v262[0] = &unk_49EFC48;
                sub_16E7A40(v262, 0, 0, 0);
                if ( !BYTE4(qword_4FBB370[2]) )
                {
                  v117 = (_DWORD *)sub_1C42D70(4, 4);
                  *v117 = 6;
                  v51 = v117;
                  sub_16D40E0(qword_4FBB370, v117);
                }
                v52 = *(__int64 (**)())(*(_QWORD *)v245 + 56LL);
                if ( v52 != sub_12D3B70 )
                {
                  v51 = v253;
                  ((void (__fastcall *)(__int64, _BYTE *, _QWORD *, _QWORD, _QWORD, __int64, _QWORD))v52)(
                    v245,
                    v253,
                    v262,
                    0,
                    0,
                    1,
                    0);
                }
                v262[0] = &unk_49EFD28;
                sub_16E7960(v262);
                if ( *(_OWORD **)s != v298 )
                  _libc_free(*(_QWORD *)s, v51);
              }
              sub_160FB70(v253, *a5, a5[1]);
              sub_1619BD0(v253, a1);
              j___libc_free_0(v274);
              if ( v270 != (char *)v272 )
                _libc_free(v270, a1);
              sub_160FE50(v253);
              if ( *(_BYTE *)(a4 + 3944) )
              {
                v251 = (_QWORD *)a1[4];
                if ( v251 != a1 + 3 )
                {
                  v248 = 0;
                  while ( 1 )
                  {
                    v53 = v251 - 7;
                    if ( !v251 )
                      v53 = 0;
                    if ( !(unsigned __int8)sub_15E4F60(v53) )
                    {
                      v54 = (_QWORD *)v53[10];
                      ++v248;
                      if ( v54 != v53 + 9 )
                        break;
                    }
LABEL_136:
                    v251 = (_QWORD *)v251[1];
                    if ( a1 + 3 == v251 )
                      goto LABEL_96;
                  }
                  v55 = 0;
                  v56 = (_QWORD *)v53[10];
                  while ( 1 )
                  {
                    if ( v56 )
                    {
                      v57 = v56 - 3;
                      if ( v54 && v57 == v54 - 3 )
                        goto LABEL_147;
                    }
                    else
                    {
                      v57 = 0;
                      if ( !v54 )
                        goto LABEL_147;
                    }
                    ++v55;
                    *(_OWORD *)s = 0;
                    memset(v298, 0, 64);
                    sprintf(s, "F%d_B%d", v248, v55);
                    v272[0] = 257;
                    if ( s[0] )
                    {
                      v270 = s;
                      LOBYTE(v272[0]) = 3;
                    }
                    sub_164B780(v57, &v270);
LABEL_147:
                    v56 = (_QWORD *)v56[1];
                    if ( v53 + 9 == v56 )
                      goto LABEL_136;
                    v54 = (_QWORD *)v53[10];
                  }
                }
              }
LABEL_96:
              if ( v295 )
                j_j___libc_free_0(v295, v296 - v295);
              if ( v293 )
                j_j___libc_free_0(v293, v294 - v293);
              if ( v292 )
              {
                v45 = v291;
                v46 = v291 + 40LL * v292;
                do
                {
                  while ( 1 )
                  {
                    if ( *(_DWORD *)v45 <= 0xFFFFFFFD )
                    {
                      v47 = *(_QWORD *)(v45 + 8);
                      if ( v47 != v45 + 24 )
                        break;
                    }
                    v45 += 40;
                    if ( v46 == v45 )
                      goto LABEL_106;
                  }
                  v48 = *(_QWORD *)(v45 + 24);
                  v45 += 40;
                  j_j___libc_free_0(v47, v48 + 1);
                }
                while ( v46 != v45 );
              }
LABEL_106:
              j___libc_free_0(v291);
              if ( v245 )
                (*(void (__fastcall **)(__int64))(*(_QWORD *)v245 + 8LL))(v245);
              v49 = v284;
              v50 = v283;
              if ( v284 != v283 )
              {
                do
                {
                  if ( (_QWORD *)*v50 != v50 + 2 )
                    j_j___libc_free_0(*v50, v50[2] + 1LL);
                  v50 += 4;
                }
                while ( v49 != v50 );
                v50 = v283;
              }
              if ( v50 )
                j_j___libc_free_0(v50, v285 - (_QWORD)v50);
              if ( v281 != &v282 )
                j_j___libc_free_0(v281, v282 + 1);
              if ( v279 != &v280 )
                j_j___libc_free_0(v279, v280 + 1);
              if ( v259 != v261 )
                j_j___libc_free_0(v259, v261[0] + 1LL);
              goto LABEL_121;
            }
            if ( *(_WORD *)v71 == 26989 && *(_BYTE *)(v71 + 2) == 100 )
            {
              if ( !*(_BYTE *)(a4 + 1960) )
              {
                v17 = &v270;
                v18 = ((__int64 (*)(void))sub_184CD60)();
                sub_12DE0B0((__int64)&v270, v18, 0, 0);
              }
              if ( !*(_BYTE *)(a4 + 2000) )
              {
                v17 = &v270;
                v18 = sub_1CB4E40(0);
                sub_12DE0B0((__int64)&v270, v18, 0, 0);
              }
              if ( !*(_BYTE *)(a4 + 2040) )
              {
                v219 = sub_1B26330(v17, v18, v23);
                sub_12DE0B0((__int64)&v270, v219, 0, 0);
              }
              v161 = sub_198E2A0();
              sub_12DE0B0((__int64)&v270, v161, 0, 0);
              v162 = ((__int64 (*)(void))sub_1CEF8F0)();
              sub_12DE0B0((__int64)&v270, v162, 0, 0);
              v163 = sub_215D9D0();
              sub_12DE0B0((__int64)&v270, v163, 0, 0);
              if ( *(_BYTE *)(a4 + 1080) )
              {
                if ( *(_BYTE *)(a4 + 1520) )
                  goto LABEL_315;
              }
              else
              {
                v216 = sub_17060B0(1, 0);
                sub_12DE0B0((__int64)&v270, v216, 0, 0);
                if ( *(_BYTE *)(a4 + 1520) )
                  goto LABEL_381;
              }
              v217 = sub_198DF00(0xFFFFFFFFLL);
              sub_12DE0B0((__int64)&v270, v217, 0, 0);
LABEL_381:
              if ( !*(_BYTE *)(a4 + 1080) )
              {
                v218 = sub_17060B0(1, 0);
                sub_12DE0B0((__int64)&v270, v218, 0, 0);
              }
LABEL_315:
              v164 = ((__int64 (*)(void))sub_1C6E800)();
              sub_12DE0B0((__int64)&v270, v164, 0, 0);
              if ( !*(_BYTE *)(a4 + 2600) )
              {
                v215 = sub_1A223D0(&v270, v164, v165);
                sub_12DE0B0((__int64)&v270, v215, 0, 0);
              }
              v166 = sub_190BB10(0, 0);
              sub_12DE0B0((__int64)&v270, v166, 0, 0);
              v167 = sub_1832270(1);
              sub_12DE0B0((__int64)&v270, v167, 0, 0);
              *(_QWORD *)&v298[0] = 0;
              v168 = sub_1A62BF0(5, 0, 0, 1, 0, 0, 1, (__int64)s);
              sub_12DE0B0((__int64)&v270, v168, 0, 0);
              if ( *(_QWORD *)&v298[0] )
                (*(void (__fastcall **)(char *, char *, __int64))&v298[0])(s, s, 3);
              if ( !*(_BYTE *)(a4 + 2000) )
              {
                v214 = sub_1CB4E40(0);
                sub_12DE0B0((__int64)&v270, v214, 0, 0);
              }
              v169 = sub_18FD350(0);
              sub_12DE0B0((__int64)&v270, v169, 0, 0);
              if ( !*(_BYTE *)(a4 + 680) )
              {
                v213 = sub_1841180(&v270, v169, v170, v171, v172, v173);
                sub_12DE0B0((__int64)&v270, v213, 0, 0);
              }
              if ( !*(_BYTE *)(a4 + 280) )
              {
                v212 = sub_18DEFF0();
                sub_12DE0B0((__int64)&v270, v212, 0, 0);
              }
              if ( !*(_BYTE *)(a4 + 1080) )
              {
                v211 = sub_17060B0(1, 0);
                sub_12DE0B0((__int64)&v270, v211, 0, 0);
              }
              if ( !*(_BYTE *)(a4 + 1960) )
              {
                v210 = ((__int64 (*)(void))sub_184CD60)();
                sub_12DE0B0((__int64)&v270, v210, 0, 0);
              }
              if ( !*(_BYTE *)(a4 + 1240) )
              {
                v209 = sub_195E880(0);
                sub_12DE0B0((__int64)&v270, v209, 0, 0);
              }
              v174 = sub_1C98160(0);
              sub_12DE0B0((__int64)&v270, v174, 0, 0);
              if ( !*(_BYTE *)(a4 + 1760) )
              {
                v208 = sub_1C8E680(0);
                sub_12DE0B0((__int64)&v270, v208, 0, 0);
              }
              if ( !*(_BYTE *)(a4 + 1280) )
              {
                v207 = sub_1B7FDF0(3);
                sub_12DE0B0((__int64)&v270, v207, 0, 0);
              }
              *(_QWORD *)&v298[0] = 0;
              v175 = (char *)&v270;
              v176 = (char *)sub_1A62BF0(8, 0, 0, 1, 1, 0, 1, (__int64)s);
              sub_12DE0B0((__int64)&v270, (__int64)v176, 0, 0);
              if ( *(_QWORD *)&v298[0] )
              {
                v176 = s;
                v175 = s;
                (*(void (__fastcall **)(char *, char *, __int64))&v298[0])(s, s, 3);
              }
              if ( !*(_BYTE *)(a4 + 880) )
              {
                v206 = sub_1857160(v175, v176, v177, v178, v179, v180);
                sub_12DE0B0((__int64)&v270, v206, 0, 0);
              }
              if ( !*(_BYTE *)(a4 + 1840) )
              {
                v205 = sub_1C6FCA0();
                sub_12DE0B0((__int64)&v270, v205, 0, 0);
              }
              if ( !*(_BYTE *)(a4 + 2720) )
              {
                v204 = sub_1A7A9F0();
                sub_12DE0B0((__int64)&v270, v204, 0, 0);
              }
              v181 = sub_18FD350(0);
              sub_12DE0B0((__int64)&v270, v181, 0, 0);
              if ( !*(_BYTE *)(a4 + 320) )
              {
                v203 = sub_1833EB0(3);
                sub_12DE0B0((__int64)&v270, v203, 0, 0);
              }
              v182 = sub_18FD350(0);
              sub_12DE0B0((__int64)&v270, v182, 0, 0);
              if ( !*(_BYTE *)(a4 + 1080) )
              {
                v202 = sub_17060B0(1, 0);
                sub_12DE0B0((__int64)&v270, v202, 0, 0);
              }
              v183 = sub_18EEA90();
              sub_12DE0B0((__int64)&v270, v183, 0, 0);
              v184 = sub_1869C50(1, 0, 1);
              sub_12DE0B0((__int64)&v270, v184, 0, 0);
              if ( !*(_BYTE *)(a4 + 1080) )
              {
                v184 = sub_17060B0(1, 0);
                sub_12DE0B0((__int64)&v270, v184, 0, 0);
              }
              if ( !*(_BYTE *)(a4 + 960) )
              {
                v184 = sub_190BB10(0, 0);
                sub_12DE0B0((__int64)&v270, v184, 0, 0);
              }
              if ( !*(_BYTE *)(a4 + 760) )
              {
                v184 = sub_18F5480(&v270, v184);
                sub_12DE0B0((__int64)&v270, v184, 0, 0);
              }
              if ( !*(_BYTE *)(a4 + 1080) )
              {
                v184 = sub_17060B0(1, 0);
                sub_12DE0B0((__int64)&v270, v184, 0, 0);
              }
              if ( !*(_BYTE *)(a4 + 2440) )
              {
                v184 = sub_1CC60B0(&v270, v184, v185);
                sub_12DE0B0((__int64)&v270, v184, 0, 0);
              }
              if ( !*(_BYTE *)(a4 + 2600) )
              {
                v201 = sub_1A223D0(&v270, v184, v185);
                sub_12DE0B0((__int64)&v270, v201, 0, 0);
              }
              v186 = sub_1C8A4D0(0);
              sub_12DE0B0((__int64)&v270, v186, 0, 0);
              if ( !*(_BYTE *)(a4 + 880) )
              {
                v198 = sub_1857160(&v270, v186, v187, v188, v189, v190);
                sub_12DE0B0((__int64)&v270, v198, 0, 0);
              }
              *(_QWORD *)&v298[0] = 0;
              v191 = (char *)&v270;
              v192 = (char *)sub_1A62BF0(8, 0, 0, 1, 1, 0, 1, (__int64)s);
              sub_12DE0B0((__int64)&v270, (__int64)v192, 0, 0);
              v195 = v229;
              v196 = v235;
              if ( *(_QWORD *)&v298[0] )
              {
                v192 = s;
                v191 = s;
                (*(void (__fastcall **)(char *, char *, __int64, __int64, __int64, __int64))&v298[0])(
                  s,
                  s,
                  3,
                  v194,
                  v229,
                  v235);
              }
              if ( !*(_BYTE *)(a4 + 2000) )
              {
                v191 = (char *)&v270;
                v192 = (char *)sub_1CB4E40(0);
                sub_12DE0B0((__int64)&v270, (__int64)v192, 0, 0);
              }
              if ( !*(_BYTE *)(a4 + 920) )
              {
                v200 = sub_185D600(v191, v192, v193, v194, v195, v196);
                v191 = (char *)&v270;
                v192 = (char *)v200;
                sub_12DE0B0((__int64)&v270, v200, 0, 0);
              }
              if ( *(_BYTE *)(a4 + 1080) )
              {
                if ( *(_BYTE *)(a4 + 1240) )
                {
LABEL_369:
                  if ( !*(_BYTE *)(a4 + 2000) )
                  {
                    v191 = (char *)&v270;
                    v192 = (char *)sub_1CB4E40(0);
                    sub_12DE0B0((__int64)&v270, (__int64)v192, 0, 0);
                  }
                  if ( !*(_BYTE *)(a4 + 2120) )
                  {
                    v199 = sub_1CB73C0(v191, v192, v193);
                    v191 = (char *)&v270;
                    v192 = (char *)v199;
                    sub_12DE0B0((__int64)&v270, v199, 0, 0);
                  }
                  if ( !*(_BYTE *)(a4 + 2320) )
                  {
                    v197 = sub_1A13320(v191, v192, v193, v194, v195, v196);
                    sub_12DE0B0((__int64)&v270, v197, 0, 0);
                  }
                  goto LABEL_84;
                }
              }
              else
              {
                v191 = (char *)&v270;
                v192 = (char *)sub_17060B0(1, 0);
                sub_12DE0B0((__int64)&v270, (__int64)v192, 0, 0);
                if ( *(_BYTE *)(a4 + 1240) )
                  goto LABEL_377;
              }
              v191 = (char *)&v270;
              v192 = (char *)sub_195E880(0);
              sub_12DE0B0((__int64)&v270, (__int64)v192, 0, 0);
LABEL_377:
              if ( !*(_BYTE *)(a4 + 1080) )
              {
                v191 = (char *)&v270;
                v192 = (char *)sub_17060B0(1, 0);
                sub_12DE0B0((__int64)&v270, (__int64)v192, 0, 0);
              }
              goto LABEL_369;
            }
            *(_QWORD *)&v298[0] = 0;
            v72 = (char *)&v270;
            v73 = (char *)sub_1A62BF0(4, 0, 0, 1, 0, 0, 1, (__int64)s);
            sub_12DE0B0((__int64)&v270, (__int64)v73, 0, 0);
            v76 = v223;
            v77 = v230;
            if ( *(_QWORD *)&v298[0] )
            {
              v73 = s;
              v72 = s;
              (*(void (__fastcall **)(char *, char *, __int64))&v298[0])(s, s, 3);
            }
            if ( !*(_BYTE *)(a4 + 880) )
            {
              v139 = sub_1857160(v72, v73, v76, v77, v74, v75);
              v72 = (char *)&v270;
              v73 = (char *)v139;
              sub_12DE0B0((__int64)&v270, v139, 0, 0);
            }
            if ( !*(_BYTE *)(a4 + 1080) )
            {
              v72 = (char *)&v270;
              v73 = (char *)sub_17060B0(1, 0);
              sub_12DE0B0((__int64)&v270, (__int64)v73, 0, 0);
            }
            if ( !*(_BYTE *)(a4 + 2000) )
            {
              v72 = (char *)&v270;
              v73 = (char *)sub_1CB4E40(0);
              sub_12DE0B0((__int64)&v270, (__int64)v73, 0, 0);
            }
            if ( !*(_BYTE *)(a4 + 880) )
            {
              v150 = sub_1857160(v72, v73, v76, v77, v74, v75);
              v72 = (char *)&v270;
              v73 = (char *)v150;
              sub_12DE0B0((__int64)&v270, v150, 0, 0);
            }
            v78 = sub_1CEF8F0(v72, v73, v76);
            sub_12DE0B0((__int64)&v270, v78, 0, 0);
            v79 = sub_215D9D0();
            sub_12DE0B0((__int64)&v270, v79, 0, 0);
            if ( !*(_BYTE *)(a4 + 2720) )
            {
              v149 = sub_1A7A9F0();
              sub_12DE0B0((__int64)&v270, v149, 0, 0);
            }
            if ( !*(_BYTE *)(a4 + 1080) )
            {
              v148 = sub_17060B0(1, 0);
              sub_12DE0B0((__int64)&v270, v148, 0, 0);
            }
            *(_QWORD *)&v298[0] = 0;
            v80 = sub_1A62BF0(5, 0, 0, 1, 0, 0, 1, (__int64)s);
            sub_12DE0B0((__int64)&v270, v80, 0, 0);
            v85 = v224;
            v86 = v231;
            if ( *(_QWORD *)&v298[0] )
            {
              v85 = s;
              v86 = s;
              (*(void (__fastcall **)(char *, char *, __int64))&v298[0])(s, s, 3);
            }
            if ( !*(_BYTE *)(a4 + 1080) )
            {
              v86 = (char *)&v270;
              v85 = (char *)sub_17060B0(1, 0);
              sub_12DE0B0((__int64)&v270, (__int64)v85, 0, 0);
            }
            if ( !*(_BYTE *)(a4 + 920) )
            {
              v147 = sub_185D600(v86, v85, v81, v82, v83, v84);
              v86 = (char *)&v270;
              v85 = (char *)v147;
              sub_12DE0B0((__int64)&v270, v147, 0, 0);
            }
            if ( !*(_BYTE *)(a4 + 2040) )
            {
              v146 = sub_1B26330(v86, v85, v81);
              v86 = (char *)&v270;
              v85 = (char *)v146;
              sub_12DE0B0((__int64)&v270, v146, 0, 0);
            }
            if ( !*(_BYTE *)(a4 + 1960) )
            {
              v145 = sub_184CD60(v86, v85);
              v86 = (char *)&v270;
              v85 = (char *)v145;
              sub_12DE0B0((__int64)&v270, v145, 0, 0);
            }
            if ( !*(_BYTE *)(a4 + 2320) )
            {
              v144 = sub_1A13320(v86, v85, v81, v82, v83, v84);
              v86 = (char *)&v270;
              v85 = (char *)v144;
              sub_12DE0B0((__int64)&v270, v144, 0, 0);
            }
            if ( !*(_BYTE *)(a4 + 320) )
            {
              v86 = (char *)&v270;
              v85 = (char *)sub_1833EB0(3);
              sub_12DE0B0((__int64)&v270, (__int64)v85, 0, 0);
            }
            v87 = sub_1C6E800(v86, v85);
            sub_12DE0B0((__int64)&v270, v87, 0, 0);
            if ( !*(_BYTE *)(a4 + 720) )
            {
              v87 = sub_1842BC0();
              sub_12DE0B0((__int64)&v270, v87, 0, 0);
            }
            if ( !*(_BYTE *)(a4 + 280) )
            {
              v87 = sub_18DEFF0();
              sub_12DE0B0((__int64)&v270, v87, 0, 0);
            }
            if ( !*(_BYTE *)(a4 + 1960) )
            {
              v143 = sub_184CD60(&v270, v87);
              sub_12DE0B0((__int64)&v270, v143, 0, 0);
            }
            v88 = sub_18FD350(0);
            sub_12DE0B0((__int64)&v270, v88, 0, 0);
            v89 = sub_18EEA90();
            sub_12DE0B0((__int64)&v270, v89, 0, 0);
            *(_QWORD *)&v298[0] = 0;
            v90 = (char *)&v270;
            v91 = (char *)sub_1A62BF0(1, 0, 0, 1, 0, 0, 1, (__int64)s);
            sub_12DE0B0((__int64)&v270, (__int64)v91, 0, 0);
            v92 = v225;
            if ( *(_QWORD *)&v298[0] )
            {
              v91 = s;
              v90 = s;
              (*(void (__fastcall **)(char *, char *, __int64))&v298[0])(s, s, 3);
            }
            v93 = sub_197E720(v90, v91, v92);
            sub_12DE0B0((__int64)&v270, v93, 0, 0);
            if ( !*(_BYTE *)(a4 + 1000) )
            {
              v93 = sub_19401A0();
              sub_12DE0B0((__int64)&v270, v93, 0, 0);
            }
            if ( !*(_BYTE *)(a4 + 880) )
            {
              v142 = sub_1857160(&v270, v93, v94, v95, v96, v97);
              sub_12DE0B0((__int64)&v270, v142, 0, 0);
            }
            if ( !*(_BYTE *)(a4 + 1080) )
            {
              v141 = sub_17060B0(1, 0);
              sub_12DE0B0((__int64)&v270, v141, 0, 0);
            }
            *(_QWORD *)&v298[0] = 0;
            v98 = sub_1A62BF0(7, 0, 0, 1, 0, 0, 1, (__int64)s);
            sub_12DE0B0((__int64)&v270, v98, 0, 0);
            if ( *(_QWORD *)&v298[0] )
              (*(void (__fastcall **)(char *, char *, __int64))&v298[0])(s, s, 3);
            v99 = sub_1C8A4D0(0);
            sub_12DE0B0((__int64)&v270, v99, 0, 0);
            if ( !*(_BYTE *)(a4 + 2600) )
            {
              v140 = sub_1A223D0(&v270, v99, v100);
              sub_12DE0B0((__int64)&v270, v140, 0, 0);
            }
            v101 = sub_1832270(1);
            sub_12DE0B0((__int64)&v270, v101, 0, 0);
            if ( !*(_BYTE *)(a4 + 1080) )
            {
              v156 = sub_17060B0(1, 0);
              sub_12DE0B0((__int64)&v270, v156, 0, 0);
            }
            v102 = sub_1869C50(1, 0, 1);
            sub_12DE0B0((__int64)&v270, v102, 0, 0);
            if ( !*(_BYTE *)(a4 + 1080) )
            {
              v155 = sub_17060B0(1, 0);
              sub_12DE0B0((__int64)&v270, v155, 0, 0);
            }
            v103 = sub_1A68E70();
            sub_12DE0B0((__int64)&v270, v103, 0, 0);
            if ( !*(_BYTE *)(a4 + 1520) )
            {
              v154 = sub_198DF00(0xFFFFFFFFLL);
              sub_12DE0B0((__int64)&v270, v154, 0, 0);
            }
            if ( !*(_BYTE *)(a4 + 1240) )
            {
              v153 = sub_195E880(0);
              sub_12DE0B0((__int64)&v270, v153, 0, 0);
            }
            if ( !*(_BYTE *)(a4 + 960) )
            {
              v152 = sub_190BB10(0, 0);
              sub_12DE0B0((__int64)&v270, v152, 0, 0);
            }
            v104 = sub_19B73C0(3, -1, -1, 0, 0, -1, 0);
            sub_12DE0B0((__int64)&v270, v104, 0, 0);
            if ( !*(_BYTE *)(a4 + 2600) )
            {
              v151 = sub_1A223D0(&v270, v104, v105);
              sub_12DE0B0((__int64)&v270, v151, 0, 0);
            }
            v106 = sub_1C98160(0);
            sub_12DE0B0((__int64)&v270, v106, 0, 0);
            if ( !*(_BYTE *)(a4 + 1760) )
            {
              v106 = sub_1C8E680(0);
              sub_12DE0B0((__int64)&v270, v106, 0, 0);
            }
            if ( !*(_BYTE *)(a4 + 1280) )
            {
              v106 = sub_1B7FDF0(3);
              sub_12DE0B0((__int64)&v270, v106, 0, 0);
            }
            if ( !*(_BYTE *)(a4 + 2640) )
            {
              v160 = sub_18B1DE0(&v270, v106, v107);
              sub_12DE0B0((__int64)&v270, v160, 0, 0);
            }
            if ( *(_BYTE *)(a4 + 1080) )
            {
              if ( *(_BYTE *)(a4 + 1160) )
              {
LABEL_266:
                v108 = sub_18FD350(0);
                sub_12DE0B0((__int64)&v270, v108, 0, 0);
                if ( !*(_BYTE *)(a4 + 1080) )
                {
                  v108 = sub_17060B0(1, 0);
                  sub_12DE0B0((__int64)&v270, v108, 0, 0);
                }
                if ( !*(_BYTE *)(a4 + 2440) )
                {
                  v135 = sub_1CC60B0(&v270, v108, v109);
                  sub_12DE0B0((__int64)&v270, v135, 0, 0);
                }
                *(_QWORD *)&v298[0] = 0;
                v110 = (char *)&v270;
                v111 = (char *)sub_1A62BF0(2, 0, 0, 1, 0, 0, 1, (__int64)s);
                sub_12DE0B0((__int64)&v270, (__int64)v111, 0, 0);
                v114 = v226;
                v115 = v232;
                if ( *(_QWORD *)&v298[0] )
                {
                  v111 = s;
                  v110 = s;
                  (*(void (__fastcall **)(char *, char *, __int64, __int64, __int64, __int64))&v298[0])(
                    s,
                    s,
                    3,
                    v113,
                    v226,
                    v232);
                }
                if ( !*(_BYTE *)(a4 + 2600) )
                {
                  v137 = sub_1A223D0(v110, v111, v112);
                  v110 = (char *)&v270;
                  v111 = (char *)v137;
                  sub_12DE0B0((__int64)&v270, v137, 0, 0);
                }
                if ( !*(_BYTE *)(a4 + 1120) )
                {
                  v136 = sub_18A3430(v110, v111, v112, v113, v114, v115);
                  sub_12DE0B0((__int64)&v270, v136, 0, 0);
                }
                if ( !*(_BYTE *)(a4 + 1080) )
                {
                  v138 = sub_17060B0(1, 0);
                  sub_12DE0B0((__int64)&v270, v138, 0, 0);
                }
                *(_QWORD *)&v298[0] = 0;
                v116 = sub_1A62BF0(4, 0, 0, 1, 1, 0, 1, (__int64)s);
                sub_12DE0B0((__int64)&v270, v116, 0, 0);
                if ( *(_QWORD *)&v298[0] )
                  (*(void (__fastcall **)(char *, char *, __int64))&v298[0])(s, s, 3);
                goto LABEL_84;
              }
            }
            else
            {
              v157 = sub_17060B0(1, 0);
              sub_12DE0B0((__int64)&v270, v157, 0, 0);
              if ( *(_BYTE *)(a4 + 1160) )
                goto LABEL_304;
            }
            v158 = sub_1952F90(0xFFFFFFFFLL);
            sub_12DE0B0((__int64)&v270, v158, 0, 0);
LABEL_304:
            if ( !*(_BYTE *)(a4 + 1080) )
            {
              v159 = sub_17060B0(1, 0);
              sub_12DE0B0((__int64)&v270, v159, 0, 0);
            }
            goto LABEL_266;
          }
LABEL_162:
          v18 = a4;
          v17 = &v270;
          sub_12DE330(&v270, (_BYTE *)a4);
          v58 = *(_BYTE *)(a4 + 3528) == 0;
          *(_BYTE *)(a4 + 4224) = 0;
          if ( !v58 )
            goto LABEL_163;
          goto LABEL_80;
        }
        v67 = v238;
        v238 = 0;
        v244 = v67;
        goto LABEL_188;
      }
      v23 = *(_QWORD *)(a4 + 3648);
      if ( *(_WORD *)v23 != 26989 || (v19 = 1, *(_BYTE *)(v23 + 2) != 100) )
      {
        if ( *(_WORD *)v23 != 26989 || (v19 = 0, *(_BYTE *)(v23 + 2) != 110) )
          v19 = 1;
        LOBYTE(v19) = (_DWORD)v19 == 0;
        if ( *(_WORD *)v23 == 24941 && *(_BYTE *)(v23 + 2) == 120 )
          goto LABEL_77;
      }
      if ( *(_WORD *)v23 == 26989 && *(_BYTE *)(v23 + 2) == 100 )
      {
LABEL_77:
        v238 = 1;
        goto LABEL_158;
      }
    }
    v23 = *(_WORD *)v23 != 26989 || *(_BYTE *)(v23 + 2) != 110;
    v238 = (_DWORD)v23 == 0;
LABEL_158:
    if ( !(_BYTE)v19 )
      goto LABEL_159;
LABEL_297:
    v20 = !v238;
    if ( !*(_BYTE *)(a4 + 4304) )
    {
      v244 = 1;
      goto LABEL_189;
    }
LABEL_196:
    if ( !v22 )
    {
      v70 = v238;
      v238 = 1;
      v244 = v70;
LABEL_188:
      v243 = v20;
      v17 = &v270;
      v18 = sub_18B3080(1);
      sub_12DE0B0((__int64)&v270, v18, 1u, 0);
      v68 = v244;
      v19 = v238;
      v20 = v243;
      v244 = v238;
      v238 = v68;
    }
LABEL_189:
    if ( !(_BYTE)v20 )
    {
      v22 = *(_BYTE *)(a4 + 4224);
LABEL_191:
      LOBYTE(v20) = v22;
      goto LABEL_38;
    }
LABEL_24:
    if ( !*(_BYTE *)(a4 + 1960) || *(_BYTE *)(a4 + 3000) )
    {
      v18 = sub_1857160(&v270, v18, v23, v19, v20, v21);
      sub_12DE0B0((__int64)&v270, v18, 1u, 0);
      if ( *(_BYTE *)(a4 + 3000) )
      {
        v18 = sub_18FD350(0);
        sub_12DE0B0((__int64)&v270, v18, 1u, 0);
      }
    }
    if ( !*(_BYTE *)(a4 + 1680) )
    {
      v69 = sub_19CE990(&v270, v18, v23);
      sub_12DE0B0((__int64)&v270, v69, 1u, 0);
    }
    v24 = sub_1CB4E40(0);
    sub_12DE0B0((__int64)&v270, v24, 1u, 0);
    v26 = sub_1B26330(&v270, v24, v25);
    sub_12DE0B0((__int64)&v270, v26, 1u, 0);
    v17 = &v270;
    v18 = sub_12D4560();
    sub_12DE0B0((__int64)&v270, v18, 1u, 0);
    if ( !*(_BYTE *)(a4 + 1960) )
    {
      v17 = &v270;
      v18 = ((__int64 (*)(void))sub_184CD60)();
      sub_12DE0B0((__int64)&v270, v18, 1u, 0);
    }
    if ( !*(_BYTE *)(a4 + 440) && !*(_BYTE *)(a4 + 400) )
    {
      v17 = &v270;
      v18 = sub_1C4B6F0();
      sub_12DE0B0((__int64)&v270, v18, 1u, 0);
    }
    if ( *(_BYTE *)(a4 + 3160) )
    {
      v17 = &v270;
      v18 = sub_17060B0(1, 0);
      sub_12DE0B0((__int64)&v270, v18, 1u, 0);
    }
    LOBYTE(v20) = *(_BYTE *)(a4 + 4224);
LABEL_38:
    v27 = *(_QWORD *)(a4 + 4488);
    if ( *(_QWORD *)(a4 + 4496) == v27 )
      goto LABEL_78;
    goto LABEL_39;
  }
  v286[0] = a5;
  v288 = 0x300000000LL;
  v289 = 0;
  v290 = 0;
  v286[1] = a2;
  v287 = a3;
  v36 = (int *)sub_16D40F0(qword_4FBB430);
  if ( v36 )
    v37 = *v36;
  else
    v37 = qword_4FBB430[2];
  LODWORD(v288) = v37;
  v38 = *(_BYTE *)(a4 + 3120);
  LOBYTE(v289) = 0;
  HIBYTE(v289) = v38;
  LOBYTE(v290) = *(_BYTE *)(a4 + 4304);
  HIBYTE(v290) = *(_BYTE *)(a4 + 3704);
  sub_12EB010(s, v286);
  sub_12EC4F0(s, a1);
  v39 = v299;
  if ( v300 )
  {
    v62 = &v299[4 * v301];
    if ( v299 != v62 )
    {
      v63 = v299;
      while ( 1 )
      {
        v64 = v63;
        if ( (unsigned int)(*v63 + 0x7FFFFFFF) <= 0xFFFFFFFD )
          break;
        v63 += 4;
        if ( v62 == v63 )
          goto LABEL_63;
      }
      if ( v63 != v62 )
      {
        do
        {
          v65 = *((_QWORD *)v64 + 1);
          if ( v65 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v65 + 16LL))(v65);
          v64 += 4;
          if ( v64 == v62 )
            break;
          while ( (unsigned int)(*v64 + 0x7FFFFFFF) > 0xFFFFFFFD )
          {
            v64 += 4;
            if ( v62 == v64 )
              goto LABEL_183;
          }
        }
        while ( v64 != v62 );
LABEL_183:
        v39 = v299;
      }
    }
  }
LABEL_63:
  j___libc_free_0(v39);
  result = (char *)v298 + 8;
  if ( *(_OWORD **)&s[8] != (_OWORD *)((char *)v298 + 8) )
    return (char *)_libc_free(*(_QWORD *)&s[8], a1);
  return result;
}
