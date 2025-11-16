// Function: sub_833B90
// Address: 0x833b90
//
void __fastcall sub_833B90(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        int a8,
        const __m128i *a9,
        __int64 a10,
        int a11,
        unsigned int a12,
        unsigned int a13,
        int a14,
        int a15,
        int a16,
        unsigned int a17,
        int a18,
        int a19,
        int a20,
        __int64 *a21,
        _DWORD *a22,
        _DWORD *a23,
        _DWORD *a24)
{
  __int64 v25; // r12
  __int64 v26; // rdx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 i; // rax
  __int64 v30; // rax
  _QWORD *v31; // rbx
  __int64 v32; // rcx
  __int64 j; // r14
  __int64 v34; // rdi
  char v35; // bl
  _BYTE *v36; // rax
  __int64 v37; // rax
  __int64 *v38; // rbx
  char v39; // r13
  __int64 v40; // rax
  char v41; // dl
  int v42; // ebx
  __int64 *v43; // r15
  int v44; // eax
  __int64 v45; // r13
  __int64 *v46; // rax
  int v47; // r15d
  __int64 v48; // rbx
  __int64 *v49; // r14
  __int64 v50; // rdi
  __int64 v51; // rcx
  __int64 v52; // rax
  __int64 *v53; // rax
  __int64 v54; // r13
  unsigned __int64 v55; // rax
  __int64 v56; // rcx
  int v57; // r14d
  __int64 v58; // rdx
  _QWORD *v59; // r15
  __int64 v60; // rdi
  _QWORD *v61; // rax
  __int64 v62; // r15
  __int64 v63; // r13
  __int64 v64; // rax
  _QWORD *v65; // r15
  __int64 v66; // rax
  __int64 *v67; // rdi
  _QWORD *v68; // rdx
  __int64 v69; // rax
  __int64 v70; // r14
  __int64 *v71; // r12
  __int64 v72; // rax
  __int64 *v73; // r14
  int v74; // r8d
  __int64 v75; // r15
  int v76; // r13d
  int v77; // eax
  int v78; // eax
  __int64 *v79; // rax
  __int64 v80; // r13
  char v81; // al
  int v82; // eax
  __int64 v83; // rax
  char v84; // al
  __int64 v85; // rdx
  __int64 v86; // rax
  __int64 v87; // r8
  __int64 *v88; // rax
  __int64 v89; // rdx
  __int64 *v90; // rax
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // r13
  unsigned __int64 v94; // rdx
  unsigned __int64 v95; // rdi
  char v96; // al
  __int64 v97; // rax
  char v98; // al
  int v99; // eax
  __int64 v100; // rax
  _QWORD *v101; // rcx
  __int64 v102; // rbx
  __int64 v103; // rcx
  unsigned __int64 v104; // rcx
  __int64 v105; // r15
  __int64 v106; // r8
  __int64 v107; // rax
  __int64 v108; // r14
  char k; // al
  __int64 v110; // rbx
  __int64 v111; // rdx
  __int64 v112; // rcx
  __int64 v113; // rdi
  __int64 v114; // rax
  __int64 v115; // r15
  char *v116; // rdi
  unsigned int v117; // ecx
  __int64 v118; // rax
  __int64 v119; // rdx
  _QWORD *v120; // r14
  __int64 v121; // rax
  __int64 *v122; // rax
  __int64 m; // rax
  __int64 v124; // rbx
  __int64 v125; // rax
  __int64 v126; // rax
  _QWORD *v127; // rax
  __int64 v128; // rbx
  _QWORD *v129; // r13
  _QWORD *v130; // rax
  _QWORD *v131; // r13
  _QWORD *v132; // rdx
  _QWORD *v133; // rax
  __int64 v134; // r14
  __m128i *v135; // rax
  __m128i *v136; // r13
  bool v137; // zf
  __int64 v138; // r12
  __int64 v139; // rax
  __int64 ii; // rax
  __int64 v141; // rdx
  __int64 v142; // rcx
  __int64 v143; // r8
  __int64 *v144; // r9
  __int64 v145; // rbx
  __m128i *v146; // r12
  __int64 v147; // r13
  char v148; // dl
  __int64 v149; // rax
  __int64 *v150; // rax
  __int64 v151; // rdx
  char v152; // al
  _QWORD *v153; // rbx
  bool v154; // dl
  __int64 v155; // rax
  _QWORD *v156; // r13
  __int64 *v157; // r15
  int v158; // eax
  __int64 v159; // rdx
  __int64 v160; // rax
  char v161; // cl
  __m128i *v162; // rdx
  __int64 jj; // rax
  int v164; // ebx
  __int64 v165; // rdx
  __int64 v166; // rcx
  __int64 v167; // r8
  __int64 *v168; // r9
  __m128i *v169; // rax
  __int64 v170; // rax
  __int64 kk; // rbx
  __int64 v172; // rbx
  __int64 v173; // rax
  int v174; // eax
  __int64 v175; // rax
  __int64 n; // rax
  __int64 v177; // r13
  __int64 v178; // rax
  unsigned int v179; // [rsp+0h] [rbp-3F0h]
  int v180; // [rsp+4h] [rbp-3ECh]
  __int64 *v181; // [rsp+8h] [rbp-3E8h]
  __int64 v182; // [rsp+10h] [rbp-3E0h]
  int v184; // [rsp+1Ch] [rbp-3D4h]
  __int64 v186; // [rsp+28h] [rbp-3C8h]
  _QWORD *v188; // [rsp+38h] [rbp-3B8h]
  __int64 v189; // [rsp+40h] [rbp-3B0h]
  __int64 *v190; // [rsp+48h] [rbp-3A8h]
  unsigned int v191; // [rsp+50h] [rbp-3A0h]
  unsigned __int8 v192; // [rsp+54h] [rbp-39Ch]
  __int64 v193; // [rsp+58h] [rbp-398h]
  __int64 *v194; // [rsp+60h] [rbp-390h]
  _BOOL4 v195; // [rsp+68h] [rbp-388h]
  _QWORD *v196; // [rsp+68h] [rbp-388h]
  _QWORD *v197; // [rsp+70h] [rbp-380h]
  __int64 v198; // [rsp+70h] [rbp-380h]
  __int64 *v199; // [rsp+78h] [rbp-378h]
  __m128i *v200; // [rsp+78h] [rbp-378h]
  unsigned __int64 v201; // [rsp+80h] [rbp-370h]
  __int64 *v202; // [rsp+80h] [rbp-370h]
  unsigned int v203; // [rsp+88h] [rbp-368h]
  _BOOL4 v204; // [rsp+8Ch] [rbp-364h]
  int v205; // [rsp+90h] [rbp-360h]
  __int64 v206; // [rsp+90h] [rbp-360h]
  int v207; // [rsp+98h] [rbp-358h]
  _QWORD *v208; // [rsp+98h] [rbp-358h]
  int v209; // [rsp+ACh] [rbp-344h] BYREF
  int v210; // [rsp+B0h] [rbp-340h] BYREF
  int v211; // [rsp+B4h] [rbp-33Ch] BYREF
  __m128i *v212; // [rsp+B8h] [rbp-338h] BYREF
  __int64 v213; // [rsp+C0h] [rbp-330h] BYREF
  __int64 v214; // [rsp+C8h] [rbp-328h]
  __int64 v215; // [rsp+D0h] [rbp-320h]
  __int64 v216; // [rsp+E0h] [rbp-310h] BYREF
  __int64 v217; // [rsp+E8h] [rbp-308h]
  __int64 v218; // [rsp+F0h] [rbp-300h]
  _BYTE v219[352]; // [rsp+100h] [rbp-2F0h] BYREF
  __m128i v220; // [rsp+260h] [rbp-190h] BYREF
  char v221; // [rsp+270h] [rbp-180h]
  __m128i v222; // [rsp+2F0h] [rbp-100h] BYREF
  __m128i v223; // [rsp+300h] [rbp-F0h] BYREF
  __m128i v224; // [rsp+310h] [rbp-E0h] BYREF
  __m128i v225; // [rsp+320h] [rbp-D0h] BYREF
  __m128i v226; // [rsp+330h] [rbp-C0h] BYREF
  __m128i v227; // [rsp+340h] [rbp-B0h] BYREF
  __m128i v228; // [rsp+350h] [rbp-A0h] BYREF
  __m128i v229; // [rsp+360h] [rbp-90h] BYREF
  __m128i v230; // [rsp+370h] [rbp-80h] BYREF
  __m128i v231; // [rsp+380h] [rbp-70h] BYREF
  __m128i v232; // [rsp+390h] [rbp-60h] BYREF
  __m128i v233; // [rsp+3A0h] [rbp-50h] BYREF
  __m128i v234[4]; // [rsp+3B0h] [rbp-40h] BYREF

  v25 = a6;
  v182 = a2;
  *a24 = 0;
  v212 = 0;
  if ( qword_4D03C50 && (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) != 0 )
  {
    sub_82D8A0(0);
    sub_725130(v212->m128i_i64);
    return;
  }
  if ( a6 )
  {
    sub_82BD70();
    v201 = 0;
    v188 = 0;
  }
  else
  {
    v84 = *(_BYTE *)(a1 + 80);
    v201 = a1;
    if ( v84 == 16 )
    {
      v201 = **(_QWORD **)(a1 + 88);
      v84 = *(_BYTE *)(v201 + 80);
    }
    if ( v84 == 24 )
    {
      v201 = *(_QWORD *)(v201 + 88);
      v84 = *(_BYTE *)(v201 + 80);
    }
    v85 = *(_QWORD *)(v201 + 88);
    if ( v84 == 20 )
      v85 = *(_QWORD *)(v85 + 176);
    v25 = *(_QWORD *)(v85 + 152);
    v26 = sub_82BD70();
    if ( (*(_BYTE *)(*(_QWORD *)(v26 + 1008) + 8 * (5LL * *(_QWORD *)(v26 + 1024) - 5)) & 1) != 0 )
    {
      v188 = 0;
      if ( !(unsigned int)sub_82BDA0(v201) )
      {
        v86 = sub_82BD70();
        v26 = 5LL * *(_QWORD *)(v86 + 1024) - 5;
        v188 = (_QWORD *)(*(_QWORD *)(v86 + 1008) + 8 * v26 + 16);
      }
    }
    else
    {
      v188 = 0;
    }
  }
  for ( i = v25; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v184 = 0;
  v30 = **(_QWORD **)(i + 168);
  if ( v30 && (*(_BYTE *)(v30 + 35) & 1) != 0 && a8 )
  {
    if ( !a9 )
    {
      v114 = sub_8D46C0(a10);
      a2 = (__int64)v219;
      sub_6EA0A0(v114, (__int64)v219);
      a9 = (const __m128i *)v219;
    }
    v31 = (_QWORD *)sub_6E3060(a9);
    *v31 = a7;
    a7 = v31;
    a8 = sub_8D2E30(*(_QWORD *)(v31[3] + 8LL));
    v184 = 1;
    if ( a8 )
    {
      v110 = v31[3];
      a8 = 0;
      *(_QWORD *)(v110 + 8) = sub_8D46C0(*(_QWORD *)(v110 + 8));
      *(_BYTE *)(a7[3] + 25LL) = 1;
    }
  }
  if ( !a1 )
  {
    if ( *(_BYTE *)(v25 + 140) == 12 )
    {
      v189 = 0;
      v39 = 0;
      LODWORD(v38) = 1;
      v203 = 0;
      v195 = 0;
      v204 = 0;
      goto LABEL_58;
    }
    v39 = 0;
    v189 = 0;
    v203 = 0;
    v190 = *(__int64 **)(v25 + 168);
    v195 = 0;
    v204 = 0;
LABEL_240:
    a2 = (__int64)a7;
    if ( sub_829250(v25, a7, v189) )
    {
      v101 = v188;
      v197 = 0;
      if ( !v188 )
        goto LABEL_132;
      goto LABEL_242;
    }
    goto LABEL_61;
  }
  v32 = (__int64)a21;
  for ( j = *a21; j; j = *(_QWORD *)j )
  {
    v34 = *(_QWORD *)(j + 8);
    if ( v34 )
    {
      if ( a1 == v34 || (a2 = a1, sub_828A00(v34, a1, v26, v32, v27)) )
      {
        if ( (a20 == 9) == ((*(_BYTE *)(j + 145) & 2) != 0) )
          goto LABEL_232;
      }
    }
  }
  if ( !a16 )
  {
    a2 = a3;
    v203 = sub_828ED0(a1, a3, a12, a14, a15, a17, a18, &v213, &v216);
    if ( !v203 )
    {
      if ( (_DWORD)v216 )
      {
        *a24 = 1;
LABEL_233:
        sub_82D8A0(0);
        goto LABEL_133;
      }
LABEL_232:
      v203 = 0;
      goto LABEL_233;
    }
  }
  v35 = *(_BYTE *)(a1 + 80);
  if ( v35 == 16 )
  {
    v201 = **(_QWORD **)(a1 + 88);
    v35 = *(_BYTE *)(v201 + 80);
  }
  else
  {
    v201 = a1;
  }
  if ( v35 == 24 )
  {
    v201 = *(_QWORD *)(v201 + 88);
    v35 = *(_BYTE *)(v201 + 80);
  }
  v204 = v35 == 20;
  if ( v35 == 20 )
  {
    if ( (*(_BYTE *)(a1 + 82) & 4) == 0 )
      goto LABEL_251;
    v189 = 0;
    goto LABEL_272;
  }
  v36 = *(_BYTE **)(v201 + 88);
  v189 = (__int64)v36;
  if ( a17 )
  {
    if ( (v36[206] & 8) != 0 && v36[174] == 5 )
    {
      v98 = v36[176];
      if ( (unsigned __int8)(v98 - 31) <= 2u || (unsigned __int8)(v98 - 16) <= 1u )
        goto LABEL_232;
    }
  }
  if ( (*(_BYTE *)(v201 + 104) & 1) != 0 )
    v203 = sub_8796F0(v201);
  else
    v203 = (*(_BYTE *)(v189 + 208) & 4) != 0;
  if ( v203 )
  {
    if ( v188 )
    {
      a2 = v201 + 48;
      sub_686E10(0xCF9u, (FILE *)(v201 + 48), v201, v188);
      v203 = 0;
      goto LABEL_233;
    }
    goto LABEL_232;
  }
  v25 = *(_QWORD *)(v189 + 152);
  if ( (*(_BYTE *)(a1 + 82) & 4) == 0 )
    goto LABEL_38;
  v37 = *(_QWORD *)(v201 + 88);
  if ( *(_BYTE *)(v201 + 80) == 20 )
LABEL_272:
    v37 = *(_QWORD *)(*(_QWORD *)(v201 + 88) + 176LL);
  if ( (*(_BYTE *)(v37 + 194) & 0x40) == 0 )
  {
    v203 = 0;
    v115 = 0;
    v180 = 0;
    goto LABEL_293;
  }
  if ( v35 == 20 )
  {
LABEL_251:
    v102 = *(_QWORD *)(v201 + 88);
    v103 = *(_QWORD *)(v102 + 176);
    v189 = v103;
    v25 = *(_QWORD *)(v103 + 152);
    if ( a4 )
    {
      if ( !dword_4F077BC || (_DWORD)qword_4F077B4 || (*(_BYTE *)(v103 + 89) & 4) != 0 )
      {
        a2 = (__int64)a7;
        v203 = sub_829250(*(_QWORD *)(v103 + 152), a7, v103);
        if ( v203 )
        {
          if ( v188 )
          {
            a2 = v201 + 48;
            sub_686E10(0xCFAu, (FILE *)(v201 + 48), v201, v188);
            v203 = 0;
            goto LABEL_233;
          }
          goto LABEL_232;
        }
      }
      else
      {
        v203 = 1;
      }
      v104 = *(unsigned int *)(v102 + 392);
      if ( v104 > unk_4D042F0 )
      {
        sub_861C90();
        v203 = 0;
        goto LABEL_233;
      }
      *(_DWORD *)(v102 + 392) = v104 + 1;
      v105 = qword_4D03C50;
      sub_865900(v201);
      a2 = a4;
      v106 = 0x20000;
      if ( a20 == 7 )
        v106 = 147456;
      v107 = sub_8B1C20(v201, a4, &v212, 0, v106);
      --*(_DWORD *)(v102 + 392);
      v25 = v107;
      qword_4D03C50 = v105;
      if ( !v107 )
      {
        if ( v188 )
        {
          a2 = v201 + 48;
          sub_6875A0(0xCFBu, (FILE *)(v201 + 48), v201, a4, v188);
          v203 = 1;
          goto LABEL_233;
        }
        goto LABEL_363;
      }
      v28 = v203;
      if ( v203 )
      {
        a2 = (__int64)a7;
        if ( sub_829250(v107, a7, v189) )
        {
          if ( v188 )
          {
            a2 = v201 + 48;
            sub_686E10(0xCFAu, (FILE *)(v201 + 48), v201, v188);
            goto LABEL_233;
          }
          goto LABEL_363;
        }
      }
      v38 = **(__int64 ***)(v25 + 168);
      if ( v38 )
      {
        do
        {
          v108 = v38[1];
          for ( k = *(_BYTE *)(v108 + 140); k == 12; k = *(_BYTE *)(v108 + 140) )
            v108 = *(_QWORD *)(v108 + 160);
          if ( (unsigned __int8)(k - 9) <= 2u )
          {
            if ( (*(_BYTE *)(v108 + 141) & 0x20) != 0 )
            {
              v220.m128i_i32[0] = 0;
              a2 = (__int64)&v220;
              sub_8AD220(v108, &v220);
              v27 = v220.m128i_u32[0];
              if ( v220.m128i_i32[0] )
              {
                if ( v188 )
                {
                  a2 = v201 + 48;
                  sub_67E730(0xD19u, (_DWORD *)(v201 + 48), v201, *((_DWORD *)v38 + 9), v108, v188);
                }
                goto LABEL_363;
              }
              if ( (*(_BYTE *)(v108 + 141) & 0x20) != 0 )
              {
                if ( v188 )
                {
                  a2 = v201 + 48;
                  sub_67E730(0xD1Au, (_DWORD *)(v201 + 48), v201, *((_DWORD *)v38 + 9), v108, v188);
                  v203 = 1;
                  goto LABEL_233;
                }
LABEL_363:
                v203 = 1;
                goto LABEL_233;
              }
            }
            if ( (*(_BYTE *)(v108 + 176) & 0x20) != 0 )
            {
              if ( v188 )
              {
                a2 = v201 + 48;
                sub_67E730(0xD1Bu, (_DWORD *)(v201 + 48), v201, *((_DWORD *)v38 + 9), v108, v188);
              }
              goto LABEL_363;
            }
          }
          v38 = (__int64 *)*v38;
        }
        while ( v38 );
        v203 = 1;
      }
      else
      {
        v203 = 1;
        LODWORD(v38) = 0;
      }
      goto LABEL_39;
    }
  }
  v203 = 0;
LABEL_38:
  LODWORD(v38) = 1;
LABEL_39:
  if ( dword_4D04474 && (*(_BYTE *)(v189 + 206) & 0x18) == 0x18 && (sub_72F570(v189) || (unsigned int)sub_72F850(v189)) )
  {
    if ( v188 )
    {
      a2 = v201 + 48;
      sub_686E10(0xCFCu, (FILE *)(v201 + 48), v201, v188);
    }
    goto LABEL_233;
  }
  v39 = dword_4D044AC;
  if ( dword_4D044AC )
  {
    v39 = 0;
    if ( (a19 & 0x400) != 0 && a7 && !*a7 && *(_BYTE *)(v189 + 174) == 1 )
      v39 = (unsigned int)sub_72F500(v189, 0, 0, 1, 0) != 0;
  }
  a2 = a17;
  if ( !a17
    || (*(_BYTE *)(v201 + 81) & 0x10) != 0
    || dword_4F077BC && qword_4F077A8 <= 0x9F5Fu
    || !*((_BYTE *)a7 + 8) && (unsigned int)sub_8D3A70(*(_QWORD *)(a7[3] + 8LL)) )
  {
    v195 = 0;
    goto LABEL_59;
  }
  v40 = *a7;
  if ( !*a7 )
    goto LABEL_55;
  v41 = *(_BYTE *)(v40 + 8);
  if ( v41 == 3 )
  {
    v40 = sub_6BBB10(a7);
    if ( !v40 )
      goto LABEL_55;
    v41 = *(_BYTE *)(v40 + 8);
  }
  if ( v41 )
  {
LABEL_55:
    v195 = a17;
    goto LABEL_59;
  }
  v195 = sub_8D3A70(*(_QWORD *)(*(_QWORD *)(v40 + 24) + 8LL)) == 0;
LABEL_59:
  while ( *(_BYTE *)(v25 + 140) == 12 )
LABEL_58:
    v25 = *(_QWORD *)(v25 + 160);
  v190 = *(__int64 **)(v25 + 168);
  if ( (_DWORD)v38 )
    goto LABEL_240;
LABEL_61:
  v180 = 0;
  v191 = 0;
  v42 = 1;
  v43 = (__int64 *)*v190;
  v44 = 0x20000;
  v194 = 0;
  if ( a20 == 7 )
    v44 = 147456;
  v186 = v25;
  v181 = (__int64 *)*v190;
  v179 = v44;
  v193 = 0;
  v197 = 0;
  v192 = v39 & 1;
  while ( 1 )
  {
    if ( a7 )
    {
      v205 = 0;
      v207 = 0;
      a2 = *((unsigned __int8 *)a7 + 8);
      v45 = (__int64)a7;
      v199 = v181;
      v46 = v43;
      v47 = v42;
      v48 = (__int64)v197;
      v49 = v46;
      while ( 1 )
      {
        ++v207;
        v50 = v45;
        while ( (_BYTE)a2 == 2 )
        {
          while ( 1 )
          {
            if ( !*(_QWORD *)v50 )
              goto LABEL_484;
            a2 = *(unsigned __int8 *)(*(_QWORD *)v50 + 8LL);
            if ( (_BYTE)a2 == 3 )
              break;
            v50 = *(_QWORD *)v50;
            if ( (_BYTE)a2 != 2 )
              goto LABEL_71;
          }
          v83 = sub_6BBB10((_QWORD *)v50);
          a2 = *(unsigned __int8 *)(v83 + 8);
          v50 = v83;
        }
LABEL_71:
        if ( !v47 )
          goto LABEL_72;
        v48 = (__int64)qword_4D03C60;
        if ( qword_4D03C60 )
          qword_4D03C60 = (_QWORD *)*qword_4D03C60;
        else
          v48 = sub_823970(104);
        sub_82D850(v48);
        if ( !v197 )
          break;
        v53 = v194;
        v194 = (__int64 *)v48;
        *v53 = v48;
        if ( !v49 )
        {
LABEL_107:
          if ( (v190[2] & 1) == 0 )
          {
            if ( v204 )
            {
              if ( !v199 )
              {
                v207 = v204;
LABEL_246:
                if ( v188 )
                {
                  a2 = v201 + 48;
                  sub_67E630(0xCFDu, (_DWORD *)(v201 + 48), v201, v207, v188);
                }
                goto LABEL_132;
              }
              if ( (*((_BYTE *)v199 + 33) & 1) != 0 )
                goto LABEL_246;
            }
            sub_721090();
          }
          *(_DWORD *)(v48 + 8) = 5;
          v205 = 1;
          goto LABEL_94;
        }
LABEL_73:
        if ( (*((_BYTE *)v49 + 33) & 3) == 1 )
        {
          v82 = 1;
          if ( !*v49 )
            v82 = v191;
          v191 = v82;
          goto LABEL_94;
        }
        if ( !v204 )
        {
          if ( v47 )
            goto LABEL_79;
          goto LABEL_78;
        }
        if ( *((char *)v199 + 32) < 0 )
        {
          if ( v47 )
          {
            if ( v205 )
              goto LABEL_96;
            v49 = (__int64 *)*v49;
            v81 = *((_BYTE *)v199 + 33) & 1;
            goto LABEL_164;
          }
LABEL_78:
          v193 = *(_QWORD *)v48;
          goto LABEL_79;
        }
        if ( !v47 )
        {
          if ( (*((_BYTE *)v199 + 33) & 1) == 0 )
          {
            if ( v205 )
            {
LABEL_95:
              v48 = *(_QWORD *)v48;
              goto LABEL_96;
            }
            v49 = (__int64 *)*v49;
            goto LABEL_165;
          }
          goto LABEL_78;
        }
LABEL_79:
        if ( a13 && a20 == 10 && (v51 = a19 & 0x1C01088B, (a19 & 0x1C01088B) == 0) )
        {
          if ( *(_BYTE *)(v50 + 8) == 1 )
            v51 = a13;
        }
        else
        {
          v51 = a13;
        }
        a2 = v49[1];
        sub_84A950(v50, a2, v49, v51, v192 & (v45 == (_QWORD)a7), v48);
        if ( dword_4F077BC && !(_DWORD)qword_4F077B4 && qword_4F077A8 && a20 == 2 && !*a7 )
        {
          v99 = 1;
          if ( *(char *)(v48 + 85) >= 0 )
            v99 = v180;
          v180 = v99;
        }
        if ( v195 )
        {
          v80 = v49[1];
          if ( (unsigned int)sub_8D2FB0(v80) )
            v80 = sub_8D46C0(v80);
          v195 = sub_8D2870(v80) == 0;
        }
        if ( !v47 )
          *(_QWORD *)v48 = v193;
        if ( *(_DWORD *)(v48 + 8) == 7 )
        {
          if ( v188 )
          {
            a2 = v201 + 48;
            sub_67E630(0xCFEu, (_DWORD *)(v201 + 48), v201, v207, v188);
          }
          goto LABEL_132;
        }
        if ( !v205 )
        {
          v49 = (__int64 *)*v49;
          if ( v204 )
          {
            v81 = *((_BYTE *)v199 + 33) & 1;
LABEL_164:
            v205 = 0;
            if ( v81 )
              goto LABEL_94;
LABEL_165:
            v205 = 0;
            v199 = (__int64 *)*v199;
          }
        }
LABEL_94:
        if ( !v47 )
          goto LABEL_95;
LABEL_96:
        v45 = *(_QWORD *)v50;
        if ( !*(_QWORD *)v50 )
          goto LABEL_109;
        a2 = *(unsigned __int8 *)(v45 + 8);
        if ( (_BYTE)a2 == 3 )
        {
          v52 = sub_6BBB10((_QWORD *)v50);
          v45 = v52;
          if ( !v52 )
          {
LABEL_109:
            v42 = v47;
            v43 = v49;
            goto LABEL_110;
          }
          a2 = *(unsigned __int8 *)(v52 + 8);
        }
      }
      v194 = (__int64 *)v48;
      v197 = (_QWORD *)v48;
LABEL_72:
      if ( !v49 )
        goto LABEL_107;
      goto LABEL_73;
    }
    v199 = v181;
LABEL_110:
    v42 = !v204 | v42 ^ 1;
    if ( v42 )
    {
      a2 = v195;
      v25 = v186;
      v122 = v199;
      if ( v195 )
        goto LABEL_132;
      while ( v122 && *v122 && (*((_BYTE *)v122 + 33) & 1) != 0 )
        v122 = (__int64 *)*v122;
      if ( v43 )
      {
        if ( v204 && (*((_BYTE *)v122 + 33) & 1) != 0 )
        {
          if ( a20 != 7 || !(unsigned int)sub_8291E0(v189) )
            goto LABEL_132;
        }
        else if ( (*((_BYTE *)v43 + 33) & 1) != 0 )
        {
          goto LABEL_132;
        }
      }
      if ( a11 )
        goto LABEL_341;
      for ( m = v186; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
        ;
      v124 = *(_QWORD *)(*(_QWORD *)(m + 168) + 40LL);
      if ( !a8 )
      {
        if ( v124 && (a17 & 1) != 0 )
          goto LABEL_132;
LABEL_341:
        v115 = (__int64)v197;
        goto LABEL_342;
      }
      v115 = (__int64)qword_4D03C60;
      if ( qword_4D03C60 )
        qword_4D03C60 = (_QWORD *)*qword_4D03C60;
      else
        v115 = sub_823970(104);
      sub_82D850(v115);
      *(_QWORD *)v115 = v197;
      if ( v124 )
      {
        a2 = a1;
        v135 = sub_82EAF0(v186, a1, 0);
        v136 = v135;
        if ( a10 )
        {
          a2 = a10;
          sub_8399C0(0, a10, 1, v135, v186, v115);
          v137 = *(_DWORD *)(v115 + 8) == 7;
          *(_QWORD *)v115 = v197;
          if ( !v137 )
            goto LABEL_342;
          v138 = sub_8D46C0(v136);
          v139 = sub_8D46C0(a10);
          a2 = v138;
          if ( !(unsigned int)sub_8D5DF0(v139) )
          {
            *a22 = 1;
            sub_82D8A0((_QWORD *)v115);
            goto LABEL_133;
          }
LABEL_437:
          *a23 = 1;
          sub_82D8A0((_QWORD *)v115);
          goto LABEL_133;
        }
        if ( a20 == 9 && !(unsigned int)sub_8D3A70(a9->m128i_i64[0]) )
        {
LABEL_417:
          sub_82D8A0((_QWORD *)v115);
          goto LABEL_133;
        }
        a2 = v189;
        sub_839CB0(a9, v189, v186, v136);
        v137 = *(_DWORD *)(v115 + 8) == 7;
        *(_QWORD *)v115 = v197;
        if ( v137 )
          goto LABEL_437;
      }
      else
      {
        if ( a5 )
        {
          *(_DWORD *)(v115 + 8) = 4;
          v172 = a5;
          if ( *(_BYTE *)(a5 + 80) == 16 )
            v172 = **(_QWORD **)(a5 + 88);
          if ( *(_BYTE *)(v172 + 80) == 24 )
            v172 = *(_QWORD *)(v172 + 88);
          if ( (*(_BYTE *)(v172 + 104) & 1) != 0 )
          {
            v174 = sub_8796F0(v172);
          }
          else
          {
            v173 = *(_QWORD *)(v172 + 88);
            if ( *(_BYTE *)(v172 + 80) == 20 )
              v173 = *(_QWORD *)(v173 + 176);
            v174 = (*(_BYTE *)(v173 + 208) & 4) != 0;
          }
          if ( v174 )
            goto LABEL_417;
          v175 = *(_QWORD *)(v172 + 88);
          *(_QWORD *)(v115 + 48) = v175;
          *(_QWORD *)(v115 + 56) = a5;
          for ( n = *(_QWORD *)(v175 + 152); *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
            ;
          v177 = *(_QWORD *)(n + 160);
          if ( (unsigned int)sub_8D3070(v177) || (unsigned int)sub_8D3190() && (unsigned int)sub_8D3110(v177) )
            *(_BYTE *)(v115 + 64) |= 4u;
        }
        else
        {
          *(_DWORD *)(v115 + 8) = 7;
        }
        *(_BYTE *)(v115 + 15) = 1;
      }
LABEL_342:
      if ( (*((_BYTE *)v190 + 20) & 2) == 0 || !v189 )
        goto LABEL_293;
      v209 = 1;
      v210 = 0;
      v200 = v212;
      v213 = 0;
      v214 = 0;
      v215 = 0;
      v125 = sub_823970(80);
      v214 = 10;
      v213 = v125;
      v216 = 0;
      v217 = 0;
      v218 = 0;
      v126 = sub_823970(0);
      v217 = 0;
      v216 = v126;
      v127 = sub_724DC0();
      v128 = v215;
      v129 = v127;
      if ( v215 == v214 )
        sub_833AE0(&v213);
      v130 = (_QWORD *)(v213 + 8 * v128);
      if ( v130 )
        *v130 = v129;
      v131 = a7;
      v215 = v128 + 1;
      while ( v131 )
      {
        v133 = sub_724DC0();
        v134 = v215;
        if ( v215 == v214 )
        {
          v208 = v133;
          sub_833AE0(&v213);
          v133 = v208;
        }
        v132 = (_QWORD *)(v213 + 8 * v134);
        if ( v132 )
          *v132 = v133;
        v215 = v134 + 1;
        if ( !*v131 )
          break;
        if ( *(_BYTE *)(*v131 + 8LL) == 3 )
          v131 = (_QWORD *)sub_6BBB10(v131);
        else
          v131 = (_QWORD *)*v131;
      }
      for ( ii = *(_QWORD *)(v189 + 152); *(_BYTE *)(ii + 140) == 12; ii = *(_QWORD *)(ii + 160) )
        ;
      v202 = sub_736C60(19, *(__int64 **)(ii + 104));
      if ( !v202 )
        goto LABEL_413;
      v206 = v115;
      while ( 1 )
      {
        v211 = 1;
        v145 = v202[4];
        if ( !v145 || *(_BYTE *)(v145 + 10) != 5 )
        {
LABEL_412:
          v115 = v206;
          v25 = v186;
          v210 = 1;
          goto LABEL_413;
        }
        v146 = *(__m128i **)(v145 + 40);
        if ( v200 )
        {
          v147 = *(_QWORD *)(v189 + 248);
          sub_892150(&v220);
          v148 = *(_BYTE *)(*(_QWORD *)v147 + 80LL);
          v149 = *(_QWORD *)(*(_QWORD *)v147 + 88LL);
          if ( v148 == 20 )
          {
            v151 = **(_QWORD **)(v149 + 328);
          }
          else
          {
            v150 = v148 == 21 ? *(__int64 **)(v149 + 232) : *(__int64 **)(v149 + 32);
            v151 = *v150;
          }
          v146 = (__m128i *)sub_743530(v146, v200, v151, 0, &v210, v220.m128i_i64);
          if ( v210 )
          {
LABEL_425:
            v115 = v206;
            v25 = v186;
LABEL_413:
            for ( jj = v215; v215; --v215 )
            {
              sub_724E30(v213 + 8 * jj - 8);
              jj = v215 - 1;
            }
            v164 = v210;
            sub_823A00(v216, 8 * v217, v141, v142, v143, v144);
            a2 = 8 * v214;
            sub_823A00(v213, 8 * v214, v165, v166, v167, v168);
            if ( v164 || !v209 )
              goto LABEL_417;
LABEL_293:
            if ( v204 )
            {
              a2 = v182;
              v67 = (__int64 *)a1;
              sub_82B9D0(a1, v182, v25, a3, (__int64)v212, v115, a21);
              if ( a11 )
                *(_BYTE *)(*a21 + 145) |= 0x40u;
            }
            else
            {
              if ( a5 )
              {
                v68 = qword_4D03C68;
                if ( qword_4D03C68 )
                  qword_4D03C68 = (_QWORD *)*qword_4D03C68;
                else
                  v68 = (_QWORD *)sub_823970(152);
                *v68 = 0;
                v68[18] = 0;
                v116 = (char *)((unsigned __int64)(v68 + 1) & 0xFFFFFFFFFFFFFFF8LL);
                v117 = (unsigned int)((_DWORD)v68 - (_DWORD)v116 + 152) >> 3;
                memset(v116, 0, 8LL * v117);
                v67 = (__int64 *)&v116[8 * v117];
                v68[15] = v115;
                v68[7] = a5;
                *v68 = *a21;
                *a21 = (__int64)v68;
              }
              else
              {
                a2 = v182;
                v67 = (__int64 *)a1;
                sub_82B8E0(a1, v182, v115, a21);
              }
              if ( a11 )
              {
                v118 = *a21;
                *(_BYTE *)(v118 + 145) |= 0x40u;
                *(_QWORD *)(v118 + 64) = v189;
              }
            }
            if ( a20 == 9 )
              *(_BYTE *)(*a21 + 145) |= 2u;
            if ( v180 )
              *(_BYTE *)(*a21 + 145) |= 0x10u;
            goto LABEL_134;
          }
        }
        v198 = v145 + 24;
        if ( (unsigned int)sub_794BC0((__int64)v146, (__int64)&v216, v145 + 24, &v211) && v211 )
          goto LABEL_379;
        v152 = *(_BYTE *)(v186 + 140);
        if ( v206 )
          break;
        if ( v152 == 12 )
        {
          v153 = 0;
          v154 = 1;
          goto LABEL_394;
        }
LABEL_428:
        v157 = a7;
LABEL_429:
        if ( v157 )
        {
          for ( kk = 16; ; kk += 8 )
          {
            sub_72C970(*(_QWORD *)(v213 + kk));
            v170 = *v157;
            if ( !*v157 )
              break;
            if ( *(_BYTE *)(v170 + 8) == 3 )
            {
              v170 = sub_6BBB10(v157);
              if ( !v170 )
                break;
            }
            v157 = (__int64 *)v170;
          }
        }
LABEL_423:
        if ( !(unsigned int)sub_794BC0((__int64)v146, (__int64)&v213, v198, &v209) )
          goto LABEL_412;
        if ( v210 )
          goto LABEL_425;
LABEL_379:
        v202 = sub_736C60(19, (__int64 *)*v202);
        if ( !v202 )
          goto LABEL_425;
      }
      v153 = (_QWORD *)v206;
      if ( *(_BYTE *)(v206 + 15) )
        v153 = *(_QWORD **)v206;
      v154 = a7 == 0 || v153 == 0;
      if ( v152 == 12 )
      {
LABEL_394:
        v155 = v186;
        do
          v155 = *(_QWORD *)(v155 + 160);
        while ( *(_BYTE *)(v155 + 140) == 12 );
      }
      else
      {
        v155 = v186;
      }
      v156 = **(_QWORD ***)(v155 + 168);
      if ( !v156 )
        goto LABEL_428;
      v157 = a7;
      if ( v154 )
        goto LABEL_428;
      while ( 1 )
      {
        v160 = v157[3];
        v159 = v213;
        v161 = *(_BYTE *)(v160 + 24);
        if ( v161 == 2 )
        {
          v162 = *(__m128i **)(v213 + 8);
          *v162 = _mm_loadu_si128((const __m128i *)(v160 + 152));
          v162[1] = _mm_loadu_si128((const __m128i *)(v160 + 168));
          v162[2] = _mm_loadu_si128((const __m128i *)(v160 + 184));
          v162[3] = _mm_loadu_si128((const __m128i *)(v160 + 200));
          v162[4] = _mm_loadu_si128((const __m128i *)(v160 + 216));
          v162[5] = _mm_loadu_si128((const __m128i *)(v160 + 232));
          v162[6] = _mm_loadu_si128((const __m128i *)(v160 + 248));
          v162[7] = _mm_loadu_si128((const __m128i *)(v160 + 264));
          v162[8] = _mm_loadu_si128((const __m128i *)(v160 + 280));
          v162[9] = _mm_loadu_si128((const __m128i *)(v160 + 296));
          v162[10] = _mm_loadu_si128((const __m128i *)(v160 + 312));
          v162[11] = _mm_loadu_si128((const __m128i *)(v160 + 328));
          v162[12] = _mm_loadu_si128((const __m128i *)(v160 + 344));
        }
        else if ( v161 != 1
               || (v158 = sub_719770(*(_QWORD *)(v160 + 152), *(_QWORD *)(v213 + 8), 1u, 0), v159 = v213, !v158) )
        {
          sub_72C970(*(_QWORD *)(v159 + 8));
          goto LABEL_404;
        }
        sub_6E6A50(*(_QWORD *)(v213 + 8), (__int64)&v220);
        v196 = (_QWORD *)sub_6E3060(&v220);
        sub_848800(v196, v156, v153 + 6, 167, &v220);
        sub_6E1990(v196);
        if ( v221 == 2 )
        {
          v169 = *(__m128i **)(v213 + 8);
          *v169 = _mm_loadu_si128(&v222);
          v169[1] = _mm_loadu_si128(&v223);
          v169[2] = _mm_loadu_si128(&v224);
          v169[3] = _mm_loadu_si128(&v225);
          v169[4] = _mm_loadu_si128(&v226);
          v169[5] = _mm_loadu_si128(&v227);
          v169[6] = _mm_loadu_si128(&v228);
          v169[7] = _mm_loadu_si128(&v229);
          v169[8] = _mm_loadu_si128(&v230);
          v169[9] = _mm_loadu_si128(&v231);
          v169[10] = _mm_loadu_si128(&v232);
          v169[11] = _mm_loadu_si128(&v233);
          v169[12] = _mm_loadu_si128(v234);
        }
        else if ( v221 != 1 || !(unsigned int)sub_719770(v222.m128i_i64[0], *(_QWORD *)(v213 + 8), 1u, 0) )
        {
          sub_72C970(*(_QWORD *)(v213 + 8));
        }
LABEL_404:
        if ( !*v157 )
          goto LABEL_423;
        if ( *(_BYTE *)(*v157 + 8) == 3 )
          v157 = (__int64 *)sub_6BBB10(v157);
        else
          v157 = (__int64 *)*v157;
        v153 = (_QWORD *)*v153;
        v156 = (_QWORD *)*v156;
        if ( v156 == 0 || v153 == 0 )
          goto LABEL_429;
        if ( !v157 )
          goto LABEL_423;
      }
    }
    v54 = *(_QWORD *)(v201 + 88);
    v55 = *(unsigned int *)(v54 + 392);
    if ( v55 > unk_4D042F0 )
    {
      sub_861C90();
      v42 = v203;
      goto LABEL_130;
    }
    if ( (*(_BYTE *)(v54 + 160) & 4) != 0 )
    {
      v42 = v203;
      goto LABEL_130;
    }
    v56 = v186;
    v57 = 0;
    v58 = *(_QWORD *)(v186 + 168);
    *(_DWORD *)(v54 + 392) = v55 + 1;
    v59 = *(_QWORD **)v58;
    v60 = (__int64)a7;
    v220.m128i_i64[0] = (__int64)a7;
    if ( v59 )
      break;
LABEL_145:
    v70 = qword_4F5F830;
    v71 = (__int64 *)v212;
    if ( qword_4F5F830 )
      qword_4F5F830 = *(_QWORD *)qword_4F5F830;
    else
      v70 = sub_822B10(32, a2, v58, v56, v27, v28);
    *(_BYTE *)(v70 + 28) &= ~1u;
    *(_QWORD *)(v70 + 16) = v71;
    *(_QWORD *)(v70 + 8) = v201;
    v72 = qword_4F5F838;
    *(_DWORD *)(v70 + 24) = 0;
    *(_QWORD *)v70 = v72;
    qword_4F5F838 = v70;
    if ( v71 )
    {
      *(_DWORD *)(v70 + 24) = sub_894790(v201, v71);
      v73 = *(__int64 **)v70;
      if ( v73 )
      {
        v74 = 0;
        v75 = v54;
        v76 = 0;
        while ( 1 )
        {
          if ( v73[1] == v201 && v73[2] )
          {
            if ( !v74 )
            {
              v76 = sub_894790(v201, v71);
              v100 = qword_4F5F838;
              *(_BYTE *)(qword_4F5F838 + 28) |= 1u;
              *(_DWORD *)(v100 + 24) = v76;
            }
            if ( (*((_BYTE *)v73 + 28) & 1) != 0 )
            {
              v77 = *((_DWORD *)v73 + 6);
            }
            else
            {
              v77 = sub_894790(v73[1], v73[2]);
              *((_BYTE *)v73 + 28) |= 1u;
              *((_DWORD *)v73 + 6) = v77;
            }
            v74 = 1;
            if ( v76 == v77 )
            {
              v60 = v73[2];
              v78 = sub_89AB40(v60, v71, (*(_BYTE *)(*(_QWORD *)(v201 + 88) + 160LL) & 0x10) == 0 ? 16 : 48);
              v74 = 1;
              if ( v78 )
                break;
            }
          }
          v73 = (__int64 *)*v73;
          if ( !v73 )
          {
            v54 = v75;
            goto LABEL_195;
          }
        }
        v79 = (__int64 *)qword_4F5F838;
        v54 = v75;
        qword_4F5F838 = *(_QWORD *)qword_4F5F838;
        v58 = qword_4F5F830;
        qword_4F5F830 = (__int64)v79;
        *v79 = v58;
        goto LABEL_128;
      }
    }
LABEL_195:
    if ( v212 && (v60 = v201, a2 = v54, (v186 = sub_894B30(v201, v54, v212, 0x20000, 0)) != 0) )
    {
      v88 = (__int64 *)qword_4F5F838;
      qword_4F5F838 = *(_QWORD *)qword_4F5F838;
      v89 = qword_4F5F830;
      qword_4F5F830 = (__int64)v88;
      *v88 = v89;
    }
    else
    {
      if ( v203 )
        qword_4D03C50 = 0;
      else
        sub_865900(v201);
      a2 = v201;
      v60 = (__int64)&v212;
      v186 = sub_8B2240(&v212, v201, 0, v179, 0);
      v90 = (__int64 *)qword_4F5F838;
      qword_4F5F838 = *(_QWORD *)qword_4F5F838;
      v58 = qword_4F5F830;
      qword_4F5F830 = (__int64)v90;
      *v90 = v58;
      if ( !v186 )
        goto LABEL_129;
      v203 = 1;
    }
    v91 = *(_QWORD *)(*(_QWORD *)(v201 + 88) + 176LL);
    v58 = *(unsigned __int8 *)(v91 + 174);
    if ( (_BYTE)v58 == 1 )
    {
      v58 = *(_QWORD *)(v186 + 168);
      v120 = *(_QWORD **)v58;
      if ( *(_QWORD *)v58 )
      {
        v60 = v120[1];
        a2 = *(_QWORD *)(*(_QWORD *)(v91 + 40) + 32LL);
        if ( (v60 == a2 || (unsigned int)sub_8D97D0(v60, a2, 32, v186, v87)) && (!*v120 || a7 && !*a7) )
          goto LABEL_128;
      }
    }
    else if ( (_BYTE)v58 == 5 )
    {
      v60 = *(unsigned __int8 *)(v91 + 176);
      v119 = 0;
      if ( (*(_BYTE *)(v91 + 89) & 4) != 0 )
        v119 = *(_QWORD *)(*(_QWORD *)(v91 + 40) + 32LL);
      a2 = v186;
      if ( (unsigned int)sub_645720(v60, v186, v119, 0) )
        goto LABEL_128;
    }
    if ( v203 )
      sub_864110(v60, a2, v58);
    --*(_DWORD *)(v54 + 392);
    if ( unk_4F04C48 != -1 )
    {
      a2 = v201;
      v92 = qword_4F04C68[0] + 776LL * unk_4F04C48;
      if ( (*(_BYTE *)(v92 + 13) & 4) != 0 && *(_QWORD *)(v92 + 368) == v201 )
      {
        a2 = *(_QWORD *)(v92 + 376);
        if ( (unsigned int)sub_89AB40(v212, a2, (*(_BYTE *)(*(_QWORD *)(v201 + 88) + 160LL) & 0x10) == 0 ? 16 : 48) )
        {
          v203 = 0;
          goto LABEL_132;
        }
      }
    }
    v28 = v191;
    if ( v191 && (a20 != 7 || !(unsigned int)sub_8291E0(v189)) )
    {
      a2 = (__int64)a7;
      if ( sub_829250(v186, a7, v189) )
      {
        v101 = v188;
        v203 = 0;
        if ( !v188 )
          goto LABEL_132;
LABEL_242:
        a2 = v201 + 48;
        sub_686E10(0xCFAu, (FILE *)(v201 + 48), v201, v101);
        goto LABEL_132;
      }
    }
    v27 = dword_4D04494;
    if ( dword_4D04494 && (_DWORD)qword_4F077B4 && qword_4F077A0 )
    {
      v93 = *(_QWORD *)(v201 + 88);
      v94 = *(unsigned int *)(v93 + 392);
      if ( v94 > unk_4D042F0 )
      {
        sub_861C90();
        v203 = 0;
        goto LABEL_132;
      }
      v95 = v201;
      a2 = (__int64)v212;
      *(_DWORD *)(v93 + 392) = v94 + 1;
      v96 = *(_BYTE *)(v201 + 80);
      if ( v96 == 16 )
      {
        v95 = **(_QWORD **)(v201 + 88);
        v96 = *(_BYTE *)(v95 + 80);
      }
      if ( v96 == 24 )
      {
        v95 = *(_QWORD *)(v95 + 88);
        v96 = *(_BYTE *)(v95 + 80);
      }
      if ( (unsigned __int8)(v96 - 10) <= 1u )
      {
        v121 = *(_QWORD *)(v95 + 88);
        if ( (*(_BYTE *)(v121 + 194) & 0x40) != 0 )
        {
          do
            v121 = *(_QWORD *)(v121 + 232);
          while ( (*(_BYTE *)(v121 + 194) & 0x40) != 0 );
          goto LABEL_317;
        }
      }
      else if ( v96 == 20 )
      {
        v178 = *(_QWORD *)(*(_QWORD *)(v95 + 88) + 176LL);
        if ( (*(_BYTE *)(v178 + 194) & 0x40) != 0 )
        {
          do
            v178 = *(_QWORD *)(v178 + 232);
          while ( (*(_BYTE *)(v178 + 194) & 0x40) != 0 );
          v121 = *(_QWORD *)(v178 + 248);
LABEL_317:
          v95 = *(_QWORD *)v121;
        }
      }
      v203 = sub_8A00C0(v95, a2, 0);
      if ( !v203 )
      {
        --*(_DWORD *)(v93 + 392);
        if ( v188 )
        {
          a2 = v201 + 48;
          sub_686E10(0xCF9u, (FILE *)(v201 + 48), v201, v188);
        }
        goto LABEL_132;
      }
      --*(_DWORD *)(v93 + 392);
    }
    v97 = v186;
    if ( *(_BYTE *)(v186 + 140) == 12 )
    {
      do
        v97 = *(_QWORD *)(v97 + 160);
      while ( *(_BYTE *)(v97 + 140) == 12 );
      v186 = v97;
    }
    v190 = *(__int64 **)(v186 + 168);
    if ( a12 | a19 & 0x400000 )
    {
      if ( *(char *)(*(_QWORD *)(v186 + 168) + 20LL) < 0 )
      {
        a2 = *(_QWORD *)(v186 + 104);
        v113 = *(_QWORD *)(sub_736C60(84, (__int64 *)a2)[4] + 40);
        if ( *(_BYTE *)(v113 + 173) != 12 )
        {
          v203 = sub_711520(v113, a2, v111, v112, v27);
          if ( !v203 )
            goto LABEL_132;
        }
      }
    }
    v203 = 0;
    v43 = (__int64 *)*v190;
  }
  v61 = v59;
  v62 = v54;
  v63 = (__int64)v61;
  if ( !a7 )
  {
LABEL_124:
    v64 = v63;
    v42 = 0;
    v54 = v62;
    v65 = (_QWORD *)v64;
    if ( (*(_WORD *)(v64 + 32) & 0x104) == 0 )
      goto LABEL_128;
    if ( (*(_BYTE *)(v64 + 33) & 1) != 0 )
    {
      v66 = *(_QWORD *)v64;
      if ( *v65 )
      {
        if ( (*(_WORD *)(v66 + 32) & 0x104) == 0 )
          goto LABEL_128;
      }
    }
    goto LABEL_145;
  }
  while ( 1 )
  {
    if ( *(_BYTE *)(v60 + 8) == 2 )
    {
      while ( *(_QWORD *)v60 )
      {
        v58 = *(unsigned __int8 *)(*(_QWORD *)v60 + 8LL);
        if ( (_BYTE)v58 == 3 )
        {
          v69 = sub_6BBB10((_QWORD *)v60);
          v58 = *(unsigned __int8 *)(v69 + 8);
          v60 = v69;
        }
        else
        {
          v60 = *(_QWORD *)v60;
        }
        v220.m128i_i64[0] = v60;
        if ( (_BYTE)v58 != 2 )
          goto LABEL_120;
      }
      v220.m128i_i64[0] = 0;
LABEL_484:
      BUG();
    }
LABEL_120:
    if ( (*(_BYTE *)(v63 + 33) & 1) != 0 && *(_QWORD *)v63 )
    {
      v63 = *(_QWORD *)v63;
      v57 = 1;
      goto LABEL_123;
    }
    a2 = (__int64)&v220;
    v60 = v63;
    if ( !(unsigned int)sub_83D790(v63, &v220, 0, v201, &v212) )
    {
      v42 = 0;
      v54 = v62;
      goto LABEL_128;
    }
    v63 = *(_QWORD *)v63;
    if ( !v63 )
      break;
LABEL_123:
    v60 = v220.m128i_i64[0];
    if ( !v220.m128i_i64[0] )
      goto LABEL_124;
  }
  v42 = 0;
  v54 = v62;
  if ( !v220.m128i_i64[0] || !v57 )
    goto LABEL_145;
LABEL_128:
  a2 = v203;
  if ( v203 )
  {
LABEL_129:
    sub_864110(v60, a2, v58);
    --*(_DWORD *)(v54 + 392);
  }
  else
  {
    --*(_DWORD *)(v54 + 392);
    v42 = 0;
  }
LABEL_130:
  v203 = v42;
  if ( v188 )
  {
    a2 = v201 + 48;
    sub_686E10(0xCFFu, (FILE *)(v201 + 48), v201, v188);
  }
LABEL_132:
  sub_82D8A0(v197);
LABEL_133:
  v67 = (__int64 *)v212;
  sub_725130(v212->m128i_i64);
LABEL_134:
  if ( v184 )
  {
    *a7 = 0;
    v67 = a7;
    sub_6E1990(a7);
  }
  if ( v203 )
    sub_864110(v67, a2, v68);
}
