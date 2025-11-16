// Function: sub_100C5A0
// Address: 0x100c5a0
//
__int64 __fastcall sub_100C5A0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, unsigned __int8 a5, __int64 *a6)
{
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 result; // rax
  __int64 v11; // rax
  __int64 *v12; // r15
  __int64 v13; // rcx
  __int64 *v14; // rbx
  bool v15; // al
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 *v18; // r8
  bool v19; // al
  __int64 v20; // rdx
  __int64 v21; // rcx
  bool v22; // al
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 *v25; // rcx
  bool v26; // al
  __int64 v27; // rdx
  __int64 v28; // rbx
  __int64 *v29; // rax
  _QWORD *v30; // rdx
  __int64 v31; // rax
  int v32; // ecx
  char v33; // dl
  int v34; // eax
  __int64 v35; // rdi
  int v36; // eax
  bool v37; // al
  __int64 v38; // rdi
  bool v39; // al
  __int64 v40; // rdi
  int v41; // eax
  bool v42; // al
  __int64 v43; // rdi
  int v44; // eax
  bool v45; // al
  __int64 v46; // rdx
  _BYTE *v47; // rax
  bool v48; // dl
  _BYTE *v49; // rax
  _BYTE *v50; // rax
  char *v51; // rax
  __int64 v52; // rax
  unsigned __int8 *v53; // rdx
  __int64 v54; // rax
  __int64 *v55; // rsi
  __int64 v56; // rax
  __int64 *v57; // rdx
  bool v58; // al
  _BYTE *v59; // r9
  __int64 *v60; // r8
  bool v61; // al
  __int64 v62; // r9
  __int64 v63; // rdx
  _BYTE *v64; // r9
  bool v65; // al
  __int64 v66; // r9
  __int64 v67; // rdx
  __int64 v68; // rax
  unsigned __int8 *v69; // rdx
  unsigned __int8 *v70; // rax
  _BYTE *v71; // rdx
  __int64 *v72; // rax
  char v73; // al
  _BYTE *v74; // rax
  int v75; // eax
  unsigned int v76; // ecx
  __int64 v77; // rdi
  int v78; // eax
  __int64 v79; // rsi
  __int64 v80; // rax
  unsigned int v81; // edx
  __int64 v82; // rsi
  bool v83; // bl
  __int64 v84; // rax
  unsigned int v85; // ebx
  int v86; // eax
  bool v87; // dl
  __int64 v88; // rsi
  bool v89; // bl
  __int64 v90; // rax
  unsigned int v91; // ebx
  int v92; // eax
  bool v93; // dl
  unsigned __int8 *v94; // rbx
  unsigned int v95; // r12d
  char v96; // r13
  __int64 v97; // rax
  unsigned int v98; // r13d
  char v99; // r8
  _BYTE *v100; // r12
  __int64 v101; // rdi
  int v102; // eax
  bool v103; // al
  unsigned int v104; // ecx
  __int64 v105; // rdi
  int v106; // eax
  bool v107; // al
  int v108; // edx
  __int64 v109; // rax
  unsigned __int8 *v110; // rdx
  unsigned __int8 *v111; // rdx
  unsigned __int8 *v112; // rdx
  signed __int64 v113; // rdx
  __int64 v114; // rdi
  _BYTE *v115; // rax
  unsigned __int8 *v116; // r9
  bool v117; // cl
  int v118; // eax
  bool v119; // al
  _BYTE *v120; // rax
  unsigned __int8 *v121; // rcx
  bool v122; // r8
  int v123; // eax
  __int64 v124; // rdi
  _BYTE *v125; // rax
  unsigned __int8 *v126; // r9
  bool v127; // cl
  unsigned int v128; // r9d
  __int64 v129; // rax
  unsigned int v130; // r9d
  int v131; // eax
  unsigned int v132; // r9d
  __int64 v133; // rax
  unsigned int v134; // r9d
  int v135; // eax
  unsigned int v136; // ebx
  unsigned __int8 *v137; // r12
  __int64 v138; // rax
  unsigned __int8 *v139; // rdx
  signed __int64 v140; // rax
  char v141; // al
  char v142; // al
  char v143; // al
  unsigned __int8 *v144; // rdi
  int v145; // eax
  int v146; // eax
  bool v147; // al
  int v148; // eax
  unsigned __int64 v149; // r12
  __int64 v150; // rax
  unsigned __int8 *v151; // rdi
  int v152; // eax
  int v153; // eax
  unsigned __int8 **v154; // rax
  unsigned __int64 v155; // rax
  __int64 *v156; // [rsp+8h] [rbp-118h]
  __int64 *v157; // [rsp+8h] [rbp-118h]
  __int64 v158; // [rsp+10h] [rbp-110h]
  __int64 v159; // [rsp+10h] [rbp-110h]
  bool v160; // [rsp+10h] [rbp-110h]
  unsigned int v161; // [rsp+10h] [rbp-110h]
  bool v162; // [rsp+10h] [rbp-110h]
  unsigned int v163; // [rsp+10h] [rbp-110h]
  bool v164; // [rsp+10h] [rbp-110h]
  int v165; // [rsp+10h] [rbp-110h]
  __int64 *v166; // [rsp+18h] [rbp-108h]
  __int64 *v167; // [rsp+18h] [rbp-108h]
  __int64 *v168; // [rsp+18h] [rbp-108h]
  unsigned int v169; // [rsp+18h] [rbp-108h]
  int v170; // [rsp+18h] [rbp-108h]
  unsigned int v171; // [rsp+18h] [rbp-108h]
  int v172; // [rsp+18h] [rbp-108h]
  int v173; // [rsp+18h] [rbp-108h]
  int v174; // [rsp+20h] [rbp-100h]
  int v175; // [rsp+20h] [rbp-100h]
  int v176; // [rsp+20h] [rbp-100h]
  int v177; // [rsp+20h] [rbp-100h]
  unsigned __int8 *v178; // [rsp+28h] [rbp-F8h]
  unsigned __int8 *v179; // [rsp+28h] [rbp-F8h]
  unsigned __int8 *v180; // [rsp+28h] [rbp-F8h]
  _QWORD *v181; // [rsp+28h] [rbp-F8h]
  unsigned __int8 *v182; // [rsp+28h] [rbp-F8h]
  _QWORD *v183; // [rsp+28h] [rbp-F8h]
  int v184; // [rsp+28h] [rbp-F8h]
  int v185; // [rsp+28h] [rbp-F8h]
  unsigned int v186; // [rsp+30h] [rbp-F0h]
  __int64 v187; // [rsp+30h] [rbp-F0h]
  __int64 *v188; // [rsp+30h] [rbp-F0h]
  __int64 v189; // [rsp+30h] [rbp-F0h]
  __int64 v190; // [rsp+30h] [rbp-F0h]
  __int64 v191; // [rsp+30h] [rbp-F0h]
  __int64 v192; // [rsp+30h] [rbp-F0h]
  __int64 v193; // [rsp+30h] [rbp-F0h]
  __int64 v194; // [rsp+30h] [rbp-F0h]
  __int64 v195; // [rsp+30h] [rbp-F0h]
  __int64 v196; // [rsp+38h] [rbp-E8h]
  __int64 v197; // [rsp+38h] [rbp-E8h]
  __int64 v198; // [rsp+38h] [rbp-E8h]
  __int64 v199; // [rsp+38h] [rbp-E8h]
  unsigned int v200; // [rsp+38h] [rbp-E8h]
  __int64 v201; // [rsp+38h] [rbp-E8h]
  __int64 v202; // [rsp+38h] [rbp-E8h]
  __int64 *v203; // [rsp+38h] [rbp-E8h]
  __int64 *v204; // [rsp+38h] [rbp-E8h]
  unsigned __int8 *v205; // [rsp+38h] [rbp-E8h]
  unsigned __int8 *v206; // [rsp+38h] [rbp-E8h]
  __int64 v207; // [rsp+38h] [rbp-E8h]
  __int64 *v208; // [rsp+40h] [rbp-E0h]
  __int64 v209; // [rsp+40h] [rbp-E0h]
  __int64 v210; // [rsp+48h] [rbp-D8h]
  __int64 *v211; // [rsp+48h] [rbp-D8h]
  unsigned int v212; // [rsp+48h] [rbp-D8h]
  __int64 *v214; // [rsp+58h] [rbp-C8h]
  __int64 *v215; // [rsp+58h] [rbp-C8h]
  char v216; // [rsp+58h] [rbp-C8h]
  __int64 v217; // [rsp+58h] [rbp-C8h]
  __int64 *v218; // [rsp+58h] [rbp-C8h]
  __int64 v219; // [rsp+58h] [rbp-C8h]
  __int64 v220; // [rsp+58h] [rbp-C8h]
  __int64 v221; // [rsp+58h] [rbp-C8h]
  unsigned int v222; // [rsp+58h] [rbp-C8h]
  bool v223; // [rsp+58h] [rbp-C8h]
  __int64 *v224; // [rsp+58h] [rbp-C8h]
  __int64 v225; // [rsp+58h] [rbp-C8h]
  int v226; // [rsp+58h] [rbp-C8h]
  unsigned int v227; // [rsp+58h] [rbp-C8h]
  char v228; // [rsp+58h] [rbp-C8h]
  __int64 v229; // [rsp+58h] [rbp-C8h]
  unsigned __int8 *v230; // [rsp+58h] [rbp-C8h]
  int v231; // [rsp+58h] [rbp-C8h]
  int v232; // [rsp+58h] [rbp-C8h]
  unsigned __int8 *v233; // [rsp+58h] [rbp-C8h]
  int v234; // [rsp+58h] [rbp-C8h]
  int v235; // [rsp+58h] [rbp-C8h]
  __int64 v236; // [rsp+58h] [rbp-C8h]
  __int64 *v237; // [rsp+58h] [rbp-C8h]
  __int64 *v238; // [rsp+58h] [rbp-C8h]
  __int64 *v239; // [rsp+58h] [rbp-C8h]
  _BYTE *v240; // [rsp+58h] [rbp-C8h]
  _BYTE *v241; // [rsp+60h] [rbp-C0h]
  __int64 v242; // [rsp+60h] [rbp-C0h]
  __int64 v243; // [rsp+60h] [rbp-C0h]
  __int64 v244; // [rsp+60h] [rbp-C0h]
  __int64 v245; // [rsp+60h] [rbp-C0h]
  int v246; // [rsp+60h] [rbp-C0h]
  __int64 v247; // [rsp+60h] [rbp-C0h]
  int v248; // [rsp+60h] [rbp-C0h]
  int v249; // [rsp+60h] [rbp-C0h]
  int v250; // [rsp+60h] [rbp-C0h]
  __int64 v251; // [rsp+60h] [rbp-C0h]
  int v252; // [rsp+60h] [rbp-C0h]
  __int64 v253; // [rsp+60h] [rbp-C0h]
  __int64 v254; // [rsp+60h] [rbp-C0h]
  int v255; // [rsp+60h] [rbp-C0h]
  __int64 *v256; // [rsp+60h] [rbp-C0h]
  __int64 *v257; // [rsp+60h] [rbp-C0h]
  __int64 *v259; // [rsp+70h] [rbp-B0h]
  __int64 v261; // [rsp+78h] [rbp-A8h]
  __int64 v262; // [rsp+78h] [rbp-A8h]
  __int64 v263; // [rsp+78h] [rbp-A8h]
  __int64 v264; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v265; // [rsp+88h] [rbp-98h]
  __int64 v266; // [rsp+90h] [rbp-90h] BYREF
  unsigned int v267; // [rsp+98h] [rbp-88h]
  __int64 *v268; // [rsp+A0h] [rbp-80h] BYREF
  __int64 *v269; // [rsp+A8h] [rbp-78h]
  __int64 v270; // [rsp+B0h] [rbp-70h]
  __int64 v271; // [rsp+C0h] [rbp-60h] BYREF
  unsigned __int8 *v272; // [rsp+C8h] [rbp-58h] BYREF
  __int64 *v273; // [rsp+D0h] [rbp-50h]
  unsigned int v274; // [rsp+D8h] [rbp-48h]
  char v275; // [rsp+E0h] [rbp-40h]

  v8 = a2;
  v9 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
    v9 = **(_QWORD **)(v9 + 16);
  if ( !a4 )
    return v8;
  v186 = *(_DWORD *)(v9 + 8) >> 8;
  v11 = sub_B4DC50(a1, (__int64)a3, a4);
  v12 = *(__int64 **)(a2 + 8);
  v196 = v11;
  v210 = 8 * a4;
  v259 = &a3[a4];
  if ( (unsigned int)*((unsigned __int8 *)v12 + 8) - 17 > 1 && a3 != &a3[a4] )
  {
    v30 = a3;
    v31 = *(_QWORD *)(*a3 + 8LL);
    v32 = *(unsigned __int8 *)(v31 + 8);
    if ( v32 == 17 )
    {
LABEL_34:
      v33 = 0;
    }
    else
    {
      while ( v32 != 18 )
      {
        if ( v259 == ++v30 )
          goto LABEL_7;
        v31 = *(_QWORD *)(*v30 + 8LL);
        v32 = *(unsigned __int8 *)(v31 + 8);
        if ( v32 == 17 )
          goto LABEL_34;
      }
      v33 = 1;
    }
    v34 = *(_DWORD *)(v31 + 32);
    BYTE4(v265) = v33;
    LODWORD(v265) = v34;
    v12 = (__int64 *)sub_BCE1B0(v12, v265);
    if ( v12 != *(__int64 **)(a2 + 8) )
      goto LABEL_19;
  }
LABEL_7:
  v13 = v210 >> 3;
  if ( v210 >> 5 <= 0 )
  {
    v14 = a3;
    goto LABEL_145;
  }
  v14 = a3;
  v208 = &a3[4 * (v210 >> 5)];
  do
  {
    if ( *(_BYTE *)*v14 > 0x15u )
      goto LABEL_18;
    v241 = (_BYTE *)*v14;
    v15 = sub_AC30F0(*v14);
    v17 = (__int64)v241;
    if ( !v15 )
    {
      if ( *v241 == 17 )
      {
        if ( *((_DWORD *)v241 + 8) <= 0x40u )
        {
          v39 = *((_QWORD *)v241 + 3) == 0;
        }
        else
        {
          v38 = (__int64)(v241 + 24);
          v248 = *((_DWORD *)v241 + 8);
          v39 = v248 == (unsigned int)sub_C444A0(v38);
        }
      }
      else
      {
        v251 = *((_QWORD *)v241 + 1);
        if ( (unsigned int)*(unsigned __int8 *)(v251 + 8) - 17 > 1 )
          goto LABEL_18;
        v178 = (unsigned __int8 *)v17;
        v47 = sub_AD7630(v17, 0, v16);
        v48 = 0;
        if ( !v47 || *v47 != 17 )
        {
          if ( *(_BYTE *)(v251 + 8) != 17 )
            goto LABEL_18;
          v174 = *(_DWORD *)(v251 + 32);
          if ( !v174 )
            goto LABEL_18;
          v79 = 0;
          do
          {
            v223 = v48;
            v80 = sub_AD69F0(v178, v79);
            if ( !v80 )
              goto LABEL_18;
            v48 = v223;
            if ( *(_BYTE *)v80 != 13 )
            {
              if ( *(_BYTE *)v80 != 17 )
                goto LABEL_18;
              v81 = *(_DWORD *)(v80 + 32);
              v48 = v81 <= 0x40 ? *(_QWORD *)(v80 + 24) == 0 : v81 == (unsigned int)sub_C444A0(v80 + 24);
              if ( !v48 )
                goto LABEL_18;
            }
            v79 = (unsigned int)(v79 + 1);
          }
          while ( v174 != (_DWORD)v79 );
          if ( !v48 )
            goto LABEL_18;
          goto LABEL_10;
        }
        if ( *((_DWORD *)v47 + 8) <= 0x40u )
        {
          v39 = *((_QWORD *)v47 + 3) == 0;
        }
        else
        {
          v252 = *((_DWORD *)v47 + 8);
          v39 = v252 == (unsigned int)sub_C444A0((__int64)(v47 + 24));
        }
      }
      if ( !v39 )
        goto LABEL_18;
    }
LABEL_10:
    v18 = v14 + 1;
    if ( *(_BYTE *)v14[1] > 0x15u )
      goto LABEL_43;
    v214 = v14 + 1;
    v242 = v14[1];
    v19 = sub_AC30F0(v242);
    v21 = v242;
    v18 = v14 + 1;
    if ( v19 )
      goto LABEL_12;
    if ( *(_BYTE *)v242 == 17 )
    {
      if ( *(_DWORD *)(v242 + 32) > 0x40u )
      {
        v35 = v242 + 24;
        v246 = *(_DWORD *)(v242 + 32);
LABEL_41:
        v36 = sub_C444A0(v35);
        v18 = v214;
        v37 = v246 == v36;
        goto LABEL_42;
      }
      v37 = *(_QWORD *)(v242 + 24) == 0;
      goto LABEL_42;
    }
    v253 = *(_QWORD *)(v242 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v253 + 8) - 17 > 1 )
      goto LABEL_43;
    v179 = (unsigned __int8 *)v21;
    v49 = sub_AD7630(v21, 0, v20);
    v18 = v14 + 1;
    if ( v49 && *v49 == 17 )
    {
      if ( *((_DWORD *)v49 + 8) > 0x40u )
      {
        v214 = v14 + 1;
        v35 = (__int64)(v49 + 24);
        v246 = *((_DWORD *)v49 + 8);
        goto LABEL_41;
      }
      v37 = *((_QWORD *)v49 + 3) == 0;
LABEL_42:
      if ( !v37 )
        goto LABEL_43;
      goto LABEL_12;
    }
    if ( *(_BYTE *)(v253 + 8) != 17 )
      goto LABEL_43;
    v175 = *(_DWORD *)(v253 + 32);
    if ( !v175 )
      goto LABEL_43;
    v82 = 0;
    v166 = v14;
    v83 = 0;
    do
    {
      v224 = v18;
      v84 = sub_AD69F0(v179, v82);
      v18 = v224;
      if ( !v84 )
        goto LABEL_43;
      if ( *(_BYTE *)v84 != 13 )
      {
        if ( *(_BYTE *)v84 != 17 )
          goto LABEL_43;
        v85 = *(_DWORD *)(v84 + 32);
        if ( v85 <= 0x40 )
        {
          v83 = *(_QWORD *)(v84 + 24) == 0;
        }
        else
        {
          v86 = sub_C444A0(v84 + 24);
          v18 = v224;
          v83 = v85 == v86;
        }
        if ( !v83 )
          goto LABEL_43;
      }
      v82 = (unsigned int)(v82 + 1);
    }
    while ( v175 != (_DWORD)v82 );
    v87 = v83;
    v14 = v166;
    if ( !v87 )
    {
LABEL_43:
      v14 = v18;
      goto LABEL_18;
    }
LABEL_12:
    v18 = v14 + 2;
    if ( *(_BYTE *)v14[2] > 0x15u )
      goto LABEL_43;
    v215 = v14 + 2;
    v243 = v14[2];
    v22 = sub_AC30F0(v243);
    v24 = v243;
    v18 = v14 + 2;
    if ( v22 )
      goto LABEL_14;
    if ( *(_BYTE *)v243 == 17 )
    {
      if ( *(_DWORD *)(v243 + 32) > 0x40u )
      {
        v40 = v243 + 24;
        v249 = *(_DWORD *)(v243 + 32);
LABEL_54:
        v41 = sub_C444A0(v40);
        v18 = v215;
        v42 = v249 == v41;
        goto LABEL_55;
      }
      v42 = *(_QWORD *)(v243 + 24) == 0;
    }
    else
    {
      v254 = *(_QWORD *)(v243 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v254 + 8) - 17 > 1 )
        goto LABEL_43;
      v180 = (unsigned __int8 *)v24;
      v50 = sub_AD7630(v24, 0, v23);
      v18 = v14 + 2;
      if ( !v50 || *v50 != 17 )
      {
        if ( *(_BYTE *)(v254 + 8) != 17 )
          goto LABEL_43;
        v176 = *(_DWORD *)(v254 + 32);
        if ( !v176 )
          goto LABEL_43;
        v88 = 0;
        v167 = v14;
        v89 = 0;
        do
        {
          v256 = v18;
          v90 = sub_AD69F0(v180, v88);
          v18 = v256;
          if ( !v90 )
            goto LABEL_43;
          if ( *(_BYTE *)v90 != 13 )
          {
            if ( *(_BYTE *)v90 != 17 )
              goto LABEL_43;
            v91 = *(_DWORD *)(v90 + 32);
            if ( v91 <= 0x40 )
            {
              v89 = *(_QWORD *)(v90 + 24) == 0;
            }
            else
            {
              v92 = sub_C444A0(v90 + 24);
              v18 = v256;
              v89 = v91 == v92;
            }
            if ( !v89 )
              goto LABEL_43;
          }
          v88 = (unsigned int)(v88 + 1);
        }
        while ( v176 != (_DWORD)v88 );
        v93 = v89;
        v14 = v167;
        if ( !v93 )
          goto LABEL_43;
        goto LABEL_14;
      }
      if ( *((_DWORD *)v50 + 8) > 0x40u )
      {
        v215 = v14 + 2;
        v40 = (__int64)(v50 + 24);
        v249 = *((_DWORD *)v50 + 8);
        goto LABEL_54;
      }
      v42 = *((_QWORD *)v50 + 3) == 0;
    }
LABEL_55:
    if ( !v42 )
    {
      v14 = v18;
      goto LABEL_18;
    }
LABEL_14:
    v25 = v14 + 3;
    if ( *(_BYTE *)v14[3] > 0x15u )
      goto LABEL_45;
    v244 = v14[3];
    v26 = sub_AC30F0(v244);
    v27 = v244;
    v25 = v14 + 3;
    if ( v26 )
      goto LABEL_16;
    if ( *(_BYTE *)v244 == 17 )
    {
      if ( *(_DWORD *)(v244 + 32) <= 0x40u )
      {
        v45 = *(_QWORD *)(v244 + 24) == 0;
      }
      else
      {
        v43 = v244 + 24;
        v250 = *(_DWORD *)(v244 + 32);
        v44 = sub_C444A0(v43);
        v25 = v14 + 3;
        v45 = v250 == v44;
      }
    }
    else
    {
      v247 = *(_QWORD *)(v244 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v247 + 8) - 17 > 1 )
        goto LABEL_45;
      v182 = (unsigned __int8 *)v27;
      v74 = sub_AD7630(v27, 0, v27);
      v25 = v14 + 3;
      if ( !v74 || *v74 != 17 )
      {
        if ( *(_BYTE *)(v247 + 8) != 17 )
          goto LABEL_45;
        v177 = *(_DWORD *)(v247 + 32);
        if ( !v177 )
          goto LABEL_45;
        v257 = v14 + 3;
        v168 = v14;
        v94 = v182;
        v225 = v8;
        v95 = 0;
        v183 = a3;
        v96 = 0;
        do
        {
          v97 = sub_AD69F0(v94, v95);
          if ( !v97 )
          {
LABEL_211:
            v25 = v257;
            v8 = v225;
            a3 = v183;
            goto LABEL_45;
          }
          if ( *(_BYTE *)v97 != 13 )
          {
            if ( *(_BYTE *)v97 != 17 )
              goto LABEL_211;
            v98 = *(_DWORD *)(v97 + 32);
            if ( v98 <= 0x40 )
            {
              if ( *(_QWORD *)(v97 + 24) )
                goto LABEL_211;
            }
            else if ( v98 != (unsigned int)sub_C444A0(v97 + 24) )
            {
              goto LABEL_211;
            }
            v96 = 1;
          }
          ++v95;
        }
        while ( v177 != v95 );
        v99 = v96;
        v14 = v168;
        v25 = v257;
        v8 = v225;
        a3 = v183;
        if ( !v99 )
        {
LABEL_45:
          v14 = v25;
          goto LABEL_18;
        }
        goto LABEL_16;
      }
      if ( *((_DWORD *)v74 + 8) <= 0x40u )
      {
        v45 = *((_QWORD *)v74 + 3) == 0;
      }
      else
      {
        v255 = *((_DWORD *)v74 + 8);
        v75 = sub_C444A0((__int64)(v74 + 24));
        v25 = v14 + 3;
        v45 = v255 == v75;
      }
    }
    if ( !v45 )
    {
      v14 = v25;
      goto LABEL_18;
    }
LABEL_16:
    v14 += 4;
  }
  while ( v208 != v14 );
  v13 = v259 - v14;
LABEL_145:
  switch ( v13 )
  {
    case 2LL:
LABEL_236:
      if ( (unsigned __int8)sub_FFFD30(*v14) )
      {
        ++v14;
LABEL_148:
        if ( (unsigned __int8)sub_FFFD30(*v14) )
          return v8;
      }
      break;
    case 3LL:
      if ( (unsigned __int8)sub_FFFD30(*v14) )
      {
        ++v14;
        goto LABEL_236;
      }
      break;
    case 1LL:
      goto LABEL_148;
    default:
      return v8;
  }
LABEL_18:
  if ( v14 == v259 )
    return v8;
LABEL_19:
  if ( *(_BYTE *)v8 == 13 )
    return sub_ACADE0((__int64 **)v12);
  v28 = v210 >> 5;
  v245 = v210 >> 3;
  if ( v210 >> 5 <= 0 )
  {
    v46 = v210 >> 3;
    v29 = a3;
LABEL_63:
    if ( v46 != 2 )
    {
      if ( v46 != 3 )
      {
        if ( v46 != 1 )
          goto LABEL_66;
        goto LABEL_141;
      }
      if ( *(_BYTE *)*v29 == 13 )
        goto LABEL_27;
      ++v29;
    }
    if ( *(_BYTE *)*v29 == 13 )
      goto LABEL_27;
    ++v29;
LABEL_141:
    if ( *(_BYTE *)*v29 == 13 )
      goto LABEL_27;
    goto LABEL_66;
  }
  v29 = a3;
  while ( 1 )
  {
    if ( *(_BYTE *)*v29 == 13 )
      goto LABEL_27;
    if ( *(_BYTE *)v29[1] == 13 )
    {
      ++v29;
      goto LABEL_27;
    }
    if ( *(_BYTE *)v29[2] == 13 )
    {
      v29 += 2;
      goto LABEL_27;
    }
    if ( *(_BYTE *)v29[3] == 13 )
      break;
    v29 += 4;
    if ( &a3[4 * (v210 >> 5)] == v29 )
    {
      v46 = v259 - v29;
      goto LABEL_63;
    }
  }
  v29 += 3;
LABEL_27:
  if ( v29 != v259 )
    return sub_ACADE0((__int64 **)v12);
LABEL_66:
  if ( (unsigned __int8)sub_1003090((__int64)a6, (unsigned __int8 *)v8) )
    return sub_ACA8A0((__int64 **)v12);
  if ( sub_BCEA30(a1) )
    goto LABEL_116;
  if ( v28 <= 0 )
  {
    v51 = (char *)a3;
LABEL_253:
    v113 = (char *)v259 - v51;
    if ( (char *)v259 - v51 != 16 )
    {
      if ( v113 != 24 )
      {
        if ( v113 != 8 )
          goto LABEL_95;
        goto LABEL_256;
      }
      if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v51 + 8LL) + 8LL) == 18 )
        goto LABEL_94;
      v51 += 8;
    }
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v51 + 8LL) + 8LL) == 18 )
      goto LABEL_94;
    v51 += 8;
LABEL_256:
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v51 + 8LL) + 8LL) == 18 )
      goto LABEL_94;
    goto LABEL_95;
  }
  v51 = (char *)a3;
  while ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v51 + 8LL) + 8LL) != 18 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(*((_QWORD *)v51 + 1) + 8LL) + 8LL) == 18 )
    {
      v51 += 8;
      break;
    }
    if ( *(_BYTE *)(*(_QWORD *)(*((_QWORD *)v51 + 2) + 8LL) + 8LL) == 18 )
    {
      v51 += 16;
      break;
    }
    if ( *(_BYTE *)(*(_QWORD *)(*((_QWORD *)v51 + 3) + 8LL) + 8LL) == 18 )
    {
      v51 += 24;
      break;
    }
    v51 += 32;
    if ( &a3[4 * v28] == (_QWORD *)v51 )
      goto LABEL_253;
  }
LABEL_94:
  if ( v51 != (char *)v259 )
    goto LABEL_116;
LABEL_95:
  if ( a4 == 1 )
  {
    if ( (v108 = *(unsigned __int8 *)(a1 + 8), (_BYTE)v108 == 12)
      || (unsigned __int8)v108 <= 3u
      || (_BYTE)v108 == 5
      || (v108 & 0xFD) == 4
      || (v108 & 0xFB) == 0xA
      || ((unsigned __int8)(*(_BYTE *)(a1 + 8) - 15) <= 3u || v108 == 20) && (unsigned __int8)sub_BCEBA0(a1, 0) )
    {
      v209 = *a6;
      v228 = sub_AE5020(*a6, a1);
      v109 = sub_9208B0(v209, a1);
      v272 = v110;
      v271 = ((1LL << v228) + ((unsigned __int64)(v109 + 7) >> 3) - 1) >> v228 << v228;
      v229 = sub_CA1930(&v271);
      if ( v229 )
      {
        v185 = sub_BCB060(*(_QWORD *)(*a3 + 8LL));
        if ( v185 != sub_AE2980(*a6, v186)[1] )
          goto LABEL_96;
        v139 = (unsigned __int8 *)*a3;
        v268 = v12;
        v269 = &v264;
        v270 = v8;
        if ( v229 == 1 )
        {
          v271 = (__int64)&v264;
          v272 = (unsigned __int8 *)v8;
          if ( sub_100C200((__int64)&v271, 15, v139) && sub_FFE530((__int64)&v268) )
            return v264;
        }
      }
      else
      {
        if ( *(__int64 **)(v8 + 8) == v12 )
          return v8;
        v184 = sub_BCB060(*(_QWORD *)(*a3 + 8LL));
        if ( sub_AE2980(*a6, v186)[1] != v184 )
          goto LABEL_96;
        v268 = v12;
        v269 = &v264;
        v270 = v8;
      }
      v111 = (unsigned __int8 *)*a3;
      v272 = (unsigned __int8 *)v8;
      v273 = &v266;
      v271 = (__int64)&v264;
      if ( !(unsigned __int8)sub_100C2B0((__int64)&v271, 27, v111) || v229 != 1LL << v266 || !sub_FFE530((__int64)&v268) )
      {
        v112 = (unsigned __int8 *)*a3;
        v272 = (unsigned __int8 *)v8;
        v271 = (__int64)&v264;
        v273 = (__int64 *)v229;
        if ( !sub_100C400((__int64)&v271, 20, v112) || !sub_FFE530((__int64)&v268) )
          goto LABEL_96;
      }
      return v264;
    }
  }
LABEL_96:
  v187 = *a6;
  v216 = sub_AE5020(*a6, v196);
  v52 = sub_9208B0(v187, v196);
  v272 = v53;
  v271 = ((1LL << v216) + ((unsigned __int64)(v52 + 7) >> 3) - 1) >> v216 << v216;
  if ( sub_CA1930(&v271) != 1 )
    goto LABEL_116;
  v54 = 8 * a4 - 8;
  v55 = (_QWORD *)((char *)a3 + v54);
  v56 = v54 >> 5;
  if ( v56 <= 0 )
  {
    v57 = a3;
LABEL_349:
    v140 = (char *)v55 - (char *)v57;
    if ( (char *)v55 - (char *)v57 != 16 )
    {
      if ( v140 != 24 )
      {
        if ( v140 != 8 )
          goto LABEL_108;
        goto LABEL_352;
      }
      v238 = v57;
      v142 = sub_FFFE90(*v57);
      v57 = v238;
      if ( !v142 )
        goto LABEL_107;
      v57 = v238 + 1;
    }
    v239 = v57;
    v143 = sub_FFFE90(*v57);
    v57 = v239;
    if ( !v143 )
      goto LABEL_107;
    v57 = v239 + 1;
LABEL_352:
    v237 = v57;
    v141 = sub_FFFE90(*v57);
    v57 = v237;
    if ( !v141 )
      goto LABEL_107;
    goto LABEL_108;
  }
  v57 = a3;
  v181 = &a3[4 * v56];
  while ( 1 )
  {
    if ( *(_BYTE *)*v57 > 0x15u )
      goto LABEL_107;
    v197 = (__int64)v57;
    v217 = *v57;
    v58 = sub_AC30F0(*v57);
    v57 = (__int64 *)v197;
    if ( !v58 )
      break;
LABEL_100:
    v59 = (_BYTE *)v57[1];
    v218 = v57;
    v60 = v57 + 1;
    if ( *v59 > 0x15u )
      goto LABEL_154;
    v188 = v57 + 1;
    v198 = v57[1];
    v61 = sub_AC30F0((__int64)v59);
    v62 = v198;
    v60 = v188;
    v63 = (__int64)v218;
    if ( v61 )
      goto LABEL_102;
    if ( *(_BYTE *)v198 == 17 )
    {
      v76 = *(_DWORD *)(v198 + 32);
      if ( v76 > 0x40 )
      {
        v77 = v198 + 24;
        v201 = (__int64)v218;
        v222 = v76;
        v78 = sub_C444A0(v77);
        v63 = v201;
        if ( v222 != v78 )
        {
          v60 = v188;
LABEL_154:
          v57 = v60;
          goto LABEL_107;
        }
        goto LABEL_102;
      }
      v119 = *(_QWORD *)(v198 + 24) == 0;
    }
    else
    {
      v158 = (__int64)v218;
      v190 = *(_QWORD *)(v198 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v190 + 8) - 17 > 1 )
        goto LABEL_154;
      v114 = v198;
      v203 = v60;
      v230 = (unsigned __int8 *)v62;
      v115 = sub_AD7630(v114, 0, v63);
      v116 = v230;
      v60 = v203;
      v117 = 0;
      v63 = v158;
      if ( !v115 || *v115 != 17 )
      {
        if ( *(_BYTE *)(v190 + 8) == 17 )
        {
          v235 = *(_DWORD *)(v190 + 32);
          if ( v235 )
          {
            v194 = v158;
            v206 = v116;
            v132 = 0;
            while ( 1 )
            {
              v157 = v60;
              v162 = v117;
              v171 = v132;
              v133 = sub_AD69F0(v206, v132);
              v60 = v157;
              if ( !v133 )
                break;
              v134 = v171;
              v117 = v162;
              if ( *(_BYTE *)v133 != 13 )
              {
                if ( *(_BYTE *)v133 != 17 )
                  break;
                if ( *(_DWORD *)(v133 + 32) <= 0x40u )
                {
                  v117 = *(_QWORD *)(v133 + 24) == 0;
                }
                else
                {
                  v163 = v171;
                  v172 = *(_DWORD *)(v133 + 32);
                  v135 = sub_C444A0(v133 + 24);
                  v134 = v163;
                  v60 = v157;
                  v117 = v172 == v135;
                }
                if ( !v117 )
                  break;
              }
              v132 = v134 + 1;
              if ( v235 == v132 )
              {
                v63 = v194;
                if ( v117 )
                  goto LABEL_102;
                goto LABEL_154;
              }
            }
          }
        }
        goto LABEL_154;
      }
      if ( *((_DWORD *)v115 + 8) <= 0x40u )
      {
        v119 = *((_QWORD *)v115 + 3) == 0;
      }
      else
      {
        v231 = *((_DWORD *)v115 + 8);
        v118 = sub_C444A0((__int64)(v115 + 24));
        v63 = v158;
        v60 = v203;
        v119 = v231 == v118;
      }
    }
    if ( !v119 )
    {
      v57 = v60;
      goto LABEL_107;
    }
LABEL_102:
    v64 = *(_BYTE **)(v63 + 16);
    v219 = v63;
    v60 = (__int64 *)(v63 + 16);
    if ( *v64 > 0x15u )
      goto LABEL_154;
    v189 = v63 + 16;
    v199 = *(_QWORD *)(v63 + 16);
    v65 = sub_AC30F0((__int64)v64);
    v66 = v199;
    v60 = (__int64 *)v189;
    v67 = v219;
    if ( v65 )
      goto LABEL_104;
    if ( *(_BYTE *)v199 == 17 )
    {
      v104 = *(_DWORD *)(v199 + 32);
      if ( v104 > 0x40 )
      {
        v105 = v199 + 24;
        v202 = v219;
        v227 = v104;
LABEL_229:
        v106 = sub_C444A0(v105);
        v67 = v202;
        v60 = (__int64 *)v189;
        v107 = v227 == v106;
        goto LABEL_230;
      }
      v107 = *(_QWORD *)(v199 + 24) == 0;
    }
    else
    {
      v159 = v219;
      v192 = *(_QWORD *)(v199 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v192 + 8) - 17 > 1 )
        goto LABEL_154;
      v124 = v199;
      v204 = v60;
      v233 = (unsigned __int8 *)v66;
      v125 = sub_AD7630(v124, 0, v67);
      v126 = v233;
      v60 = v204;
      v127 = 0;
      v67 = v159;
      if ( !v125 || *v125 != 17 )
      {
        if ( *(_BYTE *)(v192 + 8) == 17 )
        {
          v234 = *(_DWORD *)(v192 + 32);
          if ( v234 )
          {
            v193 = v159;
            v205 = v126;
            v128 = 0;
            while ( 1 )
            {
              v156 = v60;
              v160 = v127;
              v169 = v128;
              v129 = sub_AD69F0(v205, v128);
              v60 = v156;
              if ( !v129 )
                break;
              v130 = v169;
              v127 = v160;
              if ( *(_BYTE *)v129 != 13 )
              {
                if ( *(_BYTE *)v129 != 17 )
                  break;
                if ( *(_DWORD *)(v129 + 32) <= 0x40u )
                {
                  v127 = *(_QWORD *)(v129 + 24) == 0;
                }
                else
                {
                  v161 = v169;
                  v170 = *(_DWORD *)(v129 + 32);
                  v131 = sub_C444A0(v129 + 24);
                  v130 = v161;
                  v60 = v156;
                  v127 = v170 == v131;
                }
                if ( !v127 )
                  break;
              }
              v128 = v130 + 1;
              if ( v234 == v128 )
              {
                v67 = v193;
                if ( v127 )
                  goto LABEL_104;
                goto LABEL_154;
              }
            }
          }
        }
        goto LABEL_154;
      }
      if ( *((_DWORD *)v125 + 8) > 0x40u )
      {
        v189 = (__int64)v204;
        v105 = (__int64)(v125 + 24);
        v202 = v159;
        v227 = *((_DWORD *)v125 + 8);
        goto LABEL_229;
      }
      v107 = *((_QWORD *)v125 + 3) == 0;
    }
LABEL_230:
    if ( !v107 )
    {
      v57 = v60;
      goto LABEL_107;
    }
LABEL_104:
    v220 = v67;
    if ( !(unsigned __int8)sub_FFFE90(*(_QWORD *)(v67 + 24)) )
    {
      v57 = (__int64 *)(v220 + 24);
      goto LABEL_107;
    }
    v57 = (__int64 *)(v220 + 32);
    if ( v181 == (_QWORD *)(v220 + 32) )
      goto LABEL_349;
  }
  if ( *(_BYTE *)v217 == 17 )
  {
    if ( *(_DWORD *)(v217 + 32) <= 0x40u )
    {
      v103 = *(_QWORD *)(v217 + 24) == 0;
    }
    else
    {
      v101 = v217 + 24;
      v226 = *(_DWORD *)(v217 + 32);
      v102 = sub_C444A0(v101);
      v57 = (__int64 *)v197;
      v103 = v226 == v102;
    }
    goto LABEL_224;
  }
  v191 = *(_QWORD *)(v217 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v191 + 8) - 17 > 1 )
    goto LABEL_107;
  v120 = sub_AD7630(v217, 0, v197);
  v121 = (unsigned __int8 *)v217;
  v57 = (__int64 *)v197;
  v122 = 0;
  if ( v120 && *v120 == 17 )
  {
    if ( *((_DWORD *)v120 + 8) <= 0x40u )
    {
      if ( *((_QWORD *)v120 + 3) )
        goto LABEL_107;
      goto LABEL_100;
    }
    v232 = *((_DWORD *)v120 + 8);
    v123 = sub_C444A0((__int64)(v120 + 24));
    v57 = (__int64 *)v197;
    v103 = v232 == v123;
LABEL_224:
    if ( !v103 )
      goto LABEL_107;
    goto LABEL_100;
  }
  if ( *(_BYTE *)(v191 + 8) == 17 )
  {
    v173 = *(_DWORD *)(v191 + 32);
    if ( v173 )
    {
      v236 = v28;
      v136 = 0;
      v195 = v8;
      v137 = v121;
      do
      {
        v164 = v122;
        v138 = sub_AD69F0(v137, v136);
        if ( !v138
          || (v122 = v164, *(_BYTE *)v138 != 13)
          && (*(_BYTE *)v138 != 17
           || (*(_DWORD *)(v138 + 32) <= 0x40u
             ? (v122 = *(_QWORD *)(v138 + 24) == 0)
             : (v165 = *(_DWORD *)(v138 + 32), v122 = v165 == (unsigned int)sub_C444A0(v138 + 24)),
               !v122)) )
        {
          v28 = v236;
          v57 = (__int64 *)v197;
          v8 = v195;
          goto LABEL_107;
        }
        ++v136;
      }
      while ( v173 != v136 );
      v28 = v236;
      v57 = (__int64 *)v197;
      v8 = v195;
      if ( v122 )
        goto LABEL_100;
    }
  }
LABEL_107:
  if ( v55 != v57 )
    goto LABEL_116;
LABEL_108:
  v68 = *(_QWORD *)(v8 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v68 + 8) - 17 <= 1 )
    v68 = **(_QWORD **)(v68 + 16);
  v200 = sub_AE2980(*a6, *(_DWORD *)(v68 + 8) >> 8)[3];
  v221 = (__int64)&a3[(unsigned __int64)v210 / 8 - 1];
  v271 = sub_9208B0(*a6, *(_QWORD *)(*(_QWORD *)v221 + 8LL));
  v272 = v69;
  if ( sub_CA1930(&v271) != v200 )
    goto LABEL_116;
  v267 = v200;
  if ( v200 > 0x40 )
    sub_C43690((__int64)&v266, 0, 0);
  else
    v266 = 0;
  v70 = sub_BD45C0((unsigned __int8 *)v8, *a6, (__int64)&v266, 0, 0, 0, 0, 0);
  v271 = 0;
  v272 = v70;
  v211 = (__int64 *)v70;
  v71 = *(_BYTE **)v221;
  if ( **(_BYTE **)v221 != 44 )
  {
LABEL_114:
    v272 = 0;
    v271 = (__int64)v211;
    if ( *v71 != 59 )
      goto LABEL_115;
    v144 = (unsigned __int8 *)*((_QWORD *)v71 - 8);
    v145 = *v144;
    if ( (unsigned __int8)v145 <= 0x1Cu )
    {
      if ( (_BYTE)v145 == 5 )
      {
        v146 = *((unsigned __int16 *)v144 + 1);
        goto LABEL_369;
      }
      goto LABEL_115;
    }
    v146 = v145 - 29;
LABEL_369:
    if ( v146 == 47 )
    {
      v240 = v71;
      if ( v211 == *(__int64 **)sub_986520((__int64)v144) )
      {
        if ( (unsigned __int8)sub_995B10((_QWORD **)&v272, *((_QWORD *)v240 - 4)) )
        {
          if ( v267 <= 0x40 )
          {
            v147 = v266 == 1;
          }
          else
          {
            v212 = v267;
            v147 = v212 - 1 == (unsigned int)sub_C444A0((__int64)&v266);
          }
          if ( !v147 )
          {
            sub_9865C0((__int64)&v268, (__int64)&v266);
            sub_C46F20((__int64)&v268, 1u);
            v148 = (int)v269;
            LODWORD(v269) = 0;
            LODWORD(v272) = v148;
            v271 = (__int64)v268;
            v149 = sub_ACCFD0((__int64 *)*v12, (__int64)&v271);
            sub_969240(&v271);
            sub_969240((__int64 *)&v268);
            v150 = sub_AD4C70(v149, (__int64 **)v12, 0);
            goto LABEL_376;
          }
        }
      }
    }
LABEL_115:
    sub_969240(&v266);
LABEL_116:
    if ( *(_BYTE *)v8 > 0x15u )
      return 0;
    if ( v28 > 0 )
    {
      v72 = a3;
      while ( *(_BYTE *)*v72 <= 0x15u )
      {
        if ( *(_BYTE *)v72[1] > 0x15u )
        {
          ++v72;
          break;
        }
        if ( *(_BYTE *)v72[2] > 0x15u )
        {
          v72 += 2;
          break;
        }
        if ( *(_BYTE *)v72[3] > 0x15u )
        {
          v72 += 3;
          break;
        }
        v72 += 4;
        if ( &a3[4 * v28] == v72 )
        {
          v245 = v259 - v72;
          goto LABEL_215;
        }
      }
LABEL_124:
      if ( v72 == v259 )
      {
LABEL_125:
        v73 = sub_BCEA30(a1);
        v275 = 0;
        if ( v73 )
        {
          result = sub_AAB960(a1, (unsigned __int8 *)v8, (__int64)&v271, a3, a4);
          if ( v275 )
          {
            v275 = 0;
            if ( v274 > 0x40 && v273 )
            {
              v261 = result;
              j_j___libc_free_0_0(v273);
              result = v261;
            }
            if ( (unsigned int)v272 > 0x40 )
            {
              if ( v271 )
              {
                v262 = result;
                j_j___libc_free_0_0(v271);
                return v262;
              }
            }
          }
        }
        else
        {
          v100 = (_BYTE *)sub_AD9FD0(a1, (unsigned __int8 *)v8, a3, a4, a5, (__int64)&v271, 0);
          if ( v275 )
          {
            v275 = 0;
            if ( v274 > 0x40 && v273 )
              j_j___libc_free_0_0(v273);
            if ( (unsigned int)v272 > 0x40 && v271 )
              j_j___libc_free_0_0(v271);
          }
          return sub_97B670(v100, *a6, 0);
        }
        return result;
      }
      return 0;
    }
    v72 = a3;
LABEL_215:
    if ( v245 != 2 )
    {
      if ( v245 != 3 )
      {
        if ( v245 != 1 )
          goto LABEL_125;
        goto LABEL_218;
      }
      if ( *(_BYTE *)*v72 > 0x15u )
        goto LABEL_124;
      ++v72;
    }
    if ( *(_BYTE *)*v72 > 0x15u )
      goto LABEL_124;
    ++v72;
LABEL_218:
    if ( *(_BYTE *)*v72 > 0x15u )
      goto LABEL_124;
    goto LABEL_125;
  }
  v207 = *(_QWORD *)v221;
  if ( !(unsigned __int8)sub_10081F0((__int64 **)&v271, *((_QWORD *)v71 - 8)) )
  {
LABEL_384:
    v71 = *(_BYTE **)v221;
    goto LABEL_114;
  }
  v151 = *(unsigned __int8 **)(v207 - 32);
  v152 = *v151;
  if ( (unsigned __int8)v152 > 0x1Cu )
  {
    v153 = v152 - 29;
  }
  else
  {
    if ( (_BYTE)v152 != 5 )
      goto LABEL_384;
    v153 = *((unsigned __int16 *)v151 + 1);
  }
  if ( v153 != 47 )
    goto LABEL_384;
  v154 = (unsigned __int8 **)sub_986520((__int64)v151);
  if ( *v154 != v272 || sub_9867B0((__int64)&v266) )
    goto LABEL_384;
  v155 = sub_ACCFD0((__int64 *)*v12, (__int64)&v266);
  v150 = sub_AD4C70(v155, (__int64 **)v12, 0);
LABEL_376:
  v263 = v150;
  sub_969240(&v266);
  return v263;
}
