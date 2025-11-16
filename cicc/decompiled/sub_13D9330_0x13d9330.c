// Function: sub_13D9330
// Address: 0x13d9330
//
__int64 **__fastcall sub_13D9330(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, unsigned int a5)
{
  __int64 **v5; // r12
  __int64 v6; // rbx
  __int64 **result; // rax
  __int64 **v8; // r13
  __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rdi
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 *v16; // rdi
  unsigned __int8 v17; // al
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // rdi
  __int64 v21; // rax
  int v22; // eax
  unsigned __int8 v23; // al
  unsigned __int8 v24; // dl
  unsigned int v25; // r15d
  __int64 v26; // rdi
  __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // r14
  __int64 *v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rcx
  unsigned int v34; // r8d
  __int64 v35; // r11
  __int64 v36; // rdx
  __int64 v37; // rcx
  unsigned int v38; // r8d
  __int64 **v39; // rax
  __int64 v40; // rax
  unsigned int v41; // r15d
  bool v42; // al
  unsigned __int8 v43; // al
  unsigned int v44; // r15d
  bool v45; // al
  __int64 v46; // rax
  unsigned int v47; // r15d
  int v48; // eax
  __int64 **v49; // rax
  __int64 *v50; // rax
  int v51; // edx
  int v52; // edx
  __int64 *v53; // rax
  __int64 v54; // rax
  __int64 v55; // rcx
  __int64 v56; // rax
  unsigned int v57; // r15d
  unsigned __int8 v58; // al
  __int64 **v59; // rax
  __int64 v60; // r10
  char v61; // al
  _QWORD *v62; // rdx
  __int64 *v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r8
  int v66; // eax
  char v67; // al
  __int64 v68; // rsi
  unsigned int v69; // eax
  unsigned int v70; // r15d
  __int64 v71; // rax
  char v72; // dl
  bool v73; // al
  unsigned int v74; // r15d
  __int64 v75; // rax
  char v76; // dl
  bool v77; // al
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 **v82; // rsi
  __int64 **v83; // rdx
  __int64 **v84; // rdi
  __int64 v85; // rcx
  int v86; // eax
  int v87; // eax
  unsigned int v88; // r9d
  __int64 v89; // rax
  __int64 **v90; // rsi
  __int64 **v91; // rdx
  __int64 **v92; // rdi
  __int64 v93; // rcx
  int v94; // eax
  unsigned __int8 v95; // al
  unsigned int v96; // eax
  unsigned int v97; // eax
  __int64 v98; // rax
  __int64 v99; // rax
  __int64 **v100; // rsi
  __int64 **v101; // rdx
  __int64 **v102; // rdi
  __int64 v103; // rcx
  int v104; // eax
  int v105; // eax
  unsigned int v106; // r9d
  __int64 v107; // rax
  __int64 *v108; // rdx
  __int64 v109; // rax
  __int64 v110; // rax
  _QWORD *v111; // rsi
  _QWORD *v112; // rdx
  _QWORD *v113; // rdi
  __int64 v114; // rcx
  int v115; // eax
  bool v116; // zf
  __int64 v117; // rax
  __int64 v118; // rsi
  __int64 v119; // rdx
  __int64 v120; // rdi
  __int64 v121; // rcx
  int v122; // eax
  __int64 v123; // r15
  int v124; // eax
  unsigned __int8 v125; // al
  _QWORD *v126; // rdx
  unsigned __int8 v127; // al
  __int64 v128; // rax
  __int64 **v129; // rsi
  __int64 **v130; // rdx
  __int64 **v131; // rdi
  __int64 v132; // rcx
  int v133; // eax
  __int64 v134; // rax
  __int64 **v135; // rsi
  __int64 **v136; // rdx
  __int64 **v137; // rdi
  __int64 v138; // rcx
  int v139; // eax
  __int64 v140; // r15
  __int64 v141; // rax
  unsigned int v142; // eax
  __int64 v143; // rax
  _QWORD *v144; // rsi
  _QWORD *v145; // rdx
  _QWORD *v146; // rdi
  __int64 v147; // rcx
  int v148; // eax
  __int64 v149; // rax
  __int64 v150; // rsi
  __int64 v151; // rdx
  __int64 v152; // rdi
  __int64 v153; // rcx
  int v154; // eax
  __int64 v155; // rax
  _QWORD *v156; // rsi
  _QWORD *v157; // rdx
  _QWORD *v158; // rdi
  __int64 v159; // rcx
  int v160; // eax
  __int64 v161; // rax
  __int64 **v162; // rsi
  __int64 **v163; // rdx
  __int64 **v164; // rdi
  __int64 v165; // rcx
  int v166; // eax
  __int64 v167; // rax
  _QWORD *v168; // rsi
  _QWORD *v169; // rdx
  _QWORD *v170; // rdi
  __int64 v171; // rcx
  int v172; // eax
  __int64 v173; // rax
  __int64 v174; // rsi
  __int64 v175; // rdx
  __int64 v176; // rdi
  __int64 v177; // rcx
  int v178; // eax
  __int64 v179; // rax
  __int64 **v180; // rsi
  __int64 **v181; // rdx
  __int64 **v182; // rdi
  __int64 v183; // rcx
  int v184; // eax
  __int64 **v185; // rdi
  __int64 v186; // rax
  __int64 v187; // rax
  __int64 **v188; // rax
  __int64 v189; // rax
  __int64 *v190; // r14
  __int64 *v191; // rax
  __int64 v192; // rax
  __int64 **v193; // rdx
  __int64 **v194; // rcx
  __int64 **v195; // rax
  int v196; // r14d
  int v197; // eax
  __int64 *v198; // rdi
  __int64 v199; // rax
  int v200; // edx
  __int64 *v201; // rdi
  __int64 v202; // rax
  __int64 v203; // r8
  __int64 v204; // r9
  __int64 v205; // rax
  __int64 v206; // rax
  __int64 v207; // [rsp+8h] [rbp-138h]
  __int64 v208; // [rsp+18h] [rbp-128h]
  unsigned __int8 v209; // [rsp+20h] [rbp-120h]
  __int64 v210; // [rsp+20h] [rbp-120h]
  __int64 v211; // [rsp+20h] [rbp-120h]
  __int64 v212; // [rsp+20h] [rbp-120h]
  _QWORD *v213; // [rsp+28h] [rbp-118h]
  __int64 *v214; // [rsp+28h] [rbp-118h]
  unsigned int v215; // [rsp+28h] [rbp-118h]
  unsigned int v216; // [rsp+28h] [rbp-118h]
  int v217; // [rsp+28h] [rbp-118h]
  int v218; // [rsp+28h] [rbp-118h]
  unsigned int v219; // [rsp+28h] [rbp-118h]
  _QWORD *v220; // [rsp+28h] [rbp-118h]
  __int64 v221; // [rsp+30h] [rbp-110h]
  int v222; // [rsp+30h] [rbp-110h]
  int v223; // [rsp+30h] [rbp-110h]
  __int64 v224; // [rsp+30h] [rbp-110h]
  __int64 v225; // [rsp+30h] [rbp-110h]
  __int64 v226; // [rsp+30h] [rbp-110h]
  __int64 v227; // [rsp+30h] [rbp-110h]
  __int64 v228; // [rsp+30h] [rbp-110h]
  __int64 v229; // [rsp+30h] [rbp-110h]
  __int64 v230; // [rsp+30h] [rbp-110h]
  __int64 v231; // [rsp+30h] [rbp-110h]
  __int64 v232; // [rsp+30h] [rbp-110h]
  __int64 v233; // [rsp+30h] [rbp-110h]
  __int64 v234; // [rsp+30h] [rbp-110h]
  __int64 v235; // [rsp+30h] [rbp-110h]
  __int64 v236; // [rsp+30h] [rbp-110h]
  __int64 v237; // [rsp+30h] [rbp-110h]
  __int64 v238; // [rsp+38h] [rbp-108h]
  __int64 v239; // [rsp+38h] [rbp-108h]
  __int64 *v240; // [rsp+38h] [rbp-108h]
  char v241; // [rsp+38h] [rbp-108h]
  unsigned int v242; // [rsp+38h] [rbp-108h]
  _QWORD *v243; // [rsp+38h] [rbp-108h]
  __int64 v244; // [rsp+38h] [rbp-108h]
  unsigned int v245; // [rsp+38h] [rbp-108h]
  __int64 **v246; // [rsp+38h] [rbp-108h]
  __int64 **v247; // [rsp+38h] [rbp-108h]
  _QWORD *v248; // [rsp+38h] [rbp-108h]
  __int64 v249; // [rsp+38h] [rbp-108h]
  __int64 **v250; // [rsp+38h] [rbp-108h]
  __int64 **v251; // [rsp+38h] [rbp-108h]
  _QWORD *v252; // [rsp+38h] [rbp-108h]
  _QWORD *v253; // [rsp+38h] [rbp-108h]
  __int64 v254; // [rsp+38h] [rbp-108h]
  __int64 **v255; // [rsp+38h] [rbp-108h]
  __int64 **v256; // [rsp+38h] [rbp-108h]
  __int64 **v257; // [rsp+38h] [rbp-108h]
  __int64 v259; // [rsp+40h] [rbp-100h]
  __int64 v261; // [rsp+50h] [rbp-F0h] BYREF
  unsigned int v262; // [rsp+58h] [rbp-E8h]
  __int64 *v263; // [rsp+60h] [rbp-E0h] BYREF
  unsigned int v264; // [rsp+68h] [rbp-D8h]
  _QWORD **v265; // [rsp+70h] [rbp-D0h] BYREF
  unsigned int v266; // [rsp+78h] [rbp-C8h]
  __int64 v267; // [rsp+80h] [rbp-C0h] BYREF
  unsigned int v268; // [rsp+88h] [rbp-B8h]
  __int64 *v269; // [rsp+90h] [rbp-B0h] BYREF
  unsigned int v270; // [rsp+98h] [rbp-A8h]
  __int64 v271; // [rsp+A0h] [rbp-A0h] BYREF
  unsigned int v272; // [rsp+A8h] [rbp-98h]
  __int64 *v273; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v274; // [rsp+B8h] [rbp-88h]
  __int64 v275; // [rsp+C0h] [rbp-80h] BYREF
  unsigned int v276; // [rsp+C8h] [rbp-78h]
  __int64 *v277; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v278; // [rsp+E8h] [rbp-58h]
  __int64 v279; // [rsp+F0h] [rbp-50h] BYREF
  unsigned int v280; // [rsp+F8h] [rbp-48h]

  v5 = (__int64 **)a3;
  v6 = (unsigned int)a1;
  if ( *(_BYTE *)(a2 + 16) > 0x10u )
  {
    v8 = (__int64 **)a2;
  }
  else
  {
    if ( *(_BYTE *)(a3 + 16) <= 0x10u )
      return (__int64 **)sub_14D7760(a1, a2, a3, *a4, a4[1]);
    v8 = (__int64 **)a3;
    v6 = (unsigned int)sub_15FF5D0(a1);
    v5 = (__int64 **)a2;
  }
  v9 = **v8;
  if ( *((_BYTE *)*v8 + 8) == 16 )
  {
    v10 = (*v8)[4];
    v11 = sub_1643320(v9);
    v12 = sub_16463B0(v11, (unsigned int)v10);
  }
  else
  {
    v12 = sub_1643320(v9);
  }
  if ( v8 == v5 || *((_BYTE *)v5 + 16) == 9 )
  {
    v17 = sub_15FF820((unsigned int)v6);
    return (__int64 **)sub_15A0680(v12, v17, 0);
  }
  v13 = **v8;
  if ( *((_BYTE *)*v8 + 8) == 16 )
  {
    v14 = (*v8)[4];
    v15 = sub_1643320(v13);
    v238 = sub_16463B0(v15, (unsigned int)v14);
  }
  else
  {
    v238 = sub_1643320(v13);
  }
  v16 = *v8;
  if ( *((_BYTE *)*v8 + 8) == 16 )
    v16 = *(__int64 **)v16[2];
  if ( !(unsigned __int8)sub_1642F90(v16, 1) )
    goto LABEL_31;
  if ( *((_BYTE *)v5 + 16) > 0x10u )
    goto LABEL_21;
  if ( (unsigned __int8)sub_1593BB0(v5) )
  {
LABEL_16:
    switch ( (int)v6 )
    {
      case '!':
      case '"':
      case '(':
        return v8;
      case '#':
      case ')':
        goto LABEL_25;
      case '$':
      case '&':
        goto LABEL_33;
      default:
        goto LABEL_21;
    }
  }
  if ( *((_BYTE *)v5 + 16) == 13 )
  {
    v41 = *((_DWORD *)v5 + 8);
    if ( v41 <= 0x40 )
      v42 = v5[3] == 0;
    else
      v42 = v41 == (unsigned int)sub_16A57B0(v5 + 3);
    goto LABEL_86;
  }
  if ( *((_BYTE *)*v5 + 8) != 16 )
    goto LABEL_21;
  v46 = sub_15A1020(v5);
  if ( v46 && *(_BYTE *)(v46 + 16) == 13 )
  {
    v47 = *(_DWORD *)(v46 + 32);
    if ( v47 <= 0x40 )
      v42 = *(_QWORD *)(v46 + 24) == 0;
    else
      v42 = v47 == (unsigned int)sub_16A57B0(v46 + 24);
LABEL_86:
    if ( v42 )
      goto LABEL_16;
    goto LABEL_87;
  }
  v223 = (*v5)[4];
  if ( !v223 )
    goto LABEL_16;
  v74 = 0;
  while ( 1 )
  {
    v75 = sub_15A0A60(v5, v74);
    if ( !v75 )
      break;
    v76 = *(_BYTE *)(v75 + 16);
    if ( v76 != 9 )
    {
      if ( v76 != 13 )
        break;
      if ( *(_DWORD *)(v75 + 32) <= 0x40u )
      {
        v77 = *(_QWORD *)(v75 + 24) == 0;
      }
      else
      {
        v218 = *(_DWORD *)(v75 + 32);
        v77 = v218 == (unsigned int)sub_16A57B0(v75 + 24);
      }
      if ( !v77 )
        break;
    }
    if ( v223 == ++v74 )
      goto LABEL_16;
  }
LABEL_87:
  v43 = *((_BYTE *)v5 + 16);
  if ( v43 == 13 )
  {
    v44 = *((_DWORD *)v5 + 8);
    if ( v44 <= 0x40 )
    {
      if ( v5[3] == (__int64 *)1 )
      {
LABEL_91:
        switch ( (int)v6 )
        {
          case ' ':
          case '#':
          case ')':
            return v8;
          case '"':
          case '(':
LABEL_33:
            v18 = sub_15A0640(v238);
            goto LABEL_26;
          case '%':
          case '\'':
            goto LABEL_25;
          default:
            goto LABEL_21;
        }
      }
      goto LABEL_21;
    }
    v45 = v44 - 1 == (unsigned int)sub_16A57B0(v5 + 3);
    goto LABEL_90;
  }
  if ( *((_BYTE *)*v5 + 8) == 16 && v43 <= 0x10u )
  {
    v56 = sub_15A1020(v5);
    if ( v56 && *(_BYTE *)(v56 + 16) == 13 )
    {
      v57 = *(_DWORD *)(v56 + 32);
      if ( v57 <= 0x40 )
        v45 = *(_QWORD *)(v56 + 24) == 1;
      else
        v45 = v57 - 1 == (unsigned int)sub_16A57B0(v56 + 24);
LABEL_90:
      if ( v45 )
        goto LABEL_91;
      goto LABEL_21;
    }
    v70 = 0;
    v222 = (*v5)[4];
    if ( !v222 )
      goto LABEL_91;
    while ( 1 )
    {
      v71 = sub_15A0A60(v5, v70);
      if ( !v71 )
        break;
      v72 = *(_BYTE *)(v71 + 16);
      if ( v72 != 9 )
      {
        if ( v72 != 13 )
          break;
        if ( *(_DWORD *)(v71 + 32) <= 0x40u )
        {
          v73 = *(_QWORD *)(v71 + 24) == 1;
        }
        else
        {
          v217 = *(_DWORD *)(v71 + 32);
          v73 = v217 - 1 == (unsigned int)sub_16A57B0(v71 + 24);
        }
        if ( !v73 )
          break;
      }
      if ( v222 == ++v70 )
        goto LABEL_91;
    }
  }
LABEL_21:
  if ( (_DWORD)v6 == 37 || (_DWORD)v6 == 39 )
  {
    sub_14BCF40(&v277, v8, v5, *a4, 1, 0);
    if ( !BYTE1(v277) )
      goto LABEL_31;
  }
  else
  {
    if ( (_DWORD)v6 != 35 )
      goto LABEL_31;
    sub_14BCF40(&v277, v5, v8, *a4, 1, 0);
    if ( !BYTE1(v277) )
      goto LABEL_31;
  }
  if ( (_BYTE)v277 )
  {
LABEL_25:
    v18 = sub_15A0600(v238);
LABEL_26:
    if ( v18 )
      return (__int64 **)v18;
  }
LABEL_31:
  v19 = sub_13CF9C0(v6, v8, (__int64)v5, a4);
  if ( v19 )
    return (__int64 **)v19;
  v20 = **v5;
  if ( *((_BYTE *)*v5 + 8) == 16 )
  {
    v239 = (*v5)[4];
    v21 = sub_1643320(v20);
    v221 = sub_16463B0(v21, v239);
  }
  else
  {
    v221 = sub_1643320(v20);
  }
  v22 = *((unsigned __int8 *)v8 + 16);
  if ( (unsigned __int8)v22 > 0x17u )
  {
    v48 = v22 - 24;
  }
  else
  {
    if ( (_BYTE)v22 != 5 )
      goto LABEL_40;
    v48 = *((unsigned __int16 *)v8 + 9);
  }
  if ( v48 == 47 )
  {
    if ( (*((_BYTE *)v8 + 23) & 0x40) != 0 )
      v49 = (__int64 **)*(v8 - 1);
    else
      v49 = &v8[-3 * (*((_DWORD *)v8 + 5) & 0xFFFFFFF)];
    v50 = *v49;
    v51 = *((unsigned __int8 *)v50 + 16);
    if ( (unsigned __int8)v51 > 0x17u )
    {
      v52 = v51 - 24;
    }
    else
    {
      if ( (_BYTE)v51 != 5 )
        goto LABEL_40;
      v52 = *((unsigned __int16 *)v50 + 9);
    }
    if ( v52 == 41 )
    {
      v53 = (*((_BYTE *)v50 + 23) & 0x40) != 0 ? (__int64 *)*(v50 - 1) : &v50[-3 * (*((_DWORD *)v50 + 5) & 0xFFFFFFF)];
      if ( *v53 )
      {
        if ( (_DWORD)v6 == 40 )
        {
          if ( sub_13CD190((__int64)v5) )
          {
            v19 = sub_15A0640(v221);
            goto LABEL_50;
          }
        }
        else if ( (_DWORD)v6 == 38 && (unsigned __int8)sub_13CC520((__int64)v5) )
        {
          v19 = sub_15A0600(v221);
          goto LABEL_50;
        }
      }
    }
  }
LABEL_40:
  v23 = *((_BYTE *)v5 + 16);
  if ( v23 == 13 )
  {
    v213 = v5 + 3;
    goto LABEL_42;
  }
  if ( *((_BYTE *)*v5 + 8) == 16 && v23 <= 0x10u )
  {
    v54 = sub_15A1020(v5);
    if ( !v54 || *(_BYTE *)(v54 + 16) != 13 )
    {
LABEL_51:
      v23 = *((_BYTE *)v5 + 16);
      goto LABEL_52;
    }
    v213 = (_QWORD *)(v54 + 24);
LABEL_42:
    sub_158B890(&v265, (unsigned int)v6, v213);
    if ( (unsigned __int8)sub_158A120(&v265) )
    {
      v19 = sub_15A0640(v221);
      goto LABEL_44;
    }
    if ( (unsigned __int8)sub_158A0B0(&v265) )
    {
      v19 = sub_15A0600(v221);
LABEL_44:
      if ( v268 > 0x40 && v267 )
        j_j___libc_free_0_0(v267);
      if ( v266 > 0x40 && v265 )
        j_j___libc_free_0_0(v265);
LABEL_50:
      if ( v19 )
        return (__int64 **)v19;
      goto LABEL_51;
    }
    v65 = *((unsigned int *)v213 + 2);
    v262 = v65;
    if ( (unsigned int)v65 > 0x40 )
    {
      v219 = v65;
      sub_16A4EF0(&v261, 0, 0);
      v264 = v219;
      sub_16A4EF0(&v263, 0, 0);
      v66 = *((unsigned __int8 *)v8 + 16);
      v65 = v219;
      if ( (unsigned __int8)v66 <= 0x17u )
        goto LABEL_152;
    }
    else
    {
      v66 = *((unsigned __int8 *)v8 + 16);
      v261 = 0;
      v264 = v65;
      v263 = 0;
      if ( (unsigned __int8)v66 <= 0x17u )
        goto LABEL_217;
    }
    if ( (unsigned int)(v66 - 35) <= 0x11 )
    {
      v215 = v65;
      sub_13D26A0((__int64)v8, (__int64)&v261, (__int64)&v263);
      v65 = v215;
    }
LABEL_152:
    if ( v262 <= 0x40 )
    {
      if ( (__int64 *)v261 != v263 )
        goto LABEL_154;
    }
    else
    {
      v216 = v65;
      v67 = sub_16A5220(&v261, &v263);
      v65 = v216;
      if ( !v67 )
      {
LABEL_154:
        LODWORD(v278) = v264;
        if ( v264 > 0x40 )
          sub_16A4FD0(&v277, &v263);
        else
          v277 = v263;
        LODWORD(v274) = v262;
        if ( v262 > 0x40 )
          sub_16A4FD0(&v273, &v261);
        else
          v273 = (__int64 *)v261;
        sub_15898E0(&v269, &v273, &v277, v55, v65);
        if ( (unsigned int)v274 > 0x40 && v273 )
          j_j___libc_free_0_0(v273);
        if ( (unsigned int)v278 > 0x40 && v277 )
          j_j___libc_free_0_0(v277);
        goto LABEL_164;
      }
    }
LABEL_217:
    sub_15897D0(&v269, (unsigned int)v65, 1);
LABEL_164:
    if ( *((_BYTE *)v8 + 16) > 0x17u && (v8[6] || *((__int16 *)v8 + 9) < 0) )
    {
      v68 = sub_1625790(v8, 4);
      if ( v68 )
      {
        sub_1593050(&v273, v68);
        sub_158BE00(&v277, &v269, &v273);
        if ( v270 > 0x40 && v269 )
          j_j___libc_free_0_0(v269);
        v269 = v277;
        v69 = v278;
        LODWORD(v278) = 0;
        v270 = v69;
        if ( v272 > 0x40 && v271 )
          j_j___libc_free_0_0(v271);
        v271 = v279;
        v272 = v280;
        if ( (unsigned int)v278 > 0x40 && v277 )
          j_j___libc_free_0_0(v277);
        if ( v276 > 0x40 && v275 )
          j_j___libc_free_0_0(v275);
        if ( (unsigned int)v274 > 0x40 && v273 )
          j_j___libc_free_0_0(v273);
      }
    }
    if ( !(unsigned __int8)sub_158A0B0(&v269) )
    {
      if ( (unsigned __int8)sub_158BB40(&v265, &v269) )
      {
        v19 = sub_15A0600(v221);
      }
      else
      {
        sub_1590E70(&v277, &v265);
        v241 = sub_158BB40(&v277, &v269);
        if ( v280 > 0x40 && v279 )
          j_j___libc_free_0_0(v279);
        if ( (unsigned int)v278 > 0x40 && v277 )
          j_j___libc_free_0_0(v277);
        if ( v241 )
          v19 = sub_15A0640(v221);
      }
    }
    if ( v272 > 0x40 && v271 )
      j_j___libc_free_0_0(v271);
    if ( v270 > 0x40 && v269 )
      j_j___libc_free_0_0(v269);
    if ( v264 > 0x40 && v263 )
      j_j___libc_free_0_0(v263);
    if ( v262 > 0x40 && v261 )
      j_j___libc_free_0_0(v261);
    goto LABEL_44;
  }
LABEL_52:
  if ( v23 > 0x17u )
  {
    v24 = *((_BYTE *)v8 + 16);
    v25 = v6 - 32;
    if ( v24 <= 0x17u )
      goto LABEL_54;
    if ( !v5[6] && *((__int16 *)v5 + 9) >= 0 )
      goto LABEL_129;
    if ( sub_1625790(v5, 4) && sub_13CF9A0((__int64)v8, 4) )
    {
      v78 = sub_13CF9A0((__int64)v5, 4);
      sub_1593050(&v265, v78);
      v79 = sub_13CF9A0((__int64)v8, 4);
      sub_1593050(&v269, v79);
      sub_1590F80(&v273, (unsigned int)v6, &v265);
      if ( (unsigned __int8)sub_158BB40(&v273, &v269) )
      {
        v80 = sub_16498A0(v5);
        v8 = (__int64 **)sub_159C4F0(v80);
LABEL_225:
        sub_135E100(&v275);
        sub_135E100((__int64 *)&v273);
        sub_135E100(&v271);
        sub_135E100((__int64 *)&v269);
        sub_135E100(&v267);
        sub_135E100((__int64 *)&v265);
        return v8;
      }
      v97 = sub_15FF0F0((unsigned int)v6);
      sub_1590F80(&v277, v97, &v265);
      if ( (unsigned __int8)sub_158BB40(&v277, &v269) )
      {
        v98 = sub_16498A0(v5);
        v8 = (__int64 **)sub_159C540(v98);
        sub_135E100(&v279);
        sub_135E100((__int64 *)&v277);
        goto LABEL_225;
      }
      sub_135E100(&v279);
      sub_135E100((__int64 *)&v277);
      sub_135E100(&v275);
      sub_135E100((__int64 *)&v273);
      sub_135E100(&v271);
      sub_135E100((__int64 *)&v269);
      sub_135E100(&v267);
      sub_135E100((__int64 *)&v265);
    }
  }
  v24 = *((_BYTE *)v8 + 16);
  if ( v24 <= 0x17u )
    goto LABEL_143;
LABEL_129:
  if ( (unsigned int)v24 - 60 > 0xC )
    goto LABEL_143;
  v58 = *((_BYTE *)v5 + 16);
  if ( v58 > 0x10u && (unsigned __int8)(v58 - 60) > 0xCu )
    goto LABEL_143;
  v209 = v24;
  v59 = (__int64 **)sub_13CF970((__int64)v8);
  v240 = *v59;
  v60 = **v59;
  v214 = *v8;
  if ( v209 == 69 )
  {
    if ( a5 )
    {
      v212 = **v59;
      v123 = sub_127FA20(*a4, v60);
      v124 = sub_1643030(v214);
      v60 = v212;
      if ( v123 == v124 )
      {
        v125 = *((_BYTE *)v5 + 16);
        if ( v125 > 0x10u )
        {
          if ( v125 != 69 )
            goto LABEL_134;
          v126 = *(v5 - 3);
          if ( *v126 != v212 )
          {
            if ( *((_BYTE *)v8 + 16) == 62 )
            {
              v61 = 69;
LABEL_137:
              v25 = v6 - 32;
              if ( v61 != 62 )
                goto LABEL_54;
              if ( !a5 )
                goto LABEL_54;
              v62 = *(v5 - 3);
              if ( v60 != *v62 )
                goto LABEL_54;
              v63 = a4;
              v64 = a5 - 1;
              goto LABEL_141;
            }
LABEL_143:
            v25 = v6 - 32;
            goto LABEL_54;
          }
        }
        else
        {
          v126 = (_QWORD *)sub_15A3BA0(v5, v212, 0);
        }
        v18 = sub_13D9330((unsigned int)v6, v240, v126, a4, a5 - 1);
        if ( v18 )
          return (__int64 **)v18;
        v60 = v212;
      }
    }
  }
LABEL_134:
  if ( *((_BYTE *)v8 + 16) != 61 )
  {
LABEL_135:
    v25 = v6 - 32;
    if ( *((_BYTE *)v8 + 16) != 62 )
      goto LABEL_54;
    v61 = *((_BYTE *)v5 + 16);
    if ( (unsigned __int8)v61 > 0x17u )
      goto LABEL_137;
    if ( v61 != 13 )
      goto LABEL_54;
    v211 = v60;
    v107 = sub_15A43B0(v5, v60, 0);
    v108 = v214;
    v220 = (_QWORD *)v107;
    v109 = sub_15A46C0(38, v107, v108, 0);
    if ( (__int64 **)v109 != v5 )
      goto LABEL_578;
    if ( a5 )
    {
      v63 = a4;
      v64 = a5 - 1;
      v62 = v220;
LABEL_141:
      v18 = sub_13D9330((unsigned int)v6, v240, v62, v63, v64);
      if ( v18 )
        return (__int64 **)v18;
LABEL_54:
      if ( v25 <= 1 && (unsigned __int8)sub_14C00B0(v8, v5, *a4, a4[3], a4[4], a4[2]) )
      {
        if ( (_DWORD)v6 == 33 )
          return (__int64 **)sub_15A0600(v12);
        else
          return (__int64 **)sub_15A0640(v12);
      }
      v18 = sub_13DB950((unsigned int)v6, v8, v5, a4, a5);
      if ( v18 )
        return (__int64 **)v18;
      v26 = **v8;
      if ( *((_BYTE *)*v8 + 8) == 16 )
      {
        v27 = (*v8)[4];
        v28 = sub_1643320(v26);
        v29 = sub_16463B0(v28, (unsigned int)v27);
      }
      else
      {
        v29 = sub_1643320(v26);
      }
      if ( *((_BYTE *)v8 + 16) != 79 )
        goto LABEL_59;
      v81 = (__int64)*(v8 - 9);
      if ( *(_BYTE *)(v81 + 16) != 75 )
        goto LABEL_59;
      v82 = (__int64 **)*(v8 - 6);
      v83 = *(__int64 ***)(v81 - 48);
      v84 = (__int64 **)*(v8 - 3);
      v85 = *(_QWORD *)(v81 - 24);
      if ( v82 == v83 && v84 == (__int64 **)v85 )
      {
        v86 = *(unsigned __int16 *)(v81 + 18);
      }
      else
      {
        if ( v82 != (__int64 **)v85 || v84 != v83 )
          goto LABEL_59;
        v86 = *(unsigned __int16 *)(v81 + 18);
        if ( v82 != v83 )
        {
          v236 = v85;
          v256 = v83;
          v86 = sub_15FF0F0(v86 & 0xFFFF7FFF);
          v85 = v236;
          v83 = v256;
          goto LABEL_256;
        }
      }
      BYTE1(v86) &= ~0x80u;
LABEL_256:
      if ( (unsigned int)(v86 - 38) <= 1 )
      {
        if ( v83 )
        {
          v263 = (__int64 *)v83;
          if ( v85 )
          {
            v265 = (_QWORD **)v85;
            if ( v5 == v83 )
            {
LABEL_262:
              v87 = v6;
              v88 = 39;
LABEL_359:
              if ( v87 != 42 )
              {
                switch ( v87 )
                {
                  case ' ':
                  case ')':
                    break;
                  case '!':
                  case '&':
                    v88 = sub_15FF0F0(v88);
                    break;
                  case '\'':
                    goto LABEL_336;
                  case '(':
                    goto LABEL_333;
                  default:
                    goto LABEL_62;
                }
                v245 = v88;
                v18 = sub_13CB860((__int64)v8, v88, (__int64)v263, (__int64)v265);
                if ( v18 )
                  return (__int64 **)v18;
                v18 = sub_13CB860((__int64)v5, v245, (__int64)v263, (__int64)v265);
                if ( v18 )
                  return (__int64 **)v18;
                if ( a5 )
                {
                  v18 = sub_13D9330(v245, v263, v265, a4, a5 - 1);
                  if ( v18 )
                    return (__int64 **)v18;
                }
              }
LABEL_62:
              if ( *((_BYTE *)v8 + 16) != 79 )
                goto LABEL_63;
              v99 = (__int64)*(v8 - 9);
              if ( *(_BYTE *)(v99 + 16) != 75 )
                goto LABEL_63;
              v100 = (__int64 **)*(v8 - 6);
              v101 = *(__int64 ***)(v99 - 48);
              v102 = (__int64 **)*(v8 - 3);
              v103 = *(_QWORD *)(v99 - 24);
              if ( v100 == v101 && v102 == (__int64 **)v103 )
              {
                v104 = *(unsigned __int16 *)(v99 + 18);
              }
              else
              {
                if ( v100 != (__int64 **)v103 || v102 != v101 )
                  goto LABEL_63;
                v104 = *(unsigned __int16 *)(v99 + 18);
                if ( v100 != v101 )
                {
                  v235 = v103;
                  v255 = v101;
                  v104 = sub_15FF0F0(v104 & 0xFFFF7FFF);
                  v103 = v235;
                  v101 = v255;
                  goto LABEL_289;
                }
              }
              BYTE1(v104) &= ~0x80u;
LABEL_289:
              if ( (unsigned int)(v104 - 34) <= 1 )
              {
                if ( v101 )
                {
                  v263 = (__int64 *)v101;
                  if ( v103 )
                  {
                    v265 = (_QWORD **)v103;
                    if ( v5 == v101 )
                    {
LABEL_295:
                      v105 = v6;
                      v106 = 35;
                      goto LABEL_302;
                    }
                    if ( v5 == (__int64 **)v103 )
                    {
                      v263 = (__int64 *)v5;
                      v265 = v101;
                      goto LABEL_295;
                    }
                  }
                }
              }
LABEL_63:
              if ( *((_BYTE *)v5 + 16) != 79 )
                goto LABEL_64;
              v179 = (__int64)*(v5 - 9);
              if ( *(_BYTE *)(v179 + 16) != 75 )
                goto LABEL_64;
              v180 = (__int64 **)*(v5 - 6);
              v181 = *(__int64 ***)(v179 - 48);
              v182 = (__int64 **)*(v5 - 3);
              v183 = *(_QWORD *)(v179 - 24);
              if ( v180 == v181 && v182 == (__int64 **)v183 )
              {
                v184 = *(unsigned __int16 *)(v179 + 18);
              }
              else
              {
                if ( v180 != (__int64 **)v183 || v182 != v181 )
                  goto LABEL_64;
                v184 = *(unsigned __int16 *)(v179 + 18);
                if ( v180 != v181 )
                {
                  v230 = v183;
                  v250 = v181;
                  v184 = sub_15FF0F0(v184 & 0xFFFF7FFF);
                  v183 = v230;
                  v181 = v250;
                  goto LABEL_460;
                }
              }
              BYTE1(v184) &= ~0x80u;
LABEL_460:
              if ( (unsigned int)(v184 - 34) <= 1 )
              {
                if ( v181 )
                {
                  v263 = (__int64 *)v181;
                  if ( v183 )
                  {
                    v265 = (_QWORD **)v183;
                    if ( v8 == v181 )
                    {
LABEL_466:
                      v105 = sub_15FF5D0((unsigned int)v6);
                      v106 = 35;
                      goto LABEL_302;
                    }
                    if ( v8 == (__int64 **)v183 )
                    {
                      v263 = (__int64 *)v8;
                      v265 = v181;
                      goto LABEL_466;
                    }
                  }
                }
              }
LABEL_64:
              if ( *((_BYTE *)v8 + 16) != 79 )
                goto LABEL_65;
              v161 = (__int64)*(v8 - 9);
              if ( *(_BYTE *)(v161 + 16) != 75 )
                goto LABEL_65;
              v162 = (__int64 **)*(v8 - 6);
              v163 = *(__int64 ***)(v161 - 48);
              v164 = (__int64 **)*(v8 - 3);
              v165 = *(_QWORD *)(v161 - 24);
              if ( v162 == v163 && v164 == (__int64 **)v165 )
              {
                v166 = *(unsigned __int16 *)(v161 + 18);
              }
              else
              {
                if ( v162 != (__int64 **)v165 || v164 != v163 )
                  goto LABEL_65;
                v166 = *(unsigned __int16 *)(v161 + 18);
                if ( v162 != v163 )
                {
                  v231 = v165;
                  v251 = v163;
                  v166 = sub_15FF0F0(v166 & 0xFFFF7FFF);
                  v165 = v231;
                  v163 = v251;
                  goto LABEL_424;
                }
              }
              BYTE1(v166) &= ~0x80u;
LABEL_424:
              if ( (unsigned int)(v166 - 36) <= 1 )
              {
                if ( v163 )
                {
                  v263 = (__int64 *)v163;
                  if ( v165 )
                  {
                    v265 = (_QWORD **)v165;
                    if ( v5 == v163 )
                    {
LABEL_430:
                      v105 = sub_15FF5D0((unsigned int)v6);
                      v106 = 37;
                      goto LABEL_302;
                    }
                    if ( v5 == (__int64 **)v165 )
                    {
                      v263 = (__int64 *)v5;
                      v265 = v163;
                      goto LABEL_430;
                    }
                  }
                }
              }
LABEL_65:
              v277 = (__int64 *)&v263;
              v278 = (__int64)&v265;
              if ( !sub_13D6180(&v277, (__int64)v5) )
                goto LABEL_68;
              v30 = v263;
              if ( v8 != (__int64 **)v263 )
              {
                if ( v8 != v265 )
                  goto LABEL_68;
                v263 = (__int64 *)v8;
                v265 = (_QWORD **)v30;
              }
              v105 = v6;
              v106 = 37;
LABEL_302:
              if ( v105 != 42 )
              {
                switch ( v105 )
                {
                  case ' ':
                  case '%':
                    break;
                  case '!':
                  case '"':
                    v106 = sub_15FF0F0(v106);
                    break;
                  case '#':
                    goto LABEL_336;
                  case '$':
                    goto LABEL_333;
                  default:
                    goto LABEL_68;
                }
                v242 = v106;
                v18 = sub_13CB860((__int64)v8, v106, (__int64)v263, (__int64)v265);
                if ( v18 )
                  return (__int64 **)v18;
                v18 = sub_13CB860((__int64)v5, v242, (__int64)v263, (__int64)v265);
                if ( v18 )
                  return (__int64 **)v18;
                if ( a5 )
                {
                  v18 = sub_13D9330(v242, v263, v265, a4, a5 - 1);
                  if ( v18 )
                    return (__int64 **)v18;
                }
              }
LABEL_68:
              if ( *((_BYTE *)v8 + 16) != 79 )
                goto LABEL_69;
              v110 = (__int64)*(v8 - 9);
              if ( *(_BYTE *)(v110 + 16) != 75 )
                goto LABEL_69;
              v111 = *(v8 - 6);
              v112 = *(_QWORD **)(v110 - 48);
              v113 = *(v8 - 3);
              v114 = *(_QWORD *)(v110 - 24);
              if ( v111 == v112 && v113 == (_QWORD *)v114 )
              {
                v115 = *(unsigned __int16 *)(v110 + 18);
              }
              else
              {
                if ( v111 != (_QWORD *)v114 || v113 != v112 )
                  goto LABEL_69;
                v115 = *(unsigned __int16 *)(v110 + 18);
                if ( v111 != v112 )
                {
                  v233 = v114;
                  v253 = v112;
                  v115 = sub_15FF0F0(v115 & 0xFFFF7FFF);
                  v114 = v233;
                  v112 = v253;
                  goto LABEL_315;
                }
              }
              BYTE1(v115) &= ~0x80u;
LABEL_315:
              if ( (unsigned int)(v115 - 38) > 1 )
                goto LABEL_69;
              if ( !v112 )
                goto LABEL_69;
              v263 = v112;
              if ( !v114 )
                goto LABEL_69;
              v116 = *((_BYTE *)v5 + 16) == 79;
              v265 = (_QWORD **)v114;
              if ( !v116 )
                goto LABEL_69;
              v117 = (__int64)*(v5 - 9);
              if ( *(_BYTE *)(v117 + 16) != 75 )
                goto LABEL_69;
              v118 = (__int64)*(v5 - 6);
              v119 = *(_QWORD *)(v117 - 48);
              v120 = (__int64)*(v5 - 3);
              v121 = *(_QWORD *)(v117 - 24);
              if ( v118 == v119 && v120 == v121 )
              {
                v122 = *(unsigned __int16 *)(v117 + 18);
              }
              else
              {
                if ( v118 != v121 || v120 != v119 )
                  goto LABEL_69;
                v122 = *(unsigned __int16 *)(v117 + 18);
                if ( v118 != v119 )
                {
                  v234 = v121;
                  v254 = v119;
                  v122 = sub_15FF0F0(v122 & 0xFFFF7FFF);
                  v121 = v234;
                  v119 = v254;
                  goto LABEL_324;
                }
              }
              BYTE1(v122) &= ~0x80u;
LABEL_324:
              if ( (unsigned int)(v122 - 40) <= 1 )
              {
                if ( v119 )
                {
                  v269 = (__int64 *)v119;
                  if ( v121 )
                  {
                    v273 = (__int64 *)v121;
                    if ( v263 == (__int64 *)v119
                      || v263 == (__int64 *)v121
                      || v265 == (_QWORD **)v119
                      || v265 == (_QWORD **)v121 )
                    {
                      if ( (_DWORD)v6 != 39 )
                      {
                        if ( (_DWORD)v6 != 40 )
                          goto LABEL_72;
LABEL_333:
                        v18 = sub_15A0640(v29);
                        goto LABEL_334;
                      }
                      goto LABEL_336;
                    }
                  }
                }
              }
LABEL_69:
              if ( *((_BYTE *)v8 + 16) != 79 )
                goto LABEL_70;
              v167 = (__int64)*(v8 - 9);
              if ( *(_BYTE *)(v167 + 16) != 75 )
                goto LABEL_70;
              v168 = *(v8 - 6);
              v169 = *(_QWORD **)(v167 - 48);
              v170 = *(v8 - 3);
              v171 = *(_QWORD *)(v167 - 24);
              if ( v168 == v169 && v170 == (_QWORD *)v171 )
              {
                v172 = *(unsigned __int16 *)(v167 + 18);
              }
              else
              {
                if ( v168 != (_QWORD *)v171 || v170 != v169 )
                  goto LABEL_70;
                v172 = *(unsigned __int16 *)(v167 + 18);
                if ( v168 != v169 )
                {
                  v224 = v171;
                  v243 = v169;
                  v172 = sub_15FF0F0(v172 & 0xFFFF7FFF);
                  v171 = v224;
                  v169 = v243;
                  goto LABEL_436;
                }
              }
              BYTE1(v172) &= ~0x80u;
LABEL_436:
              if ( (unsigned int)(v172 - 40) > 1 )
                goto LABEL_70;
              if ( !v169 )
                goto LABEL_70;
              v263 = v169;
              if ( !v171 )
                goto LABEL_70;
              v116 = *((_BYTE *)v5 + 16) == 79;
              v265 = (_QWORD **)v171;
              if ( !v116 )
                goto LABEL_70;
              v173 = (__int64)*(v5 - 9);
              if ( *(_BYTE *)(v173 + 16) != 75 )
                goto LABEL_70;
              v174 = (__int64)*(v5 - 6);
              v175 = *(_QWORD *)(v173 - 48);
              v176 = (__int64)*(v5 - 3);
              v177 = *(_QWORD *)(v173 - 24);
              if ( v174 == v175 && v176 == v177 )
              {
                v178 = *(unsigned __int16 *)(v173 + 18);
              }
              else
              {
                if ( v174 != v177 || v176 != v175 )
                  goto LABEL_70;
                v178 = *(unsigned __int16 *)(v173 + 18);
                if ( v174 != v175 )
                {
                  v225 = v177;
                  v244 = v175;
                  v178 = sub_15FF0F0(v178 & 0xFFFF7FFF);
                  v177 = v225;
                  v175 = v244;
                  goto LABEL_445;
                }
              }
              BYTE1(v178) &= ~0x80u;
LABEL_445:
              if ( (unsigned int)(v178 - 38) <= 1 )
              {
                if ( v175 )
                {
                  v269 = (__int64 *)v175;
                  if ( v177 )
                  {
                    v273 = (__int64 *)v177;
                    if ( v263 == (__int64 *)v175
                      || v263 == (__int64 *)v177
                      || v265 == (_QWORD **)v175
                      || v265 == (_QWORD **)v177 )
                    {
                      if ( (_DWORD)v6 != 41 )
                      {
                        if ( (_DWORD)v6 != 38 )
                          goto LABEL_72;
                        goto LABEL_333;
                      }
                      goto LABEL_336;
                    }
                  }
                }
              }
LABEL_70:
              if ( *((_BYTE *)v8 + 16) != 79 )
                goto LABEL_71;
              v155 = (__int64)*(v8 - 9);
              if ( *(_BYTE *)(v155 + 16) != 75 )
                goto LABEL_71;
              v156 = *(v8 - 6);
              v157 = *(_QWORD **)(v155 - 48);
              v158 = *(v8 - 3);
              v159 = *(_QWORD *)(v155 - 24);
              if ( v156 == v157 && v158 == (_QWORD *)v159 )
              {
                v160 = *(unsigned __int16 *)(v155 + 18);
              }
              else
              {
                if ( v156 != (_QWORD *)v159 || v158 != v157 )
                  goto LABEL_71;
                v160 = *(unsigned __int16 *)(v155 + 18);
                if ( v156 != v157 )
                {
                  v232 = v159;
                  v252 = v157;
                  v160 = sub_15FF0F0(v160 & 0xFFFF7FFF);
                  v159 = v232;
                  v157 = v252;
                  goto LABEL_408;
                }
              }
              BYTE1(v160) &= ~0x80u;
LABEL_408:
              if ( (unsigned int)(v160 - 34) <= 1 )
              {
                if ( v157 )
                {
                  v263 = v157;
                  if ( v159 )
                  {
                    v265 = (_QWORD **)v159;
                    v277 = (__int64 *)&v269;
                    v278 = (__int64)&v273;
                    if ( sub_13D6180(&v277, (__int64)v5)
                      && (v263 == v269 || v263 == v273 || v269 == (__int64 *)v265 || v273 == (__int64 *)v265) )
                    {
                      if ( (_DWORD)v6 != 35 )
                      {
                        if ( (_DWORD)v6 != 36 )
                          goto LABEL_72;
                        goto LABEL_333;
                      }
                      goto LABEL_336;
                    }
                  }
                }
              }
LABEL_71:
              if ( *((_BYTE *)v8 + 16) != 79 )
                goto LABEL_72;
              v143 = (__int64)*(v8 - 9);
              if ( *(_BYTE *)(v143 + 16) != 75 )
                goto LABEL_72;
              v144 = *(v8 - 6);
              v145 = *(_QWORD **)(v143 - 48);
              v146 = *(v8 - 3);
              v147 = *(_QWORD *)(v143 - 24);
              if ( v144 == v145 && v146 == (_QWORD *)v147 )
              {
                v148 = *(unsigned __int16 *)(v143 + 18);
              }
              else
              {
                if ( v144 != (_QWORD *)v147 || v146 != v145 )
                  goto LABEL_72;
                v148 = *(unsigned __int16 *)(v143 + 18);
                if ( v144 != v145 )
                {
                  v228 = v147;
                  v248 = v145;
                  v148 = sub_15FF0F0(v148 & 0xFFFF7FFF);
                  v147 = v228;
                  v145 = v248;
                  goto LABEL_384;
                }
              }
              BYTE1(v148) &= ~0x80u;
LABEL_384:
              if ( (unsigned int)(v148 - 36) > 1 )
                goto LABEL_72;
              if ( !v145 )
                goto LABEL_72;
              v263 = v145;
              if ( !v147 )
                goto LABEL_72;
              v116 = *((_BYTE *)v5 + 16) == 79;
              v265 = (_QWORD **)v147;
              if ( !v116 )
                goto LABEL_72;
              v149 = (__int64)*(v5 - 9);
              if ( *(_BYTE *)(v149 + 16) != 75 )
                goto LABEL_72;
              v150 = (__int64)*(v5 - 6);
              v151 = *(_QWORD *)(v149 - 48);
              v152 = (__int64)*(v5 - 3);
              v153 = *(_QWORD *)(v149 - 24);
              if ( v150 == v151 && v152 == v153 )
              {
                v154 = *(unsigned __int16 *)(v149 + 18);
              }
              else
              {
                if ( v150 != v153 || v152 != v151 )
                  goto LABEL_72;
                v154 = *(unsigned __int16 *)(v149 + 18);
                if ( v150 != v151 )
                {
                  v229 = v153;
                  v249 = v151;
                  v154 = sub_15FF0F0(v154 & 0xFFFF7FFF);
                  v153 = v229;
                  v151 = v249;
                  goto LABEL_393;
                }
              }
              BYTE1(v154) &= ~0x80u;
LABEL_393:
              if ( (unsigned int)(v154 - 34) > 1 )
                goto LABEL_72;
              if ( !v151 )
                goto LABEL_72;
              v269 = (__int64 *)v151;
              if ( !v153 )
                goto LABEL_72;
              v273 = (__int64 *)v153;
              if ( v263 != (__int64 *)v151
                && v263 != (__int64 *)v153
                && v265 != (_QWORD **)v151
                && v265 != (_QWORD **)v153 )
              {
                goto LABEL_72;
              }
              if ( (_DWORD)v6 != 37 )
              {
                if ( (_DWORD)v6 != 34 )
                  goto LABEL_72;
                goto LABEL_333;
              }
LABEL_336:
              v18 = sub_15A0600(v29);
LABEL_334:
              if ( v18 )
                return (__int64 **)v18;
LABEL_72:
              v31 = (__int64)v8;
              v18 = sub_13CCF60(v6, v8, (__int64)v5, a4);
              if ( v18 )
                return (__int64 **)v18;
              if ( *((_BYTE *)*v8 + 8) == 15 )
              {
                v31 = a4[1];
                v207 = v35;
                v18 = sub_13D0E30(*a4, v31, a4[2], v6, a4[4], (__int64)v8, (__int64)v5);
                v32 = v207;
                if ( v18 )
                  return (__int64 **)v18;
              }
              if ( (unsigned __int8)sub_13CBD30((__int64)v8, v31, v32, v33, v34) )
              {
                if ( (unsigned __int8)sub_13CBD30((__int64)v5, v31, v36, v37, v38) )
                {
                  v39 = (__int64 **)sub_13CF970((__int64)v8);
                  v40 = sub_127FA20(*a4, **v39);
                  if ( v40 == sub_127FA20(*a4, (__int64)*v8) )
                  {
                    v188 = (__int64 **)sub_13CF970((__int64)v5);
                    v189 = sub_127FA20(*a4, **v188);
                    if ( v189 == sub_127FA20(*a4, (__int64)*v5) )
                    {
                      v190 = (__int64 *)sub_13CF970((__int64)v5);
                      v191 = (__int64 *)sub_13CF970((__int64)v8);
                      v18 = sub_13D0E30(*a4, a4[1], a4[2], v6, a4[4], *v191, *v190);
                      if ( v18 )
                        return (__int64 **)v18;
                    }
                  }
                }
              }
              if ( *((_BYTE *)v8 + 16) != 56 )
                goto LABEL_78;
              v127 = *((_BYTE *)v5 + 16);
              if ( v127 <= 0x17u )
              {
                if ( v127 != 5 || *((_WORD *)v5 + 9) != 32 )
                {
LABEL_80:
                  if ( *((_BYTE *)v8 + 16) == 77 )
                    return (__int64 **)sub_13D91D0((unsigned int)v6, (__int64)v8, (__int64)v5, a4, a5);
LABEL_81:
                  if ( *((_BYTE *)v5 + 16) != 77 )
                    return 0;
                  return (__int64 **)sub_13D91D0((unsigned int)v6, (__int64)v8, (__int64)v5, a4, a5);
                }
              }
              else if ( v127 != 56 )
              {
                goto LABEL_344;
              }
              if ( *(__int64 **)sub_13CF970((__int64)v5) == v8[-3 * (*((_DWORD *)v8 + 5) & 0xFFFFFFF)] )
              {
                if ( (unsigned __int8)sub_15FA290(v8) )
                {
                  v192 = 24LL * (*((_DWORD *)v5 + 5) & 0xFFFFFFF);
                  v193 = &v5[v192 / 0xFFFFFFFFFFFFFFF8LL];
                  if ( (*((_BYTE *)v5 + 23) & 0x40) != 0 )
                    v193 = (__int64 **)*(v5 - 1);
                  v194 = v193 + 3;
                  v195 = &v193[(unsigned __int64)v192 / 8];
                  while ( v195 != v194 )
                  {
                    if ( *((_BYTE *)*v194 + 16) != 13 )
                      goto LABEL_78;
                    v194 += 3;
                  }
                  if ( v25 <= 1
                    || (unsigned __int8)sub_15FA300(v8)
                    && (*((_BYTE *)v5 + 17) & 2) != 0
                    && (unsigned int)sub_15FF420((unsigned int)v6) == (_DWORD)v6 )
                  {
                    v196 = sub_15A06D0(*v8[-3 * (*((_DWORD *)v8 + 5) & 0xFFFFFFF)]);
                    v197 = *((_DWORD *)v8 + 5);
                    v274 = 0x400000000LL;
                    v273 = &v275;
                    sub_13D6230((__int64)&v273, (char *)&v8[3 * (1LL - (v197 & 0xFFFFFFF))], (char *)v8);
                    v198 = v8[7];
                    BYTE4(v277) = 0;
                    v259 = sub_15A2E80((_DWORD)v198, v196, (_DWORD)v273, v274, 0, (unsigned int)&v277, 0);
                    v199 = sub_13CF970((__int64)v5);
                    v200 = *((_DWORD *)v5 + 5);
                    v277 = &v279;
                    v278 = 0x400000000LL;
                    sub_13D6230((__int64)&v277, (char *)(v199 + 24), (char *)(v199 + 24LL * (v200 & 0xFFFFFFF)));
                    v201 = v8[7];
                    BYTE4(v269) = 0;
                    v202 = sub_15A2E80((_DWORD)v201, v196, (_DWORD)v277, v278, 0, (unsigned int)&v269, 0);
                    v8 = (__int64 **)sub_15A35F0((unsigned __int16)v6, v259, v202, 0, v203, v204);
                    if ( v277 != &v279 )
                      _libc_free((unsigned __int64)v277);
                    if ( v273 != &v275 )
                      _libc_free((unsigned __int64)v273);
                    return v8;
                  }
                }
LABEL_78:
                if ( *((_BYTE *)v8 + 16) != 79 && *((_BYTE *)v5 + 16) != 79 )
                  goto LABEL_80;
                goto LABEL_345;
              }
LABEL_344:
              if ( *((_BYTE *)v5 + 16) != 79 )
                goto LABEL_81;
LABEL_345:
              v18 = (__int64)sub_13D86C0((unsigned int)v6, (__int64)v8, (__int64)v5, a4, a5);
              if ( !v18 )
                goto LABEL_80;
              return (__int64 **)v18;
            }
            if ( v5 == (__int64 **)v85 )
            {
              v263 = (__int64 *)v5;
              v265 = v83;
              goto LABEL_262;
            }
          }
        }
      }
LABEL_59:
      if ( *((_BYTE *)v5 + 16) != 79 )
        goto LABEL_60;
      v89 = (__int64)*(v5 - 9);
      if ( *(_BYTE *)(v89 + 16) != 75 )
        goto LABEL_60;
      v90 = (__int64 **)*(v5 - 6);
      v91 = *(__int64 ***)(v89 - 48);
      v92 = (__int64 **)*(v5 - 3);
      v93 = *(_QWORD *)(v89 - 24);
      if ( v90 == v91 && v92 == (__int64 **)v93 )
      {
        v94 = *(unsigned __int16 *)(v89 + 18);
      }
      else
      {
        if ( v90 != (__int64 **)v93 || v92 != v91 )
          goto LABEL_60;
        v94 = *(unsigned __int16 *)(v89 + 18);
        if ( v90 != v91 )
        {
          v237 = v93;
          v257 = v91;
          v94 = sub_15FF0F0(v94 & 0xFFFF7FFF);
          v93 = v237;
          v91 = v257;
          goto LABEL_268;
        }
      }
      BYTE1(v94) &= ~0x80u;
LABEL_268:
      if ( (unsigned int)(v94 - 38) <= 1 )
      {
        if ( v91 )
        {
          v263 = (__int64 *)v91;
          if ( v93 )
          {
            v265 = (_QWORD **)v93;
            if ( v8 == v91 )
            {
LABEL_275:
              v87 = sub_15FF5D0((unsigned int)v6);
              v88 = 39;
              goto LABEL_359;
            }
            if ( v8 == (__int64 **)v93 )
            {
              if ( v91 != v8 )
              {
                v263 = (__int64 *)v8;
                v265 = v91;
              }
              goto LABEL_275;
            }
          }
        }
      }
LABEL_60:
      if ( *((_BYTE *)v8 + 16) != 79 )
        goto LABEL_61;
      v134 = (__int64)*(v8 - 9);
      if ( *(_BYTE *)(v134 + 16) != 75 )
        goto LABEL_61;
      v135 = (__int64 **)*(v8 - 6);
      v136 = *(__int64 ***)(v134 - 48);
      v137 = (__int64 **)*(v8 - 3);
      v138 = *(_QWORD *)(v134 - 24);
      if ( v135 == v136 && v137 == (__int64 **)v138 )
      {
        v139 = *(unsigned __int16 *)(v134 + 18);
      }
      else
      {
        if ( v135 != (__int64 **)v138 || v137 != v136 )
          goto LABEL_61;
        v139 = *(unsigned __int16 *)(v134 + 18);
        if ( v135 != v136 )
        {
          v227 = v138;
          v247 = v136;
          v139 = sub_15FF0F0(v139 & 0xFFFF7FFF);
          v138 = v227;
          v136 = v247;
          goto LABEL_366;
        }
      }
      BYTE1(v139) &= ~0x80u;
LABEL_366:
      if ( (unsigned int)(v139 - 40) <= 1 )
      {
        if ( v136 )
        {
          v263 = (__int64 *)v136;
          if ( v138 )
          {
            v265 = (_QWORD **)v138;
            if ( v5 == v136 )
            {
LABEL_372:
              v87 = sub_15FF5D0((unsigned int)v6);
              v88 = 41;
              goto LABEL_359;
            }
            if ( v5 == (__int64 **)v138 )
            {
              v263 = (__int64 *)v5;
              v265 = v136;
              goto LABEL_372;
            }
          }
        }
      }
LABEL_61:
      if ( *((_BYTE *)v5 + 16) != 79 )
        goto LABEL_62;
      v128 = (__int64)*(v5 - 9);
      if ( *(_BYTE *)(v128 + 16) != 75 )
        goto LABEL_62;
      v129 = (__int64 **)*(v5 - 6);
      v130 = *(__int64 ***)(v128 - 48);
      v131 = (__int64 **)*(v5 - 3);
      v132 = *(_QWORD *)(v128 - 24);
      if ( v129 == v130 && v131 == (__int64 **)v132 )
      {
        v133 = *(unsigned __int16 *)(v128 + 18);
      }
      else
      {
        if ( v129 != (__int64 **)v132 || v131 != v130 )
          goto LABEL_62;
        v133 = *(unsigned __int16 *)(v128 + 18);
        if ( v129 != v130 )
        {
          v226 = v132;
          v246 = v130;
          v133 = sub_15FF0F0(v133 & 0xFFFF7FFF);
          v132 = v226;
          v130 = v246;
LABEL_352:
          if ( (unsigned int)(v133 - 40) > 1 )
            goto LABEL_62;
          if ( !v130 )
            goto LABEL_62;
          v263 = (__int64 *)v130;
          if ( !v132 )
            goto LABEL_62;
          v265 = (_QWORD **)v132;
          if ( v8 != v130 )
          {
            if ( v8 != (__int64 **)v132 )
              goto LABEL_62;
            v263 = (__int64 *)v8;
            v265 = v130;
          }
          v87 = v6;
          v88 = 41;
          goto LABEL_359;
        }
      }
      BYTE1(v133) &= ~0x80u;
      goto LABEL_352;
    }
    if ( (__int64 **)v109 != v5 )
    {
LABEL_578:
      v25 = v6 - 32;
      switch ( (int)v6 )
      {
        case ' ':
          goto LABEL_472;
        case '!':
          goto LABEL_471;
        case '"':
        case '#':
          if ( !a5 )
            goto LABEL_54;
          v205 = sub_15A06D0(v211);
          v18 = sub_13D9330(40, v240, v205, a4, a5 - 1);
          if ( !v18 )
            goto LABEL_54;
          return (__int64 **)v18;
        case '$':
        case '%':
          if ( !a5 )
            goto LABEL_54;
          v206 = sub_15A06D0(v211);
          v18 = sub_13D9330(39, v240, v206, a4, a5 - 1);
          if ( !v18 )
            goto LABEL_54;
          return (__int64 **)v18;
        case '&':
        case '\'':
          goto LABEL_469;
        case '(':
        case ')':
          goto LABEL_474;
        default:
          *(_DWORD *)(v6 + 8) = (2 * (*(_DWORD *)(v6 + 8) >> 1) + 2) | *(_DWORD *)(v6 + 8) & 1;
          BUG();
      }
    }
    goto LABEL_143;
  }
  v95 = *((_BYTE *)v5 + 16);
  if ( v95 > 0x17u )
  {
    v25 = v6 - 32;
    if ( v95 != 61 || !a5 || v60 != **(v5 - 3) )
      goto LABEL_54;
    v208 = (__int64)*(v5 - 3);
    v210 = v60;
    v96 = sub_15FF470((unsigned int)v6);
    v18 = sub_13D9330(v96, v240, v208, a4, a5 - 1);
    if ( v18 )
      return (__int64 **)v18;
    goto LABEL_281;
  }
  if ( v95 != 13 )
    goto LABEL_143;
  v210 = v60;
  v140 = sub_15A43B0(v5, v60, 0);
  v141 = sub_15A46C0(37, v140, v214, 0);
  if ( (__int64 **)v141 == v5 )
  {
    v60 = v210;
    if ( a5 )
    {
      v142 = sub_15FF470((unsigned int)v6);
      v18 = sub_13D9330(v142, v240, v140, a4, a5 - 1);
      if ( v18 )
        return (__int64 **)v18;
LABEL_281:
      v60 = v210;
      goto LABEL_135;
    }
    if ( (__int64 **)v141 == v5 )
      goto LABEL_135;
  }
  switch ( (int)v6 )
  {
    case ' ':
    case '"':
    case '#':
LABEL_472:
      v185 = v5;
      goto LABEL_473;
    case '!':
    case '$':
    case '%':
LABEL_471:
      v185 = v5;
      goto LABEL_470;
    case '&':
    case '\'':
LABEL_469:
      v185 = v5;
      if ( sub_13D0200((__int64 *)v5 + 3, *((_DWORD *)v5 + 8) - 1) )
        goto LABEL_470;
      goto LABEL_473;
    case '(':
    case ')':
LABEL_474:
      v185 = v5;
      if ( sub_13D0200((__int64 *)v5 + 3, *((_DWORD *)v5 + 8) - 1) )
      {
LABEL_473:
        v187 = sub_16498A0(v185);
        result = (__int64 **)sub_159C540(v187);
      }
      else
      {
LABEL_470:
        v186 = sub_16498A0(v185);
        result = (__int64 **)sub_159C4F0(v186);
      }
      break;
  }
  return result;
}
