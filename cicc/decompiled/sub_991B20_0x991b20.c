// Function: sub_991B20
// Address: 0x991b20
//
char __fastcall sub_991B20(__int64 a1, unsigned int a2, unsigned __int8 *a3, char *a4, __int64 a5, __int64 *a6)
{
  __int64 v10; // rax
  unsigned __int8 v11; // dl
  __int64 v12; // r10
  __int64 v13; // r9
  unsigned int v14; // ecx
  unsigned __int8 v15; // al
  int v16; // edx
  unsigned __int64 *v17; // r12
  int v18; // eax
  int v19; // eax
  unsigned __int8 *v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int8 *v24; // rsi
  char v25; // al
  char v26; // al
  __int64 v27; // r9
  __int64 v28; // r10
  __int64 v29; // r10
  __int64 v30; // r9
  int v31; // eax
  unsigned int v32; // eax
  char v33; // bl
  unsigned int v34; // r12d
  unsigned __int8 *v35; // r12
  unsigned __int8 *v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rdx
  unsigned __int8 *v39; // r12
  __int64 v40; // r13
  __int64 *v41; // r12
  unsigned __int8 v42; // al
  unsigned int v43; // ecx
  __int64 *v44; // r9
  __int64 v45; // rbx
  unsigned int v46; // r12d
  __int64 v47; // rdx
  unsigned int v48; // ebx
  unsigned int v49; // ebx
  bool v50; // zf
  int v51; // eax
  unsigned __int8 *v52; // rsi
  char v53; // al
  unsigned __int8 *v54; // rsi
  unsigned int v55; // eax
  unsigned int v56; // ecx
  unsigned __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rdi
  __int64 v60; // rsi
  int v61; // edx
  unsigned __int8 *v62; // rsi
  char v63; // al
  unsigned __int8 *v64; // rsi
  char v65; // al
  int v66; // eax
  unsigned __int8 *v67; // rsi
  unsigned int v68; // eax
  char v69; // cl
  unsigned __int64 v70; // rax
  unsigned int v71; // ebx
  unsigned __int8 *v72; // rdx
  __int64 v73; // rax
  __int64 v74; // rsi
  __int64 v75; // rdx
  __int64 v76; // rax
  __int64 v77; // rdx
  unsigned __int8 *v78; // rdx
  __int64 v79; // rax
  __int64 v80; // rsi
  __int64 v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rdx
  unsigned __int8 *v84; // r12
  int v85; // eax
  unsigned __int64 v86; // rdx
  unsigned __int8 *v87; // rsi
  char v88; // al
  unsigned __int8 *v89; // r12
  int v90; // eax
  unsigned __int64 v91; // rdx
  unsigned __int8 *v92; // rsi
  char v93; // al
  unsigned __int8 *v94; // r12
  char v95; // al
  _QWORD *v96; // r15
  __int64 v97; // rdx
  __int64 v98; // rcx
  __int64 v99; // r8
  unsigned int v100; // edx
  __int64 v101; // rax
  __int64 v102; // rdi
  __int64 v103; // r15
  unsigned __int8 *v104; // rsi
  char v105; // al
  unsigned __int8 *v106; // rsi
  char v107; // al
  _QWORD *v108; // rax
  _QWORD *v109; // rax
  int v110; // eax
  unsigned __int8 *v111; // rdx
  __int64 v112; // rax
  __int64 v113; // rdx
  __int64 v114; // rax
  __int64 v115; // rdx
  unsigned __int8 *v116; // r12
  int v117; // eax
  unsigned __int8 *v118; // rdx
  __int64 v119; // rax
  __int64 v120; // rdx
  __int64 v121; // rax
  __int64 v122; // rdx
  unsigned __int8 *v123; // r12
  __int64 v124; // rsi
  __int64 v125; // rdx
  __int64 v126; // rcx
  __int64 v127; // r8
  int v128; // eax
  bool v129; // cc
  _QWORD *v130; // r9
  __int64 v131; // rsi
  __int64 v132; // rdx
  __int64 v133; // rcx
  __int64 v134; // r8
  unsigned int v135; // edx
  __int64 v136; // rax
  unsigned __int64 v137; // rax
  unsigned __int8 *v138; // rsi
  char v139; // al
  __int64 v140; // r12
  unsigned int v141; // r13d
  unsigned __int64 *v142; // rax
  unsigned __int64 v143; // rdx
  unsigned int v144; // r15d
  unsigned int v145; // ecx
  unsigned __int64 v146; // rax
  int v147; // eax
  __int64 v149; // [rsp+8h] [rbp-138h]
  __int64 v150; // [rsp+10h] [rbp-130h]
  __int64 v151; // [rsp+10h] [rbp-130h]
  __int64 v152; // [rsp+18h] [rbp-128h]
  __int64 v153; // [rsp+18h] [rbp-128h]
  __int64 v154; // [rsp+18h] [rbp-128h]
  __int64 v155; // [rsp+20h] [rbp-120h]
  __int64 v156; // [rsp+20h] [rbp-120h]
  __int64 v157; // [rsp+20h] [rbp-120h]
  __int64 v158; // [rsp+20h] [rbp-120h]
  __int64 *v159; // [rsp+28h] [rbp-118h]
  __int64 v160; // [rsp+28h] [rbp-118h]
  __int64 v161; // [rsp+28h] [rbp-118h]
  __int64 v162; // [rsp+28h] [rbp-118h]
  __int64 v163; // [rsp+28h] [rbp-118h]
  __int64 v164; // [rsp+28h] [rbp-118h]
  unsigned int v165; // [rsp+28h] [rbp-118h]
  unsigned int v166; // [rsp+28h] [rbp-118h]
  __int64 v167; // [rsp+30h] [rbp-110h]
  __int64 v168; // [rsp+30h] [rbp-110h]
  __int64 v169; // [rsp+30h] [rbp-110h]
  unsigned int v170; // [rsp+30h] [rbp-110h]
  unsigned int v171; // [rsp+30h] [rbp-110h]
  unsigned int v172; // [rsp+30h] [rbp-110h]
  __int64 v173; // [rsp+30h] [rbp-110h]
  unsigned int v174; // [rsp+30h] [rbp-110h]
  __int64 v175; // [rsp+30h] [rbp-110h]
  __int64 v176; // [rsp+30h] [rbp-110h]
  __int64 v177; // [rsp+30h] [rbp-110h]
  __int64 v178; // [rsp+30h] [rbp-110h]
  unsigned int v179; // [rsp+30h] [rbp-110h]
  unsigned int v180; // [rsp+30h] [rbp-110h]
  __int64 v181; // [rsp+30h] [rbp-110h]
  __int64 v182; // [rsp+30h] [rbp-110h]
  _QWORD *v183; // [rsp+30h] [rbp-110h]
  __int64 *v184; // [rsp+30h] [rbp-110h]
  __int64 v185; // [rsp+38h] [rbp-108h]
  __int64 v186; // [rsp+38h] [rbp-108h]
  __int64 v187; // [rsp+38h] [rbp-108h]
  __int64 v188; // [rsp+38h] [rbp-108h]
  __int64 v189; // [rsp+38h] [rbp-108h]
  __int64 v190; // [rsp+38h] [rbp-108h]
  unsigned int v191; // [rsp+38h] [rbp-108h]
  __int64 v192; // [rsp+38h] [rbp-108h]
  __int64 v193; // [rsp+38h] [rbp-108h]
  __int64 v194; // [rsp+38h] [rbp-108h]
  __int64 v195; // [rsp+38h] [rbp-108h]
  __int64 v196; // [rsp+38h] [rbp-108h]
  __int64 v197; // [rsp+38h] [rbp-108h]
  __int64 v198; // [rsp+38h] [rbp-108h]
  __int64 v199; // [rsp+38h] [rbp-108h]
  __int64 v200; // [rsp+38h] [rbp-108h]
  __int64 v201; // [rsp+38h] [rbp-108h]
  __int64 v202; // [rsp+38h] [rbp-108h]
  __int64 v203; // [rsp+38h] [rbp-108h]
  __int64 v204; // [rsp+38h] [rbp-108h]
  __int64 v205; // [rsp+38h] [rbp-108h]
  __int64 v206; // [rsp+38h] [rbp-108h]
  __int64 v207; // [rsp+38h] [rbp-108h]
  int v208; // [rsp+38h] [rbp-108h]
  __int64 v209; // [rsp+38h] [rbp-108h]
  unsigned int v210; // [rsp+38h] [rbp-108h]
  __int64 v211; // [rsp+48h] [rbp-F8h] BYREF
  __int64 v212[2]; // [rsp+50h] [rbp-F0h] BYREF
  unsigned __int64 v213; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v214; // [rsp+68h] [rbp-D8h]
  __int64 v215; // [rsp+70h] [rbp-D0h] BYREF
  unsigned int v216; // [rsp+78h] [rbp-C8h]
  unsigned __int64 v217; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v218; // [rsp+88h] [rbp-B8h]
  unsigned __int64 v219; // [rsp+90h] [rbp-B0h] BYREF
  unsigned int v220; // [rsp+98h] [rbp-A8h]
  __int64 *v221; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v222; // [rsp+A8h] [rbp-98h]
  __int64 v223; // [rsp+B0h] [rbp-90h] BYREF
  unsigned __int64 *v224; // [rsp+B8h] [rbp-88h]
  __int64 v225; // [rsp+C0h] [rbp-80h]
  __int64 v226; // [rsp+C8h] [rbp-78h]
  __int64 *v227; // [rsp+D0h] [rbp-70h] BYREF
  char v228; // [rsp+D8h] [rbp-68h]
  _QWORD v229[3]; // [rsp+E0h] [rbp-60h] BYREF
  __int64 *v230; // [rsp+F8h] [rbp-48h] BYREF
  char v231; // [rsp+100h] [rbp-40h]

  LODWORD(v10) = *(unsigned __int8 *)(*((_QWORD *)a4 + 1) + 8LL);
  if ( (_BYTE)v10 == 14 )
  {
    if ( (unsigned __int8 *)a1 != a3 || (unsigned __int8)*a4 > 0x15u )
      return v10;
    LOBYTE(v10) = sub_AC30F0(a4);
    v33 = v10;
    if ( !(_BYTE)v10 )
    {
      if ( *a4 == 17 )
      {
        v71 = *((_DWORD *)a4 + 8);
        if ( v71 <= 0x40 )
        {
          if ( *((_QWORD *)a4 + 3) )
            return v10;
        }
        else
        {
          LODWORD(v10) = sub_C444A0(a4 + 24);
          if ( v71 != (_DWORD)v10 )
            return v10;
        }
      }
      else
      {
        v103 = *((_QWORD *)a4 + 1);
        LODWORD(v10) = *(unsigned __int8 *)(v103 + 8) - 17;
        if ( (unsigned int)v10 > 1 )
          return v10;
        v10 = sub_AD7630(a4, 0);
        if ( !v10 || *(_BYTE *)v10 != 17 )
        {
          if ( *(_BYTE *)(v103 + 8) == 17 )
          {
            LODWORD(v10) = *(_DWORD *)(v103 + 32);
            v208 = v10;
            if ( (_DWORD)v10 )
            {
              v144 = 0;
              while ( 1 )
              {
                v10 = sub_AD69F0(a4, v144);
                if ( !v10 )
                  break;
                if ( *(_BYTE *)v10 != 13 )
                {
                  if ( *(_BYTE *)v10 != 17 )
                    break;
                  LOBYTE(v10) = sub_9867B0(v10 + 24);
                  v33 = v10;
                  if ( !(_BYTE)v10 )
                    break;
                }
                if ( v208 == ++v144 )
                  goto LABEL_193;
              }
            }
          }
          return v10;
        }
        LOBYTE(v10) = sub_9867B0(v10 + 24);
        v33 = v10;
LABEL_193:
        if ( !v33 )
          return v10;
      }
    }
    if ( a2 > 0x27 )
    {
      if ( a2 == 40 )
        LOBYTE(v10) = sub_987080((__int64 *)(a5 + 16), *(_DWORD *)(a5 + 24) - 1);
    }
    else if ( a2 > 0x25 )
    {
      LOBYTE(v10) = sub_987080((__int64 *)a5, *(_DWORD *)(a5 + 8) - 1);
    }
    else if ( a2 == 32 )
    {
      sub_986FF0(a5);
      LOBYTE(v10) = (unsigned __int8)sub_987100(a5 + 16);
    }
    return v10;
  }
  v11 = *a4;
  v12 = *a6;
  v13 = (__int64)(a4 + 24);
  v14 = *(_DWORD *)(a5 + 8);
  if ( v11 != 17 )
  {
    LODWORD(v10) = v10 - 17;
    v170 = *(_DWORD *)(a5 + 8);
    v188 = v12;
    if ( (unsigned int)v10 > 1 )
      return v10;
    if ( v11 > 0x15u )
      return v10;
    v10 = sub_AD7630(a4, 0);
    if ( !v10 || *(_BYTE *)v10 != 17 )
      return v10;
    v14 = v170;
    v12 = v188;
    v13 = v10 + 24;
  }
  if ( a2 == 32 )
  {
    if ( (unsigned __int8 *)a1 != a3 )
    {
      v42 = *a3;
      if ( *a3 > 0x1Cu )
      {
        v61 = v42 - 29;
      }
      else
      {
        if ( v42 != 5 )
          goto LABEL_73;
        v61 = *((unsigned __int16 *)a3 + 1);
      }
      if ( v61 != 47 )
      {
LABEL_117:
        v221 = (__int64 *)a1;
        v222 = v12;
        v223 = a1;
        v224 = (unsigned __int64 *)&v211;
        if ( v42 == 57 )
        {
          v62 = (unsigned __int8 *)*((_QWORD *)a3 - 8);
          if ( (unsigned __int8 *)a1 == v62
            || (v160 = v13,
                v171 = v14,
                v193 = v12,
                v63 = sub_9877A0((__int64)&v221, v62),
                v12 = v193,
                v14 = v171,
                v13 = v160,
                v63) )
          {
            v64 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
            if ( v64 )
            {
              *v224 = (unsigned __int64)v64;
LABEL_178:
              if ( *(_DWORD *)(a5 + 24) > 0x40u )
              {
                v203 = v13;
                sub_C43BD0(a5 + 16, v13);
                v13 = v203;
              }
              else
              {
                *(_QWORD *)(a5 + 16) |= *(_QWORD *)v13;
              }
              v178 = v13;
              v221 = v212;
              LOBYTE(v222) = 0;
              LOBYTE(v10) = sub_991580((__int64)&v221, v211);
              if ( !(_BYTE)v10 )
                return v10;
              v96 = (_QWORD *)v212[0];
              sub_9865C0((__int64)&v213, v178);
              sub_987160((__int64)&v213, v178, v97, v98, v99);
              v100 = v214;
              LODWORD(v214) = 0;
              LODWORD(v218) = v100;
              v217 = v213;
              if ( v100 > 0x40 )
              {
                sub_C43B90(&v217, v96);
                v100 = v218;
                v101 = v217;
              }
              else
              {
                v101 = *v96 & v213;
                v217 = v101;
              }
              LODWORD(v222) = v100;
              v102 = a5;
              v221 = (__int64 *)v101;
              LODWORD(v218) = 0;
LABEL_184:
              sub_984300(v102, (__int64 *)&v221);
              sub_969240((__int64 *)&v221);
              sub_969240((__int64 *)&v217);
              LOBYTE(v10) = sub_969240((__int64 *)&v213);
              return v10;
            }
          }
          else
          {
            v64 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
          }
          if ( v221 == (__int64 *)v64
            || (v161 = v13,
                v172 = v14,
                v194 = v12,
                v65 = sub_9877A0((__int64)&v221, v64),
                v12 = v194,
                v14 = v172,
                v13 = v161,
                v65) )
          {
            v137 = *((_QWORD *)a3 - 8);
            if ( v137 )
            {
              *v224 = v137;
              goto LABEL_178;
            }
          }
          v42 = *a3;
        }
LABEL_73:
        v221 = (__int64 *)a1;
        v222 = v12;
        v223 = a1;
        v224 = (unsigned __int64 *)&v211;
        if ( v42 != 58 )
          goto LABEL_74;
        v104 = (unsigned __int8 *)*((_QWORD *)a3 - 8);
        if ( (unsigned __int8 *)a1 == v104
          || (v163 = v13,
              v179 = v14,
              v201 = v12,
              v105 = sub_9877A0((__int64)&v221, v104),
              v12 = v201,
              v14 = v179,
              v13 = v163,
              v105) )
        {
          v106 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
          if ( v106 )
          {
            *v224 = (unsigned __int64)v106;
LABEL_236:
            v124 = v13;
            v206 = v13;
            sub_9865C0((__int64)&v217, v13);
            sub_987160((__int64)&v217, v124, v125, v126, v127);
            v128 = v218;
            v129 = *(_DWORD *)(a5 + 8) <= 0x40u;
            LODWORD(v218) = 0;
            v130 = (_QWORD *)v206;
            LODWORD(v222) = v128;
            v221 = (__int64 *)v217;
            if ( v129 )
            {
              *(_QWORD *)a5 |= v217;
            }
            else
            {
              sub_C43BD0(a5, &v221);
              v130 = (_QWORD *)v206;
            }
            v183 = v130;
            sub_969240((__int64 *)&v221);
            sub_969240((__int64 *)&v217);
            v221 = v212;
            LOBYTE(v222) = 0;
            LOBYTE(v10) = sub_991580((__int64)&v221, v211);
            if ( !(_BYTE)v10 )
              return v10;
            v131 = v212[0];
            sub_9865C0((__int64)&v213, v212[0]);
            sub_987160((__int64)&v213, v131, v132, v133, v134);
            v135 = v214;
            LODWORD(v214) = 0;
            LODWORD(v218) = v135;
            v217 = v213;
            if ( v135 > 0x40 )
            {
              sub_C43B90(&v217, v183);
              v135 = v218;
              v136 = v217;
            }
            else
            {
              v136 = *v183 & v213;
              v217 = v136;
            }
            LODWORD(v222) = v135;
            v102 = a5 + 16;
            v221 = (__int64 *)v136;
            LODWORD(v218) = 0;
            goto LABEL_184;
          }
        }
        else
        {
          v106 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
        }
        if ( v106 != (unsigned __int8 *)v221 )
        {
          v164 = v13;
          v180 = v14;
          v202 = v12;
          v107 = sub_9877A0((__int64)&v221, v106);
          v12 = v202;
          v14 = v180;
          v13 = v164;
          if ( !v107 )
            goto LABEL_200;
        }
        v146 = *((_QWORD *)a3 - 8);
        if ( !v146 )
        {
LABEL_200:
          v42 = *a3;
LABEL_74:
          v221 = (__int64 *)a1;
          v222 = v12;
          v223 = a1;
          v224 = &v213;
          if ( v42 != 54 )
            goto LABEL_75;
          v138 = (unsigned __int8 *)*((_QWORD *)a3 - 8);
          if ( (unsigned __int8 *)a1 == v138
            || (v157 = v13,
                v165 = v14,
                v207 = v12,
                v139 = sub_9877A0((__int64)&v221, v138),
                v12 = v207,
                v14 = v165,
                v13 = v157,
                v139) )
          {
            v140 = *((_QWORD *)a3 - 4);
            if ( *(_BYTE *)v140 == 17 )
            {
              v141 = *(_DWORD *)(v140 + 32);
              if ( v141 <= 0x40 )
              {
                v142 = v224;
                v143 = *(_QWORD *)(v140 + 24);
                goto LABEL_249;
              }
              v158 = v13;
              v166 = v14;
              v209 = v12;
              v147 = sub_C444A0(v140 + 24);
              v12 = v209;
              v14 = v166;
              v13 = v158;
              if ( v141 - v147 <= 0x40 )
              {
                v142 = v224;
                v143 = **(_QWORD **)(v140 + 24);
LABEL_249:
                *v142 = v143;
                if ( v14 > v213 )
                {
                  sub_987BA0((__int64)&v217, (__int64 *)v13);
                  v145 = v213;
                  if ( (unsigned int)v218 > 0x40 )
                  {
                    sub_C482E0(&v217, (unsigned int)v213);
                    v145 = v213;
                  }
                  else if ( (_DWORD)v213 == (_DWORD)v218 )
                  {
                    v217 = 0;
                  }
                  else
                  {
                    v217 >>= v213;
                  }
                  if ( v220 > 0x40 )
                  {
                    sub_C482E0(&v219, v145);
                  }
                  else if ( v145 == v220 )
                  {
                    v219 = 0;
                  }
                  else
                  {
                    v219 >>= v145;
                  }
                  sub_987D70((__int64)&v221, (__int64 *)a5, &v217);
                  sub_984AC0((__int64 *)a5, (__int64 *)&v221);
                  sub_969240(&v223);
                  sub_969240((__int64 *)&v221);
                  sub_969240((__int64 *)&v219);
                  LOBYTE(v10) = sub_969240((__int64 *)&v217);
                  return v10;
                }
              }
            }
          }
          v42 = *a3;
LABEL_75:
          v191 = v14;
          LOBYTE(v10) = v42 - 55;
          v159 = (__int64 *)v13;
          v221 = (__int64 *)a1;
          v222 = v12;
          v223 = a1;
          v224 = &v213;
          if ( (unsigned __int8)v10 <= 1u )
          {
            v10 = sub_986520((__int64)a3);
            v43 = v191;
            v44 = v159;
            if ( a1 != *(_QWORD *)v10 )
            {
              LOBYTE(v10) = sub_9877A0((__int64)&v221, *(unsigned __int8 **)v10);
              if ( !(_BYTE)v10 )
                return v10;
              v10 = sub_986520((__int64)a3);
              v44 = v159;
              v43 = v191;
            }
            v45 = *(_QWORD *)(v10 + 32);
            if ( *(_BYTE *)v45 == 17 )
            {
              v46 = *(_DWORD *)(v45 + 32);
              if ( v46 > 0x40 )
              {
                v184 = v44;
                v210 = v43;
                LODWORD(v10) = sub_C444A0(v45 + 24);
                if ( v46 - (unsigned int)v10 > 0x40 )
                  return v10;
                v43 = v210;
                v44 = v184;
                v10 = (__int64)v224;
                v47 = **(_QWORD **)(v45 + 24);
              }
              else
              {
                v10 = (__int64)v224;
                v47 = *(_QWORD *)(v45 + 24);
              }
              *(_QWORD *)v10 = v47;
              if ( v43 > v213 )
              {
                sub_987BA0((__int64)&v221, v44);
                v48 = v213;
                sub_9865C0((__int64)&v217, (__int64)&v221);
                sub_984A70((__int64)&v217, v48);
                sub_984300(a5, (__int64 *)&v217);
                sub_969240((__int64 *)&v217);
                v49 = v213;
                sub_9865C0((__int64)&v217, (__int64)&v223);
                sub_984A70((__int64)&v217, v49);
                sub_984300(a5 + 16, (__int64 *)&v217);
                sub_969240((__int64 *)&v217);
                sub_969240(&v223);
                LOBYTE(v10) = sub_969240((__int64 *)&v221);
              }
            }
          }
          return v10;
        }
        *v224 = v146;
        goto LABEL_236;
      }
      if ( (a3[7] & 0x40) != 0 )
        v78 = (unsigned __int8 *)*((_QWORD *)a3 - 1);
      else
        v78 = &a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
      v162 = v13;
      v174 = v14;
      v197 = v12;
      v79 = sub_9208B0(v12, *(_QWORD *)(*(_QWORD *)v78 + 8LL));
      v80 = *((_QWORD *)a3 + 1);
      v221 = (__int64 *)v79;
      v222 = v81;
      v82 = sub_9208B0(v197, v80);
      v12 = v197;
      v217 = v82;
      v14 = v174;
      v218 = v83;
      v13 = v162;
      if ( (__int64 *)v82 != v221 || (_BYTE)v218 != (_BYTE)v222 )
      {
        v42 = *a3;
        goto LABEL_117;
      }
      v108 = (_QWORD *)sub_986520((__int64)a3);
      v13 = v162;
      if ( a1 != *v108 )
      {
        v42 = *a3;
        v12 = v197;
        v14 = v174;
        goto LABEL_117;
      }
    }
    sub_987BA0((__int64)&v217, (__int64 *)v13);
    sub_987D70((__int64)&v221, (__int64 *)a5, &v217);
    LOBYTE(v10) = sub_984AC0((__int64 *)a5, (__int64 *)&v221);
    if ( (unsigned int)v224 > 0x40 && v223 )
      LOBYTE(v10) = j_j___libc_free_0_0(v223);
    if ( (unsigned int)v222 > 0x40 && v221 )
      LOBYTE(v10) = j_j___libc_free_0_0(v221);
    if ( v220 > 0x40 && v219 )
      LOBYTE(v10) = j_j___libc_free_0_0(v219);
    if ( (unsigned int)v218 > 0x40 && v217 )
      LOBYTE(v10) = j_j___libc_free_0_0(v217);
    return v10;
  }
  if ( a2 == 33 )
  {
    v34 = *(_DWORD *)(v13 + 8);
    if ( v34 <= 0x40 )
    {
      LOBYTE(v10) = *(_QWORD *)v13 == 0;
    }
    else
    {
      v189 = v12;
      LODWORD(v10) = sub_C444A0(v13);
      v12 = v189;
      LOBYTE(v10) = v34 == (_DWORD)v10;
    }
    if ( !(_BYTE)v10 || *a3 != 57 )
      return v10;
    v35 = (unsigned __int8 *)*((_QWORD *)a3 - 8);
    if ( (unsigned __int8 *)a1 != v35 )
    {
      LODWORD(v10) = *v35;
      if ( (unsigned __int8)v10 > 0x1Cu )
      {
        LODWORD(v10) = v10 - 29;
      }
      else
      {
        if ( (_BYTE)v10 != 5 )
          return v10;
        LODWORD(v10) = *((unsigned __int16 *)v35 + 1);
      }
      if ( (_DWORD)v10 != 47 )
        return v10;
      v36 = (v35[7] & 0x40) != 0
          ? (unsigned __int8 *)*((_QWORD *)v35 - 1)
          : &v35[-32 * (*((_DWORD *)v35 + 1) & 0x7FFFFFF)];
      v190 = v12;
      v221 = (__int64 *)sub_9208B0(v12, *(_QWORD *)(*(_QWORD *)v36 + 8LL));
      v222 = v37;
      v10 = sub_9208B0(v190, *((_QWORD *)v35 + 1));
      v217 = v10;
      v218 = v38;
      if ( (__int64 *)v10 != v221 )
        return v10;
      LOBYTE(v10) = v222;
      if ( (_BYTE)v218 != (_BYTE)v222 )
        return v10;
      if ( (v35[7] & 0x40) != 0 )
      {
        v39 = (unsigned __int8 *)*((_QWORD *)v35 - 1);
      }
      else
      {
        v10 = 32LL * (*((_DWORD *)v35 + 1) & 0x7FFFFFF);
        v39 = &v35[-v10];
      }
      if ( a1 != *(_QWORD *)v39 )
        return v10;
    }
    v40 = *((_QWORD *)a3 - 4);
    LOBYTE(v10) = *(_BYTE *)v40;
    if ( *(_BYTE *)v40 == 17 )
    {
      v41 = (__int64 *)(v40 + 24);
      if ( sub_986BA0(v40 + 24) )
        goto LABEL_68;
      LODWORD(v10) = *(unsigned __int8 *)(*(_QWORD *)(v40 + 8) + 8LL) - 17;
      if ( (unsigned int)v10 > 1 )
        return v10;
    }
    else if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v40 + 8) + 8LL) - 17 > 1 || (unsigned __int8)v10 > 0x15u )
    {
      return v10;
    }
    v10 = sub_AD7630(v40, 1);
    if ( !v10 )
      return v10;
    if ( *(_BYTE *)v10 != 17 )
      return v10;
    v41 = (__int64 *)(v10 + 24);
    LOBYTE(v10) = sub_986BA0(v10 + 24);
    if ( !(_BYTE)v10 )
      return v10;
LABEL_68:
    if ( *(_DWORD *)(a5 + 24) > 0x40u )
    {
      LOBYTE(v10) = sub_C43BD0(a5 + 16, v41);
    }
    else
    {
      v10 = *v41;
      *(_QWORD *)(a5 + 16) |= *v41;
    }
    return v10;
  }
  v221 = (__int64 *)a1;
  v211 = 0;
  v222 = v12;
  v223 = a1;
  v224 = (unsigned __int64 *)a1;
  v225 = v12;
  v226 = a1;
  v227 = &v211;
  v228 = 0;
  v229[0] = a1;
  v229[1] = v12;
  v229[2] = a1;
  v230 = &v211;
  v231 = 0;
  if ( (unsigned __int8 *)a1 == a3 )
    goto LABEL_24;
  v15 = *a3;
  if ( *a3 <= 0x1Cu )
  {
    if ( v15 != 5 )
      goto LABEL_97;
    v16 = *((unsigned __int16 *)a3 + 1);
  }
  else
  {
    v16 = v15 - 29;
  }
  if ( v16 == 47 )
  {
    if ( (a3[7] & 0x40) != 0 )
      v72 = (unsigned __int8 *)*((_QWORD *)a3 - 1);
    else
      v72 = &a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
    v173 = v13;
    v196 = v12;
    v73 = sub_9208B0(v12, *(_QWORD *)(*(_QWORD *)v72 + 8LL));
    v74 = *((_QWORD *)a3 + 1);
    v217 = v73;
    v218 = v75;
    v76 = sub_9208B0(v222, v74);
    v12 = v196;
    v213 = v76;
    v13 = v173;
    v214 = v77;
    if ( v76 == v217 && (_BYTE)v214 == (_BYTE)v218 )
    {
      v109 = (_QWORD *)sub_986520((__int64)a3);
      v12 = v196;
      v13 = v173;
      if ( *v109 == v223 )
        goto LABEL_24;
    }
    v15 = *a3;
  }
  if ( v15 != 42 )
    goto LABEL_19;
  v17 = (unsigned __int64 *)*((_QWORD *)a3 - 8);
  if ( v17 == v224 )
  {
LABEL_173:
    v177 = v13;
    v200 = v12;
    v95 = sub_991580((__int64)&v227, *((_QWORD *)a3 - 4));
    v12 = v200;
    v13 = v177;
    if ( !v95 )
    {
LABEL_18:
      v15 = *a3;
LABEL_19:
      if ( v15 != 58 )
        goto LABEL_97;
      if ( (a3[1] & 2) == 0 )
        goto LABEL_97;
      v24 = (unsigned __int8 *)*((_QWORD *)a3 - 8);
      if ( v24 != (unsigned __int8 *)v229[0] )
      {
        v168 = v13;
        v186 = v12;
        v25 = sub_9877A0((__int64)v229, v24);
        v12 = v186;
        v13 = v168;
        if ( !v25 )
          goto LABEL_97;
      }
      v169 = v13;
      v187 = v12;
      v26 = sub_991580((__int64)&v230, *((_QWORD *)a3 - 4));
      v12 = v187;
      v13 = v169;
      if ( !v26 )
        goto LABEL_97;
    }
LABEL_24:
    v152 = v12;
    v155 = v13;
    sub_9865C0((__int64)&v217, v13);
    sub_AADBC0(&v221, &v217);
    sub_AB15A0(&v213, a2, &v221);
    sub_969240(&v223);
    sub_969240((__int64 *)&v221);
    sub_969240((__int64 *)&v217);
    v27 = v155;
    v28 = v152;
    if ( v211 )
    {
      sub_9865C0((__int64)v212, v211);
      sub_AADBC0(&v217, v212);
      sub_AB51C0(&v221, &v213, &v217);
      v29 = v152;
      v30 = v155;
      if ( (unsigned int)v214 > 0x40 && v213 )
      {
        j_j___libc_free_0_0(v213);
        v30 = v155;
        v29 = v152;
      }
      v213 = (unsigned __int64)v221;
      v31 = v222;
      LODWORD(v222) = 0;
      LODWORD(v214) = v31;
      if ( v216 > 0x40 && v215 )
      {
        v150 = v30;
        v153 = v29;
        j_j___libc_free_0_0(v215);
        v30 = v150;
        v29 = v153;
      }
      v149 = v30;
      v151 = v29;
      v215 = v223;
      v32 = (unsigned int)v224;
      LODWORD(v224) = 0;
      v216 = v32;
      sub_969240(&v223);
      sub_969240((__int64 *)&v221);
      sub_969240((__int64 *)&v219);
      sub_969240((__int64 *)&v217);
      sub_969240(v212);
      v27 = v149;
      v28 = v151;
    }
    v154 = v27;
    v156 = v28;
    sub_AB0A90(&v217, &v213);
    sub_987D70((__int64)&v221, (__int64 *)a5, &v217);
    sub_984AC0((__int64 *)a5, (__int64 *)&v221);
    sub_969240(&v223);
    sub_969240((__int64 *)&v221);
    sub_969240((__int64 *)&v219);
    sub_969240((__int64 *)&v217);
    sub_969240(&v215);
    sub_969240((__int64 *)&v213);
    v13 = v154;
    v12 = v156;
    goto LABEL_97;
  }
  v18 = *(unsigned __int8 *)v17;
  if ( (unsigned __int8)v18 > 0x1Cu )
  {
    v19 = v18 - 29;
  }
  else
  {
    if ( (_BYTE)v18 != 5 )
      goto LABEL_97;
    v19 = *((unsigned __int16 *)v17 + 1);
  }
  if ( v19 == 47 )
  {
    if ( (*((_BYTE *)v17 + 7) & 0x40) != 0 )
      v20 = (unsigned __int8 *)*(v17 - 1);
    else
      v20 = (unsigned __int8 *)&v17[-4 * (*((_DWORD *)v17 + 1) & 0x7FFFFFF)];
    v167 = v13;
    v185 = v12;
    v217 = sub_9208B0(v225, *(_QWORD *)(*(_QWORD *)v20 + 8LL));
    v218 = v21;
    v22 = sub_9208B0(v225, v17[1]);
    v12 = v185;
    v213 = v22;
    v13 = v167;
    v214 = v23;
    if ( v22 != v217 || (_BYTE)v214 != (_BYTE)v218 )
      goto LABEL_18;
    v94 = (*((_BYTE *)v17 + 7) & 0x40) != 0
        ? (unsigned __int8 *)*(v17 - 1)
        : (unsigned __int8 *)&v17[-4 * (*((_DWORD *)v17 + 1) & 0x7FFFFFF)];
    if ( *(_QWORD *)v94 != v226 )
      goto LABEL_18;
    goto LABEL_173;
  }
LABEL_97:
  if ( a2 - 34 <= 1 )
  {
    v50 = *a3 == 57;
    v217 = a1;
    v218 = v12;
    v219 = a1;
    if ( !v50 )
    {
LABEL_126:
      v195 = v13;
      v221 = (__int64 *)a1;
      v222 = v12;
      v223 = a1;
      LOBYTE(v10) = sub_987880(a3);
      if ( !(_BYTE)v10 )
        return v10;
      v66 = *a3;
      v13 = v195;
      LODWORD(v10) = (unsigned __int8)v66 <= 0x1Cu ? *((unsigned __int16 *)a3 + 1) : v66 - 29;
      if ( (_DWORD)v10 != 15 || (a3[1] & 2) == 0 )
        return v10;
      v67 = (unsigned __int8 *)*((_QWORD *)a3 - 8);
      if ( (unsigned __int8 *)a1 != v67 )
      {
        LOBYTE(v10) = sub_9877A0((__int64)&v221, v67);
        if ( !(_BYTE)v10 )
          return v10;
        v13 = v195;
      }
LABEL_134:
      sub_9865C0((__int64)&v217, v13);
      sub_C46A40(&v217, a2 == 34);
      v68 = v218;
      LODWORD(v218) = 0;
      LODWORD(v222) = v68;
      v221 = (__int64 *)v217;
      if ( v68 > 0x40 )
      {
        v68 = sub_C44500(&v221);
      }
      else if ( v68 )
      {
        v69 = 64 - v68;
        v68 = 64;
        if ( v217 << v69 != -1 )
        {
          _BitScanReverse64(&v70, ~(v217 << v69));
          v68 = v70 ^ 0x3F;
        }
      }
      v58 = *(unsigned int *)(a5 + 24);
      v59 = a5 + 16;
      v60 = *(_DWORD *)(a5 + 24) - v68;
      goto LABEL_114;
    }
    v84 = (unsigned __int8 *)*((_QWORD *)a3 - 8);
    if ( (unsigned __int8 *)a1 == v84 )
      goto LABEL_134;
    v85 = *v84;
    if ( (unsigned __int8)v85 > 0x1Cu )
    {
      v110 = v85 - 29;
    }
    else
    {
      v86 = a1;
      if ( (_BYTE)v85 != 5 )
        goto LABEL_160;
      v110 = *((unsigned __int16 *)v84 + 1);
    }
    v86 = a1;
    if ( v110 == 47 )
    {
      v111 = (v84[7] & 0x40) != 0
           ? (unsigned __int8 *)*((_QWORD *)v84 - 1)
           : &v84[-32 * (*((_DWORD *)v84 + 1) & 0x7FFFFFF)];
      v181 = v13;
      v204 = v12;
      v112 = sub_9208B0(v12, *(_QWORD *)(*(_QWORD *)v111 + 8LL));
      v222 = v113;
      v221 = (__int64 *)v112;
      v114 = sub_9208B0(v218, *((_QWORD *)v84 + 1));
      v12 = v204;
      v13 = v181;
      v214 = v115;
      v213 = v114;
      v86 = v217;
      if ( (__int64 *)v114 == v221 && (_BYTE)v214 == (_BYTE)v222 )
      {
        if ( (v84[7] & 0x40) != 0 )
          v116 = (unsigned __int8 *)*((_QWORD *)v84 - 1);
        else
          v116 = &v84[-32 * (*((_DWORD *)v84 + 1) & 0x7FFFFFF)];
        if ( *(_QWORD *)v116 == v219 )
          goto LABEL_134;
        v86 = v217;
      }
    }
LABEL_160:
    v87 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
    if ( (unsigned __int8 *)v86 == v87 )
      goto LABEL_134;
    v175 = v13;
    v198 = v12;
    v88 = sub_9877A0((__int64)&v217, v87);
    v12 = v198;
    v13 = v175;
    if ( v88 )
      goto LABEL_134;
    goto LABEL_126;
  }
  LOBYTE(v10) = a2 - 36;
  if ( a2 - 36 > 1 )
    return v10;
  v50 = *a3 == 58;
  v217 = a1;
  v218 = v12;
  v219 = a1;
  if ( v50 )
  {
    v89 = (unsigned __int8 *)*((_QWORD *)a3 - 8);
    if ( (unsigned __int8 *)a1 == v89 )
      goto LABEL_110;
    v90 = *v89;
    if ( (unsigned __int8)v90 > 0x1Cu )
    {
      v117 = v90 - 29;
    }
    else
    {
      v91 = a1;
      if ( (_BYTE)v90 != 5 )
        goto LABEL_166;
      v117 = *((unsigned __int16 *)v89 + 1);
    }
    v91 = a1;
    if ( v117 == 47 )
    {
      v118 = (v89[7] & 0x40) != 0
           ? (unsigned __int8 *)*((_QWORD *)v89 - 1)
           : &v89[-32 * (*((_DWORD *)v89 + 1) & 0x7FFFFFF)];
      v182 = v13;
      v205 = v12;
      v119 = sub_9208B0(v12, *(_QWORD *)(*(_QWORD *)v118 + 8LL));
      v222 = v120;
      v221 = (__int64 *)v119;
      v121 = sub_9208B0(v218, *((_QWORD *)v89 + 1));
      v12 = v205;
      v13 = v182;
      v214 = v122;
      v213 = v121;
      v91 = v217;
      if ( (__int64 *)v121 == v221 && (_BYTE)v214 == (_BYTE)v222 )
      {
        if ( (v89[7] & 0x40) != 0 )
          v123 = (unsigned __int8 *)*((_QWORD *)v89 - 1);
        else
          v123 = &v89[-32 * (*((_DWORD *)v89 + 1) & 0x7FFFFFF)];
        if ( *(_QWORD *)v123 == v219 )
          goto LABEL_110;
        v91 = v217;
      }
    }
LABEL_166:
    v92 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
    if ( v92 != (unsigned __int8 *)v91 )
    {
      v176 = v13;
      v199 = v12;
      v93 = sub_9877A0((__int64)&v217, v92);
      v12 = v199;
      v13 = v176;
      if ( !v93 )
        goto LABEL_100;
    }
LABEL_110:
    sub_9865C0((__int64)&v217, v13);
    sub_C46F20(&v217, a2 == 36);
    v55 = v218;
    LODWORD(v218) = 0;
    LODWORD(v222) = v55;
    v221 = (__int64 *)v217;
    if ( v55 > 0x40 )
    {
      v55 = sub_C444A0(&v221);
    }
    else
    {
      v56 = v55 - 64;
      if ( v217 )
      {
        _BitScanReverse64(&v57, v217);
        v55 = v56 + (v57 ^ 0x3F);
      }
    }
    v58 = *(unsigned int *)(a5 + 8);
    v59 = a5;
    v60 = *(_DWORD *)(a5 + 8) - v55;
LABEL_114:
    sub_9870B0(v59, v60, v58);
    sub_969240((__int64 *)&v221);
    LOBYTE(v10) = sub_969240((__int64 *)&v217);
    return v10;
  }
LABEL_100:
  v192 = v13;
  v221 = (__int64 *)a1;
  v222 = v12;
  v223 = a1;
  LOBYTE(v10) = sub_987880(a3);
  if ( (_BYTE)v10 )
  {
    v51 = *a3;
    v13 = v192;
    LODWORD(v10) = (unsigned __int8)v51 <= 0x1Cu ? *((unsigned __int16 *)a3 + 1) : v51 - 29;
    if ( (_DWORD)v10 == 13 && (a3[1] & 2) != 0 )
    {
      v52 = (unsigned __int8 *)*((_QWORD *)a3 - 8);
      if ( (unsigned __int8 *)a1 != v52 )
      {
        v53 = sub_9877A0((__int64)&v221, v52);
        v13 = v192;
        if ( !v53 )
        {
          v54 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
          if ( v54 != (unsigned __int8 *)v221 )
          {
            LOBYTE(v10) = sub_9877A0((__int64)&v221, v54);
            if ( !(_BYTE)v10 )
              return v10;
            v13 = v192;
          }
        }
      }
      goto LABEL_110;
    }
  }
  return v10;
}
