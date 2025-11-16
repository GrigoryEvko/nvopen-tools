// Function: sub_3447D70
// Address: 0x3447d70
//
__int64 __fastcall sub_3447D70(
        unsigned int *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        _QWORD **a5,
        _QWORD **a6,
        __m128i a7,
        unsigned int a8)
{
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rsi
  unsigned int v12; // ebx
  unsigned int *v13; // r11
  _QWORD **v15; // r10
  __int64 v16; // r13
  int v17; // eax
  _QWORD *v18; // rax
  unsigned int v19; // ebx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rcx
  unsigned int v23; // r8d
  int v24; // ebx
  int v25; // r9d
  __int64 v26; // r10
  unsigned int *v27; // r11
  __int64 v28; // rdx
  __int64 result; // rax
  _QWORD *v30; // rbx
  int v31; // eax
  unsigned int v32; // ecx
  __int64 *v33; // rax
  __int64 v34; // rbx
  bool v35; // al
  int v36; // r9d
  __int64 *v37; // rax
  __int64 v38; // rbx
  __int64 v39; // rcx
  __int64 v40; // rax
  __int16 v41; // dx
  __int64 v42; // rax
  __int16 v43; // dx
  unsigned int v44; // edx
  unsigned __int16 *v45; // rax
  __int64 v46; // rcx
  __int64 v47; // rsi
  __int64 v48; // rdx
  __int16 v49; // ax
  __int64 v50; // rcx
  __int64 *v51; // rax
  __int64 v52; // rdi
  __int64 v53; // rbx
  __int64 v54; // rax
  __int64 v55; // rcx
  unsigned int v56; // r13d
  int v57; // eax
  int v58; // r9d
  __int64 v59; // r10
  __int64 (*v60)(); // rax
  __int64 v61; // r12
  __int64 v62; // r13
  __int64 v63; // r15
  char v64; // al
  int v65; // r9d
  __int64 v66; // r14
  int v67; // r12d
  _QWORD *v68; // rax
  __int64 v69; // rdx
  __int64 *v70; // rax
  __int64 v71; // r12
  __int64 v72; // r13
  int v73; // ebx
  _QWORD *v74; // rax
  __int64 v75; // rdx
  __int64 *v76; // rax
  unsigned int v77; // r15d
  char v78; // al
  int v79; // edx
  __int64 v80; // rax
  char v81; // r15
  unsigned __int64 *v82; // rsi
  int v83; // eax
  char v84; // r14
  int v85; // edx
  __int64 v86; // rax
  __int64 v87; // rdx
  unsigned __int8 *v88; // rbx
  __int64 v89; // rcx
  __int64 v90; // rax
  __int64 v91; // rdx
  _QWORD *v92; // r14
  __int64 v93; // rax
  __int64 *v94; // r12
  __int16 v95; // dx
  __int64 v96; // rax
  unsigned int v97; // eax
  __int16 *v98; // rdx
  __int16 v99; // ax
  __int64 v100; // rdx
  __int64 v101; // rbx
  __int64 i; // r12
  int v103; // r14d
  __int64 v104; // rax
  __int64 v105; // r14
  int v106; // edx
  __int64 v107; // rbx
  __int64 v108; // rax
  __int16 v109; // dx
  __int64 v110; // rax
  __int64 v111; // r12
  unsigned int v112; // eax
  __int64 v113; // rax
  _QWORD *v114; // rsi
  bool v115; // al
  char v116; // al
  bool v117; // al
  unsigned int v118; // ebx
  unsigned int v119; // eax
  int v120; // r11d
  __int64 v121; // r10
  bool v122; // al
  int v123; // r11d
  __int64 v124; // r10
  unsigned int v125; // r12d
  bool v126; // al
  int v127; // r11d
  __int64 v128; // r10
  int v129; // r11d
  __int64 v130; // r10
  unsigned int k; // r12d
  __int64 v132; // rax
  __int64 v133; // rdx
  __int64 v134; // rdx
  __int64 v135; // rcx
  __int64 v136; // r8
  __int64 v137; // rdx
  __int64 v138; // rdx
  int v139; // r13d
  __int64 v140; // rax
  unsigned int v141; // edx
  __int64 v142; // rax
  unsigned int v143; // edx
  unsigned int v144; // ebx
  int v145; // r11d
  __int64 v146; // r10
  unsigned int v147; // ecx
  unsigned int v148; // r13d
  unsigned int j; // r12d
  __int64 v150; // rax
  unsigned int v151; // edx
  unsigned int v152; // eax
  __int64 v153; // [rsp+8h] [rbp-148h]
  int v154; // [rsp+10h] [rbp-140h]
  unsigned int v155; // [rsp+18h] [rbp-138h]
  __int64 v156; // [rsp+18h] [rbp-138h]
  __int64 v157; // [rsp+20h] [rbp-130h]
  int v158; // [rsp+20h] [rbp-130h]
  int v159; // [rsp+20h] [rbp-130h]
  __int64 v160; // [rsp+20h] [rbp-130h]
  unsigned int v161; // [rsp+2Ch] [rbp-124h]
  __int64 v162; // [rsp+30h] [rbp-120h]
  unsigned int v163; // [rsp+30h] [rbp-120h]
  __int64 v164; // [rsp+30h] [rbp-120h]
  __int64 v165; // [rsp+38h] [rbp-118h]
  unsigned int v166; // [rsp+38h] [rbp-118h]
  __int64 v167; // [rsp+40h] [rbp-110h]
  int v168; // [rsp+40h] [rbp-110h]
  char v169; // [rsp+48h] [rbp-108h]
  char v170; // [rsp+48h] [rbp-108h]
  unsigned int v171; // [rsp+48h] [rbp-108h]
  int v172; // [rsp+50h] [rbp-100h]
  __int64 v173; // [rsp+50h] [rbp-100h]
  __int64 v175; // [rsp+58h] [rbp-F8h]
  _QWORD **v176; // [rsp+58h] [rbp-F8h]
  _QWORD **v177; // [rsp+58h] [rbp-F8h]
  __int64 v178; // [rsp+58h] [rbp-F8h]
  unsigned int v179; // [rsp+58h] [rbp-F8h]
  unsigned int *v180; // [rsp+58h] [rbp-F8h]
  __int64 v181; // [rsp+58h] [rbp-F8h]
  char v182; // [rsp+58h] [rbp-F8h]
  __int64 v183; // [rsp+60h] [rbp-F0h]
  unsigned int *v184; // [rsp+60h] [rbp-F0h]
  __int64 v185; // [rsp+60h] [rbp-F0h]
  __int64 v186; // [rsp+60h] [rbp-F0h]
  _QWORD **v187; // [rsp+60h] [rbp-F0h]
  int v188; // [rsp+60h] [rbp-F0h]
  _QWORD **v189; // [rsp+60h] [rbp-F0h]
  __int64 v190; // [rsp+60h] [rbp-F0h]
  int v191; // [rsp+60h] [rbp-F0h]
  int v192; // [rsp+60h] [rbp-F0h]
  int v193; // [rsp+60h] [rbp-F0h]
  unsigned __int8 *v194; // [rsp+60h] [rbp-F0h]
  char v195; // [rsp+60h] [rbp-F0h]
  unsigned __int8 *v196; // [rsp+60h] [rbp-F0h]
  _QWORD **v197; // [rsp+68h] [rbp-E8h]
  unsigned int *v198; // [rsp+68h] [rbp-E8h]
  unsigned int *v199; // [rsp+68h] [rbp-E8h]
  __int64 v200; // [rsp+68h] [rbp-E8h]
  __int64 v201; // [rsp+68h] [rbp-E8h]
  __int64 v202; // [rsp+68h] [rbp-E8h]
  __int64 v203; // [rsp+68h] [rbp-E8h]
  int v204; // [rsp+68h] [rbp-E8h]
  __int64 v205; // [rsp+68h] [rbp-E8h]
  int v206; // [rsp+68h] [rbp-E8h]
  int v207; // [rsp+68h] [rbp-E8h]
  unsigned int v208; // [rsp+68h] [rbp-E8h]
  char v209; // [rsp+68h] [rbp-E8h]
  int v210; // [rsp+68h] [rbp-E8h]
  __int64 v211; // [rsp+68h] [rbp-E8h]
  unsigned int v212; // [rsp+68h] [rbp-E8h]
  int v213; // [rsp+68h] [rbp-E8h]
  __int64 v214; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v215; // [rsp+78h] [rbp-D8h]
  unsigned __int16 v216; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v217; // [rsp+88h] [rbp-C8h]
  unsigned int v218; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v219; // [rsp+98h] [rbp-B8h]
  __int64 v220; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v221; // [rsp+A8h] [rbp-A8h]
  __int64 v222; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v223; // [rsp+B8h] [rbp-98h]
  unsigned __int64 v224; // [rsp+C0h] [rbp-90h] BYREF
  unsigned int v225; // [rsp+C8h] [rbp-88h]
  unsigned __int64 v226; // [rsp+D0h] [rbp-80h] BYREF
  unsigned int v227; // [rsp+D8h] [rbp-78h]
  unsigned __int64 v228; // [rsp+E0h] [rbp-70h] BYREF
  unsigned int v229; // [rsp+E8h] [rbp-68h]
  unsigned __int64 v230; // [rsp+F0h] [rbp-60h] BYREF
  int v231; // [rsp+F8h] [rbp-58h]
  _QWORD *v232; // [rsp+100h] [rbp-50h] BYREF
  __int64 v233; // [rsp+108h] [rbp-48h]
  __int64 v234[8]; // [rsp+110h] [rbp-40h] BYREF

  v9 = a2;
  v10 = 16LL * (unsigned int)a3 + *(_QWORD *)(a2 + 48);
  v11 = *(_QWORD *)(v10 + 8);
  LOWORD(v10) = *(_WORD *)v10;
  v215 = v11;
  LOWORD(v214) = v10;
  if ( a8 > 5 || *(_DWORD *)(v9 + 24) == 51 )
    return 0;
  v12 = *(_DWORD *)(a4 + 8);
  v13 = a1;
  v15 = a6;
  v16 = 16LL * (unsigned int)a3;
  if ( v12 <= 0x40 )
  {
    v18 = *(_QWORD **)a4;
LABEL_6:
    if ( !v18 )
      goto LABEL_16;
    goto LABEL_7;
  }
  v183 = a3;
  v17 = sub_C444A0(a4);
  v13 = a1;
  a3 = v183;
  v15 = a6;
  if ( v12 - v17 <= 0x40 )
  {
    v18 = **(_QWORD ***)a4;
    goto LABEL_6;
  }
LABEL_7:
  v19 = *((_DWORD *)a5 + 2);
  if ( v19 > 0x40 )
  {
    v176 = v15;
    v185 = a3;
    v198 = v13;
    v31 = sub_C444A0((__int64)a5);
    v13 = v198;
    a3 = v185;
    v15 = v176;
    if ( v19 - v31 > 0x40 )
    {
LABEL_10:
      v175 = a3;
      v184 = v13;
      v197 = v15;
      v21 = sub_2E79000(v15[5]);
      v24 = *(_DWORD *)(v9 + 24);
      v25 = *(_DWORD *)(a4 + 8);
      LOBYTE(v21) = *(_BYTE *)v21;
      v26 = (__int64)v197;
      v225 = 1;
      v227 = 1;
      v27 = v184;
      v224 = 0;
      v28 = v175;
      v169 = v21 ^ 1;
      LODWORD(v21) = *((_DWORD *)a5 + 2);
      v226 = 0;
      v172 = v21;
      v229 = 1;
      v228 = 0;
      v231 = 1;
      v230 = 0;
      if ( v24 <= 234 )
      {
        if ( v24 > 156 )
        {
          switch ( v24 )
          {
            case 157:
              if ( sub_3280200((__int64)&v214) )
                goto LABEL_45;
              v104 = *(_QWORD *)(v9 + 40);
              v105 = *(_QWORD *)(v104 + 80);
              v106 = *(_DWORD *)(v105 + 24);
              if ( v106 != 11 && v106 != 35 )
                goto LABEL_45;
              v107 = *(_QWORD *)v104;
              v108 = *(_QWORD *)(*(_QWORD *)v104 + 48LL) + 16LL * *(unsigned int *)(v104 + 8);
              v109 = *(_WORD *)v108;
              v110 = *(_QWORD *)(v108 + 8);
              LOWORD(v232) = v109;
              v233 = v110;
              v111 = *(_QWORD *)(v105 + 96) + 24LL;
              v112 = sub_3281500(&v232, v11);
              if ( !sub_986EE0(v111, v112) )
                goto LABEL_58;
              v113 = *(_QWORD *)(v105 + 96);
              v114 = *(_QWORD **)(v113 + 24);
              if ( *(_DWORD *)(v113 + 32) > 0x40u )
                v114 = (_QWORD *)*v114;
              v115 = sub_986C60((__int64 *)a5, (unsigned int)v114);
              v32 = v231;
              if ( v115 )
                goto LABEL_59;
              result = v107;
              goto LABEL_36;
            case 160:
              if ( sub_3280200((__int64)&v214) )
                goto LABEL_45;
              v87 = *(_QWORD *)(v9 + 40);
              v88 = *(unsigned __int8 **)v87;
              v89 = *(_QWORD *)(v87 + 40);
              v90 = *(unsigned int *)(v87 + 48);
              v91 = *(_QWORD *)(*(_QWORD *)(v87 + 80) + 96LL);
              v92 = *(_QWORD **)(v91 + 24);
              if ( *(_DWORD *)(v91 + 32) > 0x40u )
                v92 = (_QWORD *)*v92;
              v93 = *(_QWORD *)(v89 + 48) + 16 * v90;
              v94 = (__int64 *)&v232;
              v95 = *(_WORD *)v93;
              v96 = *(_QWORD *)(v93 + 8);
              LOWORD(v232) = v95;
              v233 = v96;
              v97 = sub_3281500(&v232, v11);
              sub_C440A0((__int64)&v232, (__int64 *)a5, v97, (unsigned int)v92);
              if ( sub_D94970((__int64)&v232, 0) )
              {
                v194 = v88;
                goto LABEL_86;
              }
              sub_969240((__int64 *)&v232);
              v32 = v231;
              goto LABEL_59;
            case 165:
              v98 = *(__int16 **)(v9 + 48);
              v99 = *v98;
              v100 = *((_QWORD *)v98 + 1);
              LOWORD(v232) = v99;
              v233 = v100;
              if ( v99 )
              {
                if ( (unsigned __int16)(v99 - 176) <= 0x34u )
                {
                  sub_CA17B0(
                    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be drop"
                    "ped, use EVT::getVectorElementCount() instead");
                  sub_CA17B0(
                    "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be drop"
                    "ped, use MVT::getVectorElementCount() instead");
                  v26 = (__int64)v197;
                }
              }
              else
              {
                v117 = sub_3007100((__int64)&v232);
                v26 = (__int64)v197;
                if ( v117 )
                {
                  sub_CA17B0(
                    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be drop"
                    "ped, use EVT::getVectorElementCount() instead");
                  v26 = (__int64)v197;
                }
              }
              v101 = *(_QWORD *)(v9 + 96);
              if ( !v172 )
                goto LABEL_115;
              v170 = 1;
              v182 = 1;
              v195 = 1;
              v162 = v26;
              v165 = v9;
              for ( i = 0; i != v172; ++i )
              {
                v103 = *(_DWORD *)(v101 + 4 * i);
                if ( v103 >= 0 && sub_986C60((__int64 *)a5, i) )
                {
                  v195 = 0;
                  v182 &= v103 == (_DWORD)i;
                  v170 &= v103 - v172 == (_DWORD)i;
                }
              }
              v9 = v165;
              v26 = v162;
              if ( v195 )
              {
LABEL_115:
                result = sub_3288990(
                           v26,
                           *(unsigned __int16 *)(v16 + *(_QWORD *)(v9 + 48)),
                           *(_QWORD *)(v16 + *(_QWORD *)(v9 + 48) + 8));
                goto LABEL_35;
              }
              if ( v182 )
              {
                result = **(_QWORD **)(v165 + 40);
                goto LABEL_35;
              }
              if ( !v170 )
                goto LABEL_58;
              result = *(_QWORD *)(*(_QWORD *)(v165 + 40) + 40LL);
              goto LABEL_35;
            case 186:
              sub_33D4EF0(
                (__int64)&v232,
                (__int64)v197,
                **(_QWORD **)(v9 + 40),
                *(_QWORD *)(*(_QWORD *)(v9 + 40) + 8LL),
                (__int64)a5,
                a8 + 1);
              sub_3441190((__int64)&v224, (__int64)&v232);
              sub_969240(v234);
              sub_969240((__int64 *)&v232);
              sub_33D4EF0(
                (__int64)&v232,
                (__int64)v197,
                *(_QWORD *)(*(_QWORD *)(v9 + 40) + 40LL),
                *(_QWORD *)(*(_QWORD *)(v9 + 40) + 48LL),
                (__int64)a5,
                a8 + 1);
              sub_3441190((__int64)&v228, (__int64)&v232);
              sub_969240(v234);
              sub_969240((__int64 *)&v232);
              sub_9865C0((__int64)&v222, (__int64)&v224);
              v85 = v223;
              if ( (unsigned int)v223 > 0x40 )
              {
                sub_C43BD0(&v222, (__int64 *)&v230);
                v86 = v222;
                v85 = v223;
              }
              else
              {
                v86 = v230 | v222;
                v222 |= v230;
              }
              LODWORD(v233) = v85;
              v232 = (_QWORD *)v86;
              LODWORD(v223) = 0;
              v209 = sub_10024C0((__int64 *)a4, (__int64 *)&v232);
              sub_969240((__int64 *)&v232);
              sub_969240(&v222);
              if ( v209 )
                goto LABEL_24;
              sub_9865C0((__int64)&v222, (__int64)&v228);
              v82 = &v226;
              goto LABEL_75;
            case 187:
              sub_33D4EF0(
                (__int64)&v232,
                (__int64)v197,
                **(_QWORD **)(v9 + 40),
                *(_QWORD *)(*(_QWORD *)(v9 + 40) + 8LL),
                (__int64)a5,
                a8 + 1);
              sub_3441190((__int64)&v224, (__int64)&v232);
              sub_969240(v234);
              sub_969240((__int64 *)&v232);
              sub_33D4EF0(
                (__int64)&v232,
                (__int64)v197,
                *(_QWORD *)(*(_QWORD *)(v9 + 40) + 40LL),
                *(_QWORD *)(*(_QWORD *)(v9 + 40) + 48LL),
                (__int64)a5,
                a8 + 1);
              sub_3441190((__int64)&v228, (__int64)&v232);
              sub_969240(v234);
              sub_969240((__int64 *)&v232);
              sub_9865C0((__int64)&v222, (__int64)&v226);
              v79 = v223;
              if ( (unsigned int)v223 > 0x40 )
              {
                sub_C43BD0(&v222, (__int64 *)&v228);
                v79 = v223;
                v80 = v222;
              }
              else
              {
                v80 = v228 | v222;
                v222 |= v228;
              }
              LODWORD(v233) = v79;
              v232 = (_QWORD *)v80;
              LODWORD(v223) = 0;
              v81 = sub_10024C0((__int64 *)a4, (__int64 *)&v232);
              sub_969240((__int64 *)&v232);
              sub_969240(&v222);
              if ( v81 )
                goto LABEL_24;
              sub_9865C0((__int64)&v222, (__int64)&v230);
              v82 = &v224;
LABEL_75:
              sub_343FD40(&v222, (__int64 *)v82);
              v83 = v223;
              LODWORD(v223) = 0;
              LODWORD(v233) = v83;
              v232 = (_QWORD *)v222;
              v84 = sub_10024C0((__int64 *)a4, (__int64 *)&v232);
              sub_969240((__int64 *)&v232);
              sub_969240(&v222);
              if ( !v84 )
                goto LABEL_58;
              goto LABEL_47;
            case 188:
              sub_33D4EF0(
                (__int64)&v232,
                (__int64)v197,
                **(_QWORD **)(v9 + 40),
                *(_QWORD *)(*(_QWORD *)(v9 + 40) + 8LL),
                (__int64)a5,
                a8 + 1);
              sub_3441190((__int64)&v224, (__int64)&v232);
              sub_969240(v234);
              sub_969240((__int64 *)&v232);
              sub_33D4EF0(
                (__int64)&v232,
                (__int64)v197,
                *(_QWORD *)(*(_QWORD *)(v9 + 40) + 40LL),
                *(_QWORD *)(*(_QWORD *)(v9 + 40) + 48LL),
                (__int64)a5,
                a8 + 1);
              sub_3441190((__int64)&v228, (__int64)&v232);
              sub_969240(v234);
              sub_969240((__int64 *)&v232);
              if ( sub_10024C0((__int64 *)a4, (__int64 *)&v228) )
                goto LABEL_24;
              v78 = sub_10024C0((__int64 *)a4, (__int64 *)&v224);
              v32 = v231;
              if ( !v78 )
                goto LABEL_59;
              result = *(_QWORD *)(*(_QWORD *)(v9 + 40) + 40LL);
              goto LABEL_36;
            case 190:
              v193 = v25;
              v74 = sub_33DCFD0((__int64)v197, v9, v175, (__int64)a5, a8 + 1);
              v233 = v75;
              v232 = v74;
              if ( (_BYTE)v75 )
              {
                v76 = *(__int64 **)(v9 + 40);
                v34 = *v76;
                v77 = sub_33D25A0((__int64)v197, *v76, v76[1], (__int64)a5, a8 + 1);
                if ( (unsigned int)v232 < v77 && v77 - (unsigned int)v232 >= v193 - (unsigned int)sub_D949C0(a4) )
                  goto LABEL_34;
              }
              goto LABEL_58;
            case 192:
              v192 = v25;
              v68 = sub_33DCFD0((__int64)v197, v9, v175, (__int64)a5, a8 + 1);
              v233 = v69;
              v232 = v68;
              if ( !(_BYTE)v69 )
                goto LABEL_58;
              v70 = *(__int64 **)(v9 + 40);
              v71 = *v70;
              v72 = v70[1];
              if ( (unsigned int)sub_9871A0(a4) < (unsigned int)v232 )
                goto LABEL_58;
              v73 = sub_33D25A0((__int64)v197, v71, v72, (__int64)a5, a8 + 1);
              if ( (unsigned int)sub_D949C0(a4) < v192 - v73 )
                goto LABEL_58;
              v32 = v231;
              result = v71;
              goto LABEL_36;
            case 208:
              v61 = *(_QWORD *)(v9 + 40);
              v207 = v25;
              v34 = *(_QWORD *)v61;
              v62 = *(unsigned int *)(v61 + 8);
              v63 = *(_QWORD *)(v61 + 40);
              v181 = *(_QWORD *)(v61 + 48);
              v64 = sub_986B30((__int64 *)a4, v11, v28, v22, v23);
              v65 = v207;
              if ( !v64 )
                goto LABEL_45;
              v66 = *(_QWORD *)(v61 + 40);
              v208 = *(_DWORD *)(v61 + 48);
              v67 = *(_DWORD *)(*(_QWORD *)(v61 + 80) + 96LL);
              if ( sub_34413B0(v34, v62) != v65 )
                goto LABEL_58;
              if ( (unsigned int)sub_3289F80(
                                   v184,
                                   *(unsigned __int16 *)(*(_QWORD *)(v34 + 48) + 16 * v62),
                                   *(_QWORD *)(*(_QWORD *)(v34 + 48) + 16 * v62 + 8)) != 2 )
                goto LABEL_58;
              if ( v67 != 20 )
                goto LABEL_58;
              v132 = *(_QWORD *)(v66 + 48) + 16LL * v208;
              v133 = *(_QWORD *)(v132 + 8);
              LOWORD(v132) = *(_WORD *)v132;
              v233 = v133;
              LOWORD(v232) = v132;
              if ( !sub_3280180((__int64)&v232)
                || !sub_33CF170(v63) && !(unsigned __int8)sub_33D1E40(v66, v181, v134, v135, v136) )
              {
                goto LABEL_58;
              }
              goto LABEL_34;
            case 222:
              v51 = *(__int64 **)(v9 + 40);
              v173 = v175;
              v52 = v51[1];
              v53 = *v51;
              v180 = v184;
              v54 = v51[5];
              v191 = v25;
              v55 = *(_QWORD *)(v54 + 104);
              LOWORD(v54) = *(_WORD *)(v54 + 96);
              v233 = v55;
              LOWORD(v232) = v54;
              v56 = sub_32844A0((unsigned __int16 *)&v232, v11);
              v57 = sub_9871A0(a4);
              v58 = v191;
              v59 = (__int64)v197;
              if ( v56 < *(_DWORD *)(a4 + 8) - v57
                || (v60 = *(__int64 (**)())(*(_QWORD *)v180 + 784LL), v60 != sub_2FE3290)
                && (v116 = ((__int64 (__fastcall *)(unsigned int *, __int64, __int64))v60)(v180, v9, v173),
                    v58 = v191,
                    v59 = (__int64)v197,
                    !v116) )
              {
                if ( v58 + 1 - v56 > (unsigned int)sub_33D25A0(v59, v53, v52, (__int64)a5, a8 + 1) )
                  goto LABEL_58;
              }
              v32 = v231;
              result = v53;
              goto LABEL_36;
            case 223:
            case 224:
            case 225:
              v187 = v197;
              v204 = v25;
              v35 = sub_3280200((__int64)&v214);
              v36 = v204;
              if ( v35 )
                goto LABEL_45;
              v37 = *(__int64 **)(v9 + 40);
              v38 = *v37;
              v39 = v16 + *(_QWORD *)(v9 + 48);
              v205 = v37[1];
              v40 = *(_QWORD *)(*v37 + 48) + 16LL * *((unsigned int *)v37 + 2);
              v41 = *(_WORD *)v40;
              v42 = *(_QWORD *)(v40 + 8);
              LOWORD(v218) = v41;
              v219 = v42;
              v43 = *(_WORD *)v39;
              v221 = *(_QWORD *)(v39 + 8);
              LOWORD(v220) = v43;
              if ( !v169 )
                goto LABEL_45;
              v178 = (__int64)v187;
              v188 = v36;
              if ( !sub_D94970((__int64)a5, (_QWORD *)1) )
                goto LABEL_45;
              v232 = (_QWORD *)sub_2D5B750((unsigned __int16 *)&v218);
              v233 = v137;
              v222 = sub_2D5B750((unsigned __int16 *)&v220);
              v223 = v138;
              if ( (_QWORD *)v222 != v232 || (_BYTE)v223 != (_BYTE)v233 )
                goto LABEL_45;
              v139 = sub_9871A0(a4);
              if ( (unsigned int)(v188 - v139) > (unsigned __int64)sub_32844A0((unsigned __int16 *)&v218, 1) )
                goto LABEL_58;
              result = (__int64)sub_33FB890(v178, v220, v221, v38, v205, a7);
              v32 = v231;
              goto LABEL_36;
            case 234:
              v189 = v197;
              v206 = (int)v27;
              if ( sub_3280200((__int64)&v214) )
                goto LABEL_45;
              v167 = (__int64)v189;
              v190 = sub_33CF5B0(**(_QWORD **)(v9 + 40), *(_QWORD *)(*(_QWORD *)(v9 + 40) + 8LL));
              v45 = (unsigned __int16 *)(*(_QWORD *)(v190 + 48) + 16LL * v44);
              v179 = v44;
              v46 = v16 + *(_QWORD *)(v9 + 48);
              v47 = *v45;
              v48 = *((_QWORD *)v45 + 1);
              v216 = v47;
              v217 = v48;
              v49 = *(_WORD *)v46;
              v50 = *(_QWORD *)(v46 + 8);
              LOWORD(v218) = v49;
              v219 = v50;
              if ( v49 == (_WORD)v47 && (v49 || v50 == v48) )
              {
                result = v190;
                v32 = v231;
                goto LABEL_36;
              }
              v118 = sub_32844A0(&v216, v47);
              v166 = v118;
              v163 = v118;
              v119 = sub_32844A0((unsigned __int16 *)&v218, v47);
              v120 = v206;
              v121 = v167;
              v155 = v119;
              v161 = v119;
              if ( v118 == v119 )
              {
                v47 = v190;
                v142 = sub_3447D70(v206, v190, v179, a4, (_DWORD)a5, v167, a8 + 1);
                v120 = v206;
                v121 = v167;
                if ( v142 )
                {
                  result = (__int64)sub_33FB890(v167, v218, v219, v142, v143, a7);
                  v32 = v231;
                  goto LABEL_36;
                }
              }
              v157 = v121;
              v210 = v120;
              v122 = sub_32801E0((__int64)&v216);
              v123 = v210;
              v124 = v157;
              if ( !v122 )
                goto LABEL_123;
              v158 = v161 / v118;
              if ( v161 % v118 )
                goto LABEL_123;
              v153 = v124;
              v94 = &v220;
              v154 = v210;
              v144 = sub_3281500(&v216, v47);
              sub_9691E0((__int64)&v220, v166, 0, 0, 0);
              sub_9691E0((__int64)&v222, v144, 0, 0, 0);
              v212 = 0;
              v145 = v154;
              v146 = v153;
              if ( v166 <= v155 )
              {
                do
                {
                  v147 = v158 - 1 - v212;
                  if ( v169 )
                    v147 = v212;
                  sub_C440A0((__int64)&v232, (__int64 *)a4, v163, v163 * v147);
                  if ( !sub_9867B0((__int64)&v232) )
                  {
                    sub_343FD40(&v220, (__int64 *)&v232);
                    if ( v172 )
                    {
                      v148 = v212;
                      for ( j = 0; j != v172; ++j )
                      {
                        if ( sub_986C60((__int64 *)a5, j) )
                          sub_987080(&v222, v148);
                        v148 += v158;
                      }
                    }
                  }
                  sub_969240((__int64 *)&v232);
                  ++v212;
                }
                while ( v158 != v212 );
                v145 = v154;
                v146 = v153;
                v94 = &v220;
              }
              v47 = v190;
              v160 = v146;
              v213 = v145;
              v150 = sub_3447D70(v145, v190, v179, (unsigned int)&v220, (unsigned int)&v222, v146, a8 + 1);
              if ( v150 )
              {
                v194 = sub_33FB890(v160, v218, v219, v150, v151, a7);
                sub_969240(&v222);
LABEL_86:
                sub_969240(v94);
                v32 = v231;
                result = (__int64)v194;
                goto LABEL_36;
              }
              sub_969240(&v222);
              sub_969240(&v220);
              v124 = v160;
              v123 = v213;
LABEL_123:
              v156 = v124;
              v159 = v123;
              if ( v169 )
              {
                v171 = v163 / v161;
                if ( !(v163 % v161) )
                {
                  v125 = 1;
                  v126 = sub_32801E0((__int64)&v216);
                  v127 = v159;
                  v128 = v156;
                  if ( v126 )
                  {
                    v152 = sub_3281500(&v216, v47);
                    v128 = v156;
                    v127 = v159;
                    v125 = v152;
                  }
                  v164 = v128;
                  v168 = v127;
                  sub_9691E0((__int64)&v222, v166, 0, 0, 0);
                  sub_9691E0((__int64)&v232, v125, 0, 0, 0);
                  v129 = v168;
                  v130 = v164;
                  if ( v172 )
                  {
                    for ( k = 0; k != v172; ++k )
                    {
                      if ( sub_986C60((__int64 *)a5, k) )
                      {
                        sub_C43D80((__int64)&v222, a4, v161 * (k % v171));
                        sub_987080((__int64 *)&v232, k / v171);
                      }
                    }
                    v129 = v168;
                    v130 = v164;
                  }
                  v211 = v130;
                  v140 = sub_3447D70(v129, v190, v179, (unsigned int)&v222, (unsigned int)&v232, v130, a8 + 1);
                  if ( v140 )
                  {
                    v196 = sub_33FB890(v211, v218, v219, v140, v141, a7);
                    sub_969240((__int64 *)&v232);
                    sub_969240(&v222);
                    v32 = v231;
                    result = (__int64)v196;
                    goto LABEL_36;
                  }
                  sub_969240((__int64 *)&v232);
                  sub_969240(&v222);
                }
              }
              break;
            default:
              goto LABEL_25;
          }
          goto LABEL_58;
        }
        if ( v24 == 52 )
        {
          v33 = *(__int64 **)(v9 + 40);
          v34 = *v33;
          if ( !(unsigned __int8)sub_33DE230(v197, *v33, v33[1], (__int64)a5, 0, 0) )
            goto LABEL_58;
LABEL_34:
          result = v34;
LABEL_35:
          v32 = v231;
          goto LABEL_36;
        }
        if ( v24 == 56 )
        {
          sub_33D4EF0(
            (__int64)&v232,
            (__int64)v197,
            *(_QWORD *)(*(_QWORD *)(v9 + 40) + 40LL),
            *(_QWORD *)(*(_QWORD *)(v9 + 40) + 48LL),
            (__int64)a5,
            a8 + 1);
          sub_3441190((__int64)&v228, (__int64)&v232);
          sub_969240(v234);
          sub_969240((__int64 *)&v232);
          if ( sub_986760((__int64)&v228) )
          {
LABEL_24:
            v32 = v231;
            result = **(_QWORD **)(v9 + 40);
LABEL_36:
            if ( v32 > 0x40 && v230 )
            {
              v202 = result;
              j_j___libc_free_0_0(v230);
              result = v202;
            }
            goto LABEL_39;
          }
          sub_33D4EF0(
            (__int64)&v232,
            (__int64)v197,
            **(_QWORD **)(v9 + 40),
            *(_QWORD *)(*(_QWORD *)(v9 + 40) + 8LL),
            (__int64)a5,
            a8 + 1);
          sub_3441190((__int64)&v224, (__int64)&v232);
          sub_969240(v234);
          sub_969240((__int64 *)&v232);
          if ( sub_986760((__int64)&v224) )
          {
LABEL_47:
            v32 = v231;
            result = *(_QWORD *)(*(_QWORD *)(v9 + 40) + 40LL);
            goto LABEL_36;
          }
LABEL_58:
          v32 = v231;
LABEL_59:
          result = 0;
          goto LABEL_36;
        }
      }
LABEL_25:
      v177 = v197;
      v186 = v28;
      v199 = v27;
      if ( sub_3280200((__int64)&v214) )
      {
        result = 0;
LABEL_27:
        if ( v227 > 0x40 && v226 )
        {
          v200 = result;
          j_j___libc_free_0_0(v226);
          result = v200;
        }
        if ( v225 > 0x40 )
        {
          if ( v224 )
          {
            v201 = result;
            j_j___libc_free_0_0(v224);
            return v201;
          }
        }
        return result;
      }
      if ( (unsigned int)v24 <= 0x1F3 )
      {
LABEL_45:
        result = 0;
LABEL_39:
        if ( v229 > 0x40 && v228 )
        {
          v203 = result;
          j_j___libc_free_0_0(v228);
          result = v203;
        }
        goto LABEL_27;
      }
      v34 = (*(__int64 (__fastcall **)(unsigned int *, __int64, __int64, __int64, _QWORD **, _QWORD **, _QWORD))(*(_QWORD *)v199 + 2072LL))(
              v199,
              v9,
              v186,
              a4,
              a5,
              v177,
              a8);
      if ( !v34 )
        goto LABEL_58;
      goto LABEL_34;
    }
    v20 = **a5;
  }
  else
  {
    v20 = (__int64)*a5;
  }
  if ( v20 )
    goto LABEL_10;
LABEL_16:
  v232 = 0;
  LODWORD(v233) = 0;
  v30 = sub_33F17F0(v15, 51, (__int64)&v232, v214, v215);
  if ( v232 )
    sub_B91220((__int64)&v232, (__int64)v232);
  return (__int64)v30;
}
