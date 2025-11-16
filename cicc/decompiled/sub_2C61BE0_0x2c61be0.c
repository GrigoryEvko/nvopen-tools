// Function: sub_2C61BE0
// Address: 0x2c61be0
//
__int64 __fastcall sub_2C61BE0(__int64 a1, __int64 a2)
{
  __int64 v2; // r10
  unsigned int v4; // r15d
  __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // r10
  __int64 v15; // r13
  unsigned int v16; // r14d
  _QWORD *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r13
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rsi
  __int64 v23; // rax
  unsigned __int64 v24; // r8
  unsigned __int64 v25; // r14
  _BOOL4 v26; // r13d
  _BOOL4 v27; // eax
  _BYTE *v28; // rdx
  unsigned __int64 v29; // rax
  _QWORD *v30; // rax
  _QWORD *v31; // rdx
  unsigned int v32; // edx
  __int64 v33; // r11
  __int64 v34; // rax
  unsigned int v35; // r9d
  bool v36; // zf
  __int64 *v37; // rdi
  __int64 v38; // rax
  __int64 v39; // r9
  unsigned __int64 v40; // rdx
  __int64 v41; // r10
  __int64 v42; // r11
  int v43; // r8d
  _QWORD *v44; // rsi
  __int64 v45; // rdx
  char *v46; // rax
  char *v47; // rcx
  __int64 v48; // rax
  unsigned __int64 v49; // rcx
  __int64 v50; // rdx
  int v51; // r14d
  __int64 v52; // r13
  int v53; // edx
  signed __int64 v54; // rcx
  int v55; // edx
  __int64 v56; // rax
  bool v57; // of
  unsigned __int64 v58; // r13
  unsigned __int64 v59; // r14
  signed __int64 v60; // rax
  __int64 v61; // r11
  int v62; // edx
  __int64 v63; // rax
  unsigned __int64 v64; // r13
  __int64 v65; // rcx
  signed __int64 v66; // rax
  int v67; // edx
  __int64 v68; // rax
  signed __int64 v69; // r13
  signed __int64 v70; // r14
  __int64 v71; // rdx
  __int64 v72; // r13
  __int64 v73; // rax
  __int64 v74; // r10
  int v75; // edx
  int v76; // r11d
  signed __int64 v77; // r13
  __int64 v78; // rax
  int v79; // edx
  __int64 *v80; // r13
  unsigned int **v81; // rdi
  __int64 v82; // rbx
  _BYTE *v83; // rax
  __int64 **v84; // r10
  unsigned __int64 v85; // r15
  _BYTE *v86; // rax
  __int64 v87; // r10
  __int64 v88; // r15
  __int64 i; // r13
  unsigned __int8 *v90; // rdi
  __int64 v91; // rax
  int v92; // eax
  int v93; // eax
  _BYTE **v94; // rax
  _BYTE *v95; // rax
  unsigned __int8 *v96; // rdi
  __int64 v97; // rax
  int v98; // eax
  int v99; // eax
  _BYTE **v100; // rax
  _BYTE *v101; // rax
  __int64 v102; // rax
  int v103; // edx
  unsigned __int64 v104; // r13
  unsigned __int64 v105; // r13
  unsigned __int64 v106; // r14
  signed __int64 v107; // r13
  unsigned __int64 v108; // r14
  __int64 v109; // [rsp+8h] [rbp-168h]
  __int64 v110; // [rsp+10h] [rbp-160h]
  int v111; // [rsp+18h] [rbp-158h]
  bool v112; // [rsp+1Fh] [rbp-151h]
  _QWORD **v113; // [rsp+20h] [rbp-150h]
  unsigned int v114; // [rsp+28h] [rbp-148h]
  unsigned int v115; // [rsp+28h] [rbp-148h]
  __int64 v116; // [rsp+28h] [rbp-148h]
  __int64 v117; // [rsp+30h] [rbp-140h]
  __int64 **v118; // [rsp+30h] [rbp-140h]
  int v119; // [rsp+30h] [rbp-140h]
  int v120; // [rsp+30h] [rbp-140h]
  _BOOL4 v121; // [rsp+38h] [rbp-138h]
  __int64 v122; // [rsp+38h] [rbp-138h]
  __int64 v123; // [rsp+38h] [rbp-138h]
  __int64 v124; // [rsp+38h] [rbp-138h]
  __int64 v125; // [rsp+38h] [rbp-138h]
  __int64 v126; // [rsp+40h] [rbp-130h]
  unsigned __int64 v127; // [rsp+40h] [rbp-130h]
  __int64 **v128; // [rsp+40h] [rbp-130h]
  __int64 v129; // [rsp+40h] [rbp-130h]
  __int64 v130; // [rsp+40h] [rbp-130h]
  __int64 v131; // [rsp+40h] [rbp-130h]
  __int64 v132; // [rsp+48h] [rbp-128h]
  __int64 v133; // [rsp+48h] [rbp-128h]
  _QWORD **v134; // [rsp+48h] [rbp-128h]
  __int64 v135; // [rsp+48h] [rbp-128h]
  int v136; // [rsp+48h] [rbp-128h]
  __int64 v137; // [rsp+48h] [rbp-128h]
  __int64 **v138; // [rsp+48h] [rbp-128h]
  __int64 v139; // [rsp+48h] [rbp-128h]
  __int64 v140; // [rsp+48h] [rbp-128h]
  unsigned __int64 v141; // [rsp+48h] [rbp-128h]
  __int64 v142; // [rsp+48h] [rbp-128h]
  _BYTE *v143; // [rsp+50h] [rbp-120h] BYREF
  unsigned __int64 v144; // [rsp+58h] [rbp-118h] BYREF
  _BYTE *v145; // [rsp+60h] [rbp-110h] BYREF
  unsigned __int64 v146; // [rsp+68h] [rbp-108h] BYREF
  __int64 v147; // [rsp+70h] [rbp-100h]
  int v148; // [rsp+78h] [rbp-F8h]
  _BYTE v149[32]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v150; // [rsp+A0h] [rbp-D0h]
  _QWORD *v151; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v152; // [rsp+B8h] [rbp-B8h] BYREF
  _BYTE v153[176]; // [rsp+C0h] [rbp-B0h] BYREF

  v2 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(v2 + 8) != 12 )
    return 0;
  v4 = **(unsigned __int8 **)(a1 + 184);
  if ( (_BYTE)v4 )
    return 0;
  if ( *(_BYTE *)a2 == 58 && (*(_BYTE *)(a2 + 1) & 2) != 0 )
  {
    v7 = *(_QWORD *)(a2 - 64);
    if ( *(_BYTE *)v7 > 0x1Cu )
    {
      v8 = *(_QWORD *)(a2 - 32);
      if ( *(_BYTE *)v8 > 0x1Cu )
      {
        v144 = 0;
        v9 = *(_QWORD *)(v7 + 16);
        if ( v9 )
        {
          if ( *(_QWORD *)(v9 + 8) )
            goto LABEL_11;
          if ( *(_BYTE *)v7 != 68 )
            goto LABEL_11;
          v90 = *(unsigned __int8 **)(v7 - 32);
          v91 = *((_QWORD *)v90 + 2);
          if ( !v91 || *(_QWORD *)(v91 + 8) )
            goto LABEL_11;
          v92 = *v90;
          if ( (unsigned __int8)v92 > 0x1Cu )
          {
            v93 = v92 - 29;
          }
          else
          {
            if ( (_BYTE)v92 != 5 )
              goto LABEL_11;
            v93 = *((unsigned __int16 *)v90 + 1);
          }
          if ( v93 == 49 )
          {
            v129 = v8;
            v139 = v2;
            v94 = (_BYTE **)sub_986520((__int64)v90);
            v14 = v139;
            v8 = v129;
            v95 = *v94;
            if ( v95 )
            {
              v143 = v95;
              goto LABEL_23;
            }
          }
LABEL_11:
          v151 = &v143;
          v152 = (__int64)&v144;
          if ( *(_QWORD *)(v9 + 8) )
            return v4;
          if ( *(_BYTE *)v7 != 54 )
            return v4;
          v10 = *(_QWORD *)(v7 - 64);
          v11 = *(_QWORD *)(v10 + 16);
          if ( !v11 )
            return v4;
          if ( *(_QWORD *)(v11 + 8) )
            return v4;
          if ( *(_BYTE *)v10 != 68 )
            return v4;
          v12 = *(_QWORD *)(v10 - 32);
          v13 = *(_QWORD *)(v12 + 16);
          if ( !v13 )
            return v4;
          if ( *(_QWORD *)(v13 + 8) )
            return v4;
          v132 = v8;
          if ( !(unsigned __int8)sub_2C4EFD0(&v151, (unsigned __int8 *)v12) )
            return v4;
          v15 = *(_QWORD *)(v7 - 32);
          if ( *(_BYTE *)v15 != 17 )
            return v4;
          v16 = *(_DWORD *)(v15 + 32);
          v8 = v132;
          if ( v16 > 0x40 )
          {
            v131 = v132;
            v142 = v14;
            if ( v16 - (unsigned int)sub_C444A0(v15 + 24) > 0x40 )
              return v4;
            v14 = v142;
            v8 = v131;
            v17 = (_QWORD *)v152;
            v18 = **(_QWORD **)(v15 + 24);
          }
          else
          {
            v17 = (_QWORD *)v152;
            v18 = *(_QWORD *)(v15 + 24);
          }
          *v17 = v18;
LABEL_23:
          v146 = 0;
          v19 = *(_QWORD *)(v8 + 16);
          if ( !v19 )
            return v4;
          if ( *(_QWORD *)(v19 + 8) )
            goto LABEL_25;
          if ( *(_BYTE *)v8 != 68 )
            goto LABEL_25;
          v96 = *(unsigned __int8 **)(v8 - 32);
          v97 = *((_QWORD *)v96 + 2);
          if ( !v97 || *(_QWORD *)(v97 + 8) )
            goto LABEL_25;
          v98 = *v96;
          if ( (unsigned __int8)v98 > 0x1Cu )
          {
            v99 = v98 - 29;
          }
          else
          {
            if ( (_BYTE)v98 != 5 )
              goto LABEL_25;
            v99 = *((unsigned __int16 *)v96 + 1);
          }
          if ( v99 == 49 )
          {
            v130 = v8;
            v140 = v14;
            v100 = (_BYTE **)sub_986520((__int64)v96);
            v14 = v140;
            v8 = v130;
            v101 = *v100;
            if ( v101 )
            {
              v145 = v101;
              v24 = 0;
              v121 = 0;
              goto LABEL_35;
            }
          }
LABEL_25:
          v151 = &v145;
          v152 = (__int64)&v146;
          if ( *(_QWORD *)(v19 + 8) )
            return v4;
          if ( *(_BYTE *)v8 != 54 )
            return v4;
          v20 = *(_QWORD *)(v8 - 64);
          v21 = *(_QWORD *)(v20 + 16);
          if ( !v21 )
            return v4;
          if ( *(_QWORD *)(v21 + 8) )
            return v4;
          if ( *(_BYTE *)v20 != 68 )
            return v4;
          v22 = *(_QWORD *)(v20 - 32);
          v23 = *(_QWORD *)(v22 + 16);
          if ( !v23 )
            return v4;
          if ( *(_QWORD *)(v23 + 8) )
            return v4;
          v126 = v8;
          v133 = v14;
          if ( !(unsigned __int8)sub_2C4EFD0(&v151, (unsigned __int8 *)v22)
            || !(unsigned __int8)sub_11B1B00((_QWORD **)&v152, *(_QWORD *)(v126 - 32)) )
          {
            return v4;
          }
          v24 = v146;
          v14 = v133;
          v121 = v146 != 0;
LABEL_35:
          v25 = v144;
          v26 = v144 != 0;
          v27 = v26;
          if ( v144 > v24 )
          {
            v28 = v143;
            v144 = v24;
            v26 = v121;
            v121 = v27;
            v29 = v24;
            v24 = v25;
            v143 = v145;
            v145 = v28;
            v146 = v25;
            v25 = v29;
          }
          v127 = v24;
          v134 = (_QWORD **)v14;
          v30 = (_QWORD *)sub_BCAE30(v14);
          v152 = (__int64)v31;
          v151 = v30;
          v32 = sub_CA1930(&v151);
          v33 = *((_QWORD *)v143 + 1);
          if ( *(_BYTE *)(v33 + 8) != 17 )
            return 0;
          if ( v33 != *((_QWORD *)v145 + 1) )
            return 0;
          v113 = v134;
          v114 = v32;
          v135 = *((_QWORD *)v143 + 1);
          v112 = sub_BCAC40(*(_QWORD *)(v33 + 24), 1);
          if ( !v112 )
            return 0;
          v34 = *(unsigned int *)(v135 + 32);
          v35 = *(_DWORD *)(v135 + 32);
          if ( v34 != v127 - v25 || v114 >> 1 < (unsigned int)v34 )
            return 0;
          v36 = *(_BYTE *)(v135 + 8) == 18;
          v37 = *(__int64 **)(v135 + 24);
          v109 = v135;
          LODWORD(v147) = 2 * v35;
          BYTE4(v147) = v36;
          v115 = v35;
          v110 = sub_BCE1B0(v37, v147);
          v128 = (__int64 **)sub_BCD140(*v113, *(_DWORD *)(v110 + 32));
          v38 = sub_BCD140(*v113, v115);
          v40 = *(unsigned int *)(v110 + 32);
          v41 = (__int64)v113;
          v116 = v38;
          v42 = v135;
          v43 = *(_DWORD *)(v110 + 32);
          v151 = v153;
          v152 = 0x2000000000LL;
          if ( v40 )
          {
            if ( v40 > 0x20 )
            {
              v111 = v40;
              v141 = v40;
              sub_C8D5F0((__int64)&v151, v153, v40, 4u, v40, v39);
              v43 = v111;
              v42 = v109;
              v41 = (__int64)v113;
              v40 = v141;
            }
            v44 = v151;
            v45 = 4 * v40;
            v46 = (char *)v151 + 4 * (unsigned int)v152;
            v47 = (char *)v151 + v45;
            if ( v46 != (char *)v151 + v45 )
            {
              do
              {
                if ( v46 )
                  *(_DWORD *)v46 = 0;
                v46 += 4;
              }
              while ( v47 != v46 );
              v44 = v151;
            }
            LODWORD(v152) = v43;
            v48 = 0;
            v49 = (unsigned __int64)(v45 - 4) >> 2;
            do
            {
              v50 = v48;
              *((_DWORD *)v44 + v48) = v48;
              ++v48;
            }
            while ( v50 != v49 );
          }
          v51 = v26 + v121;
          v117 = v42;
          v122 = v41;
          v52 = sub_DFD800(*(_QWORD *)(a1 + 152), 0x1Du, v41, *(_DWORD *)(a1 + 192), 0, 0, 0, 0, 0, 0);
          v136 = v53 == 1;
          v54 = sub_DFD800(*(_QWORD *)(a1 + 152), 0x19u, v122, *(_DWORD *)(a1 + 192), 0, 0, 0, 0, 0, 0);
          if ( v55 == 1 )
          {
            v56 = v54 * v51;
            if ( !is_mul_ok(v54, v51) )
            {
              if ( v51 && v54 > 0 )
              {
                v59 = 0x7FFFFFFFFFFFFFFFLL;
                v57 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v52);
                v104 = v52 + 0x7FFFFFFFFFFFFFFFLL;
                v136 = 1;
                if ( !v57 )
                  v59 = v104;
              }
              else
              {
                v59 = 0x8000000000000000LL;
                v57 = __OFADD__(0x8000000000000000LL, v52);
                v105 = v52 + 0x8000000000000000LL;
                v136 = 1;
                if ( !v57 )
                  v59 = v105;
              }
              goto LABEL_56;
            }
            v136 = 1;
          }
          else
          {
            v56 = v54 * v51;
            if ( !is_mul_ok(v54, v51) )
            {
              if ( v51 && v54 > 0 )
              {
                v59 = 0x7FFFFFFFFFFFFFFFLL;
                v57 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v52);
                v58 = v52 + 0x7FFFFFFFFFFFFFFFLL;
                if ( !v57 )
                  goto LABEL_55;
              }
              else
              {
                v59 = 0x8000000000000000LL;
                v57 = __OFADD__(0x8000000000000000LL, v52);
                v58 = v52 + 0x8000000000000000LL;
                if ( !v57 )
                  goto LABEL_55;
              }
              goto LABEL_56;
            }
          }
          v57 = __OFADD__(v56, v52);
          v58 = v56 + v52;
          if ( !v57 )
          {
LABEL_55:
            v59 = v58;
            goto LABEL_56;
          }
          v59 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v56 <= 0 )
            v59 = 0x8000000000000000LL;
LABEL_56:
          v60 = sub_DFD060(*(__int64 **)(a1 + 152), 39, v122, v116);
          v61 = v117;
          if ( v62 != 1 )
          {
            if ( !is_mul_ok(2u, v60) )
            {
              if ( v60 > 0 )
              {
                v64 = 0x7FFFFFFFFFFFFFFFLL;
                v57 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v59);
                v106 = v59 + 0x7FFFFFFFFFFFFFFFLL;
                if ( v57 )
                  goto LABEL_60;
              }
              else
              {
                v64 = 0x8000000000000000LL;
                v57 = __OFADD__(0x8000000000000000LL, v59);
                v106 = v59 + 0x8000000000000000LL;
                if ( v57 )
                  goto LABEL_60;
              }
              v64 = v106;
              goto LABEL_60;
            }
            v63 = 2 * v60;
LABEL_59:
            v64 = v63 + v59;
            if ( __OFADD__(v63, v59) )
            {
              v64 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v63 <= 0 )
                v64 = 0x8000000000000000LL;
            }
LABEL_60:
            v65 = v117;
            v118 = (__int64 **)v122;
            v123 = v61;
            v66 = sub_DFD060(*(__int64 **)(a1 + 152), 49, v116, v65);
            if ( v67 != 1 )
            {
              if ( !is_mul_ok(2u, v66) )
              {
                if ( v66 <= 0 )
                {
                  v70 = 0x8000000000000000LL;
                  v57 = __OFADD__(0x8000000000000000LL, v64);
                  v69 = v64 + 0x8000000000000000LL;
                  if ( !v57 )
                    goto LABEL_64;
                }
                else
                {
                  v70 = 0x7FFFFFFFFFFFFFFFLL;
                  v57 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v64);
                  v69 = v64 + 0x7FFFFFFFFFFFFFFFLL;
                  if ( !v57 )
                    goto LABEL_64;
                }
                goto LABEL_65;
              }
              v68 = 2 * v66;
LABEL_63:
              v57 = __OFADD__(v68, v64);
              v69 = v68 + v64;
              if ( !v57 )
              {
LABEL_64:
                v70 = v69;
                goto LABEL_65;
              }
              v70 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v68 <= 0 )
                v70 = 0x8000000000000000LL;
LABEL_65:
              v71 = v123;
              v124 = (__int64)v118;
              v72 = sub_DFBC30(
                      *(__int64 **)(a1 + 152),
                      6,
                      v71,
                      (__int64)v151,
                      (unsigned int)v152,
                      *(unsigned int *)(a1 + 192),
                      0,
                      0,
                      0,
                      0,
                      0);
              v73 = sub_DFD060(*(__int64 **)(a1 + 152), 49, (__int64)v128, v110);
              v74 = (__int64)v118;
              v76 = v75;
              if ( v75 != 1 )
                v76 = 0;
              v57 = __OFADD__(v73, v72);
              v77 = v73 + v72;
              if ( v57 )
              {
                v77 = 0x7FFFFFFFFFFFFFFFLL;
                if ( v73 <= 0 )
                  v77 = 0x8000000000000000LL;
              }
              if ( v128 != v118 )
              {
                v119 = v76;
                v78 = sub_DFD060(*(__int64 **)(a1 + 152), 39, v124, (__int64)v128);
                v76 = v119;
                v74 = v124;
                if ( v79 == 1 )
                  v76 = 1;
                v57 = __OFADD__(v78, v77);
                v77 += v78;
                if ( v57 )
                {
                  v77 = 0x7FFFFFFFFFFFFFFFLL;
                  if ( v78 <= 0 )
                    v77 = 0x8000000000000000LL;
                }
              }
              if ( v144 )
              {
                v120 = v76;
                v125 = v74;
                v102 = sub_DFD800(*(_QWORD *)(a1 + 152), 0x19u, v74, *(_DWORD *)(a1 + 192), 0, 0, 0, 0, 0, 0);
                v76 = v120;
                v74 = v125;
                if ( v103 == 1 )
                  v76 = 1;
                v57 = __OFADD__(v102, v77);
                v77 += v102;
                if ( v57 )
                {
                  v77 = 0x7FFFFFFFFFFFFFFFLL;
                  if ( v102 <= 0 )
                    v77 = 0x8000000000000000LL;
                }
              }
              if ( v76 == v136 )
              {
                if ( v77 <= v70 )
                  goto LABEL_75;
              }
              else if ( v136 >= v76 )
              {
LABEL_75:
                v80 = (__int64 *)(a1 + 8);
                v150 = 257;
                v81 = (unsigned int **)(a1 + 8);
                v137 = v74;
                v82 = a1 + 200;
                v83 = (_BYTE *)sub_A83CB0(v81, v143, v145, (__int64)v151, (unsigned int)v152, (__int64)v149);
                v84 = (__int64 **)v137;
                v85 = (unsigned __int64)v83;
                if ( *v83 > 0x1Cu )
                {
                  sub_F15FC0(v82, (__int64)v83);
                  v84 = (__int64 **)v137;
                }
                v150 = 257;
                v138 = v84;
                v86 = (_BYTE *)sub_2C511B0(v80, 0x31u, v85, v128, (__int64)v149, 0, v148, 0);
                v87 = (__int64)v138;
                v88 = (__int64)v86;
                if ( v128 != v138 )
                {
                  if ( *v86 > 0x1Cu )
                  {
                    sub_F15FC0(v82, (__int64)v86);
                    v87 = (__int64)v138;
                  }
                  v150 = 257;
                  v88 = sub_A82F30((unsigned int **)v80, v88, v87, (__int64)v149, 0);
                }
                if ( v144 )
                {
                  if ( *(_BYTE *)v88 > 0x1Cu )
                    sub_F15FC0(v82, v88);
                  v150 = 257;
                  v88 = sub_920C00((unsigned int **)v80, v88, v144, (__int64)v149, 0, 0);
                }
                sub_BD84D0(a2, v88);
                if ( *(_BYTE *)v88 > 0x1Cu )
                {
                  sub_BD6B90((unsigned __int8 *)v88, (unsigned __int8 *)a2);
                  for ( i = *(_QWORD *)(v88 + 16); i; i = *(_QWORD *)(i + 8) )
                    sub_F15FC0(v82, *(_QWORD *)(i + 24));
                  if ( *(_BYTE *)v88 > 0x1Cu )
                    sub_F15FC0(v82, v88);
                }
                v4 = v112;
                if ( *(_BYTE *)a2 > 0x1Cu )
                  sub_F15FC0(v82, a2);
              }
              if ( v151 != (_QWORD *)v153 )
                _libc_free((unsigned __int64)v151);
              return v4;
            }
            if ( is_mul_ok(2u, v66) )
            {
              v136 = 1;
              v68 = 2 * v66;
              goto LABEL_63;
            }
            if ( v66 <= 0 )
            {
              v70 = 0x8000000000000000LL;
              v57 = __OFADD__(0x8000000000000000LL, v64);
              v107 = v64 + 0x8000000000000000LL;
              if ( !v57 )
                goto LABEL_145;
            }
            else
            {
              v70 = 0x7FFFFFFFFFFFFFFFLL;
              v57 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v64);
              v107 = v64 + 0x7FFFFFFFFFFFFFFFLL;
              if ( !v57 )
              {
LABEL_145:
                v136 = 1;
                v70 = v107;
                goto LABEL_65;
              }
            }
            v136 = 1;
            goto LABEL_65;
          }
          if ( is_mul_ok(2u, v60) )
          {
            v136 = 1;
            v63 = 2 * v60;
            goto LABEL_59;
          }
          if ( v60 <= 0 )
          {
            v64 = 0x8000000000000000LL;
            v57 = __OFADD__(0x8000000000000000LL, v59);
            v108 = v59 + 0x8000000000000000LL;
            if ( v57 )
              goto LABEL_168;
          }
          else
          {
            v64 = 0x7FFFFFFFFFFFFFFFLL;
            v57 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v59);
            v108 = v59 + 0x7FFFFFFFFFFFFFFFLL;
            if ( v57 )
            {
LABEL_168:
              v136 = 1;
              goto LABEL_60;
            }
          }
          v64 = v108;
          goto LABEL_168;
        }
      }
    }
  }
  return v4;
}
