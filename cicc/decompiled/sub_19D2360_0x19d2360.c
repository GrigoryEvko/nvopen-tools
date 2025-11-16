// Function: sub_19D2360
// Address: 0x19d2360
//
__int64 __fastcall sub_19D2360(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        unsigned int a6,
        __int64 a7)
{
  __int64 v7; // r14
  __int64 v9; // r12
  unsigned __int64 i; // rbx
  __int64 v11; // rax
  bool v12; // zf
  unsigned int v13; // r15d
  __int64 v15; // rax
  __int64 v16; // r15
  unsigned int v17; // eax
  __int64 v18; // rsi
  __int64 v19; // r11
  __int64 v20; // r10
  unsigned __int64 v21; // rcx
  unsigned __int64 v22; // r8
  unsigned int v23; // r9d
  __int64 v24; // rax
  unsigned __int64 v25; // rax
  _QWORD *v26; // rcx
  unsigned __int8 v27; // al
  unsigned __int64 v28; // rax
  __int64 v29; // r10
  unsigned int v30; // r9d
  _QWORD *v31; // rdx
  __int64 v32; // r8
  __int64 v33; // rax
  signed __int64 v34; // r15
  __int64 v35; // r12
  _BYTE *v36; // rbx
  unsigned int v37; // r15d
  int v38; // r15d
  _BYTE *v39; // rdi
  __int64 v40; // rax
  __int64 v41; // r14
  unsigned __int8 v42; // al
  __int64 j; // r14
  _QWORD *v44; // rax
  int v45; // r8d
  int v46; // r9d
  __int64 v47; // rax
  unsigned int v48; // esi
  __int64 v49; // rax
  unsigned __int64 v50; // rax
  __int64 v51; // r10
  unsigned int v52; // r9d
  __int64 v53; // r8
  unsigned __int64 v54; // rax
  bool v55; // al
  unsigned __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // r14
  _QWORD *v59; // rax
  int v60; // r8d
  int v61; // r9d
  __int64 *v62; // rdi
  int v63; // eax
  __int64 v64; // rdx
  __int64 v65; // rsi
  __int64 v66; // r10
  unsigned int v67; // r9d
  unsigned int v68; // r14d
  __int64 v69; // r12
  int v70; // r13d
  unsigned int v71; // eax
  __int64 v72; // r8
  __int64 v73; // rax
  __int64 v74; // r10
  unsigned int v75; // r9d
  __int64 v76; // r15
  char v77; // al
  _QWORD *v78; // rax
  __int64 v79; // rsi
  char v80; // al
  unsigned int v81; // r9d
  __int64 v82; // rax
  unsigned int v83; // edx
  __int64 v84; // rax
  unsigned int v85; // edx
  unsigned int v86; // r14d
  __int64 v87; // rax
  int v88; // ecx
  __int64 v89; // rax
  __int64 v90; // rdx
  char v91; // al
  unsigned int v92; // r9d
  unsigned int v93; // r12d
  _QWORD *v94; // r15
  const char *v95; // rax
  __int64 v96; // rdx
  const char *v97; // rax
  __int64 v98; // rdx
  unsigned __int64 v99; // rax
  __int64 v100; // rdi
  __int64 v101; // rax
  __int64 v102; // rdx
  unsigned int v103; // [rsp+8h] [rbp-118h]
  __int64 v104; // [rsp+8h] [rbp-118h]
  unsigned int v105; // [rsp+8h] [rbp-118h]
  __int64 *v106; // [rsp+10h] [rbp-110h]
  __int64 v107; // [rsp+10h] [rbp-110h]
  unsigned int v108; // [rsp+10h] [rbp-110h]
  unsigned int v109; // [rsp+10h] [rbp-110h]
  __int64 *v110; // [rsp+18h] [rbp-108h]
  unsigned __int64 v111; // [rsp+18h] [rbp-108h]
  __int64 v112; // [rsp+18h] [rbp-108h]
  __int64 v113; // [rsp+18h] [rbp-108h]
  unsigned int v114; // [rsp+18h] [rbp-108h]
  unsigned int v115; // [rsp+18h] [rbp-108h]
  __int64 v116; // [rsp+18h] [rbp-108h]
  __int64 v117; // [rsp+20h] [rbp-100h]
  __int64 v118; // [rsp+20h] [rbp-100h]
  __int64 v120; // [rsp+20h] [rbp-100h]
  unsigned __int64 v121; // [rsp+20h] [rbp-100h]
  unsigned int v122; // [rsp+20h] [rbp-100h]
  __int64 v123; // [rsp+20h] [rbp-100h]
  unsigned int v124; // [rsp+20h] [rbp-100h]
  __int64 v125; // [rsp+20h] [rbp-100h]
  __int64 v126; // [rsp+20h] [rbp-100h]
  __int64 v127; // [rsp+28h] [rbp-F8h]
  __int64 v128; // [rsp+28h] [rbp-F8h]
  __int64 v129; // [rsp+28h] [rbp-F8h]
  __int64 v130; // [rsp+28h] [rbp-F8h]
  __int64 v131; // [rsp+28h] [rbp-F8h]
  unsigned int v133; // [rsp+30h] [rbp-F0h]
  __int64 v134; // [rsp+30h] [rbp-F0h]
  unsigned int v135; // [rsp+30h] [rbp-F0h]
  unsigned int v136; // [rsp+30h] [rbp-F0h]
  __int64 v138; // [rsp+38h] [rbp-E8h]
  int v139; // [rsp+38h] [rbp-E8h]
  _QWORD *v140; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v141; // [rsp+38h] [rbp-E8h]
  __int64 v142; // [rsp+38h] [rbp-E8h]
  unsigned int v143; // [rsp+38h] [rbp-E8h]
  __int64 v144; // [rsp+38h] [rbp-E8h]
  unsigned int v145; // [rsp+38h] [rbp-E8h]
  _QWORD *v146; // [rsp+38h] [rbp-E8h]
  __int64 v147; // [rsp+38h] [rbp-E8h]
  __int64 v148; // [rsp+40h] [rbp-E0h]
  unsigned int v149; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v150; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v151; // [rsp+40h] [rbp-E0h]
  __int64 v152; // [rsp+40h] [rbp-E0h]
  __int64 v153; // [rsp+40h] [rbp-E0h]
  __int64 v154; // [rsp+40h] [rbp-E0h]
  __int64 v155; // [rsp+48h] [rbp-D8h]
  unsigned __int64 v156; // [rsp+48h] [rbp-D8h]
  unsigned int v157; // [rsp+48h] [rbp-D8h]
  __int64 v158; // [rsp+48h] [rbp-D8h]
  __int64 *v159; // [rsp+48h] [rbp-D8h]
  __int64 v160; // [rsp+58h] [rbp-C8h] BYREF
  const char *v161; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v162; // [rsp+68h] [rbp-B8h]
  __int64 v163; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v164; // [rsp+78h] [rbp-A8h]
  __int64 v165; // [rsp+80h] [rbp-A0h]
  __int64 v166; // [rsp+88h] [rbp-98h]
  __int64 v167; // [rsp+90h] [rbp-90h]
  _BYTE *v168; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v169; // [rsp+A8h] [rbp-78h]
  _BYTE v170[112]; // [rsp+B0h] [rbp-70h] BYREF

  v7 = a1;
  v9 = a2;
  i = a4;
  v11 = *(_QWORD *)(a7 - 24);
  if ( !*(_BYTE *)(v11 + 16) && (*(_BYTE *)(v11 + 33) & 0x20) != 0 && *(_DWORD *)(v11 + 36) == 117 )
    return 0;
  v12 = *(_BYTE *)(a4 + 16) == 53;
  v160 = a7 | 4;
  if ( !v12 )
    return 0;
  v155 = *(_QWORD *)(a4 - 24);
  if ( *(_BYTE *)(v155 + 16) != 13 )
    return 0;
  v15 = sub_15F2050(a2);
  v16 = sub_1632FA0(v15);
  v117 = *(_QWORD *)(i + 56);
  v17 = sub_15A9FE0(v16, v117);
  v18 = v117;
  v127 = 1;
  v19 = v155;
  v20 = a7 | 4;
  v21 = v17;
  v22 = a5;
  v23 = a6;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v18 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v47 = v127 * *(_QWORD *)(v18 + 32);
        v18 = *(_QWORD *)(v18 + 24);
        v127 = v47;
        continue;
      case 1:
        v24 = 16;
        goto LABEL_12;
      case 2:
        v24 = 32;
        goto LABEL_12;
      case 3:
      case 9:
        v24 = 64;
        goto LABEL_12;
      case 4:
        v24 = 80;
        goto LABEL_12;
      case 5:
      case 6:
        v24 = 128;
        goto LABEL_12;
      case 7:
        v120 = v155;
        v48 = 0;
        v134 = a7 | 4;
        v141 = v21;
        v150 = v22;
        v157 = v23;
        goto LABEL_49;
      case 0xB:
        v24 = *(_DWORD *)(v18 + 8) >> 8;
        goto LABEL_12;
      case 0xD:
        v120 = v155;
        v134 = a7 | 4;
        v141 = v21;
        v150 = v22;
        v157 = v23;
        v24 = 8LL * *(_QWORD *)sub_15A9930(v16, v18);
        goto LABEL_50;
      case 0xE:
        v104 = v155;
        v111 = v21;
        v121 = a5;
        v158 = *(_QWORD *)(v18 + 32);
        v142 = *(_QWORD *)(v18 + 24);
        v151 = (unsigned int)sub_15A9FE0(v16, v142);
        v49 = sub_127FA20(v16, v142);
        v23 = a6;
        v22 = v121;
        v21 = v111;
        v20 = a7 | 4;
        v19 = v104;
        v24 = 8 * v151 * v158 * ((v151 + ((unsigned __int64)(v49 + 7) >> 3) - 1) / v151);
        goto LABEL_12;
      case 0xF:
        v120 = v155;
        v134 = a7 | 4;
        v141 = v21;
        v48 = *(_DWORD *)(v18 + 8) >> 8;
        v150 = v22;
        v157 = v23;
LABEL_49:
        v24 = 8 * (unsigned int)sub_15A9520(v16, v48);
LABEL_50:
        v23 = v157;
        v22 = v150;
        v21 = v141;
        v20 = v134;
        v19 = v120;
LABEL_12:
        v25 = v21 * ((v21 + ((unsigned __int64)(v127 * v24 + 7) >> 3) - 1) / v21);
        v26 = *(_QWORD **)(v19 + 24);
        if ( *(_DWORD *)(v19 + 32) > 0x40u )
          v26 = (_QWORD *)*v26;
        v156 = v25 * (_QWORD)v26;
        if ( v25 * (unsigned __int64)v26 > v22 )
          return 0;
        v27 = *(_BYTE *)(a3 + 16);
        if ( v27 <= 0x17u )
        {
          v143 = v23;
          v152 = v20;
          if ( v27 == 17 && !sub_15F3330(a7) )
          {
            v50 = sub_15E0380(a3);
            v29 = v152;
            v30 = v143;
            if ( v50 >= v156 )
              goto LABEL_21;
            if ( (unsigned __int8)sub_15E04F0(a3) )
            {
              v51 = v152;
              v52 = v143;
              v53 = *(_QWORD *)(*(_QWORD *)a3 + 24LL);
              v54 = *(unsigned __int8 *)(v53 + 8);
              if ( (unsigned __int8)v54 <= 0xFu )
              {
                v90 = 35454;
                if ( _bittest64(&v90, v54) )
                  goto LABEL_64;
              }
              if ( (unsigned int)(v54 - 13) <= 1 || (_DWORD)v54 == 16 )
              {
                v135 = v143;
                v144 = v152;
                v153 = *(_QWORD *)(*(_QWORD *)a3 + 24LL);
                v55 = sub_16435F0(v153, 0);
                v53 = v153;
                v51 = v144;
                v52 = v135;
                if ( v55 )
                {
LABEL_64:
                  v145 = v52;
                  v154 = v51;
                  v56 = sub_12BE0A0(v16, v53);
                  v29 = v154;
                  v30 = v145;
                  if ( v156 <= v56 )
                    goto LABEL_21;
                }
              }
            }
          }
          return 0;
        }
        if ( v27 != 53 )
          return 0;
        v133 = v23;
        v138 = v20;
        v148 = *(_QWORD *)(a3 - 24);
        if ( *(_BYTE *)(v148 + 16) != 13 )
          return 0;
        v28 = sub_12BE0A0(v16, *(_QWORD *)(a3 + 56));
        v29 = v138;
        v30 = v133;
        v31 = *(_QWORD **)(v148 + 24);
        if ( *(_DWORD *)(v148 + 32) > 0x40u )
          v31 = (_QWORD *)*v31;
        if ( v156 > (unsigned __int64)v31 * v28 )
          return 0;
LABEL_21:
        v149 = (unsigned int)(1 << *(_WORD *)(i + 18)) >> 1;
        if ( !v149 )
        {
          v136 = v30;
          v147 = v29;
          v71 = sub_15A9FE0(v16, *(_QWORD *)(i + 56));
          v30 = v136;
          v29 = v147;
          v149 = v71;
        }
        if ( v149 > v30 && *(_BYTE *)(a3 + 16) != 53 )
          return 0;
        v32 = *(_QWORD *)(i + 8);
        v168 = v170;
        v169 = 0x800000000LL;
        if ( v32 )
        {
          v33 = v32;
          v34 = 0;
          do
          {
            v33 = *(_QWORD *)(v33 + 8);
            ++v34;
          }
          while ( v33 );
          v139 = v34;
          if ( v34 > 8 )
          {
            v108 = v30;
            v113 = v32;
            v128 = v29;
            sub_16CD150((__int64)&v168, v170, v34, 8, v32, v30);
            v118 = v9;
            v72 = v113;
            v110 = (__int64 *)i;
            v36 = &v168[8 * (unsigned int)v169];
            v35 = v72;
            v37 = v108;
          }
          else
          {
            v128 = v29;
            v118 = v9;
            v35 = v32;
            v110 = (__int64 *)i;
            v36 = v170;
            v37 = v30;
          }
          do
          {
            v36 += 8;
            *((_QWORD *)v36 - 1) = sub_1648700(v35);
            v35 = *(_QWORD *)(v35 + 8);
          }
          while ( v35 );
          v29 = v128;
          v9 = v118;
          v30 = v37;
          i = (unsigned __int64)v110;
        }
        else
        {
          v139 = 0;
        }
        v38 = v169 + v139;
        LODWORD(v169) = v38;
        if ( v38 )
        {
          v106 = (__int64 *)i;
          LODWORD(i) = v38;
          v129 = v29;
          v103 = v30;
          while ( 1 )
          {
            v39 = v168;
            v40 = (unsigned int)i;
            i = (unsigned int)(i - 1);
            v41 = *(_QWORD *)&v168[8 * v40 - 8];
            LODWORD(v169) = i;
            v42 = *(_BYTE *)(v41 + 16);
            if ( v42 <= 0x17u )
              break;
            if ( (unsigned __int8)(v42 - 71) > 1u )
            {
              if ( v42 == 56 )
              {
                if ( !(unsigned __int8)sub_15FA1F0(v41) )
                  goto LABEL_84;
                v58 = *(_QWORD *)(v41 + 8);
                for ( i = (unsigned int)v169; v58; v58 = *(_QWORD *)(v58 + 8) )
                {
                  v59 = sub_1648700(v58);
                  if ( HIDWORD(v169) <= (unsigned int)i )
                  {
                    v146 = v59;
                    sub_16CD150((__int64)&v168, v170, 0, 8, v60, v61);
                    i = (unsigned int)v169;
                    v59 = v146;
                  }
                  *(_QWORD *)&v168[8 * i] = v59;
                  i = (unsigned int)(v169 + 1);
                  LODWORD(v169) = v169 + 1;
                }
              }
              else
              {
                if ( v42 != 78 )
                  break;
                v57 = *(_QWORD *)(v41 - 24);
                if ( *(_BYTE *)(v57 + 16)
                  || (*(_BYTE *)(v57 + 33) & 0x20) == 0
                  || (unsigned int)(*(_DWORD *)(v57 + 36) - 116) > 1 )
                {
                  break;
                }
              }
            }
            else
            {
              for ( j = *(_QWORD *)(v41 + 8); j; j = *(_QWORD *)(j + 8) )
              {
                v44 = sub_1648700(j);
                if ( HIDWORD(v169) <= (unsigned int)i )
                {
                  v140 = v44;
                  sub_16CD150((__int64)&v168, v170, 0, 8, v45, v46);
                  i = (unsigned int)v169;
                  v44 = v140;
                }
                *(_QWORD *)&v168[8 * i] = v44;
                i = (unsigned int)(v169 + 1);
                LODWORD(v169) = v169 + 1;
              }
            }
LABEL_36:
            if ( !(_DWORD)i )
            {
              v29 = v129;
              v7 = a1;
              i = (unsigned __int64)v106;
              v30 = v103;
              goto LABEL_79;
            }
          }
          if ( v9 != v41 && a7 != v41 )
            goto LABEL_85;
          goto LABEL_36;
        }
LABEL_79:
        v122 = v30;
        v62 = &v160;
        v130 = v29;
        v63 = sub_165AFC0(&v160);
        v65 = 0;
        v66 = v130;
        v67 = v122;
        if ( v63 )
        {
          v123 = v7;
          v68 = 0;
          v112 = v9;
          v69 = 0;
          v107 = a3;
          v70 = v63;
          v105 = v67;
          do
          {
            ++v68;
            v64 = 24LL * (*(_DWORD *)((v160 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
            v62 = (__int64 *)(v69 - v64);
            if ( i == *(_QWORD *)((v160 & 0xFFFFFFFFFFFFFFF8LL) + v69 - v64) )
            {
              v65 = v68;
              v62 = &v160;
              if ( !(unsigned __int8)sub_1779030(&v160, v68, 22) )
                goto LABEL_84;
            }
            v69 += 24;
          }
          while ( v70 != v68 );
          v66 = v130;
          v7 = v123;
          v9 = v112;
          a3 = v107;
          v67 = v105;
        }
        if ( !*(_QWORD *)(v7 + 96) )
          goto LABEL_128;
        v124 = v67;
        v62 = (__int64 *)(v7 + 80);
        v131 = v66;
        v73 = (*(__int64 (__fastcall **)(__int64, __int64))(v7 + 104))(v7 + 80, v65);
        v74 = v131;
        v75 = v124;
        v76 = v73;
        if ( *(_BYTE *)(a3 + 16) > 0x17u )
        {
          v65 = a3;
          v62 = (__int64 *)v73;
          v77 = sub_15CCEE0(v73, a3, a7);
          v74 = v131;
          v75 = v124;
          if ( !v77 )
            goto LABEL_84;
        }
        if ( !*(_QWORD *)(v7 + 32) )
LABEL_128:
          sub_4263D6(v62, v65, v64);
        v114 = v75;
        v125 = v74;
        v78 = (_QWORD *)(*(__int64 (__fastcall **)(__int64))(v7 + 40))(v7 + 16);
        v163 = a3;
        v165 = 0;
        v79 = v125;
        v126 = (__int64)v78;
        v164 = v156;
        v166 = 0;
        v167 = 0;
        v80 = sub_134F0E0(v78, v79, (__int64)&v163);
        v81 = v114;
        if ( (v80 & 3) != 0 )
        {
          v163 = a3;
          v164 = v156;
          v165 = 0;
          v166 = 0;
          v167 = 0;
          v91 = sub_13510F0(v126, a7, &v163, v76, 0);
          v39 = v168;
          v81 = v114;
          if ( (v91 & 3) != 0 )
            goto LABEL_85;
        }
        v82 = *(_QWORD *)i;
        if ( *(_BYTE *)(*(_QWORD *)i + 8LL) == 16 )
          v82 = **(_QWORD **)(v82 + 16);
        v83 = *(_DWORD *)(v82 + 8);
        v84 = *(_QWORD *)a3;
        v85 = v83 >> 8;
        if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
          v84 = **(_QWORD **)(v84 + 16);
        if ( *(_DWORD *)(v84 + 8) >> 8 == v85 )
        {
          v159 = (__int64 *)v7;
          v86 = 0;
          v115 = v81;
          while ( (unsigned int)sub_165AFC0(&v160) > v86 )
          {
            if ( i == sub_1649C60(*(_QWORD *)((v160 & 0xFFFFFFFFFFFFFFF8LL)
                                            + 24LL * v86
                                            - 24LL * (*(_DWORD *)((v160 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF))) )
            {
              v87 = *(_QWORD *)i;
              if ( *(_BYTE *)(*(_QWORD *)i + 8LL) == 16 )
                v87 = **(_QWORD **)(v87 + 16);
              v88 = *(_DWORD *)(v87 + 8) >> 8;
              v89 = **(_QWORD **)((v160 & 0xFFFFFFFFFFFFFFF8LL)
                                + 24LL * v86
                                - 24LL * (*(_DWORD *)((v160 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
              if ( *(_BYTE *)(v89 + 8) == 16 )
                v89 = **(_QWORD **)(v89 + 16);
              if ( *(_DWORD *)(v89 + 8) >> 8 != v88 )
                goto LABEL_84;
            }
            ++v86;
          }
          v13 = 0;
          v92 = v115;
          v116 = v9;
          v93 = 0;
          v109 = v92;
          while ( (unsigned int)sub_165AFC0(&v160) > v93 )
          {
            if ( i == sub_1649C60(*(_QWORD *)((v160 & 0xFFFFFFFFFFFFFFF8LL)
                                            + 8
                                            * (3LL * v93
                                             - 3LL * (*(_DWORD *)((v160 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)))) )
            {
              v94 = (_QWORD *)a3;
              if ( *(_QWORD *)a3 != *(_QWORD *)i )
              {
                v95 = sub_1649960(a3);
                LOWORD(v165) = 261;
                v161 = v95;
                v163 = (__int64)&v161;
                v162 = v96;
                v94 = (_QWORD *)sub_15FDFF0(a3, *(_QWORD *)i, (__int64)&v163, a7);
              }
              if ( *v94 == **(_QWORD **)(sub_165B7C0(&v160) + 24LL * v93) )
              {
                v102 = (__int64)v94;
                v13 = 1;
                sub_19CF7F0(v160 & 0xFFFFFFFFFFFFFFF8LL, v93, v102);
              }
              else
              {
                v97 = sub_1649960((__int64)v94);
                LOWORD(v165) = 261;
                v161 = v97;
                v162 = v98;
                v163 = (__int64)&v161;
                v99 = sub_165B7C0(&v160);
                v100 = (__int64)v94;
                v13 = 1;
                v101 = sub_15FDFF0(v100, **(_QWORD **)(v99 + 24LL * v93), (__int64)&v163, a7);
                sub_19CF7F0(v160 & 0xFFFFFFFFFFFFFFF8LL, v93, v101);
              }
            }
            ++v93;
          }
          if ( (_BYTE)v13 )
          {
            if ( v149 > v109 )
              sub_15F8A20(a3, v149);
            sub_14191F0(*v159, a7);
            v163 = 0x700000001LL;
            v164 = 0x1000000008LL;
            sub_1AEC0C0(a7, v116, &v163, 4);
            sub_14191F0(*v159, v116);
            v39 = v168;
            goto LABEL_86;
          }
        }
LABEL_84:
        v39 = v168;
LABEL_85:
        v13 = 0;
LABEL_86:
        if ( v39 != v170 )
          _libc_free((unsigned __int64)v39);
        return v13;
    }
  }
}
