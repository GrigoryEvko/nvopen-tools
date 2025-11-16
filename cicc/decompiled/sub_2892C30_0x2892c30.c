// Function: sub_2892C30
// Address: 0x2892c30
//
__int64 __fastcall sub_2892C30(__int64 a1, __m128d a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rcx
  __int64 v8; // r13
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // r12
  char v11; // al
  char *v12; // r15
  char v13; // al
  _QWORD *v14; // r15
  __int64 v15; // rax
  _QWORD *v16; // r14
  char v17; // al
  unsigned __int64 v18; // r12
  char *v19; // rbx
  unsigned __int8 v20; // al
  __int64 v21; // r11
  int v22; // r10d
  _QWORD *v23; // r8
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned int v26; // r11d
  _QWORD *v27; // rax
  __int64 v28; // rax
  __int64 v29; // r9
  unsigned __int64 v30; // r8
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // r14
  __int64 v35; // rsi
  __int64 v36; // rdx
  int v38; // ecx
  int v39; // eax
  __int64 v40; // rsi
  _BYTE *v41; // rbx
  char v42; // al
  _BYTE **v43; // rax
  _BYTE *v44; // rdx
  __int64 v45; // rsi
  __int64 v46; // rsi
  _BYTE *v47; // r8
  __int64 v48; // r11
  unsigned int v49; // r15d
  __int64 v50; // rbx
  __int64 v51; // r12
  unsigned int v52; // edi
  __int64 v53; // rcx
  _BYTE *v54; // rsi
  _BYTE *v55; // rsi
  _BYTE *v56; // rsi
  _BYTE *v57; // rsi
  __int64 v58; // r9
  __int64 v59; // r8
  unsigned int v60; // eax
  unsigned int *v61; // rcx
  unsigned int *v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rdx
  unsigned int *v66; // rdi
  __int64 v67; // rax
  __int64 v68; // rax
  _BYTE *v69; // r8
  unsigned __int64 v70; // rdx
  unsigned int v71; // eax
  __int64 v72; // r15
  __int64 v73; // r12
  __int64 v74; // rax
  unsigned int *v75; // rbx
  unsigned int *v76; // r14
  unsigned __int8 *v77; // rax
  int v78; // ecx
  __int64 v79; // rsi
  char v80; // al
  __int64 v81; // rdi
  unsigned __int64 v82; // rax
  __int64 v83; // rbx
  __int64 v84; // rax
  unsigned __int64 v85; // rdx
  __int64 v86; // rax
  unsigned int v87; // esi
  unsigned int v88; // r9d
  __int64 v89; // rax
  __int64 v90; // rax
  int v91; // ebx
  __int64 v92; // rax
  __int64 v93; // r12
  __int64 v94; // rdx
  __int64 v95; // rsi
  unsigned int v96; // r10d
  unsigned int v97; // r11d
  _QWORD *v98; // rax
  __int64 v99; // rax
  __int64 v100; // r8
  __int64 v101; // r9
  __int64 v102; // rax
  unsigned __int64 v103; // rbx
  __int64 v104; // rax
  __int64 v105; // rax
  __int64 v106; // r14
  unsigned int v107; // edx
  __int64 v108; // rax
  __int64 v109; // rdx
  __int64 v110; // rax
  __int64 v111; // rax
  __int64 v112; // rax
  __int64 v113; // rax
  __int64 v114; // rax
  __int64 v115; // rax
  unsigned int *v116; // rcx
  unsigned int *v117; // rdx
  __int64 v118; // rcx
  _BYTE *v119; // rax
  __int64 v120; // rsi
  _BYTE *v121; // rcx
  _BYTE *v122; // rcx
  __int64 v123; // [rsp+8h] [rbp-128h]
  unsigned __int64 v124; // [rsp+10h] [rbp-120h]
  _QWORD *v125; // [rsp+18h] [rbp-118h]
  __int64 v126; // [rsp+28h] [rbp-108h]
  __int64 v127; // [rsp+38h] [rbp-F8h]
  _QWORD *v128; // [rsp+38h] [rbp-F8h]
  unsigned int v129; // [rsp+38h] [rbp-F8h]
  _QWORD *v131; // [rsp+48h] [rbp-E8h]
  __int64 v132; // [rsp+48h] [rbp-E8h]
  _BYTE *v133; // [rsp+48h] [rbp-E8h]
  __int64 v134; // [rsp+48h] [rbp-E8h]
  unsigned int v135; // [rsp+48h] [rbp-E8h]
  unsigned int v136; // [rsp+48h] [rbp-E8h]
  int v137; // [rsp+48h] [rbp-E8h]
  int v138; // [rsp+50h] [rbp-E0h]
  unsigned int v139; // [rsp+50h] [rbp-E0h]
  char v140; // [rsp+50h] [rbp-E0h]
  unsigned int v141; // [rsp+50h] [rbp-E0h]
  unsigned __int64 v142; // [rsp+50h] [rbp-E0h]
  _QWORD *v143; // [rsp+50h] [rbp-E0h]
  unsigned int v144; // [rsp+50h] [rbp-E0h]
  unsigned int v145; // [rsp+50h] [rbp-E0h]
  __int64 v146; // [rsp+58h] [rbp-D8h]
  __int64 v147; // [rsp+58h] [rbp-D8h]
  __int64 v148; // [rsp+58h] [rbp-D8h]
  __int64 v149; // [rsp+58h] [rbp-D8h]
  __int64 v150; // [rsp+60h] [rbp-D0h]
  __int64 v151; // [rsp+60h] [rbp-D0h]
  __int64 v152; // [rsp+60h] [rbp-D0h]
  __int64 v153; // [rsp+60h] [rbp-D0h]
  __int64 v154; // [rsp+60h] [rbp-D0h]
  __int64 v155; // [rsp+60h] [rbp-D0h]
  char v156; // [rsp+68h] [rbp-C8h]
  _QWORD *v157; // [rsp+68h] [rbp-C8h]
  __int64 v158; // [rsp+70h] [rbp-C0h]
  __int64 v159; // [rsp+78h] [rbp-B8h]
  __int64 v160; // [rsp+88h] [rbp-A8h] BYREF
  unsigned __int64 v161; // [rsp+90h] [rbp-A0h] BYREF
  unsigned int v162; // [rsp+98h] [rbp-98h]
  unsigned __int64 v163; // [rsp+A0h] [rbp-90h] BYREF
  unsigned int v164; // [rsp+A8h] [rbp-88h]
  unsigned int *v165; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v166; // [rsp+B8h] [rbp-78h]
  _DWORD v167[28]; // [rsp+C0h] [rbp-70h] BYREF

  v158 = a4 + 72;
  v159 = *(_QWORD *)(a4 + 80);
  if ( v159 == a4 + 72 )
  {
    v35 = a1 + 32;
    v36 = a1 + 80;
    goto LABEL_190;
  }
  v156 = 0;
  do
  {
    if ( !v159 )
      BUG();
    v7 = *(_QWORD *)(v159 + 24);
    v8 = v159 + 24;
    v9 = v7 & 0xFFFFFFFFFFFFFFF8LL;
    v10 = v7 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v159 + 24 == (v7 & 0xFFFFFFFFFFFFFFF8LL) )
      goto LABEL_236;
    if ( !v9 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 > 0xA )
LABEL_236:
      BUG();
    v11 = *(_BYTE *)(v9 - 24);
    if ( v11 != 31 )
    {
      if ( v11 == 32 )
      {
        v43 = *(_BYTE ***)(v9 - 32);
        v44 = *v43;
        if ( **v43 != 85 )
          goto LABEL_12;
        v45 = *((_QWORD *)v44 - 4);
        if ( !v45 )
          goto LABEL_12;
        if ( *(_BYTE *)v45 )
          goto LABEL_12;
        if ( *(_QWORD *)(v45 + 24) != *((_QWORD *)v44 + 10) )
          goto LABEL_12;
        a7 = *(unsigned int *)(v45 + 36);
        if ( (unsigned int)(a7 - 91) > 1 )
          goto LABEL_12;
        v46 = *((_DWORD *)v44 + 1) & 0x7FFFFFF;
        v47 = *(_BYTE **)&v44[32 * (1 - v46)];
        if ( *v47 != 17 )
          goto LABEL_12;
        v153 = *(_QWORD *)&v44[-32 * v46];
        v48 = ((*(_DWORD *)(v9 - 20) & 0x7FFFFFFu) >> 1) - 1;
        v49 = (*(_DWORD *)(v9 - 20) & 0x7FFFFFFu) >> 1;
        v50 = v48 >> 2;
        if ( v48 >> 2 )
        {
          v51 = 4 * v50;
          v52 = 2;
          v53 = 0;
          while ( 1 )
          {
            v50 = v53 + 1;
            v57 = v43[4 * v52];
            if ( v57 && v47 == v57 )
            {
              v50 = v53;
              goto LABEL_79;
            }
            v54 = v43[4 * v52 + 8];
            if ( v54 && v47 == v54 )
              goto LABEL_79;
            v50 = v53 + 3;
            v55 = v43[4 * v52 + 16];
            if ( v55 )
            {
              if ( v47 == v55 )
                break;
            }
            v53 += 4;
            v56 = v43[4 * (unsigned int)(2 * v53)];
            if ( v56 && v47 == v56 )
              goto LABEL_79;
            v52 += 8;
            if ( v53 == v51 )
            {
              v50 = v53;
              goto LABEL_215;
            }
          }
          v50 = v53 + 2;
LABEL_79:
          v147 = v50;
          if ( v48 != v50 )
            goto LABEL_80;
          goto LABEL_220;
        }
LABEL_215:
        v118 = v48 - v50;
        if ( v48 - v50 == 2 )
        {
          v120 = v50;
        }
        else
        {
          if ( v118 != 3 )
          {
            if ( v118 != 1 )
              goto LABEL_220;
            goto LABEL_218;
          }
          v120 = v50 + 1;
          v121 = v43[4 * (unsigned int)(2 * (v50 + 1))];
          if ( v121 && v47 == v121 )
            goto LABEL_79;
        }
        v50 = v120 + 1;
        v122 = v43[4 * (unsigned int)(2 * (v120 + 1))];
        if ( v122 && v47 == v122 )
        {
          v50 = v120;
          goto LABEL_79;
        }
LABEL_218:
        v119 = v43[4 * (unsigned int)(2 * v50 + 2)];
        if ( v119 && v47 == v119 )
          goto LABEL_79;
LABEL_220:
        v147 = 4294967294LL;
        LODWORD(v50) = -2;
LABEL_80:
        sub_2892AE0(&v165, a7, (__int64)v44, v49, a2);
        v59 = HIDWORD(v165);
        v60 = (unsigned int)v165;
        v165 = v167;
        v166 = 0x1000000000LL;
        if ( v49 > 0x10 )
        {
          v129 = v60;
          v137 = v59;
          sub_C8D5F0((__int64)&v165, v167, v49, 4u, v59, v58);
          v116 = v165;
          LODWORD(v59) = v137;
          v117 = &v165[v49];
          do
            *v116++ = v129;
          while ( v117 != v116 );
          LODWORD(v166) = v49;
          v61 = v165;
        }
        else
        {
          v61 = v167;
          if ( v49 )
          {
            v62 = v167;
            do
              *v62++ = v60;
            while ( &v167[v49] != v62 );
            v61 = v165;
          }
          LODWORD(v166) = v49;
        }
        if ( v147 != 4294967294LL )
          v61 += (unsigned int)(v50 + 1);
        *v61 = v59;
        sub_2A3E730(v9 - 24, v165, (unsigned int)v166, 1);
        v63 = *(_QWORD *)(v9 - 32);
        if ( *(_QWORD *)v63 )
        {
          v64 = *(_QWORD *)(v63 + 8);
          **(_QWORD **)(v63 + 16) = v64;
          if ( v64 )
            *(_QWORD *)(v64 + 16) = *(_QWORD *)(v63 + 16);
        }
        *(_QWORD *)v63 = v153;
        if ( v153 )
        {
          v65 = *(_QWORD *)(v153 + 16);
          *(_QWORD *)(v63 + 8) = v65;
          if ( v65 )
            *(_QWORD *)(v65 + 16) = v63 + 8;
          *(_QWORD *)(v63 + 16) = v153 + 16;
          *(_QWORD *)(v153 + 16) = v63;
        }
        sub_BC8EC0(v9 - 24, v165, (unsigned int)v166, 1);
        v66 = v165;
        if ( v165 == v167 )
          goto LABEL_186;
LABEL_185:
        _libc_free((unsigned __int64)v66);
LABEL_186:
        v7 = *(_QWORD *)(v159 + 24);
        goto LABEL_12;
      }
LABEL_13:
      v14 = (_QWORD *)v10;
      while ( 1 )
      {
        v16 = v14;
        v17 = *((_BYTE *)v14 - 24);
        v18 = *v14 & 0xFFFFFFFFFFFFFFF8LL;
        v14 = (_QWORD *)v18;
        if ( v17 == 85 )
          break;
        if ( v17 != 86 )
          goto LABEL_17;
        v19 = (char *)*(v16 - 15);
        v20 = *v19;
        if ( (unsigned __int8)*v19 <= 0x1Cu )
          goto LABEL_17;
        if ( v20 == 82 )
        {
          v22 = *((_WORD *)v19 + 1) & 0x3F;
          if ( (unsigned int)(v22 - 32) > 1 )
            goto LABEL_17;
          v67 = *((_QWORD *)v19 - 4);
          if ( *(_BYTE *)v67 != 17 )
            goto LABEL_17;
          v21 = *((_QWORD *)v19 - 8);
          if ( *(_BYTE *)v21 != 85 || *(_DWORD *)(v67 + 32) > 0x40u )
            goto LABEL_17;
          v23 = *(_QWORD **)(v67 + 24);
LABEL_24:
          v24 = *(_QWORD *)(v21 - 32);
          if ( !v24 )
            goto LABEL_17;
          if ( *(_BYTE *)v24 )
            goto LABEL_17;
          if ( *(_QWORD *)(v24 + 24) != *(_QWORD *)(v21 + 80) )
            goto LABEL_17;
          v131 = v23;
          v138 = v22;
          v150 = *(_QWORD *)(v21 - 32);
          if ( (unsigned int)(*(_DWORD *)(v24 + 36) - 91) > 1 )
            goto LABEL_17;
          v25 = *(_DWORD *)(v21 + 4) & 0x7FFFFFF;
          a7 = *(_QWORD *)(v21 + 32 * (1 - v25));
          v127 = a7;
          if ( *(_BYTE *)a7 != 17 )
            goto LABEL_17;
          v146 = v21;
          v126 = *(_QWORD *)(v21 - 32 * v25);
          v163 = sub_BD5C60(v21);
          sub_2892AE0(&v165, *(_DWORD *)(v150 + 36), v146, 2, a2);
          v26 = HIDWORD(v165);
          LODWORD(v150) = (_DWORD)v165;
          v165 = v167;
          v166 = 0x400000000LL;
          v27 = *(_QWORD **)(v127 + 24);
          if ( *(_DWORD *)(v127 + 32) > 0x40u )
            v27 = (_QWORD *)*v27;
          if ( (v131 == v27) == (v138 == 32) )
          {
            v141 = v26;
            v90 = sub_B8C2F0(&v163, v26, v150, 1);
            LODWORD(v166) = 0;
            v29 = v90;
            v31 = 0;
            v30 = (v150 << 32) | v141;
            if ( HIDWORD(v166) > 1 )
              goto LABEL_33;
          }
          else
          {
            v139 = v26;
            v28 = sub_B8C2F0(&v163, v150, v26, 1);
            LODWORD(v166) = 0;
            v29 = v28;
            v30 = ((unsigned __int64)v139 << 32) | (unsigned int)v150;
            v31 = 0;
            if ( HIDWORD(v166) > 1 )
              goto LABEL_33;
          }
          v142 = v30;
          v154 = v29;
          sub_C8D5F0((__int64)&v165, v167, 2u, 4u, v30, v29);
          v30 = v142;
          v29 = v154;
          v31 = (unsigned int)v166;
LABEL_33:
          *(_QWORD *)&v165[v31] = v30;
          LODWORD(v166) = v166 + 2;
          if ( v19 )
          {
            if ( *((_QWORD *)v19 - 8) )
            {
              v32 = *((_QWORD *)v19 - 7);
              **((_QWORD **)v19 - 6) = v32;
              if ( v32 )
                *(_QWORD *)(v32 + 16) = *((_QWORD *)v19 - 6);
            }
            *((_QWORD *)v19 - 8) = v126;
            if ( v126 )
            {
              v33 = *(_QWORD *)(v126 + 16);
              *((_QWORD *)v19 - 7) = v33;
              if ( v33 )
                *(_QWORD *)(v33 + 16) = v19 - 56;
              *((_QWORD *)v19 - 6) = v126 + 16;
              *(_QWORD *)(v126 + 16) = v19 - 64;
            }
          }
          else
          {
            if ( *(v16 - 15) )
            {
              v111 = *(v16 - 14);
              *(_QWORD *)*(v16 - 13) = v111;
              if ( v111 )
                *(_QWORD *)(v111 + 16) = *(v16 - 13);
            }
            *(v16 - 15) = v126;
            if ( v126 )
            {
              v112 = *(_QWORD *)(v126 + 16);
              *(v16 - 14) = v112;
              if ( v112 )
                *(_QWORD *)(v112 + 16) = v16 - 14;
              *(v16 - 13) = v126 + 16;
              *(_QWORD *)(v126 + 16) = v16 - 15;
            }
          }
          v34 = (__int64)(v16 - 3);
          v151 = v29;
          sub_2A3E6C0(v34, v165, (unsigned int)v166);
          sub_B99FD0(v34, 2u, v151);
          if ( v165 == v167 )
            goto LABEL_17;
          _libc_free((unsigned __int64)v165);
          if ( v18 == v8 )
            goto LABEL_43;
        }
        else
        {
          if ( v20 == 85 )
          {
            v21 = *(v16 - 15);
            v22 = 33;
            v19 = 0;
            v23 = 0;
            goto LABEL_24;
          }
LABEL_17:
          if ( v18 == v8 )
            goto LABEL_43;
        }
      }
      v15 = *(v16 - 7);
      if ( !v15 )
        goto LABEL_17;
      if ( *(_BYTE *)v15 )
        goto LABEL_17;
      if ( *(_QWORD *)(v15 + 24) != v16[7] )
        goto LABEL_17;
      v38 = *(_DWORD *)(v15 + 36);
      if ( (unsigned int)(v38 - 91) > 1 )
        goto LABEL_17;
      v39 = *((_DWORD *)v16 - 5);
      v157 = v16 - 3;
      v40 = v39 & 0x7FFFFFF;
      v152 = v16[4 * (1 - v40) - 3];
      if ( *(_BYTE *)v152 != 17 )
      {
LABEL_59:
        sub_BD84D0((__int64)v157, v16[-4 * (v39 & 0x7FFFFFF) - 3]);
        sub_B43D60(v157);
        v156 = 1;
        goto LABEL_17;
      }
      v140 = 1;
      v41 = (_BYTE *)v16[-4 * v40 - 3];
      if ( v38 == 92 )
      {
        a2.m128d_f64[0] = sub_C41B00((__int64 *)(v157[4 * (2 - v40)] + 24LL));
        v140 = a2.m128d_f64[0] > 0.5;
      }
      v165 = v167;
      v166 = 0x400000000LL;
      v42 = *v41;
      if ( *v41 <= 0x1Cu )
      {
LABEL_58:
        v39 = *((_DWORD *)v16 - 5);
        goto LABEL_59;
      }
      if ( v42 != 84 )
      {
        while ( v42 == 68 || v42 == 69 )
        {
          v68 = (unsigned int)v166;
          v69 = (_BYTE *)*((_QWORD *)v41 - 4);
          v70 = (unsigned int)v166 + 1LL;
          if ( v70 > HIDWORD(v166) )
            goto LABEL_152;
LABEL_103:
          *(_QWORD *)&v165[2 * v68] = v41;
          v41 = v69;
          LODWORD(v166) = v166 + 1;
          v42 = *v69;
          if ( *v69 <= 0x1Cu )
            goto LABEL_56;
          if ( v42 == 84 )
            goto LABEL_105;
        }
        if ( v42 != 59 || **((_BYTE **)v41 - 4) != 17 )
          goto LABEL_56;
        v68 = (unsigned int)v166;
        v69 = (_BYTE *)*((_QWORD *)v41 - 8);
        v70 = (unsigned int)v166 + 1LL;
        if ( v70 <= HIDWORD(v166) )
          goto LABEL_103;
LABEL_152:
        v133 = v69;
        sub_C8D5F0((__int64)&v165, v167, v70, 8u, (__int64)v69, a7);
        v68 = (unsigned int)v166;
        v69 = v133;
        goto LABEL_103;
      }
LABEL_105:
      v71 = *((_DWORD *)v41 + 1) & 0x7FFFFFF;
      if ( !v71 )
      {
LABEL_56:
        if ( v165 != v167 )
          _libc_free((unsigned __int64)v165);
        goto LABEL_58;
      }
      v128 = v16;
      v132 = 8LL * v71;
      v125 = (_QWORD *)v18;
      v72 = (__int64)v41;
      v124 = v18;
      v73 = 0;
      while ( 1 )
      {
        v74 = *(_QWORD *)(*(_QWORD *)(v72 - 8) + 4 * v73);
        if ( *(_BYTE *)v74 == 17 )
          break;
LABEL_148:
        v73 += 8;
        if ( v132 == v73 )
        {
          v14 = v125;
          v16 = v128;
          v18 = v124;
          goto LABEL_56;
        }
      }
      v162 = *(_DWORD *)(v74 + 32);
      if ( v162 > 0x40 )
        sub_C43780((__int64)&v161, (const void **)(v74 + 24));
      else
        v161 = *(_QWORD *)(v74 + 24);
      v75 = v165;
      v76 = &v165[2 * (unsigned int)v166];
      if ( v165 == v76 )
      {
LABEL_123:
        if ( *(_DWORD *)(v152 + 32) <= 0x40u )
          v80 = *(_QWORD *)(v152 + 24) == v161;
        else
          v80 = sub_C43C50(v152 + 24, (const void **)&v161);
        if ( v140 == v80 )
          goto LABEL_145;
        v81 = *(_QWORD *)(*(_QWORD *)(v72 - 8) + 32LL * *(unsigned int *)(v72 + 72) + v73);
        v82 = *(_QWORD *)(v81 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v82 == v81 + 48 )
          goto LABEL_233;
        if ( !v82 )
          BUG();
        v83 = v82 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v82 - 24) - 30 > 0xA )
LABEL_233:
          BUG();
        if ( *(_BYTE *)(v82 - 24) != 31 || (*(_DWORD *)(v82 - 20) & 0x7FFFFFF) != 3 )
        {
          v84 = sub_AA54C0(v81);
          if ( !v84 )
            goto LABEL_145;
          v85 = *(_QWORD *)(v84 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v85 == v84 + 48 )
            goto LABEL_229;
          if ( !v85 )
            BUG();
          v83 = v85 - 24;
          if ( (unsigned int)*(unsigned __int8 *)(v85 - 24) - 30 > 0xA )
LABEL_229:
            BUG();
          if ( *(_BYTE *)(v85 - 24) != 31 || (*(_DWORD *)(v85 - 20) & 0x7FFFFFF) == 1 )
            goto LABEL_145;
        }
        v160 = sub_BD5C60(v72);
        v86 = *(v128 - 7);
        if ( !v86 || *(_BYTE *)v86 || *(_QWORD *)(v86 + 24) != v128[7] )
          BUG();
        v123 = *(_QWORD *)(*(_QWORD *)(v72 - 8) + 32LL * *(unsigned int *)(v72 + 72) + v73);
        sub_2892AE0(&v163, *(_DWORD *)(v86 + 36), (__int64)v157, 2, a2);
        v87 = HIDWORD(v163);
        v88 = v163;
        if ( !v140 )
        {
          v87 = v163;
          v88 = HIDWORD(v163);
        }
        v89 = *(_QWORD *)(v83 - 64);
        if ( v123 != v89 )
        {
          if ( v123 != *(_QWORD *)(v83 + 40) )
          {
            if ( v123 != *(_QWORD *)(v83 - 32) )
              goto LABEL_145;
LABEL_194:
            v107 = v87;
            v87 = v88;
            goto LABEL_188;
          }
          v109 = *(_QWORD *)(v72 + 40);
          if ( v89 != v109 )
          {
            v110 = *(_QWORD *)(v83 - 32);
            if ( v123 != v110 && v109 != v110 )
            {
LABEL_145:
              if ( v162 > 0x40 && v161 )
                j_j___libc_free_0_0(v161);
              goto LABEL_148;
            }
            goto LABEL_194;
          }
        }
        v107 = v88;
LABEL_188:
        v108 = sub_B8C2F0(&v160, v87, v107, 1);
        sub_B99FD0(v83, 2u, v108);
        goto LABEL_145;
      }
      while ( 1 )
      {
        while ( 1 )
        {
          v77 = (unsigned __int8 *)*((_QWORD *)v76 - 1);
          v78 = *v77;
          if ( v78 != 68 )
            break;
          sub_C449B0((__int64)&v163, (const void **)&v161, *(_DWORD *)(*((_QWORD *)v77 + 1) + 8LL) >> 8);
          if ( v162 > 0x40 )
          {
LABEL_113:
            if ( v161 )
              j_j___libc_free_0_0(v161);
          }
LABEL_115:
          v161 = v163;
          v162 = v164;
LABEL_116:
          v76 -= 2;
          if ( v75 == v76 )
            goto LABEL_123;
        }
        if ( v78 == 69 )
        {
          sub_C44830((__int64)&v163, &v161, *(_DWORD *)(*((_QWORD *)v77 + 1) + 8LL) >> 8);
          if ( v162 > 0x40 )
            goto LABEL_113;
          goto LABEL_115;
        }
        if ( v78 != 59 )
          BUG();
        if ( (v77[7] & 0x40) == 0 )
        {
          v79 = *(_QWORD *)&v77[-32 * (*((_DWORD *)v77 + 1) & 0x7FFFFFF) + 32];
          if ( v162 <= 0x40 )
            goto LABEL_122;
LABEL_158:
          sub_C43C10(&v161, (__int64 *)(v79 + 24));
          goto LABEL_116;
        }
        v79 = *(_QWORD *)(*((_QWORD *)v77 - 1) + 32LL);
        if ( v162 > 0x40 )
          goto LABEL_158;
LABEL_122:
        v76 -= 2;
        v161 ^= *(_QWORD *)(v79 + 24);
        if ( v75 == v76 )
          goto LABEL_123;
      }
    }
    if ( (*(_DWORD *)(v9 - 20) & 0x7FFFFFF) == 1 )
      goto LABEL_13;
    v12 = *(char **)(v9 - 120);
    v13 = *v12;
    if ( (unsigned __int8)*v12 > 0x1Cu )
    {
      if ( v13 == 82 )
      {
        v91 = *((_WORD *)v12 + 1) & 0x3F;
        if ( (unsigned int)(v91 - 32) > 1 )
          goto LABEL_12;
        v92 = *((_QWORD *)v12 - 4);
        if ( *(_BYTE *)v92 != 17 )
          goto LABEL_12;
        v93 = *((_QWORD *)v12 - 8);
        if ( *(_BYTE *)v93 != 85 || *(_DWORD *)(v92 + 32) > 0x40u )
          goto LABEL_12;
        a7 = *(_QWORD *)(v92 + 24);
      }
      else
      {
        if ( v13 != 85 )
          goto LABEL_12;
        v93 = *(_QWORD *)(v9 - 120);
        v91 = 33;
        v12 = 0;
        a7 = 0;
      }
      v94 = *(_QWORD *)(v93 - 32);
      if ( v94 )
      {
        if ( !*(_BYTE *)v94 && *(_QWORD *)(v94 + 24) == *(_QWORD *)(v93 + 80) )
        {
          v143 = (_QWORD *)a7;
          v148 = *(_QWORD *)(v93 - 32);
          if ( (unsigned int)(*(_DWORD *)(v94 + 36) - 91) <= 1 )
          {
            v95 = *(_DWORD *)(v93 + 4) & 0x7FFFFFF;
            v134 = *(_QWORD *)(v93 + 32 * (1 - v95));
            if ( *(_BYTE *)v134 == 17 )
            {
              v155 = *(_QWORD *)(v93 - 32 * v95);
              v163 = sub_BD5C60(v93);
              sub_2892AE0(&v165, *(_DWORD *)(v148 + 36), v93, 2, a2);
              v96 = HIDWORD(v165);
              v97 = (unsigned int)v165;
              v165 = v167;
              v166 = 0x400000000LL;
              v98 = *(_QWORD **)(v134 + 24);
              if ( *(_DWORD *)(v134 + 32) > 0x40u )
                v98 = (_QWORD *)*v98;
              if ( (v143 == v98) == (v91 == 32) )
              {
                v136 = v97;
                v145 = v96;
                v113 = sub_B8C2F0(&v163, v96, v97, 1);
                LODWORD(v166) = 0;
                v149 = v113;
                v102 = 0;
                v103 = ((unsigned __int64)v136 << 32) | v145;
                if ( HIDWORD(v166) > 1 )
                  goto LABEL_176;
              }
              else
              {
                v135 = v96;
                v144 = v97;
                v99 = sub_B8C2F0(&v163, v97, v96, 1);
                LODWORD(v166) = 0;
                v149 = v99;
                v102 = 0;
                v103 = ((unsigned __int64)v135 << 32) | v144;
                if ( HIDWORD(v166) > 1 )
                  goto LABEL_176;
              }
              sub_C8D5F0((__int64)&v165, v167, 2u, 4u, v100, v101);
              v102 = (unsigned int)v166;
LABEL_176:
              *(_QWORD *)&v165[v102] = v103;
              LODWORD(v166) = v166 + 2;
              if ( v12 )
              {
                if ( *((_QWORD *)v12 - 8) )
                {
                  v104 = *((_QWORD *)v12 - 7);
                  **((_QWORD **)v12 - 6) = v104;
                  if ( v104 )
                    *(_QWORD *)(v104 + 16) = *((_QWORD *)v12 - 6);
                }
                *((_QWORD *)v12 - 8) = v155;
                if ( v155 )
                {
                  v105 = *(_QWORD *)(v155 + 16);
                  *((_QWORD *)v12 - 7) = v105;
                  if ( v105 )
                    *(_QWORD *)(v105 + 16) = v12 - 56;
                  *((_QWORD *)v12 - 6) = v155 + 16;
                  *(_QWORD *)(v155 + 16) = v12 - 64;
                }
              }
              else
              {
                if ( *(_QWORD *)(v9 - 120) )
                {
                  v114 = *(_QWORD *)(v9 - 112);
                  **(_QWORD **)(v9 - 104) = v114;
                  if ( v114 )
                    *(_QWORD *)(v114 + 16) = *(_QWORD *)(v9 - 104);
                }
                *(_QWORD *)(v9 - 120) = v155;
                if ( v155 )
                {
                  v115 = *(_QWORD *)(v155 + 16);
                  *(_QWORD *)(v9 - 112) = v115;
                  if ( v115 )
                    *(_QWORD *)(v115 + 16) = v9 - 112;
                  *(_QWORD *)(v9 - 104) = v155 + 16;
                  *(_QWORD *)(v155 + 16) = v9 - 120;
                }
              }
              v106 = v9 - 24;
              sub_2A3E6C0(v106, v165, (unsigned int)v166);
              sub_B99FD0(v106, 2u, v149);
              v66 = v165;
              if ( v165 == v167 )
                goto LABEL_186;
              goto LABEL_185;
            }
          }
        }
      }
    }
LABEL_12:
    v10 = v7 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v8 != (v7 & 0xFFFFFFFFFFFFFFF8LL) )
      goto LABEL_13;
LABEL_43:
    v159 = *(_QWORD *)(v159 + 8);
  }
  while ( v158 != v159 );
  v35 = a1 + 32;
  v36 = a1 + 80;
  if ( !v156 )
  {
LABEL_190:
    *(_QWORD *)(a1 + 8) = v35;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v36;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)a1 = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    return a1;
  }
  memset((void *)a1, 0, 0x60u);
  *(_QWORD *)(a1 + 8) = v35;
  *(_DWORD *)(a1 + 16) = 2;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 56) = v36;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  return a1;
}
