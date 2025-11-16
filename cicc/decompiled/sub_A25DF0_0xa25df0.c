// Function: sub_A25DF0
// Address: 0xa25df0
//
__int64 __fastcall sub_A25DF0(__int64 *a1, unsigned int a2, unsigned int a3, char a4)
{
  __int64 v6; // rsi
  unsigned int v8; // ebx
  __int64 v9; // r12
  unsigned __int8 *v10; // r13
  __int64 v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rax
  unsigned int v14; // eax
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // r9
  unsigned __int64 v18; // rdx
  _BYTE *v19; // rcx
  __int64 i; // rax
  __int64 v21; // r14
  __int64 v22; // r13
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  _BYTE *v25; // rcx
  __int64 j; // rax
  __int64 result; // rax
  __int64 v28; // r8
  unsigned __int8 v29; // al
  __int64 v30; // rdx
  unsigned int v31; // ecx
  unsigned int v32; // r9d
  unsigned int v33; // eax
  __int64 v34; // rsi
  __int64 v35; // rcx
  int v36; // eax
  __int64 v37; // rdx
  __int64 v38; // rax
  unsigned __int8 *v39; // rsi
  _QWORD *v40; // rsi
  __int64 v41; // rsi
  unsigned int v42; // eax
  _QWORD *v43; // rax
  __int64 v44; // rdi
  volatile signed __int32 *v45; // rax
  _QWORD *v46; // rax
  __int64 v47; // rdi
  volatile signed __int32 *v48; // rax
  _QWORD *v49; // rax
  __int64 v50; // rdi
  volatile signed __int32 *v51; // rax
  _QWORD *v52; // rax
  __int64 v53; // rdi
  volatile signed __int32 *v54; // rax
  __int64 v55; // rcx
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  unsigned int m; // ebx
  __int64 v60; // rsi
  __int64 v61; // rdx
  int v62; // eax
  __int64 v63; // rax
  unsigned int v64; // eax
  __int64 v65; // rax
  int v66; // r9d
  __int64 v67; // rax
  __int64 *v68; // rdx
  __int64 *v69; // r13
  __int64 *v70; // r14
  unsigned int v71; // ebx
  __int64 v72; // r8
  unsigned int v73; // eax
  unsigned int v74; // eax
  unsigned int v75; // r14d
  unsigned int v76; // eax
  unsigned __int8 *v77; // rbx
  unsigned int v78; // r13d
  __int64 v79; // rsi
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // r14
  __int64 v84; // r13
  unsigned int v85; // eax
  __int64 v86; // rbx
  unsigned int v87; // r13d
  unsigned int v88; // eax
  __int64 v89; // rsi
  unsigned int v90; // eax
  int v91; // r9d
  unsigned int v92; // eax
  __int64 v93; // rsi
  unsigned int v94; // eax
  unsigned int v95; // eax
  unsigned int v96; // eax
  unsigned int v97; // eax
  int v98; // r14d
  char v99; // dl
  bool v100; // r8
  char v101; // r12
  unsigned __int8 *v102; // rbx
  unsigned int k; // r13d
  char v104; // dl
  _QWORD *v105; // rsi
  __int64 *v106; // r13
  __int64 v107; // rsi
  unsigned int v108; // eax
  unsigned int v109; // eax
  _QWORD *v110; // rsi
  __int128 *v111; // r13
  unsigned int v112; // eax
  __int64 v113; // rax
  __int64 v114; // rax
  unsigned int v115; // eax
  unsigned int v116; // eax
  __int64 v117; // rax
  unsigned int v118; // eax
  unsigned int v119; // eax
  __int64 v120; // rax
  unsigned int v121; // eax
  unsigned int v122; // eax
  __int64 v123; // rsi
  unsigned int v124; // eax
  unsigned int v125; // eax
  __int64 v126; // rax
  unsigned int v127; // eax
  unsigned int v128; // eax
  __int64 v129; // rax
  unsigned int v130; // eax
  unsigned int v131; // eax
  unsigned int v132; // eax
  unsigned int v133; // eax
  unsigned int v134; // eax
  unsigned int v135; // eax
  unsigned int v136; // eax
  unsigned int v137; // eax
  unsigned int v138; // [rsp+Ch] [rbp-2D4h]
  unsigned int v139; // [rsp+10h] [rbp-2D0h]
  __int64 v140; // [rsp+18h] [rbp-2C8h]
  __int64 v141; // [rsp+18h] [rbp-2C8h]
  int v142; // [rsp+20h] [rbp-2C0h]
  bool v143; // [rsp+20h] [rbp-2C0h]
  unsigned int v144; // [rsp+28h] [rbp-2B8h]
  unsigned int v145; // [rsp+28h] [rbp-2B8h]
  unsigned int v146; // [rsp+2Ch] [rbp-2B4h]
  unsigned int v147; // [rsp+30h] [rbp-2B0h]
  unsigned int v148; // [rsp+34h] [rbp-2ACh]
  __int64 v149; // [rsp+38h] [rbp-2A8h]
  unsigned int v150; // [rsp+38h] [rbp-2A8h]
  unsigned int v151; // [rsp+38h] [rbp-2A8h]
  unsigned int v152; // [rsp+38h] [rbp-2A8h]
  int v153; // [rsp+38h] [rbp-2A8h]
  unsigned int v154; // [rsp+38h] [rbp-2A8h]
  unsigned __int8 v155; // [rsp+38h] [rbp-2A8h]
  unsigned int v156; // [rsp+48h] [rbp-298h]
  __int64 v158; // [rsp+50h] [rbp-290h]
  __int64 *v159; // [rsp+60h] [rbp-280h] BYREF
  unsigned int v160; // [rsp+68h] [rbp-278h]
  _QWORD *v161; // [rsp+70h] [rbp-270h] BYREF
  volatile signed __int32 *v162; // [rsp+78h] [rbp-268h]
  __int64 v163; // [rsp+80h] [rbp-260h]
  unsigned int v164; // [rsp+88h] [rbp-258h]
  char v165; // [rsp+90h] [rbp-250h]
  _BYTE *v166; // [rsp+A0h] [rbp-240h] BYREF
  __int64 v167; // [rsp+A8h] [rbp-238h]
  _BYTE v168[560]; // [rsp+B0h] [rbp-230h] BYREF

  v6 = 11;
  sub_A19830(*a1, 0xBu, 4u);
  if ( a4 )
  {
    sub_A23770(&v161);
    sub_A186C0((__int64)v161, 7, 1);
    sub_A186C0((__int64)v161, 0, 6);
    v41 = 0;
    if ( a3 )
    {
      _BitScanReverse(&v42, a3);
      v41 = (int)(32 - (v42 ^ 0x1F));
    }
    sub_A186C0((__int64)v161, v41, 2);
    v43 = v161;
    v44 = *a1;
    v161 = 0;
    v166 = v43;
    v45 = v162;
    v162 = 0;
    v167 = (__int64)v45;
    v147 = sub_A1AB30(v44, (__int64 *)&v166);
    if ( v167 )
      sub_A191D0((volatile signed __int32 *)v167);
    sub_A23770(&v166);
    sub_A19260(&v161, (__int64 *)&v166);
    if ( v167 )
      sub_A191D0((volatile signed __int32 *)v167);
    sub_A186C0((__int64)v161, 8, 1);
    sub_A186C0((__int64)v161, 0, 6);
    sub_A186C0((__int64)v161, 8, 2);
    v46 = v161;
    v47 = *a1;
    v161 = 0;
    v166 = v46;
    v48 = v162;
    v162 = 0;
    v167 = (__int64)v48;
    v156 = sub_A1AB30(v47, (__int64 *)&v166);
    if ( v167 )
      sub_A191D0((volatile signed __int32 *)v167);
    sub_A23770(&v166);
    sub_A19260(&v161, (__int64 *)&v166);
    if ( v167 )
      sub_A191D0((volatile signed __int32 *)v167);
    sub_A186C0((__int64)v161, 9, 1);
    sub_A186C0((__int64)v161, 0, 6);
    sub_A186C0((__int64)v161, 7, 2);
    v49 = v161;
    v50 = *a1;
    v161 = 0;
    v166 = v49;
    v51 = v162;
    v162 = 0;
    v167 = (__int64)v51;
    v146 = sub_A1AB30(v50, (__int64 *)&v166);
    if ( v167 )
      sub_A191D0((volatile signed __int32 *)v167);
    sub_A23770(&v166);
    sub_A19260(&v161, (__int64 *)&v166);
    if ( v167 )
      sub_A191D0((volatile signed __int32 *)v167);
    sub_A186C0((__int64)v161, 9, 1);
    sub_A186C0((__int64)v161, 0, 6);
    sub_A186C0((__int64)v161, 0, 8);
    v52 = v161;
    v53 = *a1;
    v6 = (__int64)&v166;
    v161 = 0;
    v166 = v52;
    v54 = v162;
    v162 = 0;
    v167 = (__int64)v54;
    v148 = sub_A1AB30(v53, (__int64 *)&v166);
    if ( v167 )
      sub_A191D0((volatile signed __int32 *)v167);
    if ( v162 )
      sub_A191D0(v162);
  }
  else
  {
    v148 = 0;
    v146 = 0;
    v156 = 0;
    v147 = 0;
  }
  v8 = a2;
  v166 = v168;
  v167 = 0x4000000000LL;
  v158 = (__int64)(a1 + 3);
  if ( a3 != a2 )
  {
    v9 = 0;
    do
    {
      v10 = *(unsigned __int8 **)(a1[17] + 16LL * v8);
      v11 = v9;
      v9 = *((_QWORD *)v10 + 1);
      if ( v9 != v11 )
      {
        v12 = sub_A172F0(v158, *((_QWORD *)v10 + 1));
        sub_A188E0((__int64)&v166, v12);
        v6 = 4;
        sub_A1B020(*a1, 4u, (__int64)v166, (unsigned int)v167, 0, 0, 1u, 1);
        LODWORD(v167) = 0;
      }
      if ( *v10 == 25 )
      {
        v13 = sub_B3B7D0(v10, v6);
        v14 = sub_A172F0(v158, v13);
        sub_A188E0((__int64)&v166, v14);
        sub_A188E0(
          (__int64)&v166,
          (8 * v10[104]) | v10[96] | (2 * v10[97]) | (4 * (unsigned __int8)*((_DWORD *)v10 + 25)) & 4u);
        sub_A188E0((__int64)&v166, *((_QWORD *)v10 + 4));
        v15 = *((_QWORD *)v10 + 4);
        v16 = (unsigned int)v167;
        v17 = *((_QWORD *)v10 + 3);
        v18 = v15 + (unsigned int)v167;
        if ( v18 > HIDWORD(v167) )
        {
          v149 = *((_QWORD *)v10 + 3);
          sub_C8D5F0(&v166, v168, v18, 8);
          v16 = (unsigned int)v167;
          v17 = v149;
        }
        v19 = &v166[8 * v16];
        if ( v15 > 0 )
        {
          for ( i = 0; i != v15; ++i )
            *(_QWORD *)&v19[8 * i] = *(char *)(v17 + i);
          LODWORD(v16) = v167;
        }
        LODWORD(v167) = v16 + v15;
        sub_A188E0((__int64)&v166, *((_QWORD *)v10 + 8));
        v21 = *((_QWORD *)v10 + 7);
        v22 = *((_QWORD *)v10 + 8);
        v23 = (unsigned int)v167;
        v24 = v22 + (unsigned int)v167;
        if ( v24 > HIDWORD(v167) )
        {
          sub_C8D5F0(&v166, v168, v24, 8);
          v23 = (unsigned int)v167;
        }
        v25 = &v166[8 * v23];
        if ( v22 > 0 )
        {
          for ( j = 0; j != v22; ++j )
            *(_QWORD *)&v25[8 * j] = *(char *)(v21 + j);
          LODWORD(v23) = v167;
        }
        v6 = 30;
        LODWORD(v167) = v22 + v23;
        sub_A1FB70(*a1, 0x1Eu, (__int64)&v166, 0);
        LODWORD(v167) = 0;
        goto LABEL_21;
      }
      if ( (unsigned __int8)sub_AC30F0(v10) )
      {
        v31 = 0;
        v32 = 2;
        goto LABEL_29;
      }
      v29 = *v10;
      if ( *v10 == 13 )
      {
        v31 = 0;
        v32 = 26;
        goto LABEL_29;
      }
      LODWORD(v30) = v29;
      if ( (unsigned int)v29 - 12 <= 1 )
      {
        v31 = 0;
        v32 = 3;
        goto LABEL_29;
      }
      if ( v29 == 17 )
      {
        v33 = *((_DWORD *)v10 + 8);
        if ( v33 > 0x40 )
        {
          sub_A16F20((__int64 *)&v166, (__int64)(v10 + 24));
          v31 = 0;
          v32 = 5;
        }
        else
        {
          v34 = 0;
          if ( v33 )
            v34 = (__int64)(*((_QWORD *)v10 + 3) << (64 - (unsigned __int8)v33)) >> (64 - (unsigned __int8)v33);
          sub_A170D0((__int64)&v166, v34);
          v31 = 5;
          v32 = 4;
        }
        goto LABEL_29;
      }
      if ( v29 == 18 )
      {
        v35 = *((_QWORD *)v10 + 1);
        v36 = *(unsigned __int8 *)(v35 + 8);
        v37 = (unsigned int)(v36 - 17);
        if ( (unsigned int)v37 <= 1 )
          LOBYTE(v36) = *(_BYTE *)(**(_QWORD **)(v35 + 16) + 8LL);
        if ( (unsigned __int8)v36 > 3u )
        {
          if ( (_BYTE)v36 == 4 )
          {
            v110 = v10 + 24;
            v111 = (__int128 *)&v161;
            sub_9875A0((__int64)&v161, v110, v37, v35, v28);
            if ( (unsigned int)v162 > 0x40 )
              v111 = (__int128 *)v161;
            sub_A188E0((__int64)&v166, *v111 >> 16);
            v107 = *(unsigned __int16 *)v111;
          }
          else
          {
            if ( (unsigned __int8)(v36 - 5) > 1u )
            {
              v31 = 0;
              v32 = 6;
              goto LABEL_29;
            }
            v105 = v10 + 24;
            v106 = (__int64 *)&v161;
            sub_9875A0((__int64)&v161, v105, v37, v35, v28);
            if ( (unsigned int)v162 > 0x40 )
              v106 = v161;
            sub_A188E0((__int64)&v166, *v106);
            v107 = v106[1];
          }
          sub_A188E0((__int64)&v166, v107);
          sub_969240((__int64 *)&v161);
        }
        else
        {
          v38 = sub_C33340(v10, v6, v37, v35, v28);
          v39 = v10 + 24;
          if ( *((_QWORD *)v10 + 3) == v38 )
            sub_C3E660(&v161, v39);
          else
            sub_C3A850(&v161, v39);
          v40 = v161;
          if ( (unsigned int)v162 > 0x40 )
            v40 = (_QWORD *)*v161;
          sub_A188E0((__int64)&v166, (__int64)v40);
          if ( (unsigned int)v162 > 0x40 && v161 )
            j_j___libc_free_0_0(v161);
        }
        v31 = 0;
        v32 = 6;
      }
      else
      {
        if ( (unsigned int)v29 - 15 <= 1 )
        {
          v6 = 8;
          if ( (unsigned __int8)sub_AC5570(v10, 8) )
          {
            v98 = sub_AC5290(v10);
            v99 = sub_AC55A0(v10);
            if ( v99 )
            {
              if ( --v98 )
              {
                v31 = 0;
                v32 = 9;
LABEL_122:
                v139 = v32;
                v100 = v99;
                v145 = v31;
                v141 = v9;
                v101 = v99;
                v138 = v8;
                v102 = v10;
                for ( k = 0; k != v98; ++k )
                {
                  v143 = v100;
                  v155 = sub_AC5320(v102, k);
                  sub_A188E0((__int64)&v166, v155);
                  v100 = v143;
                  v101 &= (unsigned __int8)~v155 >> 7;
                  if ( v143 && (unsigned __int8)((v155 & 0xDF) - 65) > 0x19u && (unsigned __int8)(v155 - 48) > 9u )
                    v100 = v155 == 95 || v155 == 46;
                }
                v104 = v101;
                v32 = v139;
                v9 = v141;
                v31 = v145;
                v8 = v138;
                if ( v100 )
                {
                  v31 = v148;
                }
                else if ( v104 )
                {
                  v31 = v146;
                }
                goto LABEL_29;
              }
              v31 = v148;
              v32 = 9;
            }
            else
            {
              v31 = v156;
              v32 = 8;
              if ( v98 )
                goto LABEL_122;
            }
            goto LABEL_29;
          }
          v30 = *v10;
          v55 = (unsigned int)(v30 - 15);
          v29 = *v10;
          if ( (unsigned int)v55 <= 1 )
          {
            if ( *(_BYTE *)(sub_AC5230(v10, 8, v30, v55) + 8) == 12 )
            {
              v75 = 0;
              v153 = sub_AC5290(v10);
              if ( v153 )
              {
                v76 = v8;
                v77 = v10;
                v78 = v76;
                do
                {
                  v79 = v75++;
                  v80 = sub_AC5320(v77, v79);
                  sub_A188E0((__int64)&v166, v80);
                }
                while ( v153 != v75 );
                v8 = v78;
              }
            }
            else
            {
              v142 = sub_AC5290(v10);
              if ( v142 )
              {
                v140 = sub_C33340(v10, 8, v56, v57, v58);
                v144 = v8;
                for ( m = 0; m != v142; ++m )
                {
                  sub_AC5470(&v161, v10, m);
                  if ( (_QWORD *)v140 == v161 )
                    sub_C3E660(&v159, &v161);
                  else
                    sub_C3A850(&v159, &v161);
                  v150 = v160;
                  if ( v160 <= 0x40 )
                  {
                    v60 = (__int64)v159;
                  }
                  else
                  {
                    v60 = -1;
                    if ( v150 - (unsigned int)sub_C444A0(&v159) <= 0x40 )
                      v60 = *v159;
                  }
                  sub_A188E0((__int64)&v166, v60);
                  if ( v160 > 0x40 && v159 )
                    j_j___libc_free_0_0(v159);
                  sub_91D830(&v161);
                }
                v8 = v144;
              }
            }
            v31 = 0;
            v32 = 22;
            goto LABEL_29;
          }
        }
        v61 = (unsigned int)(v30 - 9);
        if ( (unsigned int)v61 <= 2 )
        {
          v81 = sub_986550((__int64)v10);
          v83 = v82;
          v84 = v81;
          if ( v81 != v82 )
          {
            v85 = v8;
            v86 = v84;
            v87 = v85;
            do
            {
              v86 += 32;
              v88 = sub_A3F3B0(v158);
              sub_A188E0((__int64)&v166, v88);
            }
            while ( v83 != v86 );
            v8 = v87;
          }
          v31 = v147;
          v32 = 7;
          goto LABEL_29;
        }
        if ( v29 != 5 )
        {
          switch ( v29 )
          {
            case 4u:
              v95 = sub_A172F0(v158, *(_QWORD *)(*((_QWORD *)v10 - 8) + 8LL));
              sub_A188E0((__int64)&v166, v95);
              v96 = sub_A3F3B0(v158);
              sub_A188E0((__int64)&v166, v96);
              v97 = sub_A4A530(v158, *((_QWORD *)v10 - 4));
              sub_A188E0((__int64)&v166, v97);
              v31 = 0;
              v32 = 21;
              break;
            case 6u:
              v108 = sub_A172F0(v158, *(_QWORD *)(*((_QWORD *)v10 - 4) + 8LL));
              sub_A188E0((__int64)&v166, v108);
              v109 = sub_A3F3B0(v158);
              sub_A188E0((__int64)&v166, v109);
              v31 = 0;
              v32 = 27;
              break;
            case 7u:
              v132 = sub_A172F0(v158, *(_QWORD *)(*((_QWORD *)v10 - 4) + 8LL));
              sub_A188E0((__int64)&v166, v132);
              v133 = sub_A3F3B0(v158);
              sub_A188E0((__int64)&v166, v133);
              v31 = 0;
              v32 = 29;
              break;
            case 8u:
              v134 = sub_A3F3B0(v158);
              sub_A188E0((__int64)&v166, v134);
              v135 = sub_A3F3B0(v158);
              sub_A188E0((__int64)&v166, v135);
              v136 = sub_A3F3B0(v158);
              sub_A188E0((__int64)&v166, v136);
              v137 = sub_A3F3B0(v158);
              sub_A188E0((__int64)&v166, v137);
              v31 = 0;
              v32 = 33;
              break;
            default:
              BUG();
          }
          goto LABEL_29;
        }
        v62 = *((unsigned __int16 *)v10 + 1);
        if ( (_WORD)v62 == 61 )
        {
          v114 = sub_986520((__int64)v10);
          v115 = sub_A172F0(v158, *(_QWORD *)(*(_QWORD *)v114 + 8LL));
          sub_A188E0((__int64)&v166, v115);
          sub_986520((__int64)v10);
          v116 = sub_A3F3B0(v158);
          sub_A188E0((__int64)&v166, v116);
          v117 = sub_986520((__int64)v10);
          v118 = sub_A172F0(v158, *(_QWORD *)(*(_QWORD *)(v117 + 32) + 8LL));
          sub_A188E0((__int64)&v166, v118);
          sub_986520((__int64)v10);
          v119 = sub_A3F3B0(v158);
          sub_A188E0((__int64)&v166, v119);
          v31 = 0;
          v32 = 14;
        }
        else if ( (unsigned __int16)v62 > 0x3Du )
        {
          if ( (_WORD)v62 == 62 )
          {
            sub_986520((__int64)v10);
            v124 = sub_A3F3B0(v158);
            sub_A188E0((__int64)&v166, v124);
            sub_986520((__int64)v10);
            v125 = sub_A3F3B0(v158);
            sub_A188E0((__int64)&v166, v125);
            v126 = sub_986520((__int64)v10);
            v127 = sub_A172F0(v158, *(_QWORD *)(*(_QWORD *)(v126 + 64) + 8LL));
            sub_A188E0((__int64)&v166, v127);
            sub_986520((__int64)v10);
            v128 = sub_A3F3B0(v158);
            sub_A188E0((__int64)&v166, v128);
            v31 = 0;
            v32 = 15;
          }
          else
          {
            if ( (_WORD)v62 != 63 )
              goto LABEL_146;
            v89 = *(_QWORD *)(*(_QWORD *)sub_986520((__int64)v10) + 8LL);
            if ( *((_QWORD *)v10 + 1) == v89 )
            {
              v91 = 16;
            }
            else
            {
              v90 = sub_A172F0(v158, v89);
              sub_A188E0((__int64)&v166, v90);
              sub_986520((__int64)v10);
              v91 = 19;
            }
            v154 = v91;
            v92 = sub_A3F3B0(v158);
            sub_A188E0((__int64)&v166, v92);
            sub_986520((__int64)v10);
            v93 = (unsigned int)sub_A3F3B0(v158);
            sub_A188E0((__int64)&v166, v93);
            sub_AC3600(v10);
            v94 = sub_A3F3B0(v158);
            sub_A188E0((__int64)&v166, v94);
            v32 = v154;
            v31 = 0;
          }
        }
        else
        {
          if ( (_WORD)v62 == 12 )
          {
            sub_A188E0((__int64)&v166, 0);
            sub_986520((__int64)v10);
            v112 = sub_A3F3B0(v158);
            sub_A188E0((__int64)&v166, v112);
            v113 = sub_A15DF0(v10);
            v31 = 0;
            v32 = 25;
            if ( v113 )
            {
              sub_A188E0((__int64)&v166, v113);
              v32 = 25;
              v31 = 0;
            }
            goto LABEL_29;
          }
          if ( (_WORD)v62 == 34 )
          {
            v63 = sub_BB5290(v10, v6, v61);
            v64 = sub_A172F0(v158, v63);
            sub_A188E0((__int64)&v166, v64);
            v65 = sub_A15DF0(v10);
            sub_A188E0((__int64)&v166, v65);
            sub_BB52D0(&v161, v10);
            v66 = 32;
            if ( v165 )
            {
              sub_A18930((__int64 *)&v166, (__int64)&v161, 1);
              if ( v165 )
              {
                v165 = 0;
                if ( v164 > 0x40 && v163 )
                  j_j___libc_free_0_0(v163);
                if ( (unsigned int)v162 > 0x40 && v161 )
                  j_j___libc_free_0_0(v161);
              }
              v66 = 31;
            }
            v151 = v66;
            v67 = sub_986550((__int64)v10);
            v32 = v151;
            v69 = v68;
            v70 = (__int64 *)v67;
            if ( (__int64 *)v67 != v68 )
            {
              v152 = v8;
              v71 = v32;
              do
              {
                v72 = *v70;
                v70 += 4;
                v73 = sub_A172F0(v158, *(_QWORD *)(v72 + 8));
                sub_A188E0((__int64)&v166, v73);
                v74 = sub_A3F3B0(v158);
                sub_A188E0((__int64)&v166, v74);
              }
              while ( v69 != v70 );
              v32 = v71;
              v8 = v152;
            }
            v31 = 0;
          }
          else
          {
            if ( (unsigned int)(v62 - 38) > 0xC )
            {
LABEL_146:
              v120 = (unsigned int)(v62 - 13);
              if ( (unsigned int)v120 > 0x11 )
                BUG();
              sub_A188E0((__int64)&v166, (unsigned int)dword_3F22FA0[v120]);
              sub_986520((__int64)v10);
              v121 = sub_A3F3B0(v158);
              sub_A188E0((__int64)&v166, v121);
              sub_986520((__int64)v10);
              v122 = sub_A3F3B0(v158);
              sub_A188E0((__int64)&v166, v122);
              v123 = sub_A15DF0(v10);
              if ( v123 )
                sub_A188E0((__int64)&v166, v123);
              v31 = 0;
              v32 = 10;
              goto LABEL_29;
            }
            sub_A188E0((__int64)&v166, (unsigned int)(v62 - 38));
            v129 = sub_986520((__int64)v10);
            v130 = sub_A172F0(v158, *(_QWORD *)(*(_QWORD *)v129 + 8LL));
            sub_A188E0((__int64)&v166, v130);
            sub_986520((__int64)v10);
            v131 = sub_A3F3B0(v158);
            sub_A188E0((__int64)&v166, v131);
            v31 = 6;
            v32 = 11;
          }
        }
      }
LABEL_29:
      v6 = v32;
      sub_A1FB70(*a1, v32, (__int64)&v166, v31);
      LODWORD(v167) = 0;
LABEL_21:
      ++v8;
    }
    while ( a3 != v8 );
  }
  result = sub_A192A0(*a1);
  if ( v166 != v168 )
    return _libc_free(v166, v6);
  return result;
}
