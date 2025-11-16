// Function: sub_27C2750
// Address: 0x27c2750
//
__int64 __fastcall sub_27C2750(__int64 a1, __int64 a2)
{
  unsigned int v2; // r14d
  __int64 v4; // rcx
  _BYTE *v5; // r15
  _BYTE *v6; // r13
  unsigned __int64 *v7; // r8
  _BYTE *v8; // r9
  unsigned __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  _BYTE *v12; // rbx
  __int64 v13; // r15
  unsigned __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned int v28; // eax
  __int64 *v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v33; // rax
  char v34; // al
  _BYTE *v35; // rdx
  __int64 v36; // rsi
  char v37; // al
  bool v38; // al
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r8
  _BYTE *v44; // r9
  unsigned int v45; // eax
  __int64 *v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  _BYTE *v49; // r9
  __int64 v50; // r8
  unsigned __int8 v51; // al
  _QWORD *v52; // rdx
  unsigned __int64 v53; // rax
  int v54; // edx
  unsigned __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rcx
  __int64 v59; // rcx
  __int64 v60; // rax
  __int64 v61; // rcx
  __int64 v62; // rcx
  __int16 v63; // ax
  unsigned __int64 *v64; // rsi
  __int64 v65; // rdx
  __int64 v66; // r9
  int v67; // eax
  unsigned __int64 *v68; // rdi
  unsigned __int64 v69; // rax
  unsigned __int64 v70; // rcx
  __int64 v71; // r12
  __int64 v72; // r14
  int v73; // edx
  unsigned __int64 v74; // rdi
  unsigned __int64 *v75; // rax
  int v76; // eax
  unsigned __int64 v77; // rdi
  __int64 v78; // [rsp+8h] [rbp-198h]
  unsigned int v79; // [rsp+18h] [rbp-188h]
  __int64 *v80; // [rsp+18h] [rbp-188h]
  unsigned int v81; // [rsp+20h] [rbp-180h]
  unsigned __int8 v82; // [rsp+20h] [rbp-180h]
  unsigned __int8 v83; // [rsp+2Fh] [rbp-171h]
  char v84; // [rsp+30h] [rbp-170h]
  __int64 v85; // [rsp+30h] [rbp-170h]
  _BYTE *v86; // [rsp+30h] [rbp-170h]
  __int64 v87; // [rsp+30h] [rbp-170h]
  int v88; // [rsp+30h] [rbp-170h]
  __int64 v89; // [rsp+38h] [rbp-168h]
  __int64 v90; // [rsp+38h] [rbp-168h]
  _BYTE *v91; // [rsp+38h] [rbp-168h]
  unsigned __int64 *v92; // [rsp+38h] [rbp-168h]
  unsigned __int64 *v93; // [rsp+38h] [rbp-168h]
  __int64 v94; // [rsp+40h] [rbp-160h]
  unsigned __int64 *v95; // [rsp+40h] [rbp-160h]
  _BYTE *v96; // [rsp+40h] [rbp-160h]
  __int64 v97; // [rsp+40h] [rbp-160h]
  __int64 v98; // [rsp+40h] [rbp-160h]
  __int64 v99; // [rsp+40h] [rbp-160h]
  _BYTE *v100; // [rsp+40h] [rbp-160h]
  _BYTE *v101; // [rsp+40h] [rbp-160h]
  __int64 v102; // [rsp+40h] [rbp-160h]
  __int64 v103; // [rsp+40h] [rbp-160h]
  unsigned __int64 *v104; // [rsp+40h] [rbp-160h]
  unsigned __int64 *v105; // [rsp+40h] [rbp-160h]
  unsigned __int64 *v106; // [rsp+40h] [rbp-160h]
  unsigned __int64 *v107; // [rsp+40h] [rbp-160h]
  char *v108; // [rsp+40h] [rbp-160h]
  unsigned __int64 *v109; // [rsp+40h] [rbp-160h]
  int v110; // [rsp+40h] [rbp-160h]
  _BYTE *v111; // [rsp+48h] [rbp-158h]
  unsigned __int64 *v112; // [rsp+48h] [rbp-158h]
  unsigned __int64 *v113; // [rsp+48h] [rbp-158h]
  unsigned int v114; // [rsp+48h] [rbp-158h]
  _BYTE *v115; // [rsp+48h] [rbp-158h]
  __int64 *v116; // [rsp+48h] [rbp-158h]
  unsigned __int64 *v117; // [rsp+48h] [rbp-158h]
  unsigned __int64 *v118; // [rsp+48h] [rbp-158h]
  unsigned __int64 *v119; // [rsp+48h] [rbp-158h]
  unsigned __int64 *v120; // [rsp+48h] [rbp-158h]
  _BYTE *v121; // [rsp+48h] [rbp-158h]
  _BYTE *v122; // [rsp+48h] [rbp-158h]
  _BYTE *v123; // [rsp+48h] [rbp-158h]
  _BYTE *v124; // [rsp+48h] [rbp-158h]
  _BYTE *v125; // [rsp+48h] [rbp-158h]
  __int64 v127; // [rsp+58h] [rbp-148h]
  _BYTE *v128; // [rsp+58h] [rbp-148h]
  _BYTE *v129; // [rsp+58h] [rbp-148h]
  unsigned int v130; // [rsp+58h] [rbp-148h]
  unsigned __int8 v131; // [rsp+58h] [rbp-148h]
  _BYTE *v132; // [rsp+58h] [rbp-148h]
  _BYTE *v133; // [rsp+58h] [rbp-148h]
  _BYTE *v134; // [rsp+58h] [rbp-148h]
  unsigned __int64 v135; // [rsp+68h] [rbp-138h] BYREF
  unsigned __int64 v136; // [rsp+70h] [rbp-130h] BYREF
  unsigned int v137; // [rsp+78h] [rbp-128h]
  unsigned __int64 v138; // [rsp+80h] [rbp-120h]
  unsigned int v139; // [rsp+88h] [rbp-118h]
  unsigned __int64 v140; // [rsp+90h] [rbp-110h] BYREF
  unsigned int v141; // [rsp+98h] [rbp-108h]
  unsigned __int64 v142; // [rsp+A0h] [rbp-100h] BYREF
  unsigned int v143; // [rsp+A8h] [rbp-F8h]
  unsigned __int64 v144; // [rsp+B0h] [rbp-F0h] BYREF
  __int64 v145; // [rsp+B8h] [rbp-E8h]
  unsigned __int64 v146; // [rsp+C0h] [rbp-E0h] BYREF
  unsigned int v147; // [rsp+C8h] [rbp-D8h]
  __int16 v148; // [rsp+D0h] [rbp-D0h]
  _BYTE *v149; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v150; // [rsp+E8h] [rbp-B8h]
  _BYTE v151[176]; // [rsp+F0h] [rbp-B0h] BYREF

  v2 = 0;
  v150 = 0x1000000000LL;
  v149 = v151;
  sub_D46D90(a2, (__int64)&v149);
  v5 = v149;
  v6 = &v149[8 * (unsigned int)v150];
  if ( v149 == v6 )
    goto LABEL_53;
  v7 = &v144;
  v8 = &v149[8 * (unsigned int)v150];
  do
  {
    v9 = *(_QWORD *)(*(_QWORD *)v5 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v9 == *(_QWORD *)v5 + 48LL )
      goto LABEL_166;
    if ( !v9 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 > 0xA )
LABEL_166:
      BUG();
    if ( *(_BYTE *)(v9 - 24) == 31 )
    {
      v10 = *(_QWORD *)(v9 - 120);
      if ( *(_BYTE *)v10 == 82 )
      {
        v33 = *(_QWORD *)(v10 + 16);
        if ( v33 )
        {
          if ( !*(_QWORD *)(v33 + 8) )
          {
            v95 = v7;
            v111 = v8;
            v127 = *(_QWORD *)(v10 - 64);
            v90 = *(_QWORD *)(v10 - 32);
            v34 = sub_D48480(a2, v90, v127, v4);
            v35 = (_BYTE *)v127;
            v8 = v111;
            v7 = v95;
            if ( !v34 )
            {
              v36 = v127;
              v112 = v95;
              v128 = v8;
              v96 = v35;
              v37 = sub_D48480(a2, v36, (__int64)v35, v4);
              v8 = v128;
              v7 = v112;
              if ( !v37 )
                goto LABEL_8;
              v35 = (_BYTE *)v90;
              v90 = (__int64)v96;
            }
            if ( *v35 == 68 )
            {
              v97 = *((_QWORD *)v35 - 4);
              if ( v97 )
              {
                v113 = v7;
                v129 = v8;
                v38 = sub_B532B0(*(_WORD *)(v10 + 2) & 0x3F);
                v8 = v129;
                v7 = v113;
                if ( v38 )
                {
                  v86 = v129;
                  v39 = sub_9208B0(*(_QWORD *)(a1 + 24), *(_QWORD *)(v97 + 8));
                  v145 = v40;
                  v144 = v39;
                  v130 = sub_CA1930(v113);
                  v41 = sub_9208B0(*(_QWORD *)(a1 + 24), *(_QWORD *)(v90 + 8));
                  v145 = v42;
                  v98 = (__int64)v113;
                  v144 = v41;
                  v114 = sub_CA1930(v113);
                  sub_AADB10((__int64)&v140, v130, 1);
                  sub_AB3F90(v98, (__int64)&v140, v114);
                  v43 = v98;
                  v44 = v86;
                  if ( v141 > 0x40 && v140 )
                  {
                    j_j___libc_free_0_0(v140);
                    v43 = v98;
                    v44 = v86;
                  }
                  v140 = v144;
                  v45 = v145;
                  LODWORD(v145) = 0;
                  v141 = v45;
                  if ( v143 > 0x40 && v142 )
                  {
                    v99 = v43;
                    v115 = v44;
                    j_j___libc_free_0_0(v142);
                    v44 = v115;
                    v43 = v99;
                    v142 = v146;
                    v143 = v147;
                    if ( (unsigned int)v145 > 0x40 && v144 )
                    {
                      j_j___libc_free_0_0(v144);
                      v43 = v99;
                      v44 = v115;
                    }
                  }
                  else
                  {
                    v142 = v146;
                    v143 = v147;
                  }
                  v87 = v43;
                  v100 = v44;
                  v116 = *(__int64 **)(a1 + 8);
                  v46 = sub_DD8400((__int64)v116, v90);
                  v47 = sub_DE4F70(v116, (__int64)v46, a2);
                  v48 = sub_DBB9F0((__int64)v116, v47, 0, 0);
                  v49 = v100;
                  v50 = v87;
                  LODWORD(v145) = *(_DWORD *)(v48 + 8);
                  if ( (unsigned int)v145 > 0x40 )
                  {
                    v91 = v100;
                    v103 = v48;
                    sub_C43780(v87, (const void **)v48);
                    v49 = v91;
                    v48 = v103;
                    v50 = v87;
                  }
                  else
                  {
                    v144 = *(_QWORD *)v48;
                  }
                  v147 = *(_DWORD *)(v48 + 24);
                  if ( v147 > 0x40 )
                  {
                    v102 = v50;
                    v121 = v49;
                    sub_C43780((__int64)&v146, (const void **)(v48 + 16));
                    v50 = v102;
                    v49 = v121;
                  }
                  else
                  {
                    v146 = *(_QWORD *)(v48 + 16);
                  }
                  v101 = v49;
                  v117 = (unsigned __int64 *)v50;
                  v51 = sub_AB1BB0((__int64)&v140, v50);
                  v7 = v117;
                  v8 = v101;
                  v131 = v51;
                  if ( v51 )
                  {
                    v104 = v117;
                    v122 = v8;
                    v63 = sub_B52EF0(*(_WORD *)(v10 + 2) & 0x3F);
                    v8 = v122;
                    v7 = v104;
                    *(_WORD *)(v10 + 2) = v63 | *(_WORD *)(v10 + 2) & 0xFFC0;
                    if ( v147 > 0x40 && v146 )
                    {
                      j_j___libc_free_0_0(v146);
                      v7 = v104;
                      v8 = v122;
                    }
                    if ( (unsigned int)v145 > 0x40 && v144 )
                    {
                      v105 = v7;
                      v123 = v8;
                      j_j___libc_free_0_0(v144);
                      v7 = v105;
                      v8 = v123;
                    }
                    if ( v143 > 0x40 && v142 )
                    {
                      v106 = v7;
                      v124 = v8;
                      j_j___libc_free_0_0(v142);
                      v7 = v106;
                      v8 = v124;
                    }
                    if ( v141 > 0x40 && v140 )
                    {
                      v107 = v7;
                      v125 = v8;
                      j_j___libc_free_0_0(v140);
                      v7 = v107;
                      v8 = v125;
                    }
                    v2 = v131;
                  }
                  else
                  {
                    if ( v147 > 0x40 && v146 )
                    {
                      j_j___libc_free_0_0(v146);
                      v7 = v117;
                      v8 = v101;
                    }
                    if ( (unsigned int)v145 > 0x40 && v144 )
                    {
                      v118 = v7;
                      v132 = v8;
                      j_j___libc_free_0_0(v144);
                      v7 = v118;
                      v8 = v132;
                    }
                    if ( v143 > 0x40 && v142 )
                    {
                      v119 = v7;
                      v133 = v8;
                      j_j___libc_free_0_0(v142);
                      v7 = v119;
                      v8 = v133;
                    }
                    if ( v141 > 0x40 && v140 )
                    {
                      v120 = v7;
                      v134 = v8;
                      j_j___libc_free_0_0(v140);
                      v7 = v120;
                      v8 = v134;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
LABEL_8:
    v5 += 8;
  }
  while ( v8 != v5 );
  v6 = v149;
  if ( &v149[8 * (unsigned int)v150] != v149 )
  {
    v11 = a2;
    v12 = &v149[8 * (unsigned int)v150];
    v13 = v11;
    do
    {
      v14 = *(_QWORD *)(*(_QWORD *)v6 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v14 == *(_QWORD *)v6 + 48LL )
        goto LABEL_164;
      if ( !v14 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v14 - 24) - 30 > 0xA )
LABEL_164:
        BUG();
      if ( *(_BYTE *)(v14 - 24) == 31 )
      {
        v15 = *(_QWORD *)(v14 - 120);
        if ( *(_BYTE *)v15 == 82 )
        {
          v16 = *(_QWORD *)(v15 + 16);
          if ( v16 )
          {
            if ( !*(_QWORD *)(v16 + 8) && sub_B532A0(*(_WORD *)(v15 + 2) & 0x3F) )
            {
              v94 = *(_QWORD *)(v15 - 32);
              v89 = *(_QWORD *)(v15 - 64);
              v84 = sub_D48480(v13, v89, v17, v94);
              if ( v84 != (unsigned __int8)sub_D48480(v13, v94, v18, v19) )
              {
                v83 = sub_D48480(v13, v89, v20, v21);
                if ( v83 )
                {
                  v22 = v94;
                  v94 = v89;
                  v89 = v22;
                }
                if ( *(_BYTE *)v89 == 68 )
                {
                  v85 = *(_QWORD *)(v89 - 32);
                  if ( v85 )
                  {
                    if ( (v23 = *(_QWORD *)(v89 + 16)) != 0 && !*(_QWORD *)(v23 + 8)
                      || *((_WORD *)sub_DD8400(*(_QWORD *)(a1 + 8), v85) + 12) == 8 )
                    {
                      v24 = sub_9208B0(*(_QWORD *)(a1 + 24), *(_QWORD *)(v85 + 8));
                      v145 = v25;
                      v144 = v24;
                      v81 = sub_CA1930(&v144);
                      v26 = sub_9208B0(*(_QWORD *)(a1 + 24), *(_QWORD *)(v94 + 8));
                      v145 = v27;
                      v144 = v26;
                      v79 = sub_CA1930(&v144);
                      sub_AADB10((__int64)&v136, v81, 1);
                      sub_AB3F90((__int64)&v144, (__int64)&v136, v79);
                      if ( v137 > 0x40 && v136 )
                        j_j___libc_free_0_0(v136);
                      v136 = v144;
                      v28 = v145;
                      LODWORD(v145) = 0;
                      v137 = v28;
                      if ( v139 > 0x40 && v138 )
                      {
                        j_j___libc_free_0_0(v138);
                        v138 = v146;
                        v139 = v147;
                        if ( (unsigned int)v145 > 0x40 && v144 )
                          j_j___libc_free_0_0(v144);
                      }
                      else
                      {
                        v138 = v146;
                        v139 = v147;
                      }
                      v80 = *(__int64 **)(a1 + 8);
                      v29 = sub_DD8400((__int64)v80, v94);
                      v30 = sub_DE4F70(v80, (__int64)v29, v13);
                      v31 = sub_DBB9F0((__int64)v80, v30, 0, 0);
                      v141 = *(_DWORD *)(v31 + 8);
                      if ( v141 > 0x40 )
                      {
                        v78 = v31;
                        sub_C43780((__int64)&v140, (const void **)v31);
                        v31 = v78;
                      }
                      else
                      {
                        v140 = *(_QWORD *)v31;
                      }
                      v143 = *(_DWORD *)(v31 + 24);
                      if ( v143 > 0x40 )
                        sub_C43780((__int64)&v142, (const void **)(v31 + 16));
                      else
                        v142 = *(_QWORD *)(v31 + 16);
                      v82 = sub_AB1BB0((__int64)&v136, (__int64)&v140);
                      if ( v82 )
                      {
                        v52 = (_QWORD *)(sub_D4B130(v13) + 48);
                        v53 = *v52 & 0xFFFFFFFFFFFFFFF8LL;
                        if ( (_QWORD *)v53 == v52 )
                        {
                          v55 = 0;
                        }
                        else
                        {
                          if ( !v53 )
                            BUG();
                          v54 = *(unsigned __int8 *)(v53 - 24);
                          v55 = v53 - 24;
                          if ( (unsigned int)(v54 - 30) >= 0xB )
                            v55 = 0;
                        }
                        v148 = 257;
                        v56 = sub_B51D30(38, v94, *(_QWORD *)(v85 + 8), (__int64)&v144, v55 + 24, 0);
                        v57 = v15 + 32LL * v83 - 64;
                        if ( *(_QWORD *)v57 )
                        {
                          v58 = *(_QWORD *)(v57 + 8);
                          **(_QWORD **)(v57 + 16) = v58;
                          if ( v58 )
                            *(_QWORD *)(v58 + 16) = *(_QWORD *)(v57 + 16);
                        }
                        *(_QWORD *)v57 = v85;
                        v59 = *(_QWORD *)(v85 + 16);
                        *(_QWORD *)(v57 + 8) = v59;
                        if ( v59 )
                          *(_QWORD *)(v59 + 16) = v57 + 8;
                        *(_QWORD *)(v57 + 16) = v85 + 16;
                        *(_QWORD *)(v85 + 16) = v57;
                        v60 = v15 + 32LL * (v83 ^ 1u) - 64;
                        if ( *(_QWORD *)v60 )
                        {
                          v61 = *(_QWORD *)(v60 + 8);
                          **(_QWORD **)(v60 + 16) = v61;
                          if ( v61 )
                            *(_QWORD *)(v61 + 16) = *(_QWORD *)(v60 + 16);
                        }
                        *(_QWORD *)v60 = v56;
                        if ( v56 )
                        {
                          v62 = *(_QWORD *)(v56 + 16);
                          *(_QWORD *)(v60 + 8) = v62;
                          if ( v62 )
                            *(_QWORD *)(v62 + 16) = v60 + 8;
                          *(_QWORD *)(v60 + 16) = v56 + 16;
                          *(_QWORD *)(v56 + 16) = v60;
                        }
                        *(_BYTE *)(v15 + 1) &= ~2u;
                        if ( !*(_QWORD *)(v89 + 16) )
                        {
                          v144 = 6;
                          v145 = 0;
                          v146 = v89;
                          if ( v89 != -4096 && v89 != -8192 )
                            sub_BD73F0((__int64)&v144);
                          v64 = &v144;
                          v65 = *(unsigned int *)(a1 + 64);
                          v66 = v65 + 1;
                          v67 = *(_DWORD *)(a1 + 64);
                          if ( v65 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 68) )
                          {
                            v70 = *(_QWORD *)(a1 + 56);
                            v71 = a1 + 56;
                            v72 = a1 + 72;
                            if ( v70 > (unsigned __int64)&v144 || (unsigned __int64)&v144 >= v70 + 24 * v65 )
                            {
                              v109 = (unsigned __int64 *)sub_C8D7D0(a1 + 56, a1 + 72, v65 + 1, 0x18u, &v135, v66);
                              sub_F17F80(v71, v109);
                              v76 = v135;
                              v77 = *(_QWORD *)(a1 + 56);
                              if ( v77 == v72 )
                              {
                                v64 = &v144;
                                *(_QWORD *)(a1 + 56) = v109;
                                v65 = *(unsigned int *)(a1 + 64);
                                *(_DWORD *)(a1 + 68) = v76;
                              }
                              else
                              {
                                v93 = v109;
                                v110 = v135;
                                _libc_free(v77);
                                v64 = &v144;
                                *(_QWORD *)(a1 + 56) = v93;
                                v65 = *(unsigned int *)(a1 + 64);
                                *(_DWORD *)(a1 + 68) = v110;
                              }
                              v67 = v65;
                            }
                            else
                            {
                              v108 = (char *)&v144 - v70;
                              v92 = (unsigned __int64 *)sub_C8D7D0(a1 + 56, a1 + 72, v65 + 1, 0x18u, &v135, v66);
                              sub_F17F80(v71, v92);
                              v73 = v135;
                              v74 = *(_QWORD *)(a1 + 56);
                              v75 = v92;
                              if ( v72 == v74 )
                              {
                                *(_QWORD *)(a1 + 56) = v92;
                                *(_DWORD *)(a1 + 68) = v73;
                              }
                              else
                              {
                                v88 = v135;
                                _libc_free(v74);
                                v75 = v92;
                                *(_QWORD *)(a1 + 56) = v92;
                                *(_DWORD *)(a1 + 68) = v88;
                              }
                              v64 = (unsigned __int64 *)&v108[(_QWORD)v75];
                              v65 = *(unsigned int *)(a1 + 64);
                              v67 = *(_DWORD *)(a1 + 64);
                            }
                          }
                          v68 = (unsigned __int64 *)(*(_QWORD *)(a1 + 56) + 24 * v65);
                          if ( v68 )
                          {
                            *v68 = 6;
                            v69 = v64[2];
                            v68[1] = 0;
                            v68[2] = v69;
                            if ( v69 != 0 && v69 != -4096 && v69 != -8192 )
                              sub_BD6050(v68, *v64 & 0xFFFFFFFFFFFFFFF8LL);
                            v67 = *(_DWORD *)(a1 + 64);
                          }
                          *(_DWORD *)(a1 + 64) = v67 + 1;
                          if ( v146 != 0 && v146 != -4096 && v146 != -8192 )
                            sub_BD60C0(&v144);
                        }
                        if ( v143 > 0x40 && v142 )
                          j_j___libc_free_0_0(v142);
                        if ( v141 > 0x40 && v140 )
                          j_j___libc_free_0_0(v140);
                        if ( v139 > 0x40 && v138 )
                          j_j___libc_free_0_0(v138);
                        if ( v137 > 0x40 && v136 )
                          j_j___libc_free_0_0(v136);
                        v2 = v82;
                      }
                      else
                      {
                        if ( v143 > 0x40 && v142 )
                          j_j___libc_free_0_0(v142);
                        if ( v141 > 0x40 && v140 )
                          j_j___libc_free_0_0(v140);
                        if ( v139 > 0x40 && v138 )
                          j_j___libc_free_0_0(v138);
                        if ( v137 > 0x40 && v136 )
                          j_j___libc_free_0_0(v136);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      v6 += 8;
    }
    while ( v12 != v6 );
    v6 = v149;
  }
LABEL_53:
  if ( v6 != v151 )
    _libc_free((unsigned __int64)v6);
  return v2;
}
