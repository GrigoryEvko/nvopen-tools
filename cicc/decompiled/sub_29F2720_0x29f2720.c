// Function: sub_29F2720
// Address: 0x29f2720
//
const char *__fastcall sub_29F2720(__int64 a1, __int64 a2, _QWORD *a3, char a4, __int64 a5, char a6, __int64 a7)
{
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rax
  bool v14; // cc
  _QWORD *v15; // rax
  unsigned __int64 v16; // rsi
  _QWORD *v17; // rdi
  _QWORD *v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rdx
  _QWORD *v21; // rdi
  unsigned __int64 v22; // rsi
  _QWORD *v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rsi
  unsigned __int64 v27; // r14
  unsigned __int64 v28; // rdx
  unsigned __int64 *v29; // rbx
  __int64 *v30; // rax
  __int64 *v31; // rdx
  _QWORD *v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // r12
  __int64 v35; // rax
  __int64 v36; // rbx
  char v37; // r14
  __int64 v38; // rax
  __int64 v39; // r14
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rdx
  __int16 v44; // dx
  char v45; // al
  char v46; // dl
  __int64 v47; // rcx
  __int64 v48; // rax
  __int64 v49; // r15
  __int64 v50; // rbx
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // r14
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rdx
  int v57; // eax
  __int64 v58; // r13
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rdx
  __int64 v64; // rax
  _QWORD *v65; // r9
  unsigned __int64 v66; // rdi
  _QWORD *v67; // rax
  __int64 v68; // rcx
  __int64 v69; // rdx
  __int64 v70; // rax
  __int64 v71; // rax
  _QWORD *v72; // r10
  unsigned __int64 v73; // r9
  _QWORD *v74; // rax
  __int64 v75; // rsi
  __int64 v76; // rcx
  __int64 v77; // rax
  _QWORD *v78; // rdi
  unsigned __int64 v79; // rsi
  _QWORD *v80; // rax
  __int64 v81; // rax
  unsigned __int64 v82; // rax
  unsigned __int64 v83; // rax
  unsigned __int64 v84; // rax
  unsigned __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rax
  unsigned __int64 v88; // rdi
  unsigned __int64 *v89; // r12
  unsigned __int64 v90; // rbx
  unsigned __int64 v91; // rdi
  unsigned __int64 v92; // rsi
  _QWORD *v93; // rax
  _QWORD *v94; // rdi
  unsigned __int64 v95; // rax
  __int64 v96; // r12
  int v97; // r15d
  unsigned int v98; // ebx
  __int64 *v99; // r10
  int v100; // r11d
  unsigned int v101; // edx
  __int64 *v102; // rdi
  __int64 v103; // rcx
  __int64 v104; // rax
  int v105; // r8d
  unsigned int v106; // edx
  int v107; // ecx
  __int64 v108; // r13
  __int64 *v109; // rax
  int v110; // r8d
  __int64 *v111; // rsi
  __int64 v112; // rdx
  __int64 *v113; // rsi
  int v114; // edi
  int v115; // r11d
  unsigned int v116; // edi
  __int64 v117; // [rsp+0h] [rbp-170h]
  unsigned int v118; // [rsp+8h] [rbp-168h]
  __int64 v119; // [rsp+8h] [rbp-168h]
  const char *v120; // [rsp+10h] [rbp-160h]
  __int64 v121; // [rsp+18h] [rbp-158h]
  _QWORD *v122; // [rsp+20h] [rbp-150h]
  __int64 v123; // [rsp+20h] [rbp-150h]
  _QWORD *v124; // [rsp+28h] [rbp-148h]
  __int64 v127; // [rsp+38h] [rbp-138h]
  __int64 v129; // [rsp+40h] [rbp-130h]
  __int64 v130; // [rsp+48h] [rbp-128h]
  int v131; // [rsp+54h] [rbp-11Ch] BYREF
  __int64 v132; // [rsp+58h] [rbp-118h] BYREF
  unsigned __int64 v133; // [rsp+60h] [rbp-110h] BYREF
  unsigned __int64 v134; // [rsp+68h] [rbp-108h]
  unsigned __int64 v135; // [rsp+70h] [rbp-100h]
  unsigned __int64 v136; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v137; // [rsp+88h] [rbp-E8h]
  __int64 v138; // [rsp+90h] [rbp-E0h]
  __int64 v139; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 v140; // [rsp+A8h] [rbp-C8h]
  __int64 v141; // [rsp+B0h] [rbp-C0h]
  __int64 v142; // [rsp+B8h] [rbp-B8h]
  unsigned __int64 v143[3]; // [rsp+C0h] [rbp-B0h] BYREF
  unsigned __int64 v144; // [rsp+D8h] [rbp-98h]
  __int64 v145; // [rsp+E0h] [rbp-90h]
  __int64 v146; // [rsp+E8h] [rbp-88h]
  unsigned __int64 v147; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v148; // [rsp+F8h] [rbp-78h]
  __int64 *v149; // [rsp+100h] [rbp-70h]
  __int64 *v150; // [rsp+108h] [rbp-68h]
  _QWORD *v151; // [rsp+110h] [rbp-60h]
  unsigned __int64 *v152; // [rsp+118h] [rbp-58h]
  __int64 *v153; // [rsp+120h] [rbp-50h]
  __int64 *v154; // [rsp+128h] [rbp-48h]
  _QWORD *v155; // [rsp+130h] [rbp-40h]
  unsigned __int64 *v156; // [rsp+138h] [rbp-38h]

  if ( !a3[5] )
    return sub_29F2700((unsigned __int8 *)a1, a2, a4, a5, a6, a7);
  v11 = sub_B491C0(a1);
  v12 = *(_QWORD *)(a1 - 32);
  v130 = v11;
  if ( v12 )
  {
    if ( *(_BYTE *)v12 )
    {
      v12 = 0;
    }
    else if ( *(_QWORD *)(v12 + 24) != *(_QWORD *)(a1 + 80) )
    {
      v12 = 0;
    }
  }
  v117 = *(_QWORD *)(a1 + 40);
  v132 = sub_30A7A60(v12);
  v122 = (_QWORD *)sub_30A7D00(a1);
  v13 = sub_B59BC0((__int64)v122);
  v14 = *(_DWORD *)(v13 + 32) <= 0x40u;
  v15 = *(_QWORD **)(v13 + 24);
  if ( !v14 )
    v15 = (_QWORD *)*v15;
  v131 = (int)v15;
  v16 = sub_30A7C70(a3);
  v17 = a3 + 13;
  v18 = (_QWORD *)a3[14];
  v124 = a3 + 13;
  if ( v18 )
  {
    do
    {
      while ( 1 )
      {
        v19 = v18[2];
        v20 = v18[3];
        if ( v16 <= v18[4] )
          break;
        v18 = (_QWORD *)v18[3];
        if ( !v20 )
          goto LABEL_14;
      }
      v17 = v18;
      v18 = (_QWORD *)v18[2];
    }
    while ( v19 );
LABEL_14:
    if ( v124 != v17 && v16 < v17[4] )
      v17 = a3 + 13;
  }
  else
  {
    v17 = a3 + 13;
  }
  v118 = *((_DWORD *)v17 + 10);
  v21 = a3 + 13;
  v22 = sub_30A7C70(a3);
  v23 = (_QWORD *)a3[14];
  if ( v23 )
  {
    do
    {
      while ( 1 )
      {
        v24 = v23[2];
        v25 = v23[3];
        if ( v22 <= v23[4] )
          break;
        v23 = (_QWORD *)v23[3];
        if ( !v25 )
          goto LABEL_22;
      }
      v21 = v23;
      v23 = (_QWORD *)v23[2];
    }
    while ( v24 );
LABEL_22:
    if ( v124 != v21 && v22 < v21[4] )
      v21 = a3 + 13;
  }
  v26 = a2;
  v27 = *((unsigned int *)v21 + 11);
  v120 = sub_29F2700((unsigned __int8 *)a1, a2, a4, a5, a6, a7);
  if ( !v120 )
  {
    sub_B43D60(v122);
    v28 = v118;
    v133 = 0;
    v134 = 0;
    v135 = 0;
    v136 = 0;
    v137 = 0;
    v138 = 0;
    v147 = -1;
    if ( v118 )
    {
      sub_29E54B0((__int64)&v133, 0, v118, (__int64 *)&v147);
      v26 = v137;
      v147 = -1;
      v28 = (__int64)(v137 - v136) >> 3;
      if ( v27 <= v28 )
      {
        if ( v27 < v28 && v136 + 8 * v27 != v137 )
          v137 = v136 + 8 * v27;
LABEL_30:
        v147 = 0;
        v149 = 0;
        v150 = 0;
        v151 = 0;
        v152 = 0;
        v153 = 0;
        v154 = 0;
        v155 = 0;
        v156 = 0;
        v148 = 8;
        v147 = sub_22077B0(0x40u);
        v29 = (unsigned __int64 *)(v147 + ((4 * v148 - 4) & 0xFFFFFFFFFFFFFFF8LL));
        v30 = (__int64 *)sub_22077B0(0x200u);
        v152 = v29;
        *v29 = (unsigned __int64)v30;
        v150 = v30;
        v151 = v30 + 64;
        v156 = v29;
        v154 = v30;
        v155 = v30 + 64;
        v149 = v30;
        v139 = 0;
        v140 = 0;
        v141 = 0;
        v142 = 0;
        if ( v30 )
          *v30 = v117;
        v31 = v30 + 1;
        v153 = v30 + 1;
        while ( 1 )
        {
          v123 = v130 + 16;
          while ( 1 )
          {
            if ( v30 == v31 )
            {
              v82 = v133;
              v133 = 0;
              v143[0] = v82;
              v83 = v134;
              v134 = 0;
              v143[1] = v83;
              v84 = v135;
              v135 = 0;
              v143[2] = v84;
              v85 = v136;
              v136 = 0;
              v144 = v85;
              v86 = v137;
              v137 = 0;
              v145 = v86;
              v87 = v138;
              v138 = 0;
              v146 = v87;
              sub_C7D6A0(v140, 8LL * (unsigned int)v142, 8);
              v88 = v147;
              if ( v147 )
              {
                v89 = v152;
                v90 = (unsigned __int64)(v156 + 1);
                if ( v156 + 1 > v152 )
                {
                  do
                  {
                    v91 = *v89++;
                    j_j___libc_free_0(v91);
                  }
                  while ( v90 > (unsigned __int64)v89 );
                  v88 = v147;
                }
                j_j___libc_free_0(v88);
              }
              if ( v136 )
                j_j___libc_free_0(v136);
              if ( v133 )
                j_j___libc_free_0(v133);
              v92 = sub_30A7C70(a3);
              v93 = (_QWORD *)a3[14];
              if ( v93 )
              {
                v94 = a3 + 13;
                do
                {
                  if ( v92 > v93[4] )
                  {
                    v93 = (_QWORD *)v93[3];
                  }
                  else
                  {
                    v94 = v93;
                    v93 = (_QWORD *)v93[2];
                  }
                }
                while ( v93 );
                if ( v124 != v94 )
                {
                  if ( v92 < v94[4] )
                    v94 = a3 + 13;
                  v124 = v94;
                }
              }
              LODWORD(v139) = *((_DWORD *)v124 + 10);
              v147 = (unsigned __int64)v143;
              v148 = (__int64)&v139;
              v149 = (__int64 *)&v131;
              v150 = &v132;
              sub_30A7E90(a3, sub_29E6420, &v147, v130);
              if ( v144 )
                j_j___libc_free_0(v144);
              if ( v143[0] )
                j_j___libc_free_0(v143[0]);
              return v120;
            }
            v127 = *v30;
            v32 = v151;
            v33 = (__int64)(v151 - 1);
            if ( v30 == v151 - 1 )
            {
              v26 = 512;
              j_j___libc_free_0((unsigned __int64)v150);
              v33 = *++v152 + 512;
              v150 = (__int64 *)*v152;
              v151 = (_QWORD *)v33;
              v149 = v150;
            }
            else
            {
              v149 = v30 + 1;
            }
            v34 = v127 + 48;
            v35 = sub_30A7DC0(v127, v26, v33, v32);
            v36 = v35;
            if ( !v35 )
              break;
            v37 = 0;
            if ( (unsigned __int8 *)v130 != sub_BD3990(
                                              *(unsigned __int8 **)(v35 - 32LL * (*(_DWORD *)(v35 + 4) & 0x7FFFFFF)),
                                              v26) )
            {
              v38 = sub_B59BC0(v36);
              if ( *(_DWORD *)(v38 + 32) <= 0x40u )
                v39 = *(_QWORD *)(v38 + 24);
              else
                v39 = **(_QWORD **)(v38 + 24);
              v40 = *(_QWORD *)(v133 + 8LL * (unsigned int)v39);
              if ( v40 == -1 )
              {
                v78 = a3 + 13;
                v79 = sub_30A7C70(a3);
                v80 = (_QWORD *)a3[14];
                if ( v80 )
                {
                  do
                  {
                    if ( v79 > v80[4] )
                    {
                      v80 = (_QWORD *)v80[3];
                    }
                    else
                    {
                      v78 = v80;
                      v80 = (_QWORD *)v80[2];
                    }
                  }
                  while ( v80 );
                  if ( v124 != v78 && v79 < v78[4] )
                    v78 = a3 + 13;
                }
                v81 = *((unsigned int *)v78 + 10);
                *((_DWORD *)v78 + 10) = v81 + 1;
                *(_QWORD *)(v133 + 8LL * (unsigned int)v39) = v81;
                v40 = *(_QWORD *)(v133 + 8LL * (unsigned int)v39);
              }
              v41 = v36 - 32LL * (*(_DWORD *)(v36 + 4) & 0x7FFFFFF);
              if ( *(_QWORD *)v41 )
              {
                v42 = *(_QWORD *)(v41 + 8);
                **(_QWORD **)(v41 + 16) = v42;
                if ( v42 )
                  *(_QWORD *)(v42 + 16) = *(_QWORD *)(v41 + 16);
              }
              *(_QWORD *)v41 = v130;
              if ( v130 )
              {
                v43 = *(_QWORD *)(v130 + 16);
                *(_QWORD *)(v41 + 8) = v43;
                if ( v43 )
                  *(_QWORD *)(v43 + 16) = v41 + 8;
                *(_QWORD *)(v41 + 16) = v123;
                *(_QWORD *)(v130 + 16) = v41;
              }
              v37 = 1;
              sub_B59C10(v36, v40);
            }
            v26 = sub_AA5190(v127);
            if ( v26 )
            {
              v45 = v44;
              v46 = HIBYTE(v44);
            }
            else
            {
              v46 = 0;
              v45 = 0;
            }
            v47 = v121;
            LOBYTE(v47) = v45;
            v48 = v47;
            BYTE1(v48) = v46;
            v121 = v48;
            sub_B444E0((_QWORD *)v36, v26, v48);
            v49 = *(_QWORD *)(v127 + 56);
            if ( v49 != v34 )
            {
              v129 = v36;
LABEL_56:
              while ( 2 )
              {
                v50 = *(_QWORD *)(v49 + 8);
                if ( *(_BYTE *)(v49 - 24) == 85 )
                {
                  v51 = *(_QWORD *)(v49 - 56);
                  if ( v51 )
                  {
                    if ( !*(_BYTE *)v51
                      && *(_QWORD *)(v51 + 24) == *(_QWORD *)(v49 + 56)
                      && (*(_BYTE *)(v51 + 33) & 0x20) != 0
                      && (v57 = *(_DWORD *)(v51 + 36), v26 = (unsigned int)(v57 - 198), (unsigned int)v26 <= 1) )
                    {
                      v58 = v49 - 24;
                      if ( v57 == 199 )
                      {
                        if ( *(_BYTE *)sub_B59CA0(v49 - 24) > 0x15u )
                        {
                          if ( (unsigned __int8 *)v130 != sub_BD3990(
                                                            *(unsigned __int8 **)(v58
                                                                                - 32LL
                                                                                * (*(_DWORD *)(v49 - 20) & 0x7FFFFFF)),
                                                            v26) )
                          {
                            v59 = sub_B59BC0(v49 - 24);
                            if ( *(_DWORD *)(v59 + 32) <= 0x40u )
                              v60 = *(_QWORD *)(v59 + 24);
                            else
                              v60 = **(_QWORD **)(v59 + 24);
                            v26 = *(_QWORD *)(v133 + 8LL * (unsigned int)v60);
                            if ( v26 == -1 )
                            {
                              v119 = (unsigned int)v60;
                              v71 = sub_30A7C70(a3);
                              v72 = a3 + 13;
                              v73 = v71;
                              v74 = (_QWORD *)a3[14];
                              if ( v74 )
                              {
                                do
                                {
                                  while ( 1 )
                                  {
                                    v75 = v74[2];
                                    v76 = v74[3];
                                    if ( v73 <= v74[4] )
                                      break;
                                    v74 = (_QWORD *)v74[3];
                                    if ( !v76 )
                                      goto LABEL_121;
                                  }
                                  v72 = v74;
                                  v74 = (_QWORD *)v74[2];
                                }
                                while ( v75 );
LABEL_121:
                                if ( v124 != v72 && v73 < v72[4] )
                                  v72 = a3 + 13;
                              }
                              v77 = *((unsigned int *)v72 + 10);
                              *((_DWORD *)v72 + 10) = v77 + 1;
                              *(_QWORD *)(v133 + 8 * v119) = v77;
                              v26 = *(_QWORD *)(v133 + 8 * v119);
                            }
                            v61 = v58 - 32LL * (*(_DWORD *)(v49 - 20) & 0x7FFFFFF);
                            if ( *(_QWORD *)v61 )
                            {
                              v62 = *(_QWORD *)(v61 + 8);
                              **(_QWORD **)(v61 + 16) = v62;
                              if ( v62 )
                                *(_QWORD *)(v62 + 16) = *(_QWORD *)(v61 + 16);
                            }
                            *(_QWORD *)v61 = v130;
                            if ( v130 )
                            {
                              v63 = *(_QWORD *)(v130 + 16);
                              *(_QWORD *)(v61 + 8) = v63;
                              if ( v63 )
                                *(_QWORD *)(v63 + 16) = v61 + 8;
                              *(_QWORD *)(v61 + 16) = v123;
                              *(_QWORD *)(v130 + 16) = v61;
                            }
                            sub_B59C10(v49 - 24, v26);
                          }
                        }
                        else
                        {
                          sub_B43D60((_QWORD *)(v49 - 24));
                        }
                      }
                      else if ( v129 != v58 )
                      {
                        sub_B43D60((_QWORD *)(v49 - 24));
                        if ( v50 == v34 )
                          goto LABEL_158;
                        v37 = 1;
                        goto LABEL_55;
                      }
                    }
                    else if ( !*(_BYTE *)v51
                           && *(_QWORD *)(v51 + 24) == *(_QWORD *)(v49 + 56)
                           && (*(_BYTE *)(v51 + 33) & 0x20) != 0
                           && *(_DWORD *)(v51 + 36) == 196
                           && (unsigned __int8 *)v130 != sub_BD3990(
                                                           *(unsigned __int8 **)(v49
                                                                               - 32LL
                                                                               * (*(_DWORD *)(v49 - 20) & 0x7FFFFFF)
                                                                               - 24),
                                                           v26) )
                    {
                      v52 = sub_B59BC0(v49 - 24);
                      if ( *(_DWORD *)(v52 + 32) <= 0x40u )
                        v53 = *(_QWORD *)(v52 + 24);
                      else
                        v53 = **(_QWORD **)(v52 + 24);
                      v26 = *(_QWORD *)(v136 + 8LL * (unsigned int)v53);
                      if ( v26 == -1 )
                      {
                        v64 = sub_30A7C70(a3);
                        v65 = a3 + 13;
                        v66 = v64;
                        v67 = (_QWORD *)a3[14];
                        if ( v67 )
                        {
                          do
                          {
                            while ( 1 )
                            {
                              v68 = v67[2];
                              v69 = v67[3];
                              if ( v66 <= v67[4] )
                                break;
                              v67 = (_QWORD *)v67[3];
                              if ( !v69 )
                                goto LABEL_108;
                            }
                            v65 = v67;
                            v67 = (_QWORD *)v67[2];
                          }
                          while ( v68 );
LABEL_108:
                          if ( v124 != v65 && v66 < v65[4] )
                            v65 = a3 + 13;
                        }
                        v70 = *((unsigned int *)v65 + 11);
                        *((_DWORD *)v65 + 11) = v70 + 1;
                        *(_QWORD *)(v136 + 8LL * (unsigned int)v53) = v70;
                        v26 = *(_QWORD *)(v136 + 8LL * (unsigned int)v53);
                      }
                      v54 = v49 - 24 - 32LL * (*(_DWORD *)(v49 - 20) & 0x7FFFFFF);
                      if ( *(_QWORD *)v54 )
                      {
                        v55 = *(_QWORD *)(v54 + 8);
                        **(_QWORD **)(v54 + 16) = v55;
                        if ( v55 )
                          *(_QWORD *)(v55 + 16) = *(_QWORD *)(v54 + 16);
                      }
                      *(_QWORD *)v54 = v130;
                      if ( v130 )
                      {
                        v56 = *(_QWORD *)(v130 + 16);
                        *(_QWORD *)(v54 + 8) = v56;
                        if ( v56 )
                          *(_QWORD *)(v56 + 16) = v54 + 8;
                        *(_QWORD *)(v54 + 16) = v123;
                        *(_QWORD *)(v130 + 16) = v54;
                      }
                      v37 = 1;
                      sub_B59C10(v49 - 24, v26);
                      if ( v50 == v34 )
                      {
LABEL_76:
                        v36 = v129;
                        break;
                      }
                      goto LABEL_55;
                    }
                  }
                }
                if ( v50 == v34 )
                  goto LABEL_76;
LABEL_55:
                v49 = v50;
                continue;
              }
            }
            if ( !v36 || v37 )
              goto LABEL_158;
            v31 = v153;
            v30 = v149;
          }
          v49 = *(_QWORD *)(v127 + 56);
          if ( v49 != v34 )
          {
            v129 = 0;
            v37 = 0;
            goto LABEL_56;
          }
LABEL_158:
          v95 = *(_QWORD *)(v127 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v95 != v34 )
          {
            if ( !v95 )
              BUG();
            v96 = v95 - 24;
            if ( (unsigned int)*(unsigned __int8 *)(v95 - 24) - 30 <= 0xA )
            {
              v97 = sub_B46E30(v96);
              if ( v97 )
                break;
            }
          }
LABEL_175:
          v31 = v153;
          v30 = v149;
        }
        v98 = 0;
        while ( 1 )
        {
          v104 = sub_B46EC0(v96, v98);
          v26 = (unsigned int)v142;
          v143[0] = v104;
          if ( !(_DWORD)v142 )
            break;
          v99 = 0;
          v100 = 1;
          v101 = (v142 - 1) & (((unsigned int)v104 >> 9) ^ ((unsigned int)v104 >> 4));
          v102 = (__int64 *)(v140 + 8LL * v101);
          v103 = *v102;
          if ( v104 != *v102 )
          {
            while ( v103 != -4096 )
            {
              if ( v103 != -8192 || v99 )
                v102 = v99;
              v101 = (v142 - 1) & (v100 + v101);
              v103 = *(_QWORD *)(v140 + 8LL * v101);
              if ( v104 == v103 )
                goto LABEL_164;
              ++v100;
              v99 = v102;
              v102 = (__int64 *)(v140 + 8LL * v101);
            }
            if ( !v99 )
              v99 = v102;
            ++v139;
            v107 = v141 + 1;
            if ( 4 * ((int)v141 + 1) < (unsigned int)(3 * v142) )
            {
              if ( (int)v142 - HIDWORD(v141) - v107 <= (unsigned int)v142 >> 3 )
              {
                sub_E3B4A0((__int64)&v139, v142);
                if ( !(_DWORD)v142 )
                {
LABEL_216:
                  LODWORD(v141) = v141 + 1;
                  BUG();
                }
                v110 = 1;
                v107 = v141 + 1;
                v111 = 0;
                LODWORD(v112) = (v142 - 1) & ((LODWORD(v143[0]) >> 9) ^ (LODWORD(v143[0]) >> 4));
                v99 = (__int64 *)(v140 + 8LL * (unsigned int)v112);
                v104 = *v99;
                if ( v143[0] != *v99 )
                {
                  while ( v104 != -4096 )
                  {
                    if ( v104 == -8192 && !v111 )
                      v111 = v99;
                    v112 = ((_DWORD)v142 - 1) & (unsigned int)(v112 + v110);
                    v99 = (__int64 *)(v140 + 8 * v112);
                    v104 = *v99;
                    if ( v143[0] == *v99 )
                      goto LABEL_169;
                    ++v110;
                  }
                  v104 = v143[0];
                  if ( v111 )
                    v99 = v111;
                }
              }
              goto LABEL_169;
            }
LABEL_167:
            sub_E3B4A0((__int64)&v139, 2 * v142);
            if ( !(_DWORD)v142 )
              goto LABEL_216;
            v104 = v143[0];
            v105 = v142 - 1;
            v106 = (v142 - 1) & ((LODWORD(v143[0]) >> 9) ^ (LODWORD(v143[0]) >> 4));
            v99 = (__int64 *)(v140 + 8LL * v106);
            v107 = v141 + 1;
            v108 = *v99;
            if ( *v99 != v143[0] )
            {
              v113 = (__int64 *)(v140 + 8LL * (v105 & (unsigned int)((LODWORD(v143[0]) >> 9) ^ (LODWORD(v143[0]) >> 4))));
              v114 = 1;
              v99 = 0;
              while ( v108 != -4096 )
              {
                if ( !v99 && v108 == -8192 )
                  v99 = v113;
                v115 = v114 + 1;
                v116 = v105 & (v106 + v114);
                v106 = v116;
                v113 = (__int64 *)(v140 + 8LL * v116);
                v108 = *v113;
                if ( v143[0] == *v113 )
                {
                  v99 = (__int64 *)(v140 + 8LL * v116);
                  goto LABEL_169;
                }
                v114 = v115;
              }
              if ( !v99 )
                v99 = v113;
            }
LABEL_169:
            LODWORD(v141) = v107;
            if ( *v99 != -4096 )
              --HIDWORD(v141);
            *v99 = v104;
            v26 = (__int64)v155;
            v109 = v153;
            if ( v153 == v155 - 1 )
            {
              v26 = (__int64)v143;
              sub_27698B0(&v147, v143);
            }
            else
            {
              if ( v153 )
              {
                *v153 = v143[0];
                v109 = v153;
              }
              v153 = v109 + 1;
            }
          }
LABEL_164:
          if ( v97 == ++v98 )
            goto LABEL_175;
        }
        ++v139;
        goto LABEL_167;
      }
    }
    else
    {
      if ( !v27 )
        goto LABEL_30;
      v26 = 0;
    }
    sub_29E54B0((__int64)&v136, (char *)v26, v27 - v28, (__int64 *)&v147);
    goto LABEL_30;
  }
  return v120;
}
