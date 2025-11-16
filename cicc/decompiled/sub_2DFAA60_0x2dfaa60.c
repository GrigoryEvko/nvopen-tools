// Function: sub_2DFAA60
// Address: 0x2dfaa60
//
__int64 __fastcall sub_2DFAA60(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  unsigned int v8; // r12d
  __int64 v9; // r13
  unsigned __int64 *v10; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rsi
  unsigned __int64 v16; // rcx
  unsigned int v17; // r12d
  unsigned int v18; // r13d
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rax
  int v22; // ecx
  unsigned int v23; // edi
  __int64 v24; // rdx
  unsigned int v25; // eax
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned __int64 *v30; // rcx
  __int64 v31; // rax
  _QWORD *v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // r13
  __int64 v35; // rdx
  unsigned int v37; // edi
  unsigned int v38; // eax
  __int64 v39; // rdx
  int v40; // eax
  __int64 v41; // rdx
  __int64 v42; // rcx
  unsigned int v43; // r13d
  __int64 v44; // rax
  unsigned __int64 v45; // rcx
  int v46; // eax
  __int64 v47; // rbx
  int v48; // r12d
  unsigned __int64 v49; // r14
  int v50; // r12d
  __int64 v51; // rax
  __int64 v52; // rbx
  int v53; // edx
  unsigned int v54; // eax
  int v55; // r15d
  __int64 v56; // rax
  _QWORD *v57; // rdx
  __int64 v58; // r12
  unsigned int *v59; // r11
  unsigned int *v60; // r12
  unsigned int v61; // r8d
  unsigned int v62; // r9d
  __int64 v63; // r10
  _QWORD *v64; // rdx
  int v65; // r9d
  unsigned int v66; // ecx
  __int64 v67; // rsi
  unsigned int v68; // edi
  __int64 v69; // rax
  unsigned int v70; // edi
  _QWORD *v71; // rax
  __int64 v72; // r13
  __int64 v73; // r8
  _QWORD *v74; // r15
  _QWORD *v75; // rdi
  unsigned int k; // eax
  __int64 v77; // r8
  unsigned int v78; // r9d
  unsigned int *v79; // r15
  unsigned int *v80; // r14
  unsigned int v81; // edx
  unsigned int v82; // esi
  int v83; // r11d
  __int64 v84; // rcx
  unsigned int *v85; // r10
  _QWORD *v86; // rax
  unsigned int v87; // r9d
  unsigned int v88; // edi
  unsigned int v89; // r8d
  __int64 v90; // rsi
  unsigned int v91; // r8d
  _QWORD *v92; // rsi
  __int64 v93; // r13
  __int64 v94; // r9
  _QWORD *v95; // r12
  unsigned int m; // esi
  __int64 v97; // r8
  unsigned int v98; // r14d
  __int64 v99; // r13
  __int64 v100; // rdx
  int v101; // r9d
  __int64 v102; // rdi
  __int64 v103; // rcx
  unsigned int v104; // eax
  unsigned int v105; // r9d
  _QWORD *v106; // rax
  unsigned int v107; // r15d
  __int64 v108; // r13
  __int64 v109; // rdi
  _QWORD *v110; // rcx
  unsigned int j; // eax
  __int64 v112; // rsi
  unsigned int v113; // r13d
  _QWORD *v114; // rsi
  unsigned int v115; // edi
  unsigned int v116; // r12d
  __int64 v117; // rax
  __int64 v118; // r8
  _QWORD *v119; // rsi
  unsigned int v120; // edx
  __int64 v121; // r8
  __int64 v122; // rax
  unsigned __int64 *v123; // rax
  unsigned int i; // ebx
  unsigned int v125; // eax
  __int64 v126; // rax
  unsigned __int64 v127; // rdi
  unsigned int v128; // eax
  __int64 v129; // rdi
  unsigned __int64 v130; // rcx
  __int64 v131; // [rsp+0h] [rbp-E0h]
  __int64 v132; // [rsp+8h] [rbp-D8h]
  int v133; // [rsp+10h] [rbp-D0h]
  char v134; // [rsp+17h] [rbp-C9h]
  __int64 *v135; // [rsp+28h] [rbp-B8h]
  unsigned int v136; // [rsp+30h] [rbp-B0h]
  unsigned int v137; // [rsp+34h] [rbp-ACh]
  _QWORD *v140; // [rsp+48h] [rbp-98h]
  int v141; // [rsp+48h] [rbp-98h]
  unsigned int *v142; // [rsp+50h] [rbp-90h]
  int v143; // [rsp+58h] [rbp-88h]
  _QWORD *v144; // [rsp+58h] [rbp-88h]
  unsigned int v145; // [rsp+60h] [rbp-80h]
  unsigned int v146; // [rsp+64h] [rbp-7Ch]
  _DWORD v148[4]; // [rsp+70h] [rbp-70h] BYREF
  _DWORD v149[4]; // [rsp+80h] [rbp-60h] BYREF
  _QWORD v150[10]; // [rsp+90h] [rbp-50h] BYREF

  v142 = (unsigned int *)(a1 + 8);
  if ( a2 != 1 )
  {
    v18 = a2;
    v17 = 0;
    v145 = a2 - 1;
    v19 = 16LL * (a2 - 1);
    goto LABEL_9;
  }
  v6 = *(_QWORD *)a1;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(_DWORD *)(*(_QWORD *)a1 + 164LL);
  if ( v8 > 8 )
  {
    v9 = *(unsigned int *)(v7 + 12);
    v10 = *(unsigned __int64 **)(v6 + 168);
    v11 = *v10;
    if ( *v10 )
    {
      *v10 = *(_QWORD *)v11;
    }
    else
    {
      v127 = v10[1];
      v10[11] += 192LL;
      v12 = (v127 + 63) & 0xFFFFFFFFFFFFFFC0LL;
      if ( v10[2] >= v12 + 192 && v127 )
      {
        v10[1] = v12 + 192;
        if ( !v12 )
        {
LABEL_6:
          v13 = 0;
          do
          {
            *(_QWORD *)(v11 + 8 * v13) = *(_QWORD *)(v6 + 8 * v13 + 8);
            *(_QWORD *)(v11 + 8 * v13 + 96) = *(_QWORD *)(v6 + 8 * v13 + 80);
            ++v13;
          }
          while ( v8 != (_DWORD)v13 );
          v14 = *(_QWORD *)(v12 + 160);
          ++*(_DWORD *)(v6 + 160);
          *(_QWORD *)(v6 + 8) = v12 | 8;
          v15 = v6 + 8;
          v16 = v9 << 32;
          *(_QWORD *)(v6 + 80) = v14;
          v17 = 1;
          v18 = 2;
          *(_DWORD *)(v6 + 164) = 1;
          v19 = 16;
          sub_F038C0(v142, v15, 1, v16, a5, a6);
          v145 = 1;
LABEL_9:
          if ( !*(_DWORD *)(a1 + 16) || (v20 = *(_QWORD *)(a1 + 8), *(_DWORD *)(v20 + 12) >= *(_DWORD *)(v20 + 8)) )
          {
            sub_F03AD0(v142, v145);
            ++*(_DWORD *)(v19 + *(_QWORD *)(a1 + 8) + 12);
            v20 = *(_QWORD *)(a1 + 8);
          }
          v21 = v19 + v20;
          v22 = *(_DWORD *)(v21 + 8);
          if ( v22 != 12 )
          {
LABEL_13:
            v23 = *(_DWORD *)(v21 + 12);
            v24 = *(_QWORD *)v21;
            v25 = v22 - 1;
            if ( v22 != v23 )
            {
              do
              {
                v26 = v25 + 1;
                *(_QWORD *)(v24 + 8 * v26) = *(_QWORD *)(v24 + 8LL * v25);
                *(_QWORD *)(v24 + 8 * v26 + 96) = *(_QWORD *)(v24 + 8LL * v25 + 96);
                LODWORD(v26) = v25--;
              }
              while ( v23 != (_DWORD)v26 );
            }
            *(_QWORD *)(v24 + 8LL * v23) = a3;
            *(_QWORD *)(v24 + 8LL * v23 + 96) = a4;
            v27 = v19 + *(_QWORD *)(a1 + 8);
            v28 = *(unsigned int *)(v27 + 8);
            *(_DWORD *)(v27 + 8) = v28 + 1;
            if ( v145 )
            {
              v29 = *(_QWORD *)(a1 + 8) + 16LL * (v145 - 1);
              v30 = (unsigned __int64 *)(*(_QWORD *)v29 + 8LL * *(unsigned int *)(v29 + 12));
              *v30 = *v30 & 0xFFFFFFFFFFFFFFC0LL | v28;
            }
            v31 = *(_QWORD *)(a1 + 8);
            v32 = (_QWORD *)(v31 + v19);
            v33 = *(unsigned int *)(v31 + v19 + 12);
            if ( (_DWORD)v33 == *(_DWORD *)(v31 + v19 + 8) - 1 )
            {
              sub_2DF4670(a1, v145, a4);
              v31 = *(_QWORD *)(a1 + 8);
              v32 = (_QWORD *)(v31 + v19);
              v33 = *(unsigned int *)(v31 + v19 + 12);
            }
            v34 = v31 + 16LL * v18;
            v35 = *(_QWORD *)(*v32 + 8 * v33);
            *(_QWORD *)v34 = v35 & 0xFFFFFFFFFFFFFFC0LL;
            *(_DWORD *)(v34 + 8) = (v35 & 0x3F) + 1;
            return v17;
          }
          v43 = *(_DWORD *)(v21 + 12);
          v44 = sub_F03A30((__int64 *)v142, v145);
          v132 = v44;
          if ( v44 )
          {
            v45 = v44 & 0xFFFFFFFFFFFFFFC0LL;
            v136 = 2;
            v46 = (v44 & 0x3F) + 1;
            v150[0] = v45;
            v148[0] = v46;
            v43 += v46;
            v137 = 1;
          }
          else
          {
            v136 = 1;
            v46 = 0;
            v137 = 0;
          }
          v47 = *(_QWORD *)(a1 + 8) + v19;
          v48 = *(_DWORD *)(v47 + 8);
          v148[v137] = v48;
          v49 = *(_QWORD *)v47;
          v50 = v46 + v48;
          v150[v137] = *(_QWORD *)v47;
          v51 = sub_F03C90((__int64 *)v142, v145);
          if ( v51 )
          {
            v137 += 2;
            v49 = v51 & 0xFFFFFFFFFFFFFFC0LL;
            v52 = v136;
            v53 = (v51 & 0x3F) + 1;
            v50 += v53;
            v54 = 24;
            v148[v136] = v53;
            if ( v137 != 2 )
              v54 = 36;
            v150[v136] = v49;
            if ( v50 + 1 <= v54 )
            {
              v55 = v136;
              v134 = 0;
              v136 = 0;
              v146 = v137;
              v137 = v55;
              goto LABEL_36;
            }
          }
          else
          {
            v125 = 12;
            if ( v136 != 1 )
              v125 = 24;
            if ( v50 + 1 <= v125 )
            {
              v126 = sub_F03E60(v136, v50, 12, (__int64)v148, (__int64)v149, v43, 1u);
              HIDWORD(v131) = HIDWORD(v126);
              v133 = v126;
              if ( !v137 )
              {
                v128 = v136;
                v134 = 0;
                v136 = 0;
                v146 = v128;
LABEL_70:
                if ( v132 )
                  sub_F03AD0(v142, v145);
                v98 = v145;
                v99 = 0;
                v17 = 0;
                while ( 1 )
                {
                  v100 = v150[v99];
                  v101 = v149[v99];
                  v102 = (unsigned int)(v101 - 1);
                  v103 = *(_QWORD *)(v100 + 8 * v102 + 96);
                  if ( v136 == (_DWORD)v99 && v134 )
                  {
                    ++v99;
                    v17 = sub_2DFAA60(a1, v98, v102 | v100 & 0xFFFFFFFFFFFFFFC0LL, v103);
                    v98 += (unsigned __int8)v17;
                    if ( v146 == v99 )
                      goto LABEL_100;
                  }
                  else
                  {
                    *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL * v98 + 8) = v101;
                    if ( v98 )
                    {
                      v122 = *(_QWORD *)(a1 + 8) + 16LL * (v98 - 1);
                      v123 = (unsigned __int64 *)(*(_QWORD *)v122 + 8LL * *(unsigned int *)(v122 + 12));
                      *v123 = v102 | *v123 & 0xFFFFFFFFFFFFFFC0LL;
                    }
                    ++v99;
                    sub_2DF4670(a1, v98, v103);
                    if ( v146 == v99 )
                    {
LABEL_100:
                      for ( i = v146 - 1; i != v133; --i )
                        sub_F03AD0(v142, v98);
                      *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL * v98 + 12) = HIDWORD(v131);
                      v145 += (unsigned __int8)v17;
                      v18 = v145 + 1;
                      v19 = 16LL * v145;
                      v21 = v19 + *(_QWORD *)(a1 + 8);
                      v22 = *(_DWORD *)(v21 + 8);
                      goto LABEL_13;
                    }
                  }
                  sub_F03D40((__int64 *)v142, v98);
                }
              }
              v146 = v136;
              v55 = 1;
              v134 = 0;
              v136 = 0;
LABEL_37:
              v143 = v55;
              v58 = v55;
              v140 = &v150[v55];
              v59 = &v148[v58];
              v60 = &v149[v58];
              do
              {
                v61 = *v59;
                v62 = *v60;
                --v143;
                if ( *v59 != *v60 )
                {
                  v63 = v143;
                  v64 = (_QWORD *)*v140;
                  do
                  {
                    v65 = v62 - v61;
                    v66 = v148[v63];
                    v67 = v150[v63];
                    if ( v65 <= 0 )
                    {
                      v104 = 12 - v66;
                      if ( 12 - v66 > v61 )
                        v104 = v61;
                      v105 = -v65;
                      if ( v104 <= v105 )
                        v105 = v104;
                      v106 = v64 + 12;
                      v107 = v105 + v66;
                      if ( v105 )
                      {
                        do
                        {
                          v108 = *(v106 - 12);
                          v109 = v66++;
                          ++v106;
                          *(_QWORD *)(v67 + 8 * v109) = v108;
                          *(_QWORD *)(v67 + 8 * v109 + 96) = *(v106 - 1);
                        }
                        while ( v66 != v107 );
                      }
                      v110 = v64;
                      for ( j = v105; j != v61; v110[11] = v64[v112 + 12] )
                      {
                        v112 = j++;
                        *v110++ = v64[v112];
                      }
                      v65 = -v105;
                    }
                    else
                    {
                      v68 = v61 - 1;
                      if ( v65 > v66 )
                        v65 = v148[v63];
                      if ( 12 - v61 <= v65 )
                        v65 = 12 - v61;
                      if ( v61 )
                      {
                        v69 = v68;
                        v70 = v65 + v68;
                        v71 = &v64[v69];
                        do
                        {
                          v72 = *v71;
                          v73 = v70;
                          v74 = v71;
                          --v70;
                          --v71;
                          v64[v73] = v72;
                          v64[v73 + 12] = v71[13];
                        }
                        while ( v64 != v74 );
                      }
                      v75 = v64;
                      for ( k = v66 - v65; v66 != k; v75[11] = *(_QWORD *)(v67 + 8 * v77 + 96) )
                      {
                        v77 = k++;
                        *v75++ = *(_QWORD *)(v67 + 8 * v77);
                      }
                    }
                    v148[v63] -= v65;
                    v78 = *v59 + v65;
                    *v59 = v78;
                    v61 = v78;
                    v62 = *v60;
                    if ( v61 >= *v60 )
                      break;
                    --v63;
                  }
                  while ( (_DWORD)v63 != -1 );
                }
                --v140;
                --v59;
                --v60;
              }
              while ( v143 );
              v79 = v149;
              v80 = v148;
              v141 = 0;
              v135 = v150;
              do
              {
                v81 = *v80;
                v82 = *v79;
                ++v141;
                if ( *v80 != *v79 )
                {
                  v83 = v141;
                  if ( v146 != v141 )
                  {
                    v84 = *v135;
                    do
                    {
                      v85 = &v148[v83];
                      v86 = (_QWORD *)v150[v83];
                      v87 = *v85;
                      if ( (int)(v81 - v82) <= 0 )
                      {
                        v113 = v82 - v81;
                        if ( 12 - v81 <= v82 - v81 )
                          v113 = 12 - v81;
                        v114 = v86 + 12;
                        v115 = v113;
                        if ( v87 <= v113 )
                          v115 = *v85;
                        v116 = v115 + v81;
                        if ( v115 )
                        {
                          v144 = (_QWORD *)v150[v83];
                          do
                          {
                            v117 = *(v114 - 12);
                            v118 = v81++;
                            ++v114;
                            *(_QWORD *)(v84 + 8 * v118) = v117;
                            *(_QWORD *)(v84 + 8 * v118 + 96) = *(v114 - 1);
                          }
                          while ( v81 != v116 );
                          v86 = v144;
                        }
                        v119 = v86;
                        v120 = v115;
                        if ( v87 > v113 )
                        {
                          do
                          {
                            v121 = v120++;
                            *v119++ = v86[v121];
                            v119[11] = v86[v121 + 12];
                          }
                          while ( v87 != v120 );
                        }
                        v88 = -v115;
                      }
                      else
                      {
                        v88 = v81 - v82;
                        v89 = v87 - 1;
                        if ( v81 - v82 > v81 )
                          v88 = v81;
                        if ( v88 > 12 - v87 )
                          v88 = 12 - v87;
                        if ( v87 )
                        {
                          v90 = v89;
                          v91 = v88 + v89;
                          v92 = &v86[v90];
                          do
                          {
                            v93 = *v92;
                            v94 = v91;
                            v95 = v92;
                            --v91;
                            --v92;
                            v86[v94] = v93;
                            v86[v94 + 12] = v92[13];
                          }
                          while ( v86 != v95 );
                        }
                        for ( m = v81 - v88; m != v81; v86[11] = *(_QWORD *)(v84 + 8 * v97 + 96) )
                        {
                          v97 = m++;
                          *v86++ = *(_QWORD *)(v84 + 8 * v97);
                        }
                      }
                      *v85 += v88;
                      v82 = *v79;
                      v81 = *v80 - v88;
                      *v80 = v81;
                      if ( v81 >= v82 )
                        break;
                      ++v83;
                    }
                    while ( v83 != v146 );
                  }
                }
                ++v135;
                ++v80;
                ++v79;
              }
              while ( v137 != v141 );
              goto LABEL_70;
            }
            if ( v136 == 1 )
            {
              v146 = 2;
              v53 = v148[1];
              v55 = 1;
              v52 = 1;
              v137 = 1;
              v49 = v150[1];
LABEL_32:
              v148[v137] = v53;
              v150[v137] = v49;
              v148[v52] = 0;
              v56 = *(_QWORD *)(*(_QWORD *)a1 + 168LL);
              v57 = *(_QWORD **)v56;
              if ( *(_QWORD *)v56 )
              {
                *(_QWORD *)v56 = *v57;
              }
              else
              {
                v129 = *(_QWORD *)(v56 + 8);
                *(_QWORD *)(v56 + 88) += 192LL;
                v130 = (v129 + 63) & 0xFFFFFFFFFFFFFFC0LL;
                if ( *(_QWORD *)(v56 + 16) >= v130 + 192 && v129 )
                {
                  *(_QWORD *)(v56 + 8) = v130 + 192;
                  if ( !v130 )
                    goto LABEL_35;
                  v57 = (_QWORD *)((v129 + 63) & 0xFFFFFFFFFFFFFFC0LL);
                }
                else
                {
                  v57 = (_QWORD *)sub_9D1E70(v56 + 8, 192, 192, 6);
                }
              }
              memset(v57, 0, 0xC0u);
              memset(
                (void *)((unsigned __int64)(v57 + 13) & 0xFFFFFFFFFFFFFFF8LL),
                0,
                8LL * (((unsigned int)v57 - (((_DWORD)v57 + 104) & 0xFFFFFFF8) + 192) >> 3));
LABEL_35:
              v150[v52] = v57;
              v134 = 1;
LABEL_36:
              v131 = sub_F03E60(v146, v50, 12, (__int64)v148, (__int64)v149, v43, 1u);
              v133 = v131;
              goto LABEL_37;
            }
            v52 = v137;
            v53 = v148[v137];
            v137 = 2;
            v136 = v52;
          }
          v55 = v137;
          v146 = v137 + 1;
          goto LABEL_32;
        }
        v11 = (v127 + 63) & 0xFFFFFFFFFFFFFFC0LL;
      }
      else
      {
        v11 = sub_9D1E70((__int64)(v10 + 1), 192, 192, 6);
      }
    }
    memset((void *)v11, 0, 0xC0u);
    v12 = v11 & 0xFFFFFFFFFFFFFFC0LL;
    memset(
      (void *)((v11 + 104) & 0xFFFFFFFFFFFFFFF8LL),
      0,
      8LL * (((unsigned int)v11 - (((_DWORD)v11 + 104) & 0xFFFFFFF8) + 192) >> 3));
    goto LABEL_6;
  }
  v37 = *(_DWORD *)(v7 + 12);
  v38 = v8 - 1;
  if ( v8 != v37 )
  {
    do
    {
      v39 = v38 + 1;
      *(_QWORD *)(v6 + 8 * v39 + 8) = *(_QWORD *)(v6 + 8LL * v38 + 8);
      *(_QWORD *)(v6 + 8 * v39 + 80) = *(_QWORD *)(v6 + 8LL * v38 + 80);
      LODWORD(v39) = v38--;
    }
    while ( v37 != (_DWORD)v39 );
  }
  v17 = 0;
  *(_QWORD *)(v6 + 8LL * v37 + 8) = a3;
  *(_QWORD *)(v6 + 8LL * v37 + 80) = a4;
  v40 = *(_DWORD *)(v6 + 164) + 1;
  *(_DWORD *)(v6 + 164) = v40;
  *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = v40;
  v41 = *(_QWORD *)(a1 + 8);
  v42 = *(_QWORD *)(*(_QWORD *)v41 + 8LL * *(unsigned int *)(v41 + 12));
  *(_QWORD *)(v41 + 16) = v42 & 0xFFFFFFFFFFFFFFC0LL;
  *(_DWORD *)(v41 + 24) = (v42 & 0x3F) + 1;
  return v17;
}
