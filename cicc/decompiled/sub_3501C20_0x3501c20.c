// Function: sub_3501C20
// Address: 0x3501c20
//
__int64 __fastcall sub_3501C20(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rax
  _DWORD *v10; // r13
  __int64 v11; // rdx
  _DWORD *v12; // r14
  _DWORD *v13; // r13
  __int64 v14; // r15
  __int64 v15; // r9
  __int64 v16; // rdx
  unsigned int v17; // esi
  __int64 *v18; // rcx
  int v19; // r10d
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r9
  __int64 v24; // r13
  int v25; // edx
  __int64 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // r10
  __int64 v30; // rdx
  unsigned __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r8
  unsigned int v34; // edx
  __int64 v35; // rax
  __int64 j; // r11
  __int64 *v37; // r8
  __int64 v38; // rsi
  unsigned __int64 v39; // r8
  unsigned int v40; // edx
  _QWORD *v41; // rax
  unsigned int *v42; // rdx
  unsigned int v43; // r12d
  __int64 v44; // r8
  __int64 *v45; // r14
  _QWORD *v46; // r8
  __int64 v47; // r11
  _QWORD *v48; // rsi
  unsigned int v49; // r11d
  __int64 *v50; // rax
  int v51; // edx
  __int64 v52; // rsi
  __int64 result; // rax
  __int64 *v54; // rax
  unsigned int *v55; // r15
  unsigned int *v56; // r10
  unsigned int *v57; // r15
  unsigned int *v58; // r12
  __int64 v59; // rbx
  unsigned __int64 v60; // r13
  __int64 v61; // rax
  __int64 v62; // rcx
  __int64 v63; // rax
  unsigned int v64; // edx
  __int64 v65; // rsi
  int v66; // r11d
  __int64 v67; // rax
  __int64 v68; // rdi
  unsigned int v69; // esi
  __int64 v70; // rdx
  unsigned int v71; // ecx
  __int64 v72; // rax
  __int64 v73; // r9
  char v74; // si
  __int64 v75; // rcx
  __int64 v76; // rax
  int v77; // edx
  unsigned int v78; // esi
  __int64 v79; // rcx
  __int64 v80; // rax
  __int64 v81; // r15
  unsigned __int64 v82; // r14
  __int64 v83; // r8
  __int64 *v84; // r9
  __int64 v85; // rdi
  __int64 *v86; // rax
  unsigned int v87; // esi
  __int64 v88; // rdx
  char v89; // di
  __int64 v90; // rsi
  __int64 v91; // rdi
  __int64 v92; // rax
  unsigned __int64 v93; // rdi
  unsigned int v94; // r9d
  int v95; // edx
  unsigned __int64 v96; // r15
  _DWORD *v97; // r14
  __int64 v98; // rdx
  __int64 v99; // r13
  unsigned int v101; // ebx
  __int64 v102; // rax
  __int64 v103; // rcx
  __int64 v104; // rsi
  int v105; // edi
  __int64 v106; // r9
  unsigned int i; // ecx
  __int64 v108; // r10
  __int64 *v109; // rax
  __int64 v110; // rdi
  __int64 v111; // rsi
  __int64 v112; // rax
  __int64 v113; // rcx
  __int64 v114; // rax
  unsigned int v115; // esi
  __int64 v116; // rax
  __int64 v117; // r10
  __int64 v118; // rdx
  int v119; // r11d
  __int64 v120; // rax
  __int64 v121; // rdx
  __int64 v122; // rcx
  bool v123; // zf
  __int64 v124; // [rsp+8h] [rbp-68h]
  __int64 v125; // [rsp+10h] [rbp-60h]
  _QWORD *v126; // [rsp+18h] [rbp-58h]
  __int64 *v127; // [rsp+18h] [rbp-58h]
  __int64 v128; // [rsp+18h] [rbp-58h]
  __int64 v129; // [rsp+18h] [rbp-58h]
  unsigned int v130; // [rsp+20h] [rbp-50h]
  _QWORD *v131; // [rsp+20h] [rbp-50h]
  __int64 v132; // [rsp+20h] [rbp-50h]
  __int64 v133; // [rsp+20h] [rbp-50h]
  __int64 v134; // [rsp+28h] [rbp-48h]
  __int64 v135; // [rsp+28h] [rbp-48h]
  __int64 v136; // [rsp+30h] [rbp-40h]
  __int64 v137; // [rsp+38h] [rbp-38h]
  unsigned int v138; // [rsp+38h] [rbp-38h]
  __int64 v139; // [rsp+38h] [rbp-38h]

  v5 = a2;
  v6 = a1;
  v7 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 152LL) + 16LL * a2);
  v8 = *v7;
  v134 = v7[1];
  v9 = *(_QWORD *)(a1 + 40);
  v136 = v8;
  if ( v9 != v8 )
  {
    v10 = *(_DWORD **)(a1 + 48);
    v11 = *(unsigned int *)(a1 + 56);
    if ( (v9 & 0xFFFFFFFFFFFFFFF8LL) != 0
      && (v96 = v8 & 0xFFFFFFFFFFFFFFF8LL,
          a5 = (v8 >> 1) & 3,
          (*(_DWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)a5) >= (*(_DWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL)
                                                                                           + 24)
                                                                               | (unsigned int)(v9 >> 1) & 3)) )
    {
      v97 = &v10[28 * v11];
      if ( v97 != v10 )
      {
        v98 = *(_QWORD *)(a1 + 48);
        v99 = v5;
        v101 = (v8 >> 1) & 3;
        do
        {
          v102 = *(unsigned int *)(v98 + 16);
          if ( (_DWORD)v102 )
          {
            v103 = *(_QWORD *)(v98 + 8);
            if ( *(_DWORD *)(v103 + 12) < *(_DWORD *)(v103 + 8) )
            {
              v104 = *(_QWORD *)v98;
              if ( *(_DWORD *)(*(_QWORD *)v98 + 192LL) )
              {
                v139 = v98;
                sub_2E1A970(v98, v136);
                v98 = v139;
              }
              else
              {
                v105 = *(_DWORD *)(v104 + 196);
                v106 = v103 + 16 * v102 - 16;
                for ( i = *(_DWORD *)(v106 + 12); v105 != i; ++i )
                {
                  v108 = *(_QWORD *)(v104 + 16LL * i + 8);
                  if ( (*(_DWORD *)((v108 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v108 >> 1) & 3) > (v101 | *(_DWORD *)(v96 + 24)) )
                    break;
                }
                *(_DWORD *)(v106 + 12) = i;
              }
            }
          }
          v109 = *(__int64 **)(v98 + 96);
          v110 = *(_QWORD *)(v98 + 104);
          v111 = *v109;
          v112 = 24LL * *((unsigned int *)v109 + 2);
          v113 = v111 + v112;
          if ( v110 != v111 + v112 )
          {
            v114 = *(_QWORD *)(v111 + v112 - 16);
            v115 = v101 | *(_DWORD *)(v96 + 24);
            if ( v115 < (*(_DWORD *)((v114 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v114 >> 1) & 3) )
            {
              v113 = *(_QWORD *)(v98 + 104);
              if ( v115 >= (*(_DWORD *)((*(_QWORD *)(v110 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                          | (unsigned int)(*(__int64 *)(v110 + 8) >> 1) & 3) )
              {
                do
                {
                  v116 = *(_QWORD *)(v113 + 32);
                  v113 += 24;
                }
                while ( v115 >= (*(_DWORD *)((v116 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v116 >> 1) & 3) );
              }
            }
            *(_QWORD *)(v98 + 104) = v113;
          }
          v98 += 112;
        }
        while ( v97 != (_DWORD *)v98 );
        v6 = a1;
        v5 = v99;
        *(_QWORD *)(v6 + 40) = v136;
        goto LABEL_17;
      }
    }
    else
    {
      v12 = &v10[28 * v11];
      if ( v12 != v10 )
      {
        v13 = v10 + 2;
        v14 = (v8 >> 1) & 3;
        while ( 1 )
        {
          v15 = *((_QWORD *)v13 - 1);
          v16 = *(unsigned int *)(v15 + 192);
          v17 = *(_DWORD *)(v15 + 196);
          if ( (_DWORD)v16 )
          {
            v117 = v15 + 8;
            if ( v17 )
            {
              v15 += 96;
              v118 = 0;
              do
              {
                if ( (*(_DWORD *)((*(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                    | (unsigned int)(*(__int64 *)v15 >> 1) & 3) > ((unsigned int)v14
                                                                 | *(_DWORD *)((v136 & 0xFFFFFFFFFFFFFFF8LL) + 24)) )
                  break;
                v118 = (unsigned int)(v118 + 1);
                v15 += 8;
              }
              while ( (_DWORD)v118 != v17 );
            }
            else
            {
              v118 = 0;
            }
            v119 = v13[3];
            v13[2] = 0;
            v120 = (v118 << 32) | v17;
            v121 = 0;
            if ( !v119 )
            {
              v129 = v120;
              v133 = v117;
              sub_C8D5F0((__int64)v13, v13 + 4, 1u, 0x10u, a5, v15);
              v120 = v129;
              v117 = v133;
              v121 = 16LL * (unsigned int)v13[2];
            }
            v122 = *(_QWORD *)v13;
            *(_QWORD *)(v122 + v121) = v117;
            *(_QWORD *)(v122 + v121 + 8) = v120;
            v123 = v13[2]++ == -1;
            if ( !v123 && *(_DWORD *)(*(_QWORD *)v13 + 12LL) < *(_DWORD *)(*(_QWORD *)v13 + 8LL) )
              sub_2E1A640((__int64)(v13 - 2), v136, v121, v122, a5);
          }
          else
          {
            if ( v17 )
            {
              v18 = (__int64 *)(v15 + 8);
              do
              {
                if ( (*(_DWORD *)((*v18 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v18 >> 1) & 3) > ((unsigned int)v14 | *(_DWORD *)((v136 & 0xFFFFFFFFFFFFFFF8LL) + 24)) )
                  break;
                v16 = (unsigned int)(v16 + 1);
                v18 += 2;
              }
              while ( (_DWORD)v16 != v17 );
            }
            v19 = v13[3];
            v13[2] = 0;
            v20 = (v16 << 32) | v17;
            v21 = 0;
            if ( !v19 )
            {
              v128 = v20;
              v132 = v15;
              sub_C8D5F0((__int64)v13, v13 + 4, 1u, 0x10u, a5, v15);
              v20 = v128;
              v15 = v132;
              v21 = 16LL * (unsigned int)v13[2];
            }
            v22 = *(_QWORD *)v13;
            *(_QWORD *)(v22 + v21) = v15;
            *(_QWORD *)(v22 + v21 + 8) = v20;
            ++v13[2];
          }
          *((_QWORD *)v13 + 12) = sub_2E09D00(*((__int64 **)v13 + 11), v136);
          if ( v12 == v13 + 26 )
            break;
          v13 += 28;
        }
        v6 = a1;
      }
    }
    *(_QWORD *)(v6 + 40) = v136;
  }
LABEL_17:
  v23 = v134;
  v24 = *(_QWORD *)(v6 + 512) + 24 * v5;
  v25 = *(_DWORD *)(v6 + 4);
  v137 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v6 + 16) + 96LL) + 8 * v5);
  while ( 1 )
  {
    *(_DWORD *)v24 = v25;
    *(_QWORD *)(v24 + 16) = 0;
    *(_QWORD *)(v24 + 8) = 0;
    v26 = *(_QWORD *)(v6 + 48);
    v27 = v26 + 112LL * *(unsigned int *)(v6 + 56);
    if ( v26 == v27 )
    {
      v31 = 0;
    }
    else
    {
      v28 = 0;
      v29 = (v23 >> 1) & 3;
      do
      {
        v30 = *(unsigned int *)(v26 + 16);
        v31 = v28 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (_DWORD)v30 )
        {
          v32 = *(_QWORD *)(v26 + 8);
          if ( *(_DWORD *)(v32 + 12) < *(_DWORD *)(v32 + 8) )
          {
            v33 = *(_QWORD *)(*(_QWORD *)(v32 + 16 * v30 - 16) + 16LL * *(unsigned int *)(v32 + 16 * v30 - 16 + 12));
            v34 = *(_DWORD *)((v33 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v33 >> 1) & 3;
            if ( v34 < ((unsigned int)v29 | *(_DWORD *)((v23 & 0xFFFFFFFFFFFFFFF8LL) + 24))
              && (!v31 || v34 < (*(_DWORD *)(v31 + 24) | (unsigned int)(v28 >> 1) & 3)) )
            {
              *(_QWORD *)(v24 + 8) = v33;
              v31 = v33 & 0xFFFFFFFFFFFFFFF8LL;
              v28 = v33;
            }
          }
        }
        v26 += 112;
      }
      while ( v26 != v27 );
      v35 = *(_QWORD *)(v6 + 48);
      for ( j = v35 + 112LL * *(unsigned int *)(v6 + 56); j != v35; v35 += 112 )
      {
        v37 = *(__int64 **)(v35 + 104);
        v31 = v28 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v37 != (__int64 *)(**(_QWORD **)(v35 + 96) + 24LL * *(unsigned int *)(*(_QWORD *)(v35 + 96) + 8LL)) )
        {
          v38 = *v37;
          v39 = *v37 & 0xFFFFFFFFFFFFFFF8LL;
          v40 = *(_DWORD *)(v39 + 24) | (v38 >> 1) & 3;
          if ( v40 < ((unsigned int)v29 | *(_DWORD *)((v23 & 0xFFFFFFFFFFFFFFF8LL) + 24))
            && (!v31 || v40 < (*(_DWORD *)(v31 + 24) | (unsigned int)(v28 >> 1) & 3)) )
          {
            *(_QWORD *)(v24 + 8) = v38;
            v31 = v39;
            v28 = v38;
          }
        }
      }
    }
    v41 = *(_QWORD **)(v6 + 32);
    v42 = (unsigned int *)(v41[43] + 8 * v5);
    v43 = v42[1];
    v44 = 8LL * *v42;
    v45 = (__int64 *)(v44 + v41[23]);
    v46 = (_QWORD *)(v41[33] + v44);
    v47 = v31 ? *(_QWORD *)(v24 + 8) : v23;
    if ( v43 )
      break;
LABEL_41:
    *(_QWORD *)(v6 + 40) = v23;
    if ( (*(_QWORD *)(v24 + 8) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      goto LABEL_47;
LABEL_42:
    v52 = *(_QWORD *)(v137 + 8);
    result = *(_QWORD *)(v6 + 16) + 320LL;
    v137 = v52;
    if ( v52 == result )
      return result;
    v5 = *(unsigned int *)(v52 + 24);
    result = *(_QWORD *)(v6 + 512);
    v24 = result + 24 * v5;
    v25 = *(_DWORD *)(v6 + 4);
    if ( *(_DWORD *)v24 == v25 )
      return result;
    v54 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(v6 + 24) + 152LL) + 16 * v5);
    v23 = v54[1];
    v136 = *v54;
  }
  v48 = v46;
  v49 = *(_DWORD *)((v47 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v47 >> 1) & 3;
  v50 = v45;
  while ( 1 )
  {
    if ( (*(_DWORD *)((*v50 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v50 >> 1) & 3) >= v49 )
      goto LABEL_41;
    v51 = *(_DWORD *)(*v48 + 4LL * (*(_DWORD *)v6 >> 5));
    if ( !_bittest(&v51, *(_DWORD *)v6) )
      break;
    ++v50;
    ++v48;
    if ( &v45[v43] == v50 )
      goto LABEL_41;
  }
  *(_QWORD *)(v24 + 8) = *v50;
  *(_QWORD *)(v6 + 40) = v23;
  if ( (*(_QWORD *)(v24 + 8) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_42;
LABEL_47:
  v55 = *(unsigned int **)(v6 + 48);
  v135 = v23;
  v56 = &v55[28 * *(unsigned int *)(v6 + 56)];
  if ( v56 != v55 )
  {
    v126 = v46;
    v57 = v55 + 2;
    v130 = v43;
    v58 = v56;
    v125 = v6;
    v59 = v24;
    v124 = (v23 >> 1) & 3;
    v138 = (v23 >> 1) & 3;
    v60 = v23 & 0xFFFFFFFFFFFFFFF8LL;
    while ( 2 )
    {
      v61 = v57[2];
      if ( !(_DWORD)v61 )
        goto LABEL_70;
      if ( *(_DWORD *)(*(_QWORD *)v57 + 12LL) >= *(_DWORD *)(*(_QWORD *)v57 + 8LL) )
        goto LABEL_70;
      v62 = *(_QWORD *)v57 + 16 * v61 - 16;
      v63 = *(unsigned int *)(v62 + 12);
      v64 = *(_DWORD *)(v62 + 12);
      if ( (*(_DWORD *)((*(_QWORD *)(*(_QWORD *)v62 + 16 * v63) & 0xFFFFFFFFFFFFFFF8LL) + 24)
          | (unsigned int)(*(__int64 *)(*(_QWORD *)v62 + 16 * v63) >> 1) & 3) >= (*(_DWORD *)(v60 + 24) | v138) )
        goto LABEL_70;
      v65 = *((_QWORD *)v57 - 1);
      if ( *(_DWORD *)(v65 + 192) )
      {
        sub_2E1A970((__int64)(v57 - 2), v135);
      }
      else
      {
        v66 = *(_DWORD *)(v65 + 196);
        if ( v66 != (_DWORD)v63 )
        {
          while ( (*(_DWORD *)((*(_QWORD *)(v65 + 16 * v63 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                 | (unsigned int)(*(__int64 *)(v65 + 16 * v63 + 8) >> 1) & 3) <= (*(_DWORD *)(v60 + 24)
                                                                                | (unsigned int)v124) )
          {
            if ( v66 == ++v64 )
              break;
            v63 = v64;
          }
        }
        *(_DWORD *)(v62 + 12) = v64;
      }
      v67 = v57[2];
      v68 = *(_QWORD *)v57;
      v69 = *(_DWORD *)(*((_QWORD *)v57 - 1) + 192LL);
      if ( (_DWORD)v67 )
      {
        v70 = v68 + 16 * v67 - 16;
        v71 = *(_DWORD *)(v70 + 12);
        if ( *(_DWORD *)(v68 + 12) < *(_DWORD *)(v68 + 8) )
        {
          v72 = v71;
          v73 = *(_QWORD *)(*(_QWORD *)v70 + 16LL * v71);
          if ( (*(_DWORD *)((v73 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v73 >> 1) & 3) < (*(_DWORD *)(v60 + 24)
                                                                                                | v138) )
          {
            v74 = 0;
LABEL_63:
            v75 = *(_QWORD *)(*(_QWORD *)v70 + 16 * v72 + 8);
            if ( (*(_QWORD *)(v59 + 16) & 0xFFFFFFFFFFFFFFF8LL) == 0
              || (*(_DWORD *)((v75 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v75 >> 1) & 3) > (*(_DWORD *)((*(_QWORD *)(v59 + 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                  | (unsigned int)(*(__int64 *)(v59 + 16) >> 1) & 3) )
            {
              *(_QWORD *)(v59 + 16) = v75;
            }
            if ( v74 )
            {
              v76 = *(_QWORD *)v57 + 16LL * v57[2] - 16;
              v77 = *(_DWORD *)(v76 + 12) + 1;
              *(_DWORD *)(v76 + 12) = v77;
              if ( v77 == *(_DWORD *)(*(_QWORD *)v57 + 16LL * v57[2] - 8) )
              {
                v78 = *(_DWORD *)(*((_QWORD *)v57 - 1) + 192LL);
                if ( v78 )
                  sub_F03D40((__int64 *)v57, v78);
              }
            }
LABEL_70:
            if ( v58 == v57 + 26 )
            {
              v24 = v59;
              v6 = v125;
              v43 = v130;
              v46 = v126;
              v79 = *(_QWORD *)(v125 + 48);
              v80 = 112LL * *(unsigned int *)(v125 + 56);
              v81 = v79 + v80;
              if ( v79 + v80 != v79 )
              {
                v131 = v126;
                v127 = v45;
                v82 = v135 & 0xFFFFFFFFFFFFFFF8LL;
                do
                {
                  v83 = *(_QWORD *)(v79 + 96);
                  v84 = *(__int64 **)(v79 + 104);
                  v85 = 24LL * *(unsigned int *)(v83 + 8);
                  v86 = (__int64 *)(*(_QWORD *)v83 + v85);
                  if ( v84 != v86 )
                  {
                    v87 = v138 | *(_DWORD *)(v82 + 24);
                    if ( (*(_DWORD *)((*v84 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v84 >> 1) & 3) < v87 )
                    {
                      if ( v87 < (*(_DWORD *)((*(_QWORD *)(*(_QWORD *)v83 + v85 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                | (unsigned int)(*(__int64 *)(*(_QWORD *)v83 + v85 - 16) >> 1) & 3) )
                      {
                        v86 = *(__int64 **)(v79 + 104);
                        if ( (*(_DWORD *)((v84[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v84[1] >> 1) & 3) <= v87 )
                        {
                          do
                          {
                            v88 = v86[4];
                            v86 += 3;
                          }
                          while ( v87 >= (*(_DWORD *)((v88 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v88 >> 1) & 3) );
                        }
                      }
                      *(_QWORD *)(v79 + 104) = v86;
                      if ( v86 == (__int64 *)(*(_QWORD *)v83 + 24LL * *(unsigned int *)(v83 + 8))
                        || (v89 = 0,
                            (*(_DWORD *)((*v86 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v86 >> 1) & 3) >= (v138 | *(_DWORD *)(v82 + 24))) )
                      {
                        v86 -= 3;
                        v89 = 1;
                        *(_QWORD *)(v79 + 104) = v86;
                      }
                      v90 = v86[1];
                      if ( (*(_QWORD *)(v24 + 16) & 0xFFFFFFFFFFFFFFF8LL) == 0
                        || (*(_DWORD *)((v90 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v90 >> 1) & 3) > (*(_DWORD *)((*(_QWORD *)(v24 + 16) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)(v24 + 16) >> 1) & 3) )
                      {
                        *(_QWORD *)(v24 + 16) = v90;
                      }
                      if ( v89 )
                        *(_QWORD *)(v79 + 104) += 24LL;
                    }
                  }
                  v79 += 112;
                }
                while ( v79 != v81 );
                v46 = v131;
                v45 = v127;
              }
              goto LABEL_90;
            }
            v57 += 28;
            continue;
          }
          if ( !v71 )
          {
LABEL_129:
            sub_F03AD0(v57, v69);
            goto LABEL_127;
          }
LABEL_126:
          *(_DWORD *)(v70 + 12) = v71 - 1;
LABEL_127:
          v74 = 1;
          v70 = *(_QWORD *)v57 + 16LL * v57[2] - 16;
          v72 = *(unsigned int *)(v70 + 12);
          goto LABEL_63;
        }
        if ( !v71 )
          goto LABEL_129;
      }
      else
      {
        v71 = *(_DWORD *)(v68 - 4);
        v70 = v68 - 16;
        if ( !v71 )
          goto LABEL_129;
      }
      break;
    }
    if ( v69 )
      goto LABEL_129;
    goto LABEL_126;
  }
LABEL_90:
  v91 = *(_QWORD *)(v24 + 16);
  if ( (v91 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    v91 = v136;
  v92 = v91;
  v93 = v91 & 0xFFFFFFFFFFFFFFF8LL;
  result = v92 >> 1;
  v94 = result & 3;
  while ( v43 )
  {
    result = v45[--v43] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_DWORD *)(result + 24) | 3u) <= (v94 | *(_DWORD *)(v93 + 24)) )
      break;
    v95 = *(_DWORD *)(v46[v43] + 4LL * (*(_DWORD *)v6 >> 5));
    if ( !_bittest(&v95, *(_DWORD *)v6) )
    {
      result |= 6uLL;
      *(_QWORD *)(v24 + 16) = result;
      return result;
    }
  }
  return result;
}
