// Function: sub_20F85B0
// Address: 0x20f85b0
//
__int64 __fastcall sub_20F85B0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, unsigned __int64 a5, int a6)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 *v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v13; // r15
  __int64 v14; // r13
  __int64 v15; // r10
  __int64 v16; // rdx
  unsigned int v17; // esi
  __int64 *v18; // rcx
  int v19; // r9d
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r13
  int v25; // edx
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rdx
  __int64 v30; // r11
  __int64 v31; // rdi
  unsigned __int64 v32; // rsi
  __int64 v33; // r9
  __int64 v34; // rcx
  __int64 v35; // r9
  __int64 v36; // r9
  unsigned int v37; // ecx
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r14
  __int64 v41; // rdx
  __int64 *v42; // r9
  __int64 v43; // rdi
  unsigned __int64 v44; // r9
  unsigned int v45; // edx
  _QWORD *v46; // rdx
  unsigned int *v47; // rcx
  unsigned int v48; // r12d
  __int64 v49; // r9
  __int64 *v50; // r14
  _QWORD *v51; // r9
  _QWORD *v52; // rdi
  unsigned int v53; // eax
  __int64 *v54; // rdx
  int v55; // ecx
  __int64 v56; // rdi
  __int64 result; // rax
  __int64 *v58; // rax
  __int64 v59; // r15
  __int64 v60; // r12
  __int64 v61; // r14
  __int64 v62; // r13
  __int64 *v63; // rbx
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rsi
  __int64 v67; // rax
  unsigned int v68; // edx
  unsigned int v69; // edi
  __int64 v70; // r8
  int v71; // r11d
  __int64 v72; // rdx
  __int64 v73; // rax
  unsigned int v74; // r8d
  __int64 v75; // rdx
  unsigned int v76; // esi
  __int64 v77; // rax
  __int64 v78; // r10
  char v79; // di
  __int64 v80; // rsi
  __int64 v81; // rax
  int v82; // edx
  __int64 v83; // rsi
  __int64 v84; // r8
  unsigned __int64 v85; // r15
  __int64 v86; // rdi
  __int64 v87; // r9
  __int64 *v88; // r10
  __int64 v89; // rsi
  unsigned int v90; // ecx
  __int64 v91; // rdx
  char v92; // si
  __int64 v93; // rcx
  __int64 v94; // rdi
  __int64 v95; // r8
  unsigned __int64 v96; // rdi
  unsigned int v97; // r8d
  int v98; // edx
  unsigned __int64 v99; // r15
  __int64 v100; // r14
  __int64 v101; // r11
  __int64 v102; // rdx
  unsigned int v103; // r12d
  __int64 v104; // r13
  __int64 v105; // rcx
  __int64 v106; // rax
  __int64 v107; // rsi
  __int64 v108; // rdi
  int v109; // r8d
  __int64 v110; // r10
  unsigned int i; // esi
  __int64 v112; // r11
  __int64 *v113; // rax
  __int64 v114; // r8
  __int64 v115; // rdi
  __int64 v116; // rax
  __int64 v117; // rsi
  __int64 v118; // rax
  unsigned int v119; // edi
  __int64 v120; // rax
  __int64 v121; // rax
  __int64 *v122; // r10
  __int64 v123; // rcx
  int v124; // r11d
  __int64 v125; // rdx
  __int64 v126; // r10
  __int64 v127; // rcx
  bool v128; // zf
  __int64 *v129; // [rsp+8h] [rbp-68h]
  _QWORD *v130; // [rsp+10h] [rbp-60h]
  __int64 v131; // [rsp+10h] [rbp-60h]
  __int64 v132; // [rsp+10h] [rbp-60h]
  unsigned int v133; // [rsp+18h] [rbp-58h]
  __int64 v134; // [rsp+18h] [rbp-58h]
  __int64 v135; // [rsp+18h] [rbp-58h]
  __int64 v136; // [rsp+20h] [rbp-50h]
  unsigned int v137; // [rsp+20h] [rbp-50h]
  __int64 v138; // [rsp+20h] [rbp-50h]
  __int64 v139; // [rsp+28h] [rbp-48h]
  __int64 v140; // [rsp+28h] [rbp-48h]
  __int64 v141; // [rsp+30h] [rbp-40h]
  __int64 v142; // [rsp+38h] [rbp-38h]
  __int64 v143; // [rsp+38h] [rbp-38h]
  unsigned __int64 v144; // [rsp+38h] [rbp-38h]
  __int64 v145; // [rsp+38h] [rbp-38h]
  __int64 v146; // [rsp+38h] [rbp-38h]

  v6 = a2;
  v7 = a1;
  v8 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 392LL) + 16LL * a2);
  v9 = *v8;
  v139 = v8[1];
  v10 = *(_QWORD *)(v7 + 40);
  v141 = v9;
  if ( v10 != v9 )
  {
    v11 = *(unsigned int *)(v7 + 56);
    if ( (v10 & 0xFFFFFFFFFFFFFFF8LL) != 0
      && (v99 = v9 & 0xFFFFFFFFFFFFFFF8LL,
          a5 = (v9 >> 1) & 3,
          (*(_DWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)a5) >= (*(_DWORD *)((v10 & 0xFFFFFFFFFFFFFFF8LL)
                                                                                           + 24)
                                                                               | (unsigned int)(v10 >> 1) & 3)) )
    {
      if ( (_DWORD)v11 )
      {
        v100 = 0;
        v101 = 7 * v11;
        v102 = a2;
        v103 = (v9 >> 1) & 3;
        v104 = 16 * v101;
        do
        {
          v105 = v100 + *(_QWORD *)(v7 + 48);
          v106 = *(unsigned int *)(v105 + 16);
          if ( (_DWORD)v106 )
          {
            v107 = *(_QWORD *)(v105 + 8);
            if ( *(_DWORD *)(v107 + 12) < *(_DWORD *)(v107 + 8) )
            {
              v108 = *(_QWORD *)v105;
              if ( *(_DWORD *)(*(_QWORD *)v105 + 192LL) )
              {
                v138 = v102;
                v146 = v100 + *(_QWORD *)(v7 + 48);
                sub_20F82D0(v146, v141);
                v105 = v146;
                v102 = v138;
              }
              else
              {
                v109 = *(_DWORD *)(v108 + 196);
                v110 = v107 + 16 * v106 - 16;
                for ( i = *(_DWORD *)(v110 + 12); v109 != i; ++i )
                {
                  v112 = *(_QWORD *)(v108 + 16LL * i + 8);
                  if ( (*(_DWORD *)((v112 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v112 >> 1) & 3) > (v103 | *(_DWORD *)(v99 + 24)) )
                    break;
                }
                *(_DWORD *)(v110 + 12) = i;
              }
            }
          }
          v113 = *(__int64 **)(v105 + 96);
          v114 = *(_QWORD *)(v105 + 104);
          v115 = *v113;
          v116 = 24LL * *((unsigned int *)v113 + 2);
          v117 = v115 + v116;
          if ( v114 != v115 + v116 )
          {
            v118 = *(_QWORD *)(v115 + v116 - 16);
            v119 = v103 | *(_DWORD *)(v99 + 24);
            if ( v119 < (*(_DWORD *)((v118 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v118 >> 1) & 3) )
            {
              v117 = *(_QWORD *)(v105 + 104);
              if ( (*(_DWORD *)((*(_QWORD *)(v114 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                  | (unsigned int)(*(__int64 *)(v114 + 8) >> 1) & 3) <= v119 )
              {
                do
                {
                  v120 = *(_QWORD *)(v117 + 32);
                  v117 += 24;
                }
                while ( v119 >= (*(_DWORD *)((v120 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v120 >> 1) & 3) );
              }
            }
            *(_QWORD *)(v105 + 104) = v117;
          }
          v100 += 112;
        }
        while ( v104 != v100 );
        v6 = v102;
      }
    }
    else if ( (_DWORD)v11 )
    {
      v136 = a2;
      v142 = 112 * v11;
      v12 = (v9 >> 1) & 3;
      v13 = 0;
      do
      {
        v14 = v13 + *(_QWORD *)(v7 + 48);
        v15 = *(_QWORD *)v14;
        v16 = *(unsigned int *)(*(_QWORD *)v14 + 192LL);
        v17 = *(_DWORD *)(*(_QWORD *)v14 + 196LL);
        if ( (_DWORD)v16 )
        {
          v121 = v15 + 8;
          if ( v17 )
          {
            v122 = (__int64 *)(v15 + 96);
            v123 = 0;
            do
            {
              a5 = *v122 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_DWORD *)(a5 + 24) | (unsigned int)(*v122 >> 1) & 3) > ((unsigned int)v12
                                                                            | *(_DWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL)
                                                                                        + 24)) )
                break;
              v123 = (unsigned int)(v123 + 1);
              ++v122;
            }
            while ( (_DWORD)v123 != v17 );
          }
          else
          {
            v123 = 0;
          }
          v124 = *(_DWORD *)(v14 + 20);
          v125 = 0;
          *(_DWORD *)(v14 + 16) = 0;
          v126 = (v123 << 32) | v17;
          if ( !v124 )
          {
            v132 = (v123 << 32) | v17;
            v135 = v121;
            sub_16CD150(v14 + 8, (const void *)(v14 + 24), 0, 16, a5, a6);
            v126 = v132;
            v121 = v135;
            v125 = 16LL * *(unsigned int *)(v14 + 16);
          }
          v127 = *(_QWORD *)(v14 + 8);
          *(_QWORD *)(v127 + v125) = v121;
          *(_QWORD *)(v127 + v125 + 8) = v126;
          v128 = (*(_DWORD *)(v14 + 16))++ == -1;
          if ( !v128 && *(_DWORD *)(*(_QWORD *)(v14 + 8) + 12LL) < *(_DWORD *)(*(_QWORD *)(v14 + 8) + 8LL) )
            sub_1EC1C10(v14, v9, v125, v127, a5);
        }
        else
        {
          if ( v17 )
          {
            v18 = (__int64 *)(v15 + 8);
            do
            {
              a5 = *v18 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_DWORD *)(a5 + 24) | (unsigned int)(*v18 >> 1) & 3) > ((unsigned int)v12
                                                                           | *(_DWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 24)) )
                break;
              v16 = (unsigned int)(v16 + 1);
              v18 += 2;
            }
            while ( (_DWORD)v16 != v17 );
          }
          v19 = *(_DWORD *)(v14 + 20);
          *(_DWORD *)(v14 + 16) = 0;
          v20 = (v16 << 32) | v17;
          v21 = 0;
          if ( !v19 )
          {
            v131 = v20;
            v134 = v15;
            sub_16CD150(v14 + 8, (const void *)(v14 + 24), 0, 16, a5, 0);
            v20 = v131;
            v15 = v134;
            v21 = 16LL * *(unsigned int *)(v14 + 16);
          }
          v22 = *(_QWORD *)(v14 + 8);
          *(_QWORD *)(v22 + v21) = v15;
          *(_QWORD *)(v22 + v21 + 8) = v20;
          ++*(_DWORD *)(v14 + 16);
        }
        v13 += 112;
        *(_QWORD *)(v14 + 104) = sub_1DB3C70(*(__int64 **)(v14 + 96), v9);
      }
      while ( v142 != v13 );
      v6 = v136;
    }
    *(_QWORD *)(v7 + 40) = v141;
  }
  v23 = v139;
  v24 = *(_QWORD *)(v7 + 512) + 24 * v6;
  v25 = *(_DWORD *)(v7 + 4);
  v143 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 16) + 96LL) + 8 * v6);
  while ( 1 )
  {
    *(_DWORD *)v24 = v25;
    *(_QWORD *)(v24 + 16) = 0;
    *(_QWORD *)(v24 + 8) = 0;
    v26 = *(unsigned int *)(v7 + 56);
    if ( (_DWORD)v26 )
    {
      v27 = 0;
      v28 = 7 * v26;
      v29 = 0;
      v30 = (v23 >> 1) & 3;
      v31 = 16 * v28;
      do
      {
        v32 = v27 & 0xFFFFFFFFFFFFFFF8LL;
        v33 = v29 + *(_QWORD *)(v7 + 48);
        v34 = *(unsigned int *)(v33 + 16);
        if ( (_DWORD)v34 )
        {
          v35 = *(_QWORD *)(v33 + 8);
          if ( *(_DWORD *)(v35 + 12) < *(_DWORD *)(v35 + 8) )
          {
            v36 = *(_QWORD *)(*(_QWORD *)(v35 + 16 * v34 - 16) + 16LL * *(unsigned int *)(v35 + 16 * v34 - 16 + 12));
            v37 = *(_DWORD *)((v36 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v36 >> 1) & 3;
            if ( v37 < ((unsigned int)v30 | *(_DWORD *)((v23 & 0xFFFFFFFFFFFFFFF8LL) + 24))
              && (!v32 || v37 < (*(_DWORD *)(v32 + 24) | (unsigned int)(v27 >> 1) & 3)) )
            {
              *(_QWORD *)(v24 + 8) = v36;
              v32 = v36 & 0xFFFFFFFFFFFFFFF8LL;
              v27 = v36;
            }
          }
        }
        v29 += 112;
      }
      while ( v31 != v29 );
      v38 = *(unsigned int *)(v7 + 56);
      if ( (_DWORD)v38 )
      {
        v39 = 0;
        v40 = 112 * v38;
        do
        {
          v32 = v27 & 0xFFFFFFFFFFFFFFF8LL;
          v41 = v39 + *(_QWORD *)(v7 + 48);
          v42 = *(__int64 **)(v41 + 104);
          if ( v42 != (__int64 *)(**(_QWORD **)(v41 + 96) + 24LL * *(unsigned int *)(*(_QWORD *)(v41 + 96) + 8LL)) )
          {
            v43 = *v42;
            v44 = *v42 & 0xFFFFFFFFFFFFFFF8LL;
            v45 = *(_DWORD *)(v44 + 24) | (v43 >> 1) & 3;
            if ( v45 < ((unsigned int)v30 | *(_DWORD *)((v23 & 0xFFFFFFFFFFFFFFF8LL) + 24))
              && (!v32 || v45 < (*(_DWORD *)(v32 + 24) | (unsigned int)(v27 >> 1) & 3)) )
            {
              *(_QWORD *)(v24 + 8) = v43;
              v32 = v44;
              v27 = v43;
            }
          }
          v39 += 112;
        }
        while ( v40 != v39 );
      }
    }
    else
    {
      v27 = 0;
      v32 = 0;
    }
    v46 = *(_QWORD **)(v7 + 32);
    v47 = (unsigned int *)(v46[74] + 8 * v6);
    v48 = v47[1];
    v49 = 8LL * *v47;
    v50 = (__int64 *)(v49 + v46[54]);
    v51 = (_QWORD *)(v46[64] + v49);
    if ( !v32 )
      v27 = v23;
    if ( v48 )
      break;
LABEL_41:
    *(_QWORD *)(v7 + 40) = v23;
    if ( (*(_QWORD *)(v24 + 8) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      goto LABEL_46;
LABEL_42:
    v56 = *(_QWORD *)(v143 + 8);
    result = *(_QWORD *)(v7 + 16) + 320LL;
    v143 = v56;
    if ( v56 == result )
      return result;
    v6 = *(unsigned int *)(v56 + 48);
    result = *(_QWORD *)(v7 + 512);
    v24 = result + 24 * v6;
    v25 = *(_DWORD *)(v7 + 4);
    if ( *(_DWORD *)v24 == v25 )
      return result;
    v58 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(v7 + 24) + 392LL) + 16 * v6);
    v23 = v58[1];
    v141 = *v58;
  }
  v52 = v51;
  v53 = *(_DWORD *)((v27 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v27 >> 1) & 3;
  v54 = v50;
  while ( 1 )
  {
    if ( (*(_DWORD *)((*v54 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v54 >> 1) & 3) >= v53 )
      goto LABEL_41;
    v55 = *(_DWORD *)(*v52 + 4LL * (*(_DWORD *)v7 >> 5));
    if ( !_bittest(&v55, *(_DWORD *)v7) )
      break;
    ++v54;
    ++v52;
    if ( &v50[v48] == v54 )
      goto LABEL_41;
  }
  *(_QWORD *)(v24 + 8) = *v54;
  *(_QWORD *)(v7 + 40) = v23;
  if ( (*(_QWORD *)(v24 + 8) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_42;
LABEL_46:
  result = *(unsigned int *)(v7 + 56);
  v140 = v23;
  if ( (_DWORD)result )
  {
    v130 = v51;
    v133 = v48;
    v59 = 0;
    v60 = v24;
    v129 = v50;
    v144 = v23 & 0xFFFFFFFFFFFFFFF8LL;
    v61 = v7;
    v62 = 112 * result;
    v137 = (v23 >> 1) & 3;
    while ( 1 )
    {
      v63 = (__int64 *)(v59 + *(_QWORD *)(v61 + 48));
      v64 = *((unsigned int *)v63 + 4);
      if ( (_DWORD)v64 )
      {
        v65 = v63[1];
        if ( *(_DWORD *)(v65 + 12) < *(_DWORD *)(v65 + 8) )
        {
          v66 = v65 + 16 * v64 - 16;
          v67 = *(unsigned int *)(v66 + 12);
          v68 = *(_DWORD *)(v66 + 12);
          v69 = *(_DWORD *)(v144 + 24) | v137;
          if ( (*(_DWORD *)((*(_QWORD *)(*(_QWORD *)v66 + 16 * v67) & 0xFFFFFFFFFFFFFFF8LL) + 24)
              | (unsigned int)(*(__int64 *)(*(_QWORD *)v66 + 16 * v67) >> 1) & 3) < v69 )
            break;
        }
      }
LABEL_65:
      v59 += 112;
      if ( v62 == v59 )
      {
        v7 = v61;
        v24 = v60;
        v51 = v130;
        v48 = v133;
        result = *(unsigned int *)(v61 + 56);
        v50 = v129;
        if ( (_DWORD)result )
        {
          v84 = 0;
          v85 = v140 & 0xFFFFFFFFFFFFFFF8LL;
          v145 = 112 * result;
          do
          {
            v86 = v84 + *(_QWORD *)(v7 + 48);
            v87 = *(_QWORD *)(v86 + 96);
            v88 = *(__int64 **)(v86 + 104);
            v89 = 24LL * *(unsigned int *)(v87 + 8);
            result = *(_QWORD *)v87 + v89;
            if ( v88 != (__int64 *)result )
            {
              v90 = v137 | *(_DWORD *)(v85 + 24);
              if ( (*(_DWORD *)((*v88 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v88 >> 1) & 3) < v90 )
              {
                if ( v90 < (*(_DWORD *)((*(_QWORD *)(*(_QWORD *)v87 + v89 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                          | (unsigned int)(*(__int64 *)(*(_QWORD *)v87 + v89 - 16) >> 1) & 3) )
                {
                  result = *(_QWORD *)(v86 + 104);
                  if ( (*(_DWORD *)((v88[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v88[1] >> 1) & 3) <= v90 )
                  {
                    do
                    {
                      v91 = *(_QWORD *)(result + 32);
                      result += 24;
                    }
                    while ( v90 >= (*(_DWORD *)((v91 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v91 >> 1) & 3) );
                  }
                }
                *(_QWORD *)(v86 + 104) = result;
                if ( result == *(_QWORD *)v87 + 24LL * *(unsigned int *)(v87 + 8)
                  || (v92 = 0,
                      (*(_DWORD *)((*(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL) + 24)
                     | (unsigned int)(*(__int64 *)result >> 1) & 3) >= (v137 | *(_DWORD *)(v85 + 24))) )
                {
                  result -= 24;
                  v92 = 1;
                  *(_QWORD *)(v86 + 104) = result;
                }
                v93 = *(_QWORD *)(result + 8);
                result = *(_QWORD *)(v24 + 16);
                if ( (result & 0xFFFFFFFFFFFFFFF8LL) == 0
                  || (result = *(_DWORD *)((*(_QWORD *)(v24 + 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                             | (unsigned int)(result >> 1) & 3,
                      (*(_DWORD *)((v93 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v93 >> 1) & 3) > (unsigned int)result) )
                {
                  *(_QWORD *)(v24 + 16) = v93;
                }
                if ( v92 )
                  *(_QWORD *)(v86 + 104) += 24LL;
              }
            }
            v84 += 112;
          }
          while ( v145 != v84 );
          v51 = v130;
          v50 = v129;
        }
        goto LABEL_83;
      }
    }
    v70 = *v63;
    if ( *(_DWORD *)(*v63 + 192) )
    {
      sub_20F82D0(v59 + *(_QWORD *)(v61 + 48), v140);
    }
    else
    {
      v71 = *(_DWORD *)(v70 + 196);
      if ( v71 != (_DWORD)v67 )
      {
        while ( v69 >= (*(_DWORD *)((*(_QWORD *)(v70 + 16 * v67 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                      | (unsigned int)(*(__int64 *)(v70 + 16 * v67 + 8) >> 1) & 3) )
        {
          if ( v71 == ++v68 )
            break;
          v67 = v68;
        }
      }
      *(_DWORD *)(v66 + 12) = v68;
    }
    v72 = *((unsigned int *)v63 + 4);
    v73 = v63[1];
    v74 = *(_DWORD *)(*v63 + 192);
    if ( (_DWORD)v72 )
    {
      v75 = v73 + 16 * v72 - 16;
      v76 = *(_DWORD *)(v75 + 12);
      if ( *(_DWORD *)(v73 + 12) < *(_DWORD *)(v73 + 8) )
      {
        v77 = v76;
        v78 = *(_QWORD *)(*(_QWORD *)v75 + 16LL * v76);
        if ( (*(_DWORD *)((v78 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v78 >> 1) & 3) < (*(_DWORD *)(v144 + 24)
                                                                                              | v137) )
        {
          v79 = 0;
          goto LABEL_58;
        }
        if ( !v76 )
        {
LABEL_127:
          sub_3945E40(v63 + 1, v74);
          goto LABEL_125;
        }
LABEL_124:
        *(_DWORD *)(v75 + 12) = v76 - 1;
LABEL_125:
        v79 = 1;
        v75 = v63[1] + 16LL * *((unsigned int *)v63 + 4) - 16;
        v77 = *(unsigned int *)(v75 + 12);
LABEL_58:
        v80 = *(_QWORD *)(*(_QWORD *)v75 + 16 * v77 + 8);
        if ( (*(_QWORD *)(v60 + 16) & 0xFFFFFFFFFFFFFFF8LL) == 0
          || (*(_DWORD *)((v80 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v80 >> 1) & 3) > (*(_DWORD *)((*(_QWORD *)(v60 + 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                              | (unsigned int)(*(__int64 *)(v60 + 16) >> 1)
                                                                                              & 3) )
        {
          *(_QWORD *)(v60 + 16) = v80;
        }
        if ( v79 )
        {
          v81 = v63[1] + 16LL * *((unsigned int *)v63 + 4) - 16;
          v82 = *(_DWORD *)(v81 + 12) + 1;
          *(_DWORD *)(v81 + 12) = v82;
          if ( v82 == *(_DWORD *)(v63[1] + 16LL * *((unsigned int *)v63 + 4) - 8) )
          {
            v83 = *(unsigned int *)(*v63 + 192);
            if ( (_DWORD)v83 )
              sub_39460A0(v63 + 1, v83);
          }
        }
        goto LABEL_65;
      }
      if ( !v76 )
        goto LABEL_127;
    }
    else
    {
      v76 = *(_DWORD *)(v73 - 4);
      v75 = v73 - 16;
      if ( !v76 )
        goto LABEL_127;
    }
    if ( v74 )
      goto LABEL_127;
    goto LABEL_124;
  }
LABEL_83:
  v94 = *(_QWORD *)(v24 + 16);
  if ( (v94 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    v94 = v141;
  v95 = v94;
  v96 = v94 & 0xFFFFFFFFFFFFFFF8LL;
  v97 = (v95 >> 1) & 3;
  while ( v48 )
  {
    result = v50[--v48] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_DWORD *)(result + 24) | 3u) <= (v97 | *(_DWORD *)(v96 + 24)) )
      break;
    v98 = *(_DWORD *)(v51[v48] + 4LL * (*(_DWORD *)v7 >> 5));
    if ( !_bittest(&v98, *(_DWORD *)v7) )
    {
      result |= 6uLL;
      *(_QWORD *)(v24 + 16) = result;
      return result;
    }
  }
  return result;
}
