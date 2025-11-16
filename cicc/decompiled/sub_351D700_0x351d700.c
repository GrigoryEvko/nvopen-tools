// Function: sub_351D700
// Address: 0x351d700
//
_QWORD *__fastcall sub_351D700(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 *v8; // r12
  __int64 v9; // rdx
  __int64 i; // rax
  char v11; // dl
  __int64 v12; // r13
  char v13; // r12
  _QWORD **v14; // r14
  unsigned __int8 v15; // al
  unsigned int v16; // esi
  int v17; // r11d
  __int64 v18; // r8
  _QWORD *v19; // rdx
  unsigned int v20; // edi
  _QWORD *v21; // rax
  _QWORD **v22; // rcx
  __int64 v23; // r14
  __int64 *v24; // r13
  __int64 v25; // rax
  __int64 *v26; // r12
  __int64 v27; // rdx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 *v30; // r13
  __int64 v31; // r11
  int v32; // r10d
  __int64 *v33; // rdx
  unsigned int v34; // edi
  __int64 *v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rax
  __int64 v38; // r14
  __int64 v39; // r12
  unsigned int v40; // esi
  int v41; // esi
  int v42; // esi
  unsigned int v43; // ecx
  int v44; // eax
  __int64 v45; // rdi
  int v46; // eax
  int v47; // ecx
  int v48; // ecx
  __int64 v49; // rdi
  unsigned int v50; // r15d
  int v51; // r10d
  __int64 v52; // rsi
  int v53; // eax
  int v54; // eax
  _QWORD *result; // rax
  __int64 *v56; // r12
  __int64 *v57; // r10
  unsigned int v58; // esi
  unsigned int v59; // r13d
  __int64 v60; // r9
  _QWORD *v61; // rdi
  int v62; // r11d
  unsigned int v63; // edx
  _QWORD *v64; // rax
  __int64 v65; // r8
  _QWORD **v66; // rax
  int v67; // r10d
  int v68; // r10d
  __int64 v69; // rsi
  unsigned int v70; // r9d
  int v71; // eax
  __int64 v72; // rcx
  __int64 v73; // rax
  int v74; // eax
  int v75; // r10d
  int v76; // r10d
  _QWORD *v77; // r11
  int v78; // r13d
  __int64 v79; // rsi
  unsigned int v80; // r9d
  __int64 v81; // rcx
  _QWORD *v82; // r12
  _QWORD *v83; // r10
  unsigned int v84; // esi
  unsigned int v85; // r14d
  __int64 v86; // rcx
  _QWORD *v87; // r11
  int v88; // r9d
  unsigned int v89; // edx
  _QWORD *v90; // rax
  __int64 v91; // r8
  int v92; // ecx
  int v93; // ecx
  __int64 v94; // rdi
  unsigned int v95; // edx
  int v96; // eax
  __int64 v97; // rsi
  int v98; // eax
  int v99; // r8d
  int v100; // r8d
  __int64 v101; // rsi
  _QWORD *v102; // rcx
  int v103; // r9d
  unsigned int v104; // r13d
  __int64 v105; // rdx
  char v106; // al
  __int64 v107; // rax
  bool v108; // al
  int v109; // esi
  int v110; // esi
  __int64 v111; // r8
  unsigned int v112; // ecx
  __int64 v113; // rdi
  int v114; // r10d
  _QWORD *v115; // r9
  int v116; // ecx
  int v117; // ecx
  __int64 v118; // rdi
  _QWORD *v119; // r8
  unsigned int v120; // r12d
  int v121; // r9d
  _QWORD **v122; // rsi
  int v123; // r15d
  int v124; // r13d
  _QWORD *v125; // r9
  int v126; // r10d
  __int64 v127; // [rsp+10h] [rbp-80h]
  char v129; // [rsp+28h] [rbp-68h]
  __int64 v130; // [rsp+28h] [rbp-68h]
  __int64 v131; // [rsp+28h] [rbp-68h]
  __int64 v132; // [rsp+28h] [rbp-68h]
  __int64 v134; // [rsp+38h] [rbp-58h]
  void *v135; // [rsp+38h] [rbp-58h]
  __int64 *v136; // [rsp+38h] [rbp-58h]
  __int64 v137; // [rsp+38h] [rbp-58h]
  char v138; // [rsp+4Fh] [rbp-41h] BYREF
  _QWORD *v139; // [rsp+50h] [rbp-40h] BYREF
  _QWORD v140[7]; // [rsp+58h] [rbp-38h] BYREF

  v4 = a1;
  v139 = *(_QWORD **)(*(_QWORD *)(a1 + 520) + 328LL);
  if ( a4 )
    v140[0] = *(_QWORD *)(a4 + 32);
  v6 = *(unsigned int *)(a3 + 8);
  v7 = *(_QWORD *)a3 + 8 * v6;
  if ( *(_QWORD *)a3 != v7 )
  {
    v134 = *(_QWORD *)a3 + 8 * v6;
    v8 = *(__int64 **)a3;
    do
    {
      v9 = *v8++;
      sub_3514A50(a1, (__int64 **)a3, v9, a2, a4);
    }
    while ( (__int64 *)v134 != v8 );
    v7 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
  }
  v127 = a1 + 888;
  v135 = *(void **)(v7 - 8);
LABEL_8:
  for ( i = sub_351A710(v4, (__int64)v135, (__int64 ***)a3, a4); ; i = sub_351A710(
                                                                         v4,
                                                                         (__int64)v135,
                                                                         (__int64 ***)a3,
                                                                         a4) )
  {
    v129 = v11;
    v12 = i;
    v13 = v11;
    v14 = (_QWORD **)i;
    if ( !(_BYTE)qword_503C648 || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v4 + 520) + 8LL) + 688LL) & 1) != 0 )
    {
      if ( i )
        goto LABEL_18;
    }
    else if ( i )
    {
      v15 = sub_2FD62C0(i);
      if ( *(_DWORD *)(v12 + 120) != 1 && (unsigned __int8)sub_2FD64C0((__int64 *)(v4 + 600), v15, (__int64 *)v12) )
        v129 = v13 | sub_3515CB0(v4, (__int64)v135, (__int64 *)v12, a3, a4);
      goto LABEL_15;
    }
    v14 = (_QWORD **)sub_3513A60(v4, a3, v4 + 200);
    if ( !v14 )
    {
      v14 = (_QWORD **)sub_3513A60(v4, a3, v4 + 344);
      if ( !v14 )
      {
        if ( !a4 )
        {
          result = *(_QWORD **)(v4 + 520);
          v82 = v139;
          v83 = result + 40;
          if ( v139 == result + 40 )
            return result;
          v84 = *(_DWORD *)(v4 + 912);
          v85 = v84 - 1;
          while ( v84 )
          {
            v86 = *(_QWORD *)(v4 + 896);
            v87 = 0;
            v88 = 1;
            v89 = v85 & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
            v90 = (_QWORD *)(v86 + 16LL * v89);
            v91 = *v90;
            if ( (_QWORD *)*v90 != v82 )
            {
              while ( v91 != -4096 )
              {
                if ( !v87 && v91 == -8192 )
                  v87 = v90;
                v89 = v85 & (v88 + v89);
                v90 = (_QWORD *)(v86 + 16LL * v89);
                v91 = *v90;
                if ( (_QWORD *)*v90 == v82 )
                  goto LABEL_106;
                ++v88;
              }
              if ( !v87 )
                v87 = v90;
              v98 = *(_DWORD *)(v4 + 904);
              ++*(_QWORD *)(v4 + 888);
              v96 = v98 + 1;
              if ( 4 * v96 < 3 * v84 )
              {
                if ( v84 - *(_DWORD *)(v4 + 908) - v96 <= v84 >> 3 )
                {
                  sub_3512300(v127, v84);
                  v99 = *(_DWORD *)(v4 + 912);
                  if ( v99 )
                  {
                    v100 = v99 - 1;
                    v101 = *(_QWORD *)(v4 + 896);
                    v102 = 0;
                    v103 = 1;
                    v104 = v100 & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
                    v96 = *(_DWORD *)(v4 + 904) + 1;
                    v87 = (_QWORD *)(v101 + 16LL * v104);
                    v105 = *v87;
                    if ( v82 != (_QWORD *)*v87 )
                    {
                      while ( v105 != -4096 )
                      {
                        if ( !v102 && v105 == -8192 )
                          v102 = v87;
                        v104 = v100 & (v103 + v104);
                        v87 = (_QWORD *)(v101 + 16LL * v104);
                        v105 = *v87;
                        if ( (_QWORD *)*v87 == v82 )
                          goto LABEL_112;
                        ++v103;
                      }
                      if ( v102 )
                        v87 = v102;
                    }
                    goto LABEL_112;
                  }
LABEL_213:
                  ++*(_DWORD *)(v4 + 904);
                  BUG();
                }
LABEL_112:
                *(_DWORD *)(v4 + 904) = v96;
                if ( *v87 != -4096 )
                  --*(_DWORD *)(v4 + 908);
                *v87 = v82;
                result = 0;
                v87[1] = 0;
LABEL_115:
                v139 = v82;
                result = (_QWORD *)*result;
                v14 = (_QWORD **)*result;
                goto LABEL_85;
              }
LABEL_110:
              sub_3512300(v127, 2 * v84);
              v92 = *(_DWORD *)(v4 + 912);
              if ( !v92 )
                goto LABEL_213;
              v93 = v92 - 1;
              v94 = *(_QWORD *)(v4 + 896);
              v95 = v93 & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
              v96 = *(_DWORD *)(v4 + 904) + 1;
              v87 = (_QWORD *)(v94 + 16LL * v95);
              v97 = *v87;
              if ( (_QWORD *)*v87 != v82 )
              {
                v125 = 0;
                v126 = 1;
                while ( v97 != -4096 )
                {
                  if ( !v125 && v97 == -8192 )
                    v125 = v87;
                  v95 = v93 & (v126 + v95);
                  v87 = (_QWORD *)(v94 + 16LL * v95);
                  v97 = *v87;
                  if ( (_QWORD *)*v87 == v82 )
                    goto LABEL_112;
                  ++v126;
                }
                if ( v125 )
                  v87 = v125;
              }
              goto LABEL_112;
            }
LABEL_106:
            result = (_QWORD *)v90[1];
            if ( (_QWORD *)a3 != result )
              goto LABEL_115;
            v82 = (_QWORD *)v82[1];
            if ( v83 == v82 )
              return result;
          }
          ++*(_QWORD *)(v4 + 888);
          goto LABEL_110;
        }
        result = *(_QWORD **)(a4 + 32);
        v56 = (__int64 *)v140[0];
        v57 = &result[*(unsigned int *)(a4 + 40)];
        if ( (__int64 *)v140[0] == v57 )
          return result;
        v58 = *(_DWORD *)(v4 + 912);
        v59 = v58 - 1;
        while ( 1 )
        {
          if ( !v58 )
          {
            ++*(_QWORD *)(v4 + 888);
LABEL_79:
            sub_3512300(v127, 2 * v58);
            v67 = *(_DWORD *)(v4 + 912);
            if ( v67 )
            {
              v68 = v67 - 1;
              v69 = *(_QWORD *)(v4 + 896);
              v70 = v68 & (((unsigned int)*v56 >> 9) ^ ((unsigned int)*v56 >> 4));
              v71 = *(_DWORD *)(v4 + 904) + 1;
              v61 = (_QWORD *)(v69 + 16LL * v70);
              v72 = *v61;
              if ( *v56 != *v61 )
              {
                v77 = 0;
                v124 = 1;
                while ( v72 != -4096 )
                {
                  if ( v72 == -8192 && !v77 )
                    v77 = v61;
                  v70 = v68 & (v124 + v70);
                  v61 = (_QWORD *)(v69 + 16LL * v70);
                  v72 = *v61;
                  if ( *v56 == *v61 )
                    goto LABEL_81;
                  ++v124;
                }
                goto LABEL_100;
              }
              goto LABEL_81;
            }
LABEL_212:
            ++*(_DWORD *)(v4 + 904);
            BUG();
          }
          v60 = *(_QWORD *)(v4 + 896);
          v61 = 0;
          v62 = 1;
          v63 = v59 & (((unsigned int)*v56 >> 9) ^ ((unsigned int)*v56 >> 4));
          v64 = (_QWORD *)(v60 + 16LL * v63);
          v65 = *v64;
          if ( *v56 != *v64 )
            break;
LABEL_75:
          v66 = (_QWORD **)v64[1];
          if ( (_QWORD **)a3 != v66 )
          {
            v14 = v66;
            goto LABEL_84;
          }
          result = (_QWORD *)v140[0];
          v56 = (__int64 *)(v140[0] + 8LL);
          v140[0] = v56;
          if ( v56 == v57 )
            return result;
        }
        while ( v65 != -4096 )
        {
          if ( v65 == -8192 && !v61 )
            v61 = v64;
          v63 = v59 & (v62 + v63);
          v64 = (_QWORD *)(v60 + 16LL * v63);
          v65 = *v64;
          if ( *v56 == *v64 )
            goto LABEL_75;
          ++v62;
        }
        if ( !v61 )
          v61 = v64;
        v74 = *(_DWORD *)(v4 + 904);
        ++*(_QWORD *)(v4 + 888);
        v71 = v74 + 1;
        if ( 4 * v71 >= 3 * v58 )
          goto LABEL_79;
        if ( v58 - *(_DWORD *)(v4 + 908) - v71 > v58 >> 3 )
          goto LABEL_81;
        sub_3512300(v127, v58);
        v75 = *(_DWORD *)(v4 + 912);
        if ( !v75 )
          goto LABEL_212;
        v76 = v75 - 1;
        v77 = 0;
        v78 = 1;
        v79 = *(_QWORD *)(v4 + 896);
        v80 = v76 & (((unsigned int)*v56 >> 9) ^ ((unsigned int)*v56 >> 4));
        v71 = *(_DWORD *)(v4 + 904) + 1;
        v61 = (_QWORD *)(v79 + 16LL * v80);
        v81 = *v61;
        if ( *v56 != *v61 )
        {
          while ( v81 != -4096 )
          {
            if ( !v77 && v81 == -8192 )
              v77 = v61;
            v80 = v76 & (v78 + v80);
            v61 = (_QWORD *)(v79 + 16LL * v80);
            v81 = *v61;
            if ( *v56 == *v61 )
              goto LABEL_81;
            ++v78;
          }
LABEL_100:
          if ( v77 )
            v61 = v77;
        }
LABEL_81:
        *(_DWORD *)(v4 + 904) = v71;
        if ( *v61 != -4096 )
          --*(_DWORD *)(v4 + 908);
        v73 = *v56;
        v61[1] = 0;
        *v61 = v73;
LABEL_84:
        result = *v14;
        v14 = (_QWORD **)**v14;
LABEL_85:
        if ( !v14 )
          break;
      }
    }
LABEL_15:
    if ( !(_BYTE)qword_503C648 || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v4 + 520) + 8LL) + 688LL) & 1) != 0 || !v129 )
    {
LABEL_18:
      v16 = *(_DWORD *)(v4 + 912);
      if ( !v16 )
        goto LABEL_143;
      goto LABEL_19;
    }
    if ( !(unsigned __int8)sub_351C710(v4, (__int64)v14, v135, a3, a4, (__int64)&v139, (__int64)v140, &v138) )
      goto LABEL_140;
    if ( v138 )
    {
      do
      {
        v107 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
        if ( *(_QWORD *)a3 == v107 - 8 )
        {
          v135 = *(void **)(v107 - 8);
          sub_3514A50(v4, (__int64 **)a3, (__int64)v135, a2, a4);
          goto LABEL_140;
        }
        v106 = sub_351C710(
                 v4,
                 *(_QWORD *)(v107 - 8),
                 *(void **)(v107 - 16),
                 a3,
                 a4,
                 (__int64)&v139,
                 (__int64)v140,
                 &v138);
      }
      while ( v138 && v106 );
      v135 = *(void **)(*(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8) - 8);
      sub_3514A50(v4, (__int64 **)a3, (__int64)v135, a2, a4);
LABEL_140:
      v108 = sub_2E322C0((__int64)v135, (__int64)v14);
      goto LABEL_141;
    }
    v135 = *(void **)(*(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8) - 8);
    v108 = sub_2E322C0((__int64)v135, (__int64)v14);
LABEL_141:
    if ( !v108 )
      goto LABEL_8;
    v16 = *(_DWORD *)(v4 + 912);
    if ( !v16 )
    {
LABEL_143:
      ++*(_QWORD *)(v4 + 888);
      goto LABEL_144;
    }
LABEL_19:
    v17 = 1;
    v18 = *(_QWORD *)(v4 + 896);
    v19 = 0;
    v20 = (v16 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v21 = (_QWORD *)(v18 + 16LL * v20);
    v22 = (_QWORD **)*v21;
    if ( v14 == (_QWORD **)*v21 )
    {
LABEL_20:
      v23 = v21[1];
      goto LABEL_21;
    }
    while ( v22 != (_QWORD **)-4096LL )
    {
      if ( v22 == (_QWORD **)-8192LL && !v19 )
        v19 = v21;
      v20 = (v16 - 1) & (v17 + v20);
      v21 = (_QWORD *)(v18 + 16LL * v20);
      v22 = (_QWORD **)*v21;
      if ( (_QWORD **)*v21 == v14 )
        goto LABEL_20;
      ++v17;
    }
    if ( !v19 )
      v19 = v21;
    v53 = *(_DWORD *)(v4 + 904);
    ++*(_QWORD *)(v4 + 888);
    v54 = v53 + 1;
    if ( 4 * v54 < 3 * v16 )
    {
      if ( v16 - *(_DWORD *)(v4 + 908) - v54 <= v16 >> 3 )
      {
        sub_3512300(v127, v16);
        v116 = *(_DWORD *)(v4 + 912);
        if ( !v116 )
        {
LABEL_214:
          ++*(_DWORD *)(v4 + 904);
          BUG();
        }
        v117 = v116 - 1;
        v118 = *(_QWORD *)(v4 + 896);
        v119 = 0;
        v120 = v117 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v121 = 1;
        v54 = *(_DWORD *)(v4 + 904) + 1;
        v19 = (_QWORD *)(v118 + 16LL * v120);
        v122 = (_QWORD **)*v19;
        if ( (_QWORD **)*v19 != v14 )
        {
          while ( v122 != (_QWORD **)-4096LL )
          {
            if ( v122 == (_QWORD **)-8192LL && !v119 )
              v119 = v19;
            v120 = v117 & (v121 + v120);
            v19 = (_QWORD *)(v118 + 16LL * v120);
            v122 = (_QWORD **)*v19;
            if ( (_QWORD **)*v19 == v14 )
              goto LABEL_65;
            ++v121;
          }
          if ( v119 )
            v19 = v119;
        }
      }
      goto LABEL_65;
    }
LABEL_144:
    sub_3512300(v127, 2 * v16);
    v109 = *(_DWORD *)(v4 + 912);
    if ( !v109 )
      goto LABEL_214;
    v110 = v109 - 1;
    v111 = *(_QWORD *)(v4 + 896);
    v112 = v110 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v54 = *(_DWORD *)(v4 + 904) + 1;
    v19 = (_QWORD *)(v111 + 16LL * v112);
    v113 = *v19;
    if ( (_QWORD **)*v19 != v14 )
    {
      v114 = 1;
      v115 = 0;
      while ( v113 != -4096 )
      {
        if ( v113 == -8192 && !v115 )
          v115 = v19;
        v112 = v110 & (v114 + v112);
        v19 = (_QWORD *)(v111 + 16LL * v112);
        v113 = *v19;
        if ( (_QWORD **)*v19 == v14 )
          goto LABEL_65;
        ++v114;
      }
      if ( v115 )
        v19 = v115;
    }
LABEL_65:
    *(_DWORD *)(v4 + 904) = v54;
    if ( *v19 != -4096 )
      --*(_DWORD *)(v4 + 908);
    *v19 = v14;
    v23 = 0;
    v19[1] = 0;
LABEL_21:
    v24 = *(__int64 **)v23;
    v25 = *(unsigned int *)(v23 + 8);
    *(_DWORD *)(v23 + 56) = 0;
    if ( v24 != &v24[v25] )
    {
      v136 = &v24[v25];
      v26 = v24;
      do
      {
        v27 = *v26++;
        sub_3514A50(v4, (__int64 **)v23, v27, a2, a4);
      }
      while ( v136 != v26 );
      v30 = *(__int64 **)v23;
      v31 = *(_QWORD *)v23 + 8LL * *(unsigned int *)(v23 + 8);
      if ( *(_QWORD *)v23 != v31 )
      {
        v137 = v4;
        while ( 1 )
        {
          v37 = *(unsigned int *)(a3 + 8);
          v38 = *v30;
          if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
          {
            v132 = v31;
            sub_C8D5F0(a3, (const void *)(a3 + 16), v37 + 1, 8u, v28, v29);
            v37 = *(unsigned int *)(a3 + 8);
            v31 = v132;
          }
          *(_QWORD *)(*(_QWORD *)a3 + 8 * v37) = v38;
          v39 = *(_QWORD *)(a3 + 48);
          ++*(_DWORD *)(a3 + 8);
          v40 = *(_DWORD *)(v39 + 24);
          if ( !v40 )
            break;
          v29 = v40 - 1;
          v28 = *(_QWORD *)(v39 + 8);
          v32 = 1;
          v33 = 0;
          v34 = v29 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
          v35 = (__int64 *)(v28 + 16LL * v34);
          v36 = *v35;
          if ( v38 == *v35 )
          {
LABEL_27:
            ++v30;
            v35[1] = a3;
            if ( (__int64 *)v31 == v30 )
              goto LABEL_37;
          }
          else
          {
            while ( v36 != -4096 )
            {
              if ( v36 == -8192 && !v33 )
                v33 = v35;
              v34 = v29 & (v32 + v34);
              v35 = (__int64 *)(v28 + 16LL * v34);
              v36 = *v35;
              if ( v38 == *v35 )
                goto LABEL_27;
              ++v32;
            }
            if ( !v33 )
              v33 = v35;
            v46 = *(_DWORD *)(v39 + 16);
            ++*(_QWORD *)v39;
            v44 = v46 + 1;
            if ( 4 * v44 < 3 * v40 )
            {
              if ( v40 - *(_DWORD *)(v39 + 20) - v44 <= v40 >> 3 )
              {
                v131 = v31;
                sub_3512300(v39, v40);
                v47 = *(_DWORD *)(v39 + 24);
                if ( !v47 )
                {
LABEL_215:
                  ++*(_DWORD *)(v39 + 16);
                  BUG();
                }
                v48 = v47 - 1;
                v49 = *(_QWORD *)(v39 + 8);
                v28 = 0;
                v50 = v48 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
                v31 = v131;
                v51 = 1;
                v44 = *(_DWORD *)(v39 + 16) + 1;
                v33 = (__int64 *)(v49 + 16LL * v50);
                v52 = *v33;
                if ( v38 != *v33 )
                {
                  while ( v52 != -4096 )
                  {
                    if ( v52 != -8192 || v28 )
                      v33 = (__int64 *)v28;
                    v28 = (unsigned int)(v51 + 1);
                    v50 = v48 & (v51 + v50);
                    v29 = v49 + 16LL * v50;
                    v52 = *(_QWORD *)v29;
                    if ( v38 == *(_QWORD *)v29 )
                    {
                      v33 = (__int64 *)(v49 + 16LL * v50);
                      goto LABEL_34;
                    }
                    ++v51;
                    v28 = (__int64)v33;
                    v33 = (__int64 *)(v49 + 16LL * v50);
                  }
                  if ( v28 )
                    v33 = (__int64 *)v28;
                }
              }
              goto LABEL_34;
            }
LABEL_32:
            v130 = v31;
            sub_3512300(v39, 2 * v40);
            v41 = *(_DWORD *)(v39 + 24);
            if ( !v41 )
              goto LABEL_215;
            v42 = v41 - 1;
            v28 = *(_QWORD *)(v39 + 8);
            v31 = v130;
            v43 = v42 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
            v44 = *(_DWORD *)(v39 + 16) + 1;
            v33 = (__int64 *)(v28 + 16LL * v43);
            v45 = *v33;
            if ( v38 != *v33 )
            {
              v123 = 1;
              v29 = 0;
              while ( v45 != -4096 )
              {
                if ( v45 == -8192 && !v29 )
                  v29 = (__int64)v33;
                v43 = v42 & (v123 + v43);
                v33 = (__int64 *)(v28 + 16LL * v43);
                v45 = *v33;
                if ( v38 == *v33 )
                  goto LABEL_34;
                ++v123;
              }
              if ( v29 )
                v33 = (__int64 *)v29;
            }
LABEL_34:
            *(_DWORD *)(v39 + 16) = v44;
            if ( *v33 != -4096 )
              --*(_DWORD *)(v39 + 20);
            ++v30;
            *v33 = v38;
            v33[1] = 0;
            v33[1] = a3;
            if ( (__int64 *)v31 == v30 )
            {
LABEL_37:
              v4 = v137;
              goto LABEL_38;
            }
          }
        }
        ++*(_QWORD *)v39;
        goto LABEL_32;
      }
    }
LABEL_38:
    v135 = *(void **)(*(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8) - 8);
  }
  return result;
}
