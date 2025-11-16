// Function: sub_1BB12A0
// Address: 0x1bb12a0
//
__int64 __fastcall sub_1BB12A0(__int64 a1, int a2)
{
  __int64 v2; // r14
  int *v3; // rax
  void *v4; // rdi
  int *v5; // rbx
  __int64 v6; // rdx
  unsigned int v7; // eax
  __int64 v8; // rdi
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 *v15; // rax
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // r15
  _QWORD *v20; // rax
  unsigned __int8 v21; // dl
  __int64 v22; // rdx
  int v23; // edx
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 *v27; // rsi
  __int64 *v28; // rcx
  __int64 *v29; // r15
  __int64 *v30; // rbx
  __int64 v31; // rsi
  __int64 *v32; // r12
  __int64 *v33; // rax
  __int64 *v34; // rdx
  __int64 *v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // r12
  __int64 v38; // rax
  __int64 *v39; // rbx
  __int64 v40; // r13
  __int64 v41; // rsi
  __int64 v42; // r15
  _QWORD *v43; // rdx
  _QWORD *v44; // rax
  _QWORD *v45; // r14
  __int64 v46; // rax
  __int64 v47; // r13
  _QWORD *v48; // r12
  _QWORD *v49; // rax
  __int64 v50; // rax
  int v51; // edi
  unsigned int v52; // edx
  _QWORD *v53; // rcx
  _QWORD *v54; // r15
  __int64 v55; // r14
  __int64 v56; // rbx
  _QWORD *v57; // rdx
  _QWORD *v58; // rdx
  __int64 v59; // rdx
  __int64 v60; // rsi
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rbx
  char v65; // r8
  unsigned int v66; // edi
  __int64 v67; // rdx
  __int64 v68; // rax
  __int64 v69; // rcx
  __int64 v70; // rax
  __int64 v71; // r15
  __int64 v72; // rbx
  _QWORD *v73; // r15
  _QWORD *v74; // rax
  __int64 v75; // rax
  int v76; // r8d
  unsigned int v77; // eax
  __int64 v78; // rsi
  _QWORD *v79; // rax
  __int64 v80; // r13
  __int64 v81; // r12
  __int64 v82; // rsi
  _QWORD *v83; // rdx
  __int64 v84; // rdx
  __int64 v85; // rsi
  __int64 v86; // rdx
  __int64 v87; // rsi
  int *v88; // rax
  __int64 *v89; // r13
  __int64 *v90; // rbx
  __int64 v91; // r12
  __int64 v92; // rsi
  __int64 v94; // r13
  _QWORD *v95; // r12
  _QWORD *v96; // rax
  __int64 v97; // rax
  int v98; // esi
  unsigned int v99; // eax
  __int64 v100; // rcx
  _QWORD *v101; // rax
  __int64 v102; // r15
  __int64 v103; // r8
  __int64 v104; // rsi
  _QWORD *v105; // rdx
  __int64 v106; // rdx
  __int64 v107; // rsi
  __int64 v108; // rax
  _QWORD *v109; // rdx
  _QWORD *v110; // rdx
  _QWORD *v111; // rdx
  __int64 v112; // [rsp+18h] [rbp-1F8h]
  __int64 v113; // [rsp+20h] [rbp-1F0h]
  unsigned int v114; // [rsp+30h] [rbp-1E0h]
  __int64 v115; // [rsp+38h] [rbp-1D8h]
  __int64 *v116; // [rsp+38h] [rbp-1D8h]
  __int64 v117; // [rsp+40h] [rbp-1D0h]
  __int64 *v118; // [rsp+40h] [rbp-1D0h]
  __int64 v119; // [rsp+40h] [rbp-1D0h]
  __int64 v120; // [rsp+40h] [rbp-1D0h]
  __int64 v121; // [rsp+48h] [rbp-1C8h]
  __int64 *v122; // [rsp+48h] [rbp-1C8h]
  __int64 *v123; // [rsp+48h] [rbp-1C8h]
  __int64 v124; // [rsp+50h] [rbp-1C0h]
  __int64 v125; // [rsp+50h] [rbp-1C0h]
  int v126[3]; // [rsp+5Ch] [rbp-1B4h] BYREF
  __int64 v127; // [rsp+68h] [rbp-1A8h] BYREF
  __int64 v128; // [rsp+70h] [rbp-1A0h] BYREF
  __int64 v129; // [rsp+78h] [rbp-198h] BYREF
  _QWORD *v130; // [rsp+80h] [rbp-190h] BYREF
  int v131; // [rsp+88h] [rbp-188h]
  __int64 v132; // [rsp+90h] [rbp-180h] BYREF
  __int64 v133; // [rsp+98h] [rbp-178h]
  __int64 v134; // [rsp+A0h] [rbp-170h]
  __int64 v135; // [rsp+A8h] [rbp-168h]
  __int64 *v136; // [rsp+B0h] [rbp-160h]
  __int64 *v137; // [rsp+B8h] [rbp-158h]
  __int64 v138; // [rsp+C0h] [rbp-150h]
  __int64 v139; // [rsp+D0h] [rbp-140h] BYREF
  __int64 *v140; // [rsp+D8h] [rbp-138h]
  __int64 *v141; // [rsp+E0h] [rbp-130h]
  __int64 v142; // [rsp+E8h] [rbp-128h]
  int v143; // [rsp+F0h] [rbp-120h]
  _BYTE v144[72]; // [rsp+F8h] [rbp-118h] BYREF
  __int64 v145; // [rsp+140h] [rbp-D0h] BYREF
  __int64 v146; // [rsp+148h] [rbp-C8h]
  __int64 v147; // [rsp+150h] [rbp-C0h] BYREF
  __int64 *v148; // [rsp+190h] [rbp-80h] BYREF
  __int64 v149; // [rsp+198h] [rbp-78h]
  _BYTE v150[112]; // [rsp+1A0h] [rbp-70h] BYREF

  v2 = a1;
  v126[0] = a2;
  v113 = a1 + 168;
  v3 = sub_1BA5D70(a1 + 168, v126);
  ++*((_QWORD *)v3 + 1);
  v4 = (void *)*((_QWORD *)v3 + 3);
  v5 = v3;
  if ( v4 == *((void **)v3 + 2) )
    goto LABEL_6;
  v6 = (unsigned int)v3[8];
  v7 = 4 * (v3[9] - v3[10]);
  if ( v7 < 0x20 )
    v7 = 32;
  if ( (unsigned int)v6 <= v7 )
  {
    memset(v4, -1, 8 * v6);
LABEL_6:
    *(_QWORD *)(v5 + 9) = 0;
    goto LABEL_7;
  }
  sub_16CC920((__int64)(v5 + 2));
LABEL_7:
  v8 = *(_QWORD *)(v2 + 296);
  v132 = 0;
  v133 = 0;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v137 = 0;
  v138 = 0;
  v112 = sub_13FCB50(v8);
  v9 = sub_157EBA0(v112);
  v10 = *(_QWORD *)sub_13CF970(v9);
  if ( *(_BYTE *)(v10 + 16) <= 0x17u )
  {
    v127 = 0;
  }
  else
  {
    v127 = v10;
    if ( sub_1377F70(*(_QWORD *)(v2 + 296) + 56LL, *(_QWORD *)(v10 + 40)) )
    {
      v11 = *(_QWORD *)(v127 + 8);
      if ( v11 )
      {
        if ( !*(_QWORD *)(v11 + 8) )
          sub_1BB1140((__int64)&v132, &v127);
      }
    }
  }
  v145 = 0;
  v12 = (unsigned __int64 *)&v147;
  v146 = 1;
  do
    *v12++ = -8;
  while ( v12 != (unsigned __int64 *)&v148 );
  v143 = 0;
  v148 = (__int64 *)v150;
  v149 = 0x800000000LL;
  v140 = (__int64 *)v144;
  v141 = (__int64 *)v144;
  v13 = *(_QWORD *)(v2 + 296);
  v139 = 0;
  v142 = 8;
  v117 = *(_QWORD *)(v13 + 40);
  v121 = *(_QWORD *)(v13 + 32);
  if ( v117 == v121 )
    goto LABEL_62;
  do
  {
    v14 = *(_QWORD *)(*(_QWORD *)v121 + 48LL);
    v124 = *(_QWORD *)v121 + 40LL;
    while ( v124 != v14 )
    {
LABEL_20:
      v16 = v14 - 24;
      if ( !v14 )
        v16 = 0;
      v17 = sub_13A4950(v16);
      v18 = v17;
      if ( !v17 || *(_BYTE *)(v17 + 16) <= 0x17u )
        goto LABEL_19;
      v128 = v17;
      v19 = *(_QWORD *)(v17 + 8);
      if ( v19 )
      {
        while ( 1 )
        {
          v20 = sub_1648700(v19);
          v21 = *((_BYTE *)v20 + 16);
          if ( v21 <= 0x17u )
            break;
          if ( v21 == 54 || v21 == 55 )
          {
            v26 = *(v20 - 3);
            if ( !v26 )
              break;
          }
          else
          {
            if ( v21 != 78 )
              break;
            v22 = *(v20 - 3);
            if ( *(_BYTE *)(v22 + 16) )
              break;
            v23 = *(_DWORD *)(v22 + 36);
            if ( v23 == 4085 || v23 == 4057 )
            {
              v24 = 1;
              v25 = *((_DWORD *)v20 + 5) & 0xFFFFFFF;
            }
            else
            {
              if ( v23 != 4503 && v23 != 4492 )
                break;
              v24 = 2;
              v25 = *((_DWORD *)v20 + 5) & 0xFFFFFFF;
            }
            v26 = v20[3 * (v24 - v25)];
            if ( !v26 )
              break;
          }
          if ( v18 != v26 )
            break;
          v19 = *(_QWORD *)(v19 + 8);
          if ( !v19 )
            goto LABEL_38;
        }
        v15 = v140;
        if ( v141 == v140 )
          goto LABEL_42;
      }
      else
      {
LABEL_38:
        v130 = (_QWORD *)v16;
        v131 = v126[0];
        if ( (unsigned __int8)sub_1B99450(v2 + 264, (__int64 *)&v130, &v129)
          && v129 != *(_QWORD *)(v2 + 272) + 24LL * *(unsigned int *)(v2 + 288)
          && (unsigned int)(*(_DWORD *)(v129 + 16) - 1) <= 2 )
        {
          sub_1BAFD60((__int64)&v145, &v128);
          goto LABEL_19;
        }
        v18 = v128;
        v15 = v140;
        if ( v141 == v140 )
        {
LABEL_42:
          v27 = &v15[HIDWORD(v142)];
          if ( v15 != v27 )
          {
            v28 = 0;
            while ( *v15 != v18 )
            {
              if ( *v15 == -2 )
                v28 = v15;
              if ( v27 == ++v15 )
              {
                if ( !v28 )
                  goto LABEL_214;
                *v28 = v18;
                --v143;
                v14 = *(_QWORD *)(v14 + 8);
                ++v139;
                if ( v124 != v14 )
                  goto LABEL_20;
                goto LABEL_50;
              }
            }
            goto LABEL_19;
          }
LABEL_214:
          if ( HIDWORD(v142) < (unsigned int)v142 )
          {
            ++HIDWORD(v142);
            *v27 = v18;
            ++v139;
            goto LABEL_19;
          }
        }
      }
      sub_16CCBA0((__int64)&v139, v18);
LABEL_19:
      v14 = *(_QWORD *)(v14 + 8);
    }
LABEL_50:
    v121 += 8;
  }
  while ( v117 != v121 );
  v29 = v148;
  v30 = &v148[(unsigned int)v149];
  if ( v30 != v148 )
  {
    do
    {
      v31 = *v29;
      v130 = (_QWORD *)*v29;
      if ( v141 == v140 )
        v32 = &v141[HIDWORD(v142)];
      else
        v32 = &v141[(unsigned int)v142];
      v33 = sub_15CC2D0((__int64)&v139, v31);
      if ( v141 == v140 )
        v34 = &v141[HIDWORD(v142)];
      else
        v34 = &v141[(unsigned int)v142];
      for ( ; v34 != v33; ++v33 )
      {
        if ( (unsigned __int64)*v33 < 0xFFFFFFFFFFFFFFFELL )
          break;
      }
      if ( v33 == v32 )
        sub_1BB1140((__int64)&v132, (__int64 *)&v130);
      ++v29;
    }
    while ( v30 != v29 );
  }
LABEL_62:
  v35 = v136;
  v36 = 0;
  v114 = 0;
  if ( v137 == v136 )
    goto LABEL_107;
  v125 = v2;
  do
  {
    v37 = v35[v36];
    ++v114;
    v38 = 3LL * (*(_DWORD *)(v37 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v37 + 23) & 0x40) != 0 )
    {
      v39 = *(__int64 **)(v37 - 8);
      v118 = &v39[v38];
    }
    else
    {
      v118 = (__int64 *)v35[v36];
      v39 = (__int64 *)(v37 - v38 * 8);
    }
    v122 = v39;
    if ( v118 != v39 )
    {
      while ( 1 )
      {
        v40 = *v122;
        if ( *(_BYTE *)(*v122 + 16) > 0x17u )
          break;
LABEL_103:
        v122 += 3;
        if ( v118 == v122 )
        {
          v35 = v136;
          goto LABEL_105;
        }
      }
      v41 = *(_QWORD *)(v40 + 40);
      v42 = *(_QWORD *)(v125 + 296);
      v43 = *(_QWORD **)(v42 + 72);
      v44 = *(_QWORD **)(v42 + 64);
      if ( v43 == v44 )
      {
        v45 = &v44[*(unsigned int *)(v42 + 84)];
        if ( v44 == v45 )
        {
          v109 = *(_QWORD **)(v42 + 64);
        }
        else
        {
          do
          {
            if ( v41 == *v44 )
              break;
            ++v44;
          }
          while ( v45 != v44 );
          v109 = v45;
        }
      }
      else
      {
        v115 = *(_QWORD *)(v40 + 40);
        v45 = &v43[*(unsigned int *)(v42 + 80)];
        v44 = sub_16CC9F0(v42 + 56, v41);
        if ( v115 == *v44 )
        {
          v84 = *(_QWORD *)(v42 + 72);
          if ( v84 == *(_QWORD *)(v42 + 64) )
            v85 = *(unsigned int *)(v42 + 84);
          else
            v85 = *(unsigned int *)(v42 + 80);
          v109 = (_QWORD *)(v84 + 8 * v85);
        }
        else
        {
          v46 = *(_QWORD *)(v42 + 72);
          if ( v46 != *(_QWORD *)(v42 + 64) )
          {
            v44 = (_QWORD *)(v46 + 8LL * *(unsigned int *)(v42 + 80));
            goto LABEL_72;
          }
          v44 = (_QWORD *)(v46 + 8LL * *(unsigned int *)(v42 + 84));
          v109 = v44;
        }
      }
      while ( v109 != v44 && *v44 >= 0xFFFFFFFFFFFFFFFELL )
        ++v44;
LABEL_72:
      if ( v45 == v44 || *(_BYTE *)(v40 + 16) == 77 && (unsigned __int8)sub_1BF28F0(*(_QWORD *)(v125 + 320), v40) )
        goto LABEL_103;
      v128 = v40;
      v47 = *(_QWORD *)(v40 + 8);
      if ( !v47 )
      {
LABEL_142:
        sub_1BB1140((__int64)&v132, &v128);
        goto LABEL_103;
      }
      while ( 1 )
      {
        v54 = sub_1648700(v47);
        v55 = v54[5];
        v56 = *(_QWORD *)(v125 + 296);
        v57 = *(_QWORD **)(v56 + 72);
        v49 = *(_QWORD **)(v56 + 64);
        if ( v57 == v49 )
        {
          v58 = &v49[*(unsigned int *)(v56 + 84)];
          if ( v49 == v58 )
          {
            v48 = *(_QWORD **)(v56 + 64);
          }
          else
          {
            do
            {
              if ( v55 == *v49 )
                break;
              ++v49;
            }
            while ( v58 != v49 );
            v48 = v58;
          }
          goto LABEL_91;
        }
        v48 = &v57[*(unsigned int *)(v56 + 80)];
        v49 = sub_16CC9F0(v56 + 56, v55);
        if ( v55 == *v49 )
          break;
        v50 = *(_QWORD *)(v56 + 72);
        if ( v50 == *(_QWORD *)(v56 + 64) )
        {
          v49 = (_QWORD *)(v50 + 8LL * *(unsigned int *)(v56 + 84));
          v58 = v49;
LABEL_91:
          while ( v58 != v49 && *v49 >= 0xFFFFFFFFFFFFFFFELL )
            ++v49;
          goto LABEL_79;
        }
        v49 = (_QWORD *)(v50 + 8LL * *(unsigned int *)(v56 + 80));
LABEL_79:
        if ( v48 == v49 )
        {
LABEL_82:
          v47 = *(_QWORD *)(v47 + 8);
          if ( !v47 )
            goto LABEL_142;
        }
        else
        {
          if ( (_DWORD)v135 )
          {
            v51 = 1;
            v52 = (v135 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
            v53 = *(_QWORD **)(v133 + 8LL * v52);
            if ( v54 == v53 )
              goto LABEL_82;
            while ( v53 != (_QWORD *)-8LL )
            {
              v52 = (v135 - 1) & (v51 + v52);
              v53 = *(_QWORD **)(v133 + 8LL * v52);
              if ( v54 == v53 )
                goto LABEL_82;
              ++v51;
            }
          }
          v61 = sub_13A4950((__int64)v54);
          if ( v128 != v61 )
            goto LABEL_103;
          v130 = v54;
          v131 = v126[0];
          if ( !(unsigned __int8)sub_1B99450(v125 + 264, (__int64 *)&v130, &v129)
            || v129 == *(_QWORD *)(v125 + 272) + 24LL * *(unsigned int *)(v125 + 288)
            || (unsigned int)(*(_DWORD *)(v129 + 16) - 1) > 2 )
          {
            goto LABEL_103;
          }
          v47 = *(_QWORD *)(v47 + 8);
          if ( !v47 )
            goto LABEL_142;
        }
      }
      v59 = *(_QWORD *)(v56 + 72);
      if ( v59 == *(_QWORD *)(v56 + 64) )
        v60 = *(unsigned int *)(v56 + 84);
      else
        v60 = *(unsigned int *)(v56 + 80);
      v58 = (_QWORD *)(v59 + 8 * v60);
      goto LABEL_91;
    }
LABEL_105:
    v36 = v114;
  }
  while ( v114 != v137 - v35 );
  v2 = v125;
LABEL_107:
  v62 = *(_QWORD *)(v2 + 320);
  v116 = *(__int64 **)(v62 + 144);
  v123 = *(__int64 **)(v62 + 136);
  if ( v116 != v123 )
  {
    while ( 1 )
    {
      v63 = 0x17FFFFFFE8LL;
      v64 = *v123;
      v65 = *(_BYTE *)(*v123 + 23) & 0x40;
      v66 = *(_DWORD *)(*v123 + 20) & 0xFFFFFFF;
      if ( v66 )
      {
        v67 = 24LL * *(unsigned int *)(v64 + 56) + 8;
        v68 = 0;
        do
        {
          v69 = v64 - 24LL * v66;
          if ( v65 )
            v69 = *(_QWORD *)(v64 - 8);
          if ( v112 == *(_QWORD *)(v69 + v67) )
          {
            v63 = 24 * v68;
            goto LABEL_115;
          }
          ++v68;
          v67 += 8;
        }
        while ( v66 != (_DWORD)v68 );
        v63 = 0x17FFFFFFE8LL;
      }
LABEL_115:
      if ( v65 )
        v70 = *(_QWORD *)(v64 - 8);
      else
        v70 = v64 - 24LL * v66;
      v71 = *(_QWORD *)(v70 + v63);
      v129 = v71;
      if ( *(_QWORD *)(v64 + 8) )
      {
        v119 = v64;
        v72 = *(_QWORD *)(v64 + 8);
        while ( 1 )
        {
          v79 = sub_1648700(v72);
          v80 = (__int64)v79;
          if ( v79 == (_QWORD *)v71 )
            goto LABEL_126;
          v81 = *(_QWORD *)(v2 + 296);
          v82 = v79[5];
          v83 = *(_QWORD **)(v81 + 72);
          v74 = *(_QWORD **)(v81 + 64);
          if ( v83 == v74 )
          {
            v73 = &v74[*(unsigned int *)(v81 + 84)];
            if ( v74 == v73 )
            {
              v110 = *(_QWORD **)(v81 + 64);
            }
            else
            {
              do
              {
                if ( v82 == *v74 )
                  break;
                ++v74;
              }
              while ( v73 != v74 );
              v110 = v73;
            }
            goto LABEL_136;
          }
          v73 = &v83[*(unsigned int *)(v81 + 80)];
          v74 = sub_16CC9F0(v81 + 56, v82);
          if ( v82 == *v74 )
            break;
          v75 = *(_QWORD *)(v81 + 72);
          if ( v75 == *(_QWORD *)(v81 + 64) )
          {
            v110 = (_QWORD *)(v75 + 8LL * *(unsigned int *)(v81 + 84));
            v74 = v110;
LABEL_136:
            while ( v110 != v74 && *v74 >= 0xFFFFFFFFFFFFFFFELL )
              ++v74;
            goto LABEL_122;
          }
          v74 = (_QWORD *)(v75 + 8LL * *(unsigned int *)(v81 + 80));
LABEL_122:
          if ( v73 != v74 )
          {
            if ( !(_DWORD)v135 )
              goto LABEL_165;
            v76 = 1;
            v77 = (v135 - 1) & (((unsigned int)v80 >> 9) ^ ((unsigned int)v80 >> 4));
            v78 = *(_QWORD *)(v133 + 8LL * v77);
            if ( v80 != v78 )
            {
              while ( v78 != -8 )
              {
                v77 = (v135 - 1) & (v76 + v77);
                v78 = *(_QWORD *)(v133 + 8LL * v77);
                if ( v80 == v78 )
                  goto LABEL_125;
                ++v76;
              }
LABEL_165:
              if ( v119 != sub_13A4950(v80) || (unsigned int)sub_1B99570(v2, v80, v126[0]) - 1 > 2 )
                goto LABEL_166;
            }
          }
LABEL_125:
          v71 = v129;
LABEL_126:
          v72 = *(_QWORD *)(v72 + 8);
          if ( !v72 )
          {
            v64 = v119;
            goto LABEL_181;
          }
        }
        v86 = *(_QWORD *)(v81 + 72);
        if ( v86 == *(_QWORD *)(v81 + 64) )
          v87 = *(unsigned int *)(v81 + 84);
        else
          v87 = *(unsigned int *)(v81 + 80);
        v110 = (_QWORD *)(v86 + 8 * v87);
        goto LABEL_136;
      }
LABEL_181:
      v94 = *(_QWORD *)(v71 + 8);
      if ( v94 )
        break;
LABEL_213:
      v130 = (_QWORD *)v64;
      sub_1BB1140((__int64)&v132, (__int64 *)&v130);
      sub_1BB1140((__int64)&v132, &v129);
LABEL_166:
      v123 += 11;
      if ( v116 == v123 )
        goto LABEL_167;
    }
    while ( 1 )
    {
      v101 = sub_1648700(v94);
      v102 = (__int64)v101;
      if ( (_QWORD *)v64 == v101 )
        goto LABEL_189;
      v103 = *(_QWORD *)(v2 + 296);
      v104 = v101[5];
      v105 = *(_QWORD **)(v103 + 72);
      v96 = *(_QWORD **)(v103 + 64);
      if ( v105 == v96 )
      {
        v95 = &v96[*(unsigned int *)(v103 + 84)];
        if ( v96 == v95 )
        {
          v111 = *(_QWORD **)(v103 + 64);
        }
        else
        {
          do
          {
            if ( v104 == *v96 )
              break;
            ++v96;
          }
          while ( v95 != v96 );
          v111 = v95;
        }
        goto LABEL_199;
      }
      v120 = *(_QWORD *)(v2 + 296);
      v95 = &v105[*(unsigned int *)(v103 + 80)];
      v96 = sub_16CC9F0(v103 + 56, v104);
      if ( v104 == *v96 )
        break;
      v97 = *(_QWORD *)(v120 + 72);
      if ( v97 == *(_QWORD *)(v120 + 64) )
      {
        v96 = (_QWORD *)(v97 + 8LL * *(unsigned int *)(v120 + 84));
        v111 = v96;
LABEL_199:
        while ( v111 != v96 && *v96 >= 0xFFFFFFFFFFFFFFFELL )
          ++v96;
        goto LABEL_186;
      }
      v96 = (_QWORD *)(v97 + 8LL * *(unsigned int *)(v120 + 80));
LABEL_186:
      if ( v95 == v96 )
      {
LABEL_189:
        v94 = *(_QWORD *)(v94 + 8);
        if ( !v94 )
          goto LABEL_213;
      }
      else
      {
        if ( (_DWORD)v135 )
        {
          v98 = 1;
          v99 = (v135 - 1) & (((unsigned int)v102 >> 9) ^ ((unsigned int)v102 >> 4));
          v100 = *(_QWORD *)(v133 + 8LL * v99);
          if ( v102 == v100 )
            goto LABEL_189;
          while ( v100 != -8 )
          {
            v99 = (v135 - 1) & (v98 + v99);
            v100 = *(_QWORD *)(v133 + 8LL * v99);
            if ( v102 == v100 )
              goto LABEL_189;
            ++v98;
          }
        }
        v108 = sub_13A4950(v102);
        if ( v129 != v108 || (unsigned int)sub_1B99570(v2, v102, v126[0]) - 1 > 2 )
          goto LABEL_166;
        v94 = *(_QWORD *)(v94 + 8);
        if ( !v94 )
          goto LABEL_213;
      }
    }
    v106 = *(_QWORD *)(v120 + 72);
    if ( v106 == *(_QWORD *)(v120 + 64) )
      v107 = *(unsigned int *)(v120 + 84);
    else
      v107 = *(unsigned int *)(v120 + 80);
    v111 = (_QWORD *)(v106 + 8 * v107);
    goto LABEL_199;
  }
LABEL_167:
  v88 = sub_1BA5D70(v113, v126);
  v89 = v137;
  v90 = v136;
  v91 = (__int64)(v88 + 2);
  while ( v89 != v90 )
  {
    v92 = *v90++;
    sub_1412190(v91, v92);
  }
  if ( v141 != v140 )
    _libc_free((unsigned __int64)v141);
  if ( v148 != (__int64 *)v150 )
    _libc_free((unsigned __int64)v148);
  if ( (v146 & 1) == 0 )
    j___libc_free_0(v147);
  if ( v136 )
    j_j___libc_free_0(v136, v138 - (_QWORD)v136);
  return j___libc_free_0(v133);
}
