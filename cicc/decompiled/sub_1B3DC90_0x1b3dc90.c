// Function: sub_1B3DC90
// Address: 0x1b3dc90
//
__int64 __fastcall sub_1B3DC90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r9
  unsigned int v7; // esi
  __int64 v8; // rbx
  int v9; // r8d
  __int64 v10; // rdi
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  unsigned int v15; // ebx
  __int64 v16; // rdx
  __int64 v17; // r13
  __int64 v18; // r8
  unsigned __int64 v19; // rbx
  __int64 v20; // r12
  __int64 v21; // rdi
  int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // r14
  __int64 v25; // rax
  __int64 v26; // r12
  _QWORD *v27; // rax
  int v28; // r8d
  __int64 v29; // r13
  __int64 v30; // rax
  int v31; // edx
  unsigned int v32; // r13d
  __int64 v33; // r8
  __int64 v34; // rdi
  unsigned int v35; // ecx
  _QWORD *v36; // rbx
  __int64 v37; // rdx
  __int64 v38; // rax
  unsigned int v39; // esi
  __int64 v40; // r12
  int v41; // ecx
  int v42; // ecx
  __int64 v43; // rdi
  unsigned int v44; // eax
  int v45; // edx
  __int64 v46; // rsi
  __int64 v47; // rax
  int v48; // edx
  __int64 v49; // rsi
  int v50; // edx
  unsigned int v51; // ecx
  __int64 *v52; // rax
  __int64 v53; // r9
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // r8
  __int64 *v59; // r9
  int v60; // edx
  __int64 v61; // r13
  __int64 v62; // rax
  __int64 v63; // rbx
  bool v64; // zf
  __int64 v65; // rbx
  unsigned __int64 v66; // rax
  __int64 v67; // r13
  unsigned int v68; // r12d
  int v69; // r14d
  unsigned int v70; // edx
  __int64 *v71; // rax
  __int64 v72; // rdi
  __int64 v73; // rbx
  __int64 v74; // rax
  __int64 v75; // rax
  unsigned int v76; // esi
  __int64 v77; // rbx
  int v78; // eax
  int v79; // esi
  __int64 v80; // rdx
  unsigned int v81; // eax
  int v82; // edi
  __int64 *v83; // rcx
  _BYTE *v84; // rdi
  __int64 v86; // rdx
  int v87; // eax
  int v88; // r11d
  _QWORD *v89; // r10
  int v90; // edi
  int v91; // ecx
  int v92; // ecx
  __int64 v93; // rdi
  _QWORD *v94; // r10
  int v95; // r11d
  unsigned int v96; // eax
  __int64 v97; // rsi
  int v98; // r11d
  int v99; // eax
  int v100; // eax
  int v101; // eax
  int v102; // r10d
  unsigned int v103; // edx
  __int64 v104; // rsi
  __int64 v105; // rax
  int v106; // r11d
  _QWORD *v107; // r9
  int v108; // edi
  int v109; // eax
  int v110; // r10d
  int v111; // r11d
  _QWORD *v112; // r10
  int v113; // edi
  int v114; // edx
  int v115; // r11d
  int v116; // r11d
  __int64 v117; // r10
  __int64 v118; // rdi
  int v119; // esi
  _QWORD *v120; // rcx
  int v121; // r10d
  int v122; // r10d
  unsigned int v123; // r13d
  int v124; // esi
  __int64 v125; // rdi
  __int64 v127; // [rsp+30h] [rbp-320h]
  __int64 v128; // [rsp+38h] [rbp-318h]
  __int64 v129; // [rsp+38h] [rbp-318h]
  __int64 v130; // [rsp+38h] [rbp-318h]
  unsigned int v131; // [rsp+38h] [rbp-318h]
  __int64 v132; // [rsp+38h] [rbp-318h]
  __int64 v133; // [rsp+40h] [rbp-310h]
  __int64 v134; // [rsp+40h] [rbp-310h]
  __int64 v135; // [rsp+40h] [rbp-310h]
  __int64 v136; // [rsp+40h] [rbp-310h]
  unsigned int v137; // [rsp+40h] [rbp-310h]
  __int64 v138; // [rsp+40h] [rbp-310h]
  __int64 v139; // [rsp+40h] [rbp-310h]
  __int64 v140; // [rsp+40h] [rbp-310h]
  __int64 *v141; // [rsp+48h] [rbp-308h]
  int v142; // [rsp+48h] [rbp-308h]
  _BYTE *v143; // [rsp+50h] [rbp-300h] BYREF
  __int64 v144; // [rsp+58h] [rbp-2F8h]
  _BYTE v145[80]; // [rsp+60h] [rbp-2F0h] BYREF
  _BYTE *v146; // [rsp+B0h] [rbp-2A0h] BYREF
  __int64 v147; // [rsp+B8h] [rbp-298h]
  _BYTE v148[80]; // [rsp+C0h] [rbp-290h] BYREF
  _BYTE *v149; // [rsp+110h] [rbp-240h] BYREF
  __int64 v150; // [rsp+118h] [rbp-238h]
  _BYTE v151[560]; // [rsp+120h] [rbp-230h] BYREF

  v143 = v145;
  v144 = 0xA00000000LL;
  v149 = v151;
  v150 = 0x4000000000LL;
  v141 = (__int64 *)(a1 + 56);
  v5 = sub_145CBF0((__int64 *)(a1 + 56), 64, 16);
  v7 = *(_DWORD *)(a1 + 48);
  *(_QWORD *)v5 = a2;
  v8 = v5;
  *(_QWORD *)(v5 + 8) = 0;
  *(_QWORD *)(v5 + 16) = 0;
  *(_DWORD *)(v5 + 24) = 0;
  *(_QWORD *)(v5 + 32) = 0;
  *(_DWORD *)(v5 + 40) = 0;
  *(_QWORD *)(v5 + 48) = 0;
  *(_QWORD *)(v5 + 56) = 0;
  v127 = a1 + 24;
  if ( v7 )
  {
    v9 = v7 - 1;
    v10 = *(_QWORD *)(a1 + 32);
    LODWORD(v11) = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v12 = (_QWORD *)(v10 + 16LL * (unsigned int)v11);
    v13 = *v12;
    if ( *v12 == a2 )
      goto LABEL_3;
    v111 = 1;
    v112 = 0;
    while ( v13 != -8 )
    {
      if ( !v112 && v13 == -16 )
        v112 = v12;
      LODWORD(v6) = v111 + 1;
      v11 = v9 & (unsigned int)(v11 + v111);
      v12 = (_QWORD *)(v10 + 16 * v11);
      v13 = *v12;
      if ( *v12 == a2 )
        goto LABEL_3;
      ++v111;
    }
    v113 = *(_DWORD *)(a1 + 40);
    if ( v112 )
      v12 = v112;
    ++*(_QWORD *)(a1 + 24);
    v114 = v113 + 1;
    if ( 4 * (v113 + 1) < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a1 + 44) - v114 > v7 >> 3 )
        goto LABEL_151;
      sub_1B3C650(v127, v7);
      v121 = *(_DWORD *)(a1 + 48);
      if ( !v121 )
        goto LABEL_193;
      v122 = v121 - 1;
      v6 = *(_QWORD *)(a1 + 32);
      v120 = 0;
      v123 = v122 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v114 = *(_DWORD *)(a1 + 40) + 1;
      v124 = 1;
      v12 = (_QWORD *)(v6 + 16LL * v123);
      v125 = *v12;
      if ( *v12 == a2 )
        goto LABEL_151;
      while ( v125 != -8 )
      {
        if ( !v120 && v125 == -16 )
          v120 = v12;
        v9 = v124 + 1;
        v123 = v122 & (v124 + v123);
        v12 = (_QWORD *)(v6 + 16LL * v123);
        v125 = *v12;
        if ( *v12 == a2 )
          goto LABEL_151;
        ++v124;
      }
      goto LABEL_163;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 24);
  }
  sub_1B3C650(v127, 2 * v7);
  v115 = *(_DWORD *)(a1 + 48);
  if ( !v115 )
    goto LABEL_193;
  v116 = v115 - 1;
  v117 = *(_QWORD *)(a1 + 32);
  LODWORD(v6) = v116 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v114 = *(_DWORD *)(a1 + 40) + 1;
  v12 = (_QWORD *)(v117 + 16LL * (unsigned int)v6);
  v118 = *v12;
  if ( *v12 == a2 )
    goto LABEL_151;
  v119 = 1;
  v120 = 0;
  while ( v118 != -8 )
  {
    if ( !v120 && v118 == -16 )
      v120 = v12;
    v9 = v119 + 1;
    LODWORD(v6) = v116 & (v119 + v6);
    v12 = (_QWORD *)(v117 + 16LL * (unsigned int)v6);
    v118 = *v12;
    if ( *v12 == a2 )
      goto LABEL_151;
    ++v119;
  }
LABEL_163:
  if ( v120 )
    v12 = v120;
LABEL_151:
  *(_DWORD *)(a1 + 40) = v114;
  if ( *v12 != -8 )
    --*(_DWORD *)(a1 + 44);
  *v12 = a2;
  v12[1] = 0;
LABEL_3:
  v12[1] = v8;
  v14 = (unsigned int)v150;
  if ( (unsigned int)v150 >= HIDWORD(v150) )
  {
    sub_16CD150((__int64)&v149, v151, 0, 8, v9, v6);
    v14 = (unsigned int)v150;
  }
  *(_QWORD *)&v149[8 * v14] = v8;
  v15 = v150 + 1;
  v146 = v148;
  LODWORD(v150) = v15;
  v147 = 0xA00000000LL;
  if ( !v15 )
    goto LABEL_50;
  do
  {
    while ( 1 )
    {
      v23 = v15--;
      v24 = *(_QWORD *)&v149[8 * v23 - 8];
      LODWORD(v150) = v15;
      LODWORD(v147) = 0;
      v25 = *(_QWORD *)(*(_QWORD *)v24 + 48LL);
      if ( !v25 )
        BUG();
      if ( *(_BYTE *)(v25 - 8) == 77 )
      {
        v16 = *(_DWORD *)(v25 - 4) & 0xFFFFFFF;
        if ( (*(_BYTE *)(v25 - 1) & 0x40) != 0 )
          v17 = *(_QWORD *)(v25 - 32);
        else
          v17 = v25 - 24 - 24 * v16;
        v18 = 8 * v16;
        v19 = *(_DWORD *)(v25 - 4) & 0xFFFFFFF;
        v20 = *(unsigned int *)(v25 + 32);
        v21 = 0;
        v22 = 0;
        if ( v19 > HIDWORD(v147) )
        {
          v138 = 8 * v16;
          sub_16CD150((__int64)&v146, v148, v19, 8, v18, v6);
          v22 = v147;
          v18 = v138;
          v21 = 8LL * (unsigned int)v147;
        }
        if ( v18 )
        {
          memcpy(&v146[v21], (const void *)(v17 + 24 * v20 + 8), v18);
          v22 = v147;
        }
        LODWORD(v147) = v19 + v22;
        LODWORD(v19) = v19 + v22;
      }
      else
      {
        v26 = *(_QWORD *)(*(_QWORD *)v24 + 8LL);
        if ( !v26 )
        {
LABEL_28:
          *(_DWORD *)(v24 + 40) = 0;
          goto LABEL_16;
        }
        while ( 1 )
        {
          v27 = sub_1648700(v26);
          if ( (unsigned __int8)(*((_BYTE *)v27 + 16) - 25) <= 9u )
            break;
          v26 = *(_QWORD *)(v26 + 8);
          if ( !v26 )
            goto LABEL_28;
        }
        v19 = 0;
LABEL_25:
        v29 = v27[5];
        if ( (unsigned int)v19 >= HIDWORD(v147) )
        {
          sub_16CD150((__int64)&v146, v148, 0, 8, v28, v6);
          v19 = (unsigned int)v147;
        }
        *(_QWORD *)&v146[8 * v19] = v29;
        v19 = (unsigned int)(v147 + 1);
        LODWORD(v147) = v147 + 1;
        while ( 1 )
        {
          v26 = *(_QWORD *)(v26 + 8);
          if ( !v26 )
            break;
          v27 = sub_1648700(v26);
          if ( (unsigned __int8)(*((_BYTE *)v27 + 16) - 25) <= 9u )
            goto LABEL_25;
        }
      }
      *(_DWORD *)(v24 + 40) = v19;
      if ( (_DWORD)v19 )
        break;
      v15 = v150;
LABEL_16:
      *(_QWORD *)(v24 + 48) = 0;
      if ( !v15 )
        goto LABEL_50;
    }
    v30 = sub_145CBF0(v141, 8LL * (unsigned int)v19, 8);
    v31 = *(_DWORD *)(v24 + 40);
    *(_QWORD *)(v24 + 48) = v30;
    if ( !v31 )
      goto LABEL_49;
    v32 = 0;
    v33 = v24;
    while ( 2 )
    {
      while ( 2 )
      {
        v39 = *(_DWORD *)(a1 + 48);
        v40 = *(_QWORD *)&v146[8 * v32];
        if ( !v39 )
        {
          ++*(_QWORD *)(a1 + 24);
          goto LABEL_37;
        }
        LODWORD(v6) = v39 - 1;
        v34 = *(_QWORD *)(a1 + 32);
        v35 = (v39 - 1) & (((unsigned int)v40 >> 4) ^ ((unsigned int)v40 >> 9));
        v36 = (_QWORD *)(v34 + 16LL * v35);
        v37 = *v36;
        if ( v40 == *v36 )
          goto LABEL_32;
        v88 = 1;
        v89 = 0;
        while ( 1 )
        {
          if ( v37 == -8 )
          {
            v90 = *(_DWORD *)(a1 + 40);
            if ( v89 )
              v36 = v89;
            ++*(_QWORD *)(a1 + 24);
            v45 = v90 + 1;
            if ( 4 * (v90 + 1) < 3 * v39 )
            {
              if ( v39 - *(_DWORD *)(a1 + 44) - v45 > v39 >> 3 )
                goto LABEL_39;
              v129 = v33;
              v137 = ((unsigned int)v40 >> 4) ^ ((unsigned int)v40 >> 9);
              sub_1B3C650(v127, v39);
              v91 = *(_DWORD *)(a1 + 48);
              if ( v91 )
              {
                v92 = v91 - 1;
                v93 = *(_QWORD *)(a1 + 32);
                v94 = 0;
                v33 = v129;
                v95 = 1;
                v96 = v92 & v137;
                v45 = *(_DWORD *)(a1 + 40) + 1;
                v36 = (_QWORD *)(v93 + 16LL * (v92 & v137));
                v97 = *v36;
                if ( v40 == *v36 )
                  goto LABEL_39;
                while ( v97 != -8 )
                {
                  if ( v94 || v97 != -16 )
                    v36 = v94;
                  v96 = v92 & (v95 + v96);
                  v107 = (_QWORD *)(v93 + 16LL * v96);
                  v97 = *v107;
                  if ( v40 == *v107 )
                  {
LABEL_129:
                    v36 = v107;
                    goto LABEL_39;
                  }
                  ++v95;
                  v94 = v36;
                  v36 = (_QWORD *)(v93 + 16LL * v96);
                }
                goto LABEL_101;
              }
LABEL_193:
              ++*(_DWORD *)(a1 + 40);
              BUG();
            }
LABEL_37:
            v133 = v33;
            sub_1B3C650(v127, 2 * v39);
            v41 = *(_DWORD *)(a1 + 48);
            if ( v41 )
            {
              v42 = v41 - 1;
              v43 = *(_QWORD *)(a1 + 32);
              v33 = v133;
              v44 = v42 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
              v45 = *(_DWORD *)(a1 + 40) + 1;
              v36 = (_QWORD *)(v43 + 16LL * v44);
              v46 = *v36;
              if ( v40 == *v36 )
              {
LABEL_39:
                *(_DWORD *)(a1 + 40) = v45;
                if ( *v36 != -8 )
                  --*(_DWORD *)(a1 + 44);
                *v36 = v40;
                v36[1] = 0;
                goto LABEL_42;
              }
              v106 = 1;
              v94 = 0;
              while ( v46 != -8 )
              {
                if ( v46 != -16 || v94 )
                  v36 = v94;
                v44 = v42 & (v106 + v44);
                v107 = (_QWORD *)(v43 + 16LL * v44);
                v46 = *v107;
                if ( v40 == *v107 )
                  goto LABEL_129;
                ++v106;
                v94 = v36;
                v36 = (_QWORD *)(v43 + 16LL * v44);
              }
LABEL_101:
              if ( v94 )
                v36 = v94;
              goto LABEL_39;
            }
            goto LABEL_193;
          }
          if ( v89 || v37 != -16 )
            v36 = v89;
          v35 = v6 & (v88 + v35);
          v37 = *(_QWORD *)(v34 + 16LL * v35);
          if ( v40 == v37 )
            break;
          ++v88;
          v89 = v36;
          v36 = (_QWORD *)(v34 + 16LL * v35);
        }
        v36 = (_QWORD *)(v34 + 16LL * v35);
LABEL_32:
        v38 = v36[1];
        if ( v38 )
        {
          *(_QWORD *)(*(_QWORD *)(v33 + 48) + 8LL * v32) = v38;
          goto LABEL_34;
        }
LABEL_42:
        v47 = *(_QWORD *)(a1 + 8);
        v48 = *(_DWORD *)(v47 + 24);
        if ( v48 )
        {
          v49 = *(_QWORD *)(v47 + 8);
          v50 = v48 - 1;
          v51 = v50 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
          v52 = (__int64 *)(v49 + 16LL * v51);
          v53 = *v52;
          if ( v40 == *v52 )
          {
LABEL_44:
            v128 = v33;
            v134 = v52[1];
            v54 = sub_145CBF0(v141, 64, 16);
            v33 = v128;
            *(_QWORD *)v54 = v40;
            v55 = v54;
            *(_QWORD *)(v54 + 8) = v134;
            if ( v134 )
              goto LABEL_45;
            goto LABEL_91;
          }
          v87 = 1;
          while ( v53 != -8 )
          {
            v108 = v87 + 1;
            v51 = v50 & (v87 + v51);
            v52 = (__int64 *)(v49 + 16LL * v51);
            v53 = *v52;
            if ( v40 == *v52 )
              goto LABEL_44;
            v87 = v108;
          }
        }
        v136 = v33;
        v54 = sub_145CBF0(v141, 64, 16);
        *(_QWORD *)v54 = v40;
        *(_QWORD *)(v54 + 8) = 0;
        v33 = v136;
LABEL_91:
        v55 = 0;
LABEL_45:
        *(_QWORD *)(v54 + 16) = v55;
        *(_DWORD *)(v54 + 24) = 0;
        *(_QWORD *)(v54 + 32) = 0;
        *(_DWORD *)(v54 + 40) = 0;
        *(_QWORD *)(v54 + 48) = 0;
        *(_QWORD *)(v54 + 56) = 0;
        v36[1] = v54;
        *(_QWORD *)(*(_QWORD *)(v33 + 48) + 8LL * v32) = v54;
        if ( *(_QWORD *)(v54 + 8) )
        {
          v56 = (unsigned int)v144;
          if ( (unsigned int)v144 >= HIDWORD(v144) )
          {
            v130 = v33;
            v139 = v54;
            sub_16CD150((__int64)&v143, v145, 0, 8, v33, v6);
            v56 = (unsigned int)v144;
            v33 = v130;
            v54 = v139;
          }
          ++v32;
          *(_QWORD *)&v143[8 * v56] = v54;
          LODWORD(v144) = v144 + 1;
          if ( *(_DWORD *)(v33 + 40) == v32 )
            goto LABEL_49;
          continue;
        }
        break;
      }
      v86 = (unsigned int)v150;
      if ( (unsigned int)v150 >= HIDWORD(v150) )
      {
        v132 = v33;
        v140 = v54;
        sub_16CD150((__int64)&v149, v151, 0, 8, v33, v6);
        v86 = (unsigned int)v150;
        v33 = v132;
        v54 = v140;
      }
      *(_QWORD *)&v149[8 * v86] = v54;
      LODWORD(v150) = v150 + 1;
LABEL_34:
      if ( *(_DWORD *)(v33 + 40) != ++v32 )
        continue;
      break;
    }
LABEL_49:
    v15 = v150;
  }
  while ( (_DWORD)v150 );
LABEL_50:
  v57 = sub_145CBF0(v141, 64, 16);
  v60 = v144;
  v135 = v57;
  v61 = v57;
  *(_QWORD *)v57 = 0;
  *(_QWORD *)(v57 + 8) = 0;
  *(_QWORD *)(v57 + 16) = 0;
  *(_DWORD *)(v57 + 24) = 0;
  *(_QWORD *)(v57 + 32) = 0;
  *(_DWORD *)(v57 + 40) = 0;
  *(_QWORD *)(v57 + 48) = 0;
  *(_QWORD *)(v57 + 56) = 0;
  v62 = (unsigned int)v150;
  if ( v60 )
  {
    do
    {
      v63 = *(_QWORD *)&v143[8 * v60 - 8];
      LODWORD(v144) = v60 - 1;
      *(_QWORD *)(v63 + 32) = v61;
      *(_DWORD *)(v63 + 24) = -1;
      if ( (unsigned int)v62 >= HIDWORD(v150) )
      {
        sub_16CD150((__int64)&v149, v151, 0, 8, v58, (int)v59);
        v62 = (unsigned int)v150;
      }
      *(_QWORD *)&v149[8 * v62] = v63;
      v60 = v144;
      v62 = (unsigned int)(v150 + 1);
      LODWORD(v150) = v150 + 1;
    }
    while ( (_DWORD)v144 );
  }
  v142 = 1;
  if ( !(_DWORD)v62 )
  {
    v109 = 1;
    goto LABEL_78;
  }
  while ( 2 )
  {
    while ( 2 )
    {
      v65 = *(_QWORD *)&v149[8 * v62 - 8];
      if ( *(_DWORD *)(v65 + 24) != -2 )
      {
        *(_DWORD *)(v65 + 24) = -2;
        v66 = sub_157EBA0(*(_QWORD *)v65);
        v67 = v66;
        if ( !v66 || (v68 = 0, (v69 = sub_15F4D60(v66)) == 0) )
        {
LABEL_76:
          v62 = (unsigned int)v150;
          if ( !(_DWORD)v150 )
            goto LABEL_77;
          continue;
        }
        while ( 1 )
        {
          v75 = sub_15F4DF0(v67, v68);
          v76 = *(_DWORD *)(a1 + 48);
          v77 = v75;
          if ( !v76 )
            break;
          v59 = *(__int64 **)(a1 + 32);
          v70 = ((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4);
          LODWORD(v58) = (v76 - 1) & v70;
          v71 = &v59[2 * (unsigned int)v58];
          v72 = *v71;
          if ( v77 != *v71 )
          {
            v98 = 1;
            v83 = 0;
            while ( v72 != -8 )
            {
              if ( v72 == -16 && !v83 )
                v83 = v71;
              LODWORD(v58) = (v76 - 1) & (v98 + v58);
              v71 = &v59[2 * (unsigned int)v58];
              v72 = *v71;
              if ( v77 == *v71 )
                goto LABEL_63;
              ++v98;
            }
            if ( !v83 )
              v83 = v71;
            v99 = *(_DWORD *)(a1 + 40);
            ++*(_QWORD *)(a1 + 24);
            v82 = v99 + 1;
            if ( 4 * (v99 + 1) < 3 * v76 )
            {
              LODWORD(v58) = v76 >> 3;
              if ( v76 - *(_DWORD *)(a1 + 44) - v82 <= v76 >> 3 )
              {
                v131 = v70;
                sub_1B3C650(v127, v76);
                v100 = *(_DWORD *)(a1 + 48);
                if ( !v100 )
                  goto LABEL_193;
                v101 = v100 - 1;
                v58 = *(_QWORD *)(a1 + 32);
                v59 = 0;
                v102 = 1;
                v103 = v101 & v131;
                v82 = *(_DWORD *)(a1 + 40) + 1;
                v83 = (__int64 *)(v58 + 16LL * (v101 & v131));
                v104 = *v83;
                if ( v77 != *v83 )
                {
                  while ( v104 != -8 )
                  {
                    if ( v104 == -16 && !v59 )
                      v59 = v83;
                    v103 = v101 & (v102 + v103);
                    v83 = (__int64 *)(v58 + 16LL * v103);
                    v104 = *v83;
                    if ( v77 == *v83 )
                      goto LABEL_73;
                    ++v102;
                  }
LABEL_114:
                  if ( v59 )
                    v83 = v59;
                }
              }
LABEL_73:
              *(_DWORD *)(a1 + 40) = v82;
              if ( *v83 != -8 )
                --*(_DWORD *)(a1 + 44);
              ++v68;
              *v83 = v77;
              v83[1] = 0;
              if ( v69 == v68 )
                goto LABEL_76;
              continue;
            }
LABEL_71:
            sub_1B3C650(v127, 2 * v76);
            v78 = *(_DWORD *)(a1 + 48);
            if ( !v78 )
              goto LABEL_193;
            v79 = v78 - 1;
            v80 = *(_QWORD *)(a1 + 32);
            v81 = (v78 - 1) & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
            v82 = *(_DWORD *)(a1 + 40) + 1;
            v83 = (__int64 *)(v80 + 16LL * v81);
            v58 = *v83;
            if ( v77 != *v83 )
            {
              v110 = 1;
              v59 = 0;
              while ( v58 != -8 )
              {
                if ( !v59 && v58 == -16 )
                  v59 = v83;
                v81 = v79 & (v110 + v81);
                v83 = (__int64 *)(v80 + 16LL * v81);
                v58 = *v83;
                if ( v77 == *v83 )
                  goto LABEL_73;
                ++v110;
              }
              goto LABEL_114;
            }
            goto LABEL_73;
          }
LABEL_63:
          v73 = v71[1];
          if ( v73 && !*(_DWORD *)(v73 + 24) )
          {
            *(_DWORD *)(v73 + 24) = -1;
            v74 = (unsigned int)v150;
            if ( (unsigned int)v150 >= HIDWORD(v150) )
            {
              sub_16CD150((__int64)&v149, v151, 0, 8, v58, (int)v59);
              v74 = (unsigned int)v150;
            }
            *(_QWORD *)&v149[8 * v74] = v73;
            LODWORD(v150) = v150 + 1;
          }
          if ( v69 == ++v68 )
            goto LABEL_76;
        }
        ++*(_QWORD *)(a1 + 24);
        goto LABEL_71;
      }
      break;
    }
    v64 = *(_QWORD *)(v65 + 8) == 0;
    *(_DWORD *)(v65 + 24) = v142;
    if ( v64 )
    {
      v105 = *(unsigned int *)(a3 + 8);
      if ( (unsigned int)v105 >= *(_DWORD *)(a3 + 12) )
      {
        sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v58, (int)v59);
        v105 = *(unsigned int *)(a3 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a3 + 8 * v105) = v65;
      ++*(_DWORD *)(a3 + 8);
    }
    ++v142;
    v62 = (unsigned int)(v150 - 1);
    LODWORD(v150) = v62;
    if ( (_DWORD)v62 )
      continue;
    break;
  }
LABEL_77:
  v109 = v142;
LABEL_78:
  v84 = v146;
  *(_DWORD *)(v135 + 24) = v109;
  if ( v84 != v148 )
    _libc_free((unsigned __int64)v84);
  if ( v149 != v151 )
    _libc_free((unsigned __int64)v149);
  if ( v143 != v145 )
    _libc_free((unsigned __int64)v143);
  return v135;
}
