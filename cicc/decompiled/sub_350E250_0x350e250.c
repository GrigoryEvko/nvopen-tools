// Function: sub_350E250
// Address: 0x350e250
//
__int64 __fastcall sub_350E250(__int64 a1)
{
  __int64 (*v1)(void); // rax
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // rdx
  _DWORD *v7; // rax
  _DWORD *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r15
  int v11; // eax
  unsigned int v12; // esi
  unsigned int v13; // r9d
  __int64 v14; // r8
  int v15; // r11d
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rdi
  int v20; // r11d
  __int64 v21; // rbx
  __int64 v22; // r12
  unsigned __int8 v23; // al
  int v24; // r13d
  int *v25; // rax
  int v26; // edi
  _DWORD *v27; // rax
  int v28; // esi
  __int64 v29; // rbx
  __int64 v30; // r12
  __int64 v31; // r13
  int v32; // esi
  __int64 v33; // rdi
  __int64 v34; // rsi
  int v36; // eax
  int v37; // edi
  __int64 v38; // rsi
  __int64 v39; // rbx
  int v40; // r10d
  __int64 *v41; // rdx
  unsigned int v42; // edi
  __int64 *v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // rax
  int v47; // ecx
  __int64 v48; // r8
  unsigned int v49; // ebx
  unsigned int v50; // r8d
  __int64 *v51; // rax
  __int64 v52; // rdi
  unsigned int v53; // eax
  int v54; // eax
  __int64 v55; // r12
  unsigned __int16 v56; // cx
  int v57; // r13d
  __int64 v58; // r13
  unsigned __int64 v59; // rdx
  __int64 v60; // rcx
  int v61; // r9d
  __int64 v62; // r12
  __int64 *v63; // r8
  __int64 v64; // rsi
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // r8
  __int64 (__fastcall *v68)(__int64); // rax
  unsigned int v69; // edi
  unsigned int v70; // ecx
  __int64 *v71; // rax
  __int64 v72; // r9
  unsigned int v73; // ecx
  __int64 *v74; // rdx
  __int64 v75; // r9
  unsigned int v76; // r10d
  int v77; // eax
  unsigned int v78; // r10d
  unsigned int v79; // ecx
  _DWORD *v80; // rdi
  __int64 v81; // rsi
  unsigned int v82; // eax
  int v83; // eax
  unsigned __int64 v84; // rax
  __int64 v85; // rax
  unsigned int v86; // ebx
  __int64 v87; // r12
  _DWORD *v88; // rax
  _DWORD *i; // rdx
  int v90; // edi
  __int64 v91; // rsi
  int v92; // edi
  int v93; // esi
  unsigned int v94; // ebx
  __int64 v95; // rdi
  __int64 v96; // rax
  __int64 v97; // rdx
  unsigned int v98; // ecx
  __int64 *v99; // rdx
  __int64 v100; // r9
  int v101; // edx
  int v102; // eax
  int v103; // r10d
  int v104; // r10d
  __int64 *v105; // r9
  int v106; // edx
  int v107; // r11d
  __int16 v108; // ax
  int v109; // ecx
  __int64 v110; // rax
  __int64 v111; // rdx
  __int64 v112; // rsi
  __int64 v113; // rax
  signed __int64 v114; // rdx
  __int64 v115; // rdi
  int v116; // ecx
  __int64 *v117; // rdx
  int v118; // edx
  unsigned int v119; // ecx
  int v120; // edi
  __int64 *v121; // rsi
  __int64 v122; // r8
  unsigned int v123; // ecx
  __int64 v124; // r8
  int v125; // edi
  _DWORD *v126; // rsi
  int v127; // r10d
  __int64 v128; // [rsp+0h] [rbp-100h]
  __int64 v129; // [rsp+8h] [rbp-F8h]
  _QWORD *v130; // [rsp+10h] [rbp-F0h]
  int v131; // [rsp+20h] [rbp-E0h]
  __int64 v132; // [rsp+20h] [rbp-E0h]
  int v133; // [rsp+20h] [rbp-E0h]
  unsigned int v134; // [rsp+30h] [rbp-D0h]
  int v135; // [rsp+30h] [rbp-D0h]
  __int64 v136; // [rsp+38h] [rbp-C8h]
  int v137; // [rsp+38h] [rbp-C8h]
  int v138; // [rsp+38h] [rbp-C8h]
  __int64 v139; // [rsp+40h] [rbp-C0h]
  __int64 v140; // [rsp+48h] [rbp-B8h]
  __int64 v141; // [rsp+50h] [rbp-B0h]
  unsigned int v142; // [rsp+58h] [rbp-A8h]
  int v143; // [rsp+58h] [rbp-A8h]
  char v144; // [rsp+6Fh] [rbp-91h] BYREF
  __int64 v145[2]; // [rsp+70h] [rbp-90h] BYREF
  char v146; // [rsp+80h] [rbp-80h]
  __int64 v147; // [rsp+90h] [rbp-70h] BYREF
  __int64 v148; // [rsp+98h] [rbp-68h]
  __int64 v149; // [rsp+A0h] [rbp-60h]
  unsigned int v150; // [rsp+A8h] [rbp-58h]
  __int64 v151; // [rsp+B0h] [rbp-50h] BYREF
  _DWORD *v152; // [rsp+B8h] [rbp-48h]
  __int64 v153; // [rsp+C0h] [rbp-40h]
  unsigned int v154; // [rsp+C8h] [rbp-38h]

  v128 = 0;
  v130 = *(_QWORD **)(a1 + 32);
  v1 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 16) + 128LL);
  if ( v1 != sub_2DAC790 )
    v128 = v1();
  v2 = *(_QWORD *)(a1 + 328);
  v147 = 0;
  v148 = 0;
  v149 = 0;
  v150 = 0;
  v151 = 0;
  v152 = 0;
  v153 = 0;
  v154 = 0;
  v141 = v2;
  v129 = a1 + 320;
  if ( v2 == a1 + 320 )
  {
    v33 = 0;
    v34 = 0;
    goto LABEL_59;
  }
  do
  {
    v140 = v141 + 48;
    if ( v141 + 48 == (*(_QWORD *)(v141 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
      goto LABEL_57;
    v3 = *(_QWORD *)(v141 + 56);
    if ( *(_BYTE *)(v141 + 216) )
    {
      v3 = sub_2E312E0(v141, *(_QWORD *)(v141 + 56), 0, 1);
      if ( v141 + 48 == v3 )
        goto LABEL_57;
    }
    sub_350D990(v3, (__int64)&v147);
    v4 = sub_2E312E0(v141, v3, 0, 1);
    ++v151;
    v5 = v4;
    if ( !(_DWORD)v153 )
    {
      if ( !HIDWORD(v153) )
        goto LABEL_13;
      v6 = v154;
      if ( v154 > 0x40 )
      {
        sub_C7D6A0((__int64)v152, 24LL * v154, 8);
        v152 = 0;
        v153 = 0;
        v154 = 0;
        goto LABEL_13;
      }
LABEL_10:
      v7 = v152;
      v8 = &v152[6 * v6];
      if ( v152 != v8 )
      {
        do
        {
          *v7 = -1;
          v7 += 6;
        }
        while ( v8 != v7 );
      }
      v153 = 0;
      goto LABEL_13;
    }
    v79 = 4 * v153;
    v6 = v154;
    if ( (unsigned int)(4 * v153) < 0x40 )
      v79 = 64;
    if ( v79 >= v154 )
      goto LABEL_10;
    v80 = v152;
    v81 = 6LL * v154;
    if ( (_DWORD)v153 == 1 )
    {
      v87 = 3072;
      v86 = 128;
LABEL_170:
      sub_C7D6A0((__int64)v152, v81 * 4, 8);
      v154 = v86;
      v88 = (_DWORD *)sub_C7D670(v87, 8);
      v153 = 0;
      v152 = v88;
      for ( i = &v88[6 * v154]; i != v88; v88 += 6 )
      {
        if ( v88 )
          *v88 = -1;
      }
      goto LABEL_13;
    }
    _BitScanReverse(&v82, v153 - 1);
    v83 = 1 << (33 - (v82 ^ 0x1F));
    if ( v83 < 64 )
      v83 = 64;
    if ( v83 != v154 )
    {
      v84 = ((unsigned __int64)(4 * v83 / 3u + 1) >> 1)
          | (4 * v83 / 3u + 1)
          | ((((unsigned __int64)(4 * v83 / 3u + 1) >> 1) | (4 * v83 / 3u + 1)) >> 2);
      v85 = (((v84 | (v84 >> 4)) >> 8) | v84 | (v84 >> 4) | ((((v84 | (v84 >> 4)) >> 8) | v84 | (v84 >> 4)) >> 16)) + 1;
      v86 = v85;
      v87 = 24 * v85;
      goto LABEL_170;
    }
    v153 = 0;
    v126 = &v152[v81];
    do
    {
      if ( v80 )
        *v80 = -1;
      v80 += 6;
    }
    while ( v126 != v80 );
LABEL_13:
    v144 = 0;
    if ( v140 == v5 )
      goto LABEL_57;
    while ( 1 )
    {
      if ( !v5 )
        BUG();
      v9 = v5;
      if ( (*(_BYTE *)v5 & 4) == 0 && (*(_BYTE *)(v5 + 44) & 8) != 0 )
      {
        do
          v9 = *(_QWORD *)(v9 + 8);
        while ( (*(_BYTE *)(v9 + 44) & 8) != 0 );
      }
      v10 = sub_2E312E0(v141, *(_QWORD *)(v9 + 8), 0, 1);
      if ( (unsigned int)*(unsigned __int16 *)(v5 + 68) - 1 > 1 || (*(_BYTE *)(*(_QWORD *)(v5 + 32) + 64LL) & 0x10) == 0 )
      {
        v11 = *(_DWORD *)(v5 + 44);
        if ( (v11 & 4) != 0 || (v11 & 8) == 0 )
        {
          if ( (*(_QWORD *)(*(_QWORD *)(v5 + 16) + 24LL) & 0x100000LL) == 0 )
          {
LABEL_23:
            v12 = v150;
            if ( !v150 )
              goto LABEL_62;
            goto LABEL_24;
          }
        }
        else if ( !sub_2E88A90(v5, 0x100000, 1) )
        {
          goto LABEL_23;
        }
      }
      v12 = v150;
      v144 = 1;
      if ( !v150 )
      {
LABEL_62:
        ++v147;
        goto LABEL_63;
      }
LABEL_24:
      v13 = v12 - 1;
      v14 = v148;
      v15 = 1;
      v16 = 0;
      v17 = (v12 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v18 = v148 + 16 * v17;
      v19 = *(_QWORD *)v18;
      if ( *(_QWORD *)v18 == v5 )
        goto LABEL_25;
      while ( 1 )
      {
        if ( v19 == -4096 )
        {
          if ( !v16 )
            v16 = v18;
          ++v147;
          v36 = v149 + 1;
          if ( 4 * ((int)v149 + 1) < 3 * v12 )
          {
            v17 = v12 - HIDWORD(v149) - v36;
            if ( (unsigned int)v17 > v12 >> 3 )
            {
LABEL_155:
              LODWORD(v149) = v36;
              if ( *(_QWORD *)v16 != -4096 )
                --HIDWORD(v149);
              *(_QWORD *)v16 = v5;
              v20 = 0;
              *(_DWORD *)(v16 + 8) = 0;
              goto LABEL_26;
            }
            sub_2E261E0((__int64)&v147, v12);
            if ( v150 )
            {
              v14 = v148;
              v93 = 1;
              v94 = (v150 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
              v17 = 0;
              v36 = v149 + 1;
              v16 = v148 + 16LL * v94;
              v95 = *(_QWORD *)v16;
              if ( *(_QWORD *)v16 != v5 )
              {
                while ( v95 != -4096 )
                {
                  if ( !v17 && v95 == -8192 )
                    v17 = v16;
                  v94 = (v150 - 1) & (v93 + v94);
                  v16 = v148 + 16LL * v94;
                  v95 = *(_QWORD *)v16;
                  if ( *(_QWORD *)v16 == v5 )
                    goto LABEL_155;
                  ++v93;
                }
                if ( v17 )
                  v16 = v17;
              }
              goto LABEL_155;
            }
LABEL_353:
            LODWORD(v149) = v149 + 1;
            BUG();
          }
LABEL_63:
          sub_2E261E0((__int64)&v147, 2 * v12);
          if ( v150 )
          {
            v17 = (v150 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
            v36 = v149 + 1;
            v16 = v148 + 16 * v17;
            v14 = *(_QWORD *)v16;
            if ( *(_QWORD *)v16 != v5 )
            {
              v37 = 1;
              v38 = 0;
              while ( v14 != -4096 )
              {
                if ( !v38 && v14 == -8192 )
                  v38 = v16;
                v17 = (v150 - 1) & (v37 + (_DWORD)v17);
                v16 = v148 + 16LL * (unsigned int)v17;
                v14 = *(_QWORD *)v16;
                if ( *(_QWORD *)v16 == v5 )
                  goto LABEL_155;
                ++v37;
              }
              if ( v38 )
                v16 = v38;
            }
            goto LABEL_155;
          }
          goto LABEL_353;
        }
        if ( v16 || v19 != -8192 )
          v18 = v16;
        v16 = (unsigned int)(v15 + 1);
        v107 = v17 + v15;
        v17 = v13 & v107;
        v19 = *(_QWORD *)(v148 + 16 * v17);
        if ( v19 == v5 )
          break;
        v15 = v16;
        v16 = v18;
        v18 = v148 + 16 * v17;
      }
      v18 = v148 + 16LL * (v13 & v107);
LABEL_25:
      v20 = *(_DWORD *)(v18 + 8);
LABEL_26:
      v21 = *(_QWORD *)(v5 + 32);
      v139 = 0;
      v142 = 0;
      v22 = v21 + 40LL * (*(_DWORD *)(v5 + 40) & 0xFFFFFF);
      while ( v22 != v21 )
      {
        while ( *(_BYTE *)v21 || (*(_BYTE *)(v21 + 4) & 8) != 0 )
        {
LABEL_29:
          v21 += 40;
          if ( v22 == v21 )
            goto LABEL_37;
        }
        v23 = *(_BYTE *)(v21 + 3);
        if ( (v23 & 0x10) != 0 )
        {
          v16 = *(unsigned __int8 *)(v21 + 3);
          LOBYTE(v16) = v23 >> 6;
          if ( (((v23 & 0x10) != 0) & (v23 >> 6)) != 0 )
          {
            v16 = *(unsigned int *)(v21 + 8);
            if ( v154 )
            {
              v28 = (v154 - 1) & (37 * v16);
              v17 = (__int64)&v152[6 * v28];
              v14 = *(unsigned int *)v17;
              if ( (_DWORD)v16 == (_DWORD)v14 )
              {
LABEL_45:
                if ( (_DWORD *)v17 != &v152[6 * v154] && *(_DWORD *)(v17 + 8) > v142 )
                {
                  v142 = *(_DWORD *)(v17 + 8);
                  v139 = *(_QWORD *)(v17 + 16);
                }
              }
              else
              {
                v17 = 1;
                while ( (_DWORD)v14 != -1 )
                {
                  v78 = v17 + 1;
                  v28 = (v154 - 1) & (v28 + v17);
                  v17 = (__int64)&v152[6 * v28];
                  v14 = *(unsigned int *)v17;
                  if ( (_DWORD)v16 == (_DWORD)v14 )
                    goto LABEL_45;
                  v17 = v78;
                }
              }
            }
          }
          goto LABEL_29;
        }
        v24 = *(_DWORD *)(v21 + 8);
        if ( !v154 )
        {
          ++v151;
          goto LABEL_176;
        }
        v17 = (unsigned int)(37 * v24);
        v14 = (v154 - 1) & (37 * v24);
        v25 = &v152[6 * v14];
        v26 = *v25;
        if ( v24 != *v25 )
        {
          v137 = 1;
          v16 = 0;
          while ( v26 != -1 )
          {
            if ( v26 == -2 && !v16 )
              v16 = (__int64)v25;
            v14 = (v154 - 1) & (v137 + (_DWORD)v14);
            v25 = &v152[6 * (unsigned int)v14];
            v26 = *v25;
            if ( v24 == *v25 )
              goto LABEL_35;
            ++v137;
          }
          if ( !v16 )
            v16 = (__int64)v25;
          ++v151;
          v77 = v153 + 1;
          if ( 4 * ((int)v153 + 1) < 3 * v154 )
          {
            v14 = v154 >> 3;
            if ( v154 - HIDWORD(v153) - v77 <= (unsigned int)v14 )
            {
              v135 = v20;
              sub_350E070((__int64)&v151, v154);
              if ( !v154 )
              {
LABEL_352:
                LODWORD(v153) = v153 + 1;
                BUG();
              }
              v91 = 0;
              v20 = v135;
              v92 = 1;
              v17 = (v154 - 1) & (37 * v24);
              v16 = (__int64)&v152[6 * v17];
              v14 = *(unsigned int *)v16;
              v77 = v153 + 1;
              if ( v24 != (_DWORD)v14 )
              {
                while ( (_DWORD)v14 != -1 )
                {
                  if ( (_DWORD)v14 == -2 && !v91 )
                    v91 = v16;
                  v17 = (v154 - 1) & (v92 + (_DWORD)v17);
                  v16 = (__int64)&v152[6 * (unsigned int)v17];
                  v14 = *(unsigned int *)v16;
                  if ( v24 == (_DWORD)v14 )
                    goto LABEL_146;
                  ++v92;
                }
                goto LABEL_180;
              }
            }
            goto LABEL_146;
          }
LABEL_176:
          v138 = v20;
          sub_350E070((__int64)&v151, 2 * v154);
          if ( !v154 )
            goto LABEL_352;
          v20 = v138;
          v17 = (v154 - 1) & (37 * v24);
          v16 = (__int64)&v152[6 * v17];
          v14 = *(unsigned int *)v16;
          v77 = v153 + 1;
          if ( v24 != (_DWORD)v14 )
          {
            v90 = 1;
            v91 = 0;
            while ( (_DWORD)v14 != -1 )
            {
              if ( (_DWORD)v14 == -2 && !v91 )
                v91 = v16;
              v17 = (v154 - 1) & (v90 + (_DWORD)v17);
              v16 = (__int64)&v152[6 * (unsigned int)v17];
              v14 = *(unsigned int *)v16;
              if ( v24 == (_DWORD)v14 )
                goto LABEL_146;
              ++v90;
            }
LABEL_180:
            if ( v91 )
              v16 = v91;
          }
LABEL_146:
          LODWORD(v153) = v77;
          if ( *(_DWORD *)v16 != -1 )
            --HIDWORD(v153);
          *(_DWORD *)v16 = v24;
          v27 = (_DWORD *)(v16 + 8);
          *(_DWORD *)(v16 + 8) = 0;
          *(_QWORD *)(v16 + 16) = 0;
          goto LABEL_36;
        }
LABEL_35:
        v27 = v25 + 2;
LABEL_36:
        v21 += 40;
        *v27 = v20;
        *((_QWORD *)v27 + 1) = v5;
      }
LABEL_37:
      if ( (unsigned __int8)sub_2E8B400(v5, (__int64)&v144, v16, v17, (_QWORD *)v14) )
        break;
      if ( !sub_2E8B090(v5) || *(_WORD *)(v5 + 68) == 24 )
        goto LABEL_56;
      if ( v140 == v10 )
        goto LABEL_57;
      sub_350D990(v10, (__int64)&v147);
      v144 = 0;
LABEL_42:
      v5 = v10;
    }
    v29 = *(_QWORD *)(v5 + 32);
    v30 = v29 + 40LL * (*(_DWORD *)(v5 + 40) & 0xFFFFFF);
    if ( v29 == v30 )
      goto LABEL_56;
    v136 = 0;
    v31 = 0;
    v134 = 0;
    do
    {
      while ( 1 )
      {
        while ( 1 )
        {
          if ( !*(_BYTE *)v29
            && (((*(_BYTE *)(v29 + 3) & 0x10) != 0) & (*(_BYTE *)(v29 + 3) >> 6)) == 0
            && (*(_BYTE *)(v29 + 4) & 8) == 0 )
          {
            v32 = *(_DWORD *)(v29 + 8);
            if ( v32 >= 0 )
            {
              if ( v32 && !(unsigned __int8)sub_2EBF3A0(v130, v32) )
                goto LABEL_56;
              goto LABEL_73;
            }
            if ( (*(_BYTE *)(v29 + 3) & 0x10) != 0 )
            {
              if ( v31 )
                goto LABEL_56;
              v31 = v29;
              goto LABEL_73;
            }
            v131 = *(_DWORD *)(v29 + 8);
            if ( !(unsigned __int8)sub_2EBEF70((__int64)v130, v32) )
              goto LABEL_56;
            if ( !sub_2DADE10((__int64)v130, v131) )
              goto LABEL_56;
            if ( !v31 )
              goto LABEL_56;
            v65 = v130[7];
            if ( (*(_QWORD *)(v65 + 16LL * (*(_DWORD *)(v31 + 8) & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL) != (*(_QWORD *)(v65 + 16LL * (*(_DWORD *)(v29 + 8) & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL) )
              goto LABEL_56;
            v66 = *(_QWORD *)(v65 + 16LL * (v131 & 0x7FFFFFFF) + 8);
            if ( v66 )
            {
              if ( (*(_BYTE *)(v66 + 3) & 0x10) == 0 )
              {
                v66 = *(_QWORD *)(v66 + 32);
                if ( v66 )
                {
                  if ( (*(_BYTE *)(v66 + 3) & 0x10) == 0 )
                    BUG();
                }
              }
            }
            v67 = *(_QWORD *)(v66 + 16);
            if ( *(_WORD *)(v67 + 68) != 20 )
            {
              v68 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v128 + 520LL);
              if ( v68 == sub_2DCA430
                || (v132 = v67,
                    ((void (__fastcall *)(__int64 *, __int64, __int64))v68)(v145, v128, v67),
                    v67 = v132,
                    !v146) )
              {
                ++v134;
              }
            }
            if ( v150 )
            {
              v69 = v150 - 1;
              v70 = (v150 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
              v71 = (__int64 *)(v148 + 16LL * v70);
              v72 = *v71;
              if ( v67 != *v71 )
              {
                v102 = 1;
                while ( v72 != -4096 )
                {
                  v103 = v102 + 1;
                  v70 = v69 & (v102 + v70);
                  v71 = (__int64 *)(v148 + 16LL * v70);
                  v72 = *v71;
                  if ( v67 == *v71 )
                    goto LABEL_132;
                  v102 = v103;
                }
                goto LABEL_73;
              }
LABEL_132:
              if ( (__int64 *)(v148 + 16LL * v150) != v71 )
                break;
            }
          }
LABEL_73:
          v29 += 40;
          if ( v30 == v29 )
            goto LABEL_74;
        }
        if ( v136 )
          break;
LABEL_191:
        v29 += 40;
        v136 = v67;
        if ( v30 == v29 )
          goto LABEL_74;
      }
      v73 = v69 & (((unsigned int)v136 >> 9) ^ ((unsigned int)v136 >> 4));
      v74 = (__int64 *)(v148 + 16LL * v73);
      v75 = *v74;
      if ( *v74 != v136 )
      {
        v106 = 1;
        while ( v75 != -4096 )
        {
          v127 = v106 + 1;
          v73 = v69 & (v106 + v73);
          v74 = (__int64 *)(v148 + 16LL * v73);
          v75 = *v74;
          if ( *v74 == v136 )
            goto LABEL_135;
          v106 = v127;
        }
        v74 = (__int64 *)(v148 + 16LL * v150);
      }
LABEL_135:
      v76 = *((_DWORD *)v71 + 2);
      if ( *((_DWORD *)v74 + 2) == v76 )
      {
        v96 = *(_QWORD *)(v136 + 8);
        if ( v96 == *(_QWORD *)(v136 + 24) + 48LL )
          v96 = 0;
        while ( 1 )
        {
          v98 = v69 & (((unsigned int)v96 >> 9) ^ ((unsigned int)v96 >> 4));
          v99 = (__int64 *)(v148 + 16LL * v98);
          v100 = *v99;
          if ( v96 != *v99 )
          {
            v101 = 1;
            while ( v100 != -4096 )
            {
              v98 = v69 & (v101 + v98);
              v133 = v101 + 1;
              v99 = (__int64 *)(v148 + 16LL * v98);
              v100 = *v99;
              if ( v96 == *v99 )
                goto LABEL_205;
              v101 = v133;
            }
            v99 = (__int64 *)(v148 + 16LL * v150);
          }
LABEL_205:
          if ( v76 != *((_DWORD *)v99 + 2) )
            goto LABEL_73;
          if ( v67 == v96 )
            goto LABEL_191;
          v97 = *(_QWORD *)(v96 + 24);
          v96 = *(_QWORD *)(v96 + 8);
          if ( v96 == v97 + 48 )
            v96 = 0;
        }
      }
      if ( *((_DWORD *)v74 + 2) >= v76 )
        v67 = v136;
      v29 += 40;
      v136 = v67;
    }
    while ( v30 != v29 );
LABEL_74:
    v39 = v136;
    while ( v39 )
    {
      if ( !v150 )
      {
        ++v147;
        goto LABEL_82;
      }
      v40 = 1;
      v41 = 0;
      v42 = (v150 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
      v43 = (__int64 *)(v148 + 16LL * v42);
      v44 = *v43;
      if ( *v43 != v39 )
      {
        while ( v44 != -4096 )
        {
          if ( v44 == -8192 && !v41 )
            v41 = v43;
          v42 = (v150 - 1) & (v40 + v42);
          v43 = (__int64 *)(v148 + 16LL * v42);
          v44 = *v43;
          if ( *v43 == v39 )
            goto LABEL_76;
          ++v40;
        }
        if ( !v41 )
          v41 = v43;
        ++v147;
        v47 = v149 + 1;
        if ( 4 * ((int)v149 + 1) >= 3 * v150 )
        {
LABEL_82:
          sub_2E261E0((__int64)&v147, 2 * v150);
          if ( !v150 )
            goto LABEL_349;
          LODWORD(v46) = (v150 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
          v47 = v149 + 1;
          v41 = (__int64 *)(v148 + 16LL * (unsigned int)v46);
          v48 = *v41;
          if ( *v41 != v39 )
          {
            v104 = 1;
            v105 = 0;
            while ( v48 != -4096 )
            {
              if ( v48 == -8192 && !v105 )
                v105 = v41;
              v46 = (v150 - 1) & ((_DWORD)v46 + v104);
              v41 = (__int64 *)(v148 + 16 * v46);
              v48 = *v41;
              if ( *v41 == v39 )
                goto LABEL_84;
              ++v104;
            }
            if ( v105 )
              v41 = v105;
          }
        }
        else if ( v150 - HIDWORD(v149) - v47 <= v150 >> 3 )
        {
          sub_2E261E0((__int64)&v147, v150);
          if ( !v150 )
          {
LABEL_349:
            LODWORD(v149) = v149 + 1;
            BUG();
          }
          v61 = 1;
          LODWORD(v62) = (v150 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
          v63 = 0;
          v47 = v149 + 1;
          v41 = (__int64 *)(v148 + 16LL * (unsigned int)v62);
          v64 = *v41;
          if ( v39 != *v41 )
          {
            while ( v64 != -4096 )
            {
              if ( v64 == -8192 && !v63 )
                v63 = v41;
              v62 = (v150 - 1) & ((_DWORD)v62 + v61);
              v41 = (__int64 *)(v148 + 16 * v62);
              v64 = *v41;
              if ( *v41 == v39 )
                goto LABEL_84;
              ++v61;
            }
            if ( v63 )
              v41 = v63;
          }
        }
LABEL_84:
        LODWORD(v149) = v47;
        if ( *v41 != -4096 )
          --HIDWORD(v149);
        *v41 = v39;
        *((_DWORD *)v41 + 2) = 0;
        if ( v142 )
          break;
        goto LABEL_77;
      }
LABEL_76:
      if ( v142 != *((_DWORD *)v43 + 2) )
        break;
LABEL_77:
      if ( v39 == v139 )
        goto LABEL_56;
      v45 = *(_QWORD *)(v39 + 24);
      v39 = *(_QWORD *)(v39 + 8);
      if ( v39 == v45 + 48 )
        break;
    }
    if ( v134 <= 1 || v136 == 0 || v31 == 0 )
      goto LABEL_56;
    if ( !v150 )
    {
      ++v147;
      goto LABEL_277;
    }
    v49 = ((unsigned int)v136 >> 9) ^ ((unsigned int)v136 >> 4);
    v50 = (v150 - 1) & v49;
    v51 = (__int64 *)(v148 + 16LL * v50);
    v52 = *v51;
    if ( *v51 != v136 )
    {
      v116 = 1;
      v117 = 0;
      while ( v52 != -4096 )
      {
        if ( v52 == -8192 && !v117 )
          v117 = v51;
        v50 = (v150 - 1) & (v116 + v50);
        v51 = (__int64 *)(v148 + 16LL * v50);
        v52 = *v51;
        if ( *v51 == v136 )
          goto LABEL_90;
        ++v116;
      }
      if ( v117 )
        v51 = v117;
      ++v147;
      v118 = v149 + 1;
      if ( 4 * ((int)v149 + 1) < 3 * v150 )
      {
        if ( v150 - HIDWORD(v149) - v118 > v150 >> 3 )
        {
LABEL_265:
          LODWORD(v149) = v118;
          if ( *v51 != -4096 )
            --HIDWORD(v149);
          *((_DWORD *)v51 + 2) = 0;
          *v51 = v136;
          v53 = 0;
          goto LABEL_91;
        }
        sub_2E261E0((__int64)&v147, v150);
        if ( v150 )
        {
          v119 = (v150 - 1) & v49;
          v120 = 1;
          v118 = v149 + 1;
          v121 = 0;
          v51 = (__int64 *)(v148 + 16LL * v119);
          v122 = *v51;
          if ( *v51 == v136 )
            goto LABEL_265;
          while ( v122 != -4096 )
          {
            if ( v122 == -8192 && !v121 )
              v121 = v51;
            v119 = (v150 - 1) & (v120 + v119);
            v51 = (__int64 *)(v148 + 16LL * v119);
            v122 = *v51;
            if ( *v51 == v136 )
              goto LABEL_265;
            ++v120;
          }
LABEL_281:
          if ( v121 )
            v51 = v121;
          goto LABEL_265;
        }
        goto LABEL_350;
      }
LABEL_277:
      sub_2E261E0((__int64)&v147, 2 * v150);
      if ( v150 )
      {
        v118 = v149 + 1;
        v123 = (v150 - 1) & (((unsigned int)v136 >> 9) ^ ((unsigned int)v136 >> 4));
        v51 = (__int64 *)(v148 + 16LL * v123);
        v124 = *v51;
        if ( *v51 == v136 )
          goto LABEL_265;
        v125 = 1;
        v121 = 0;
        while ( v124 != -4096 )
        {
          if ( !v121 && v124 == -8192 )
            v121 = v51;
          v123 = (v150 - 1) & (v125 + v123);
          v51 = (__int64 *)(v148 + 16LL * v123);
          v124 = *v51;
          if ( *v51 == v136 )
            goto LABEL_265;
          ++v125;
        }
        goto LABEL_281;
      }
LABEL_350:
      LODWORD(v149) = v149 + 1;
      BUG();
    }
LABEL_90:
    v53 = *((_DWORD *)v51 + 2);
LABEL_91:
    if ( v53 >= v142 )
    {
      v54 = v136;
      v55 = *(_QWORD *)(v136 + 8);
      if ( v140 != v55 )
      {
        while ( 1 )
        {
          v56 = *(_WORD *)(v55 + 68);
          LOBYTE(v54) = v56 == 68;
          if ( v56 <= 0x18u )
            v54 |= (0x107C001uLL >> v56) & 1;
          if ( !(_BYTE)v54 )
            break;
          if ( (*(_BYTE *)v55 & 4) != 0 )
          {
            v55 = *(_QWORD *)(v55 + 8);
            if ( v140 == v55 )
              break;
          }
          else
          {
            while ( (*(_BYTE *)(v55 + 44) & 8) != 0 )
              v55 = *(_QWORD *)(v55 + 8);
            v55 = *(_QWORD *)(v55 + 8);
            if ( v140 == v55 )
              break;
          }
        }
      }
      if ( v5 != v55 )
      {
        v145[0] = v55;
        v57 = *(_DWORD *)sub_350DE20((__int64)&v147, v145);
        v145[0] = v5;
        v143 = v57;
        *(_DWORD *)sub_350DE20((__int64)&v147, v145) = v57;
        v58 = *(_QWORD *)(v5 + 8);
        if ( **(_BYTE **)(v5 + 32) )
        {
LABEL_102:
          if ( v5 != v58 && v55 != v58 )
          {
            sub_2E310C0((__int64 *)(v141 + 40), (__int64 *)(v141 + 40), v5, v58);
            v59 = *(_QWORD *)v58 & 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)((*(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v58;
            *(_QWORD *)v58 = *(_QWORD *)v58 & 7LL | *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL;
            v60 = *(_QWORD *)v55;
            *(_QWORD *)(v59 + 8) = v55;
            v60 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)v5 = v60 | *(_QWORD *)v5 & 7LL;
            *(_QWORD *)(v60 + 8) = v5;
            *(_QWORD *)v55 = v59 | *(_QWORD *)v55 & 7LL;
          }
          goto LABEL_56;
        }
        while ( 2 )
        {
          if ( v140 == v58 )
            goto LABEL_102;
          v108 = *(_WORD *)(v58 + 68);
          if ( v108 == 14 )
          {
            v109 = *(_DWORD *)(*(_QWORD *)(v5 + 32) + 8LL);
            v113 = *(_QWORD *)(v58 + 32);
            v112 = v113 + 40;
          }
          else
          {
            if ( v108 != 15 )
              goto LABEL_102;
            v109 = *(_DWORD *)(*(_QWORD *)(v5 + 32) + 8LL);
            v110 = *(_QWORD *)(v58 + 32);
            v111 = 40LL * (*(_DWORD *)(v58 + 40) & 0xFFFFFF);
            v112 = v110 + v111;
            v113 = v110 + 80;
            v114 = 0xCCCCCCCCCCCCCCCDLL * ((v111 - 80) >> 3);
            v115 = v114 >> 2;
            if ( v114 >> 2 > 0 )
            {
              while ( *(_BYTE *)v113 || v109 != *(_DWORD *)(v113 + 8) )
              {
                if ( !*(_BYTE *)(v113 + 40) && v109 == *(_DWORD *)(v113 + 48) )
                {
                  v113 += 40;
                  goto LABEL_256;
                }
                if ( !*(_BYTE *)(v113 + 80) && v109 == *(_DWORD *)(v113 + 88) )
                {
                  v113 += 80;
                  goto LABEL_256;
                }
                if ( !*(_BYTE *)(v113 + 120) && v109 == *(_DWORD *)(v113 + 128) )
                {
                  v113 += 120;
                  goto LABEL_256;
                }
                v113 += 160;
                if ( !--v115 )
                {
                  v114 = 0xCCCCCCCCCCCCCCCDLL * ((v112 - v113) >> 3);
                  goto LABEL_295;
                }
              }
              goto LABEL_256;
            }
LABEL_295:
            switch ( v114 )
            {
              case 2LL:
LABEL_300:
                if ( !*(_BYTE *)v113 && v109 == *(_DWORD *)(v113 + 8) )
                  goto LABEL_256;
                v113 += 40;
                break;
              case 3LL:
                if ( *(_BYTE *)v113 || v109 != *(_DWORD *)(v113 + 8) )
                {
                  v113 += 40;
                  goto LABEL_300;
                }
LABEL_256:
                if ( v113 == v112 )
                  goto LABEL_102;
                v145[0] = v58;
                *(_DWORD *)sub_350DE20((__int64)&v147, v145) = v143;
                if ( (*(_BYTE *)v58 & 4) == 0 )
                {
                  while ( (*(_BYTE *)(v58 + 44) & 8) != 0 )
                    v58 = *(_QWORD *)(v58 + 8);
                }
                v58 = *(_QWORD *)(v58 + 8);
                continue;
              case 1LL:
                break;
              default:
                goto LABEL_102;
            }
          }
          break;
        }
        if ( *(_BYTE *)v113 || *(_DWORD *)(v113 + 8) != v109 )
          goto LABEL_102;
        goto LABEL_256;
      }
    }
LABEL_56:
    if ( v140 != v10 )
      goto LABEL_42;
LABEL_57:
    v141 = *(_QWORD *)(v141 + 8);
  }
  while ( v129 != v141 );
  v33 = (__int64)v152;
  v34 = 24LL * v154;
LABEL_59:
  sub_C7D6A0(v33, v34, 8);
  return sub_C7D6A0(v148, 16LL * v150, 8);
}
