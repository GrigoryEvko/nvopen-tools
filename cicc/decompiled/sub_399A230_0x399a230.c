// Function: sub_399A230
// Address: 0x399a230
//
void __fastcall sub_399A230(__int64 a1, __int64 *a2)
{
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r9
  __int64 v9; // r13
  int v10; // r8d
  int v11; // r9d
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // r15
  __int64 v16; // rax
  _BYTE **v17; // r12
  _BYTE *v18; // rdx
  __int64 v19; // r14
  unsigned __int64 v20; // r13
  unsigned __int64 v21; // rsi
  int v22; // r8d
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rax
  _QWORD *v25; // rcx
  unsigned __int64 v26; // rbx
  unsigned int j; // edi
  _QWORD *v28; // rax
  _BYTE *v29; // r9
  __int64 v30; // rax
  __int64 *v31; // rsi
  _QWORD *v32; // rsi
  int v33; // ecx
  int v34; // r8d
  int v35; // r10d
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rdi
  unsigned __int64 v38; // rax
  int v39; // edi
  _QWORD *v40; // r9
  unsigned int kk; // eax
  _BYTE *v42; // r11
  __int64 *v43; // rax
  char v44; // dl
  __int64 v45; // rdi
  int v46; // r13d
  _QWORD *v47; // rbx
  _QWORD *v48; // r12
  unsigned int v49; // eax
  unsigned __int64 v50; // rdi
  void *v51; // rdi
  unsigned int v52; // eax
  __int64 v53; // rdx
  void *v54; // rdi
  unsigned int v55; // eax
  __int64 v56; // rdx
  unsigned __int64 v57; // rdi
  unsigned int v58; // edi
  unsigned __int64 v59; // rcx
  unsigned __int64 v60; // rax
  _QWORD *v61; // rax
  __int64 *v62; // r14
  _QWORD *ii; // rcx
  __int64 v64; // rax
  __int64 v65; // r9
  int v66; // r15d
  __int64 *v67; // rbx
  __int64 v68; // rdi
  unsigned __int64 v69; // rdi
  unsigned __int64 v70; // rdi
  unsigned int jj; // ecx
  __int64 *v72; // rdi
  __int64 v73; // r10
  __int64 *v74; // rsi
  unsigned int v75; // edi
  __int64 *v76; // rcx
  __int64 v77; // rax
  unsigned __int64 v78; // rdi
  unsigned __int64 v79; // r15
  unsigned __int64 v80; // rdi
  unsigned int v81; // eax
  unsigned int v82; // ecx
  __int64 v83; // rdx
  int v84; // ebx
  unsigned int v85; // eax
  _QWORD *v86; // rdi
  unsigned __int64 v87; // rdx
  unsigned __int64 v88; // rax
  _QWORD *v89; // rax
  __int64 v90; // rcx
  _QWORD *i; // rdx
  unsigned __int64 v92; // rax
  __int64 v93; // rax
  _QWORD *v94; // rax
  __int64 *v95; // r11
  _QWORD *k; // rcx
  __int64 *v97; // rcx
  __int64 v98; // r14
  __int64 v99; // rdi
  int v100; // r15d
  __int64 *v101; // r10
  __int64 v102; // rdx
  unsigned __int64 v103; // rdx
  unsigned __int64 v104; // rdx
  unsigned int m; // eax
  __int64 *v106; // rdx
  __int64 v107; // r8
  _QWORD *v108; // rsi
  int v109; // ecx
  int v110; // eax
  int v111; // r9d
  _QWORD *v112; // rdi
  unsigned int n; // ebx
  _BYTE *v114; // r10
  unsigned int v115; // ebx
  int v116; // edi
  int v117; // edi
  unsigned int v118; // eax
  int v119; // edx
  int v120; // r10d
  _QWORD *v121; // rax
  _BYTE *v122; // [rsp+8h] [rbp-98h]
  _BYTE *v123; // [rsp+10h] [rbp-90h]
  _BYTE *v124; // [rsp+10h] [rbp-90h]
  __int64 v125; // [rsp+10h] [rbp-90h]
  _BYTE *v126; // [rsp+10h] [rbp-90h]
  __int64 v127; // [rsp+10h] [rbp-90h]
  __int64 v128; // [rsp+18h] [rbp-88h]
  __int64 v129; // [rsp+20h] [rbp-80h]
  __int64 *v130; // [rsp+28h] [rbp-78h]
  __int64 v131; // [rsp+30h] [rbp-70h]
  __int64 *v132; // [rsp+38h] [rbp-68h]
  __int64 v133; // [rsp+40h] [rbp-60h]
  __int64 v134; // [rsp+48h] [rbp-58h]
  __int64 v135; // [rsp+50h] [rbp-50h] BYREF
  _QWORD *v136; // [rsp+58h] [rbp-48h]
  __int64 v137; // [rsp+60h] [rbp-40h]
  __int64 v138; // [rsp+68h] [rbp-38h]

  v2 = a1;
  v129 = sub_1626D20(*a2);
  *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL) + 8LL) + 1164LL) = 0;
  v128 = *(_QWORD *)(a1 + 288);
  v3 = *(unsigned int *)(a1 + 544);
  if ( !(_DWORD)v3 )
    goto LABEL_228;
  v4 = *(_QWORD *)(v129 + 8 * (5LL - *(unsigned int *)(v129 + 8)));
  v5 = *(_QWORD *)(v2 + 528);
  v6 = (v3 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v7 = (__int64 *)(v5 + 16LL * v6);
  v8 = *v7;
  if ( v4 != *v7 )
  {
    v119 = 1;
    while ( v8 != -8 )
    {
      v120 = v119 + 1;
      v6 = (v3 - 1) & (v119 + v6);
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( v4 == *v7 )
        goto LABEL_3;
      v119 = v120;
    }
LABEL_228:
    BUG();
  }
LABEL_3:
  if ( v7 == (__int64 *)(v5 + 16 * v3) )
    goto LABEL_228;
  v9 = *(_QWORD *)(*(_QWORD *)(v2 + 552) + 16LL * *((unsigned int *)v7 + 2) + 8);
  if ( *(_DWORD *)(*(_QWORD *)(v9 + 80) + 36LL) != 3 )
  {
    v135 = 0;
    v136 = 0;
    v137 = 0;
    v138 = 0;
    sub_3994310(v2, v9, v129, (__int64)&v135);
    sub_39C7B40(v9, *(_QWORD *)(*(_QWORD *)(v2 + 8) + 384LL), *(_QWORD *)(*(_QWORD *)(v2 + 8) + 392LL));
    v12 = *(_QWORD *)(v9 + 80);
    v13 = *(unsigned int *)(v2 + 248);
    if ( !*(_BYTE *)(v12 + 49) && *(_DWORD *)(v12 + 36) == 2 && !(_DWORD)v13 )
    {
      if ( !*(_BYTE *)(v2 + 5409) )
        goto LABEL_67;
      goto LABEL_34;
    }
    v14 = *(__int64 **)(v2 + 240);
    v130 = &v14[v13];
    if ( v130 == v14 )
    {
LABEL_34:
      v43 = *(__int64 **)(v2 + 3704);
      if ( *(__int64 **)(v2 + 3712) != v43 )
        goto LABEL_35;
      v74 = &v43[*(unsigned int *)(v2 + 3724)];
      v75 = *(_DWORD *)(v2 + 3724);
      if ( v43 != v74 )
      {
        v10 = v129;
        v76 = 0;
        while ( v129 != *v43 )
        {
          if ( *v43 == -2 )
            v76 = v43;
          if ( v74 == ++v43 )
          {
            if ( !v76 )
              goto LABEL_212;
            *v76 = v129;
            --*(_DWORD *)(v2 + 3728);
            ++*(_QWORD *)(v2 + 3696);
            goto LABEL_111;
          }
        }
LABEL_36:
        sub_39CEBA0(v9, v129, v128);
        v45 = *(_QWORD *)(v9 + 616);
        if ( v45 && *(_DWORD *)(v2 + 248) && *(_BYTE *)(*(_QWORD *)(v9 + 80) + 48LL) )
          sub_39CEBA0(v45, v129, v128);
        v46 = *(_DWORD *)(v2 + 4320);
        ++*(_QWORD *)(v2 + 4304);
        if ( v46 || *(_DWORD *)(v2 + 4324) )
        {
          v47 = *(_QWORD **)(v2 + 4312);
          v48 = &v47[17 * *(unsigned int *)(v2 + 4328)];
          v49 = 4 * v46;
          if ( (unsigned int)(4 * v46) < 0x40 )
            v49 = 64;
          if ( *(_DWORD *)(v2 + 4328) <= v49 )
          {
            while ( v47 != v48 )
            {
              if ( *v47 != -8 )
              {
                if ( *v47 != -16 )
                {
                  v50 = v47[7];
                  if ( (_QWORD *)v50 != v47 + 9 )
                    _libc_free(v50);
                  sub_3985EB0(v47[3]);
                }
                *v47 = -8;
              }
              v47 += 17;
            }
          }
          else
          {
            do
            {
              if ( *v47 != -16 && *v47 != -8 )
              {
                v78 = v47[7];
                if ( (_QWORD *)v78 != v47 + 9 )
                  _libc_free(v78);
                v79 = v47[3];
                while ( v79 )
                {
                  sub_3985EB0(*(_QWORD *)(v79 + 24));
                  v80 = v79;
                  v79 = *(_QWORD *)(v79 + 16);
                  j_j___libc_free_0(v80);
                }
              }
              v47 += 17;
            }
            while ( v47 != v48 );
            v83 = *(unsigned int *)(v2 + 4328);
            if ( v46 )
            {
              v84 = 64;
              if ( v46 != 1 )
              {
                _BitScanReverse(&v85, v46 - 1);
                v84 = 1 << (33 - (v85 ^ 0x1F));
                if ( v84 < 64 )
                  v84 = 64;
              }
              v86 = *(_QWORD **)(v2 + 4312);
              if ( (_DWORD)v83 == v84 )
              {
                *(_QWORD *)(v2 + 4320) = 0;
                v121 = &v86[17 * v83];
                do
                {
                  if ( v86 )
                    *v86 = -8;
                  v86 += 17;
                }
                while ( v121 != v86 );
              }
              else
              {
                j___libc_free_0((unsigned __int64)v86);
                v87 = ((((((((4 * v84 / 3u + 1) | ((unsigned __int64)(4 * v84 / 3u + 1) >> 1)) >> 2)
                         | (4 * v84 / 3u + 1)
                         | ((unsigned __int64)(4 * v84 / 3u + 1) >> 1)) >> 4)
                       | (((4 * v84 / 3u + 1) | ((unsigned __int64)(4 * v84 / 3u + 1) >> 1)) >> 2)
                       | (4 * v84 / 3u + 1)
                       | ((unsigned __int64)(4 * v84 / 3u + 1) >> 1)) >> 8)
                     | (((((4 * v84 / 3u + 1) | ((unsigned __int64)(4 * v84 / 3u + 1) >> 1)) >> 2)
                       | (4 * v84 / 3u + 1)
                       | ((unsigned __int64)(4 * v84 / 3u + 1) >> 1)) >> 4)
                     | (((4 * v84 / 3u + 1) | ((unsigned __int64)(4 * v84 / 3u + 1) >> 1)) >> 2)
                     | (4 * v84 / 3u + 1)
                     | ((unsigned __int64)(4 * v84 / 3u + 1) >> 1)) >> 16;
                v88 = (v87
                     | (((((((4 * v84 / 3u + 1) | ((unsigned __int64)(4 * v84 / 3u + 1) >> 1)) >> 2)
                         | (4 * v84 / 3u + 1)
                         | ((unsigned __int64)(4 * v84 / 3u + 1) >> 1)) >> 4)
                       | (((4 * v84 / 3u + 1) | ((unsigned __int64)(4 * v84 / 3u + 1) >> 1)) >> 2)
                       | (4 * v84 / 3u + 1)
                       | ((unsigned __int64)(4 * v84 / 3u + 1) >> 1)) >> 8)
                     | (((((4 * v84 / 3u + 1) | ((unsigned __int64)(4 * v84 / 3u + 1) >> 1)) >> 2)
                       | (4 * v84 / 3u + 1)
                       | ((unsigned __int64)(4 * v84 / 3u + 1) >> 1)) >> 4)
                     | (((4 * v84 / 3u + 1) | ((unsigned __int64)(4 * v84 / 3u + 1) >> 1)) >> 2)
                     | (4 * v84 / 3u + 1)
                     | ((unsigned __int64)(4 * v84 / 3u + 1) >> 1))
                    + 1;
                *(_DWORD *)(v2 + 4328) = v88;
                v89 = (_QWORD *)sub_22077B0(136 * v88);
                v90 = *(unsigned int *)(v2 + 4328);
                *(_QWORD *)(v2 + 4320) = 0;
                *(_QWORD *)(v2 + 4312) = v89;
                for ( i = &v89[17 * v90]; i != v89; v89 += 17 )
                {
                  if ( v89 )
                    *v89 = -8;
                }
              }
              goto LABEL_55;
            }
            if ( (_DWORD)v83 )
            {
              j___libc_free_0(*(_QWORD *)(v2 + 4312));
              *(_QWORD *)(v2 + 4312) = 0;
              *(_QWORD *)(v2 + 4320) = 0;
              *(_DWORD *)(v2 + 4328) = 0;
              goto LABEL_55;
            }
          }
          *(_QWORD *)(v2 + 4320) = 0;
        }
LABEL_55:
        ++*(_QWORD *)(v2 + 6592);
        v51 = *(void **)(v2 + 6608);
        if ( v51 != *(void **)(v2 + 6600) )
        {
          v52 = 4 * (*(_DWORD *)(v2 + 6620) - *(_DWORD *)(v2 + 6624));
          v53 = *(unsigned int *)(v2 + 6616);
          if ( v52 < 0x20 )
            v52 = 32;
          if ( v52 < (unsigned int)v53 )
          {
            sub_16CC920(v2 + 6592);
LABEL_61:
            ++*(_QWORD *)(v2 + 6664);
            v54 = *(void **)(v2 + 6680);
            if ( v54 != *(void **)(v2 + 6672) )
            {
              v55 = 4 * (*(_DWORD *)(v2 + 6692) - *(_DWORD *)(v2 + 6696));
              v56 = *(unsigned int *)(v2 + 6688);
              if ( v55 < 0x20 )
                v55 = 32;
              if ( v55 < (unsigned int)v56 )
              {
                sub_16CC920(v2 + 6664);
                goto LABEL_67;
              }
              memset(v54, -1, 8 * v56);
            }
            *(_QWORD *)(v2 + 6692) = 0;
LABEL_67:
            *(_QWORD *)(v2 + 32) = 0;
            v57 = (unsigned __int64)v136;
            *(_QWORD *)(v2 + 4008) = 0;
            j___libc_free_0(v57);
            return;
          }
          memset(v51, -1, 8 * v53);
        }
        *(_QWORD *)(v2 + 6620) = 0;
        goto LABEL_61;
      }
LABEL_212:
      if ( v75 < *(_DWORD *)(v2 + 3720) )
      {
        *(_DWORD *)(v2 + 3724) = v75 + 1;
        *v74 = v129;
        ++*(_QWORD *)(v2 + 3696);
      }
      else
      {
LABEL_35:
        sub_16CCBA0(v2 + 3696, v129);
        if ( !v44 )
          goto LABEL_36;
      }
LABEL_111:
      v77 = *(unsigned int *)(v2 + 3872);
      if ( (unsigned int)v77 >= *(_DWORD *)(v2 + 3876) )
      {
        sub_16CD150(v2 + 3864, (const void *)(v2 + 3880), 0, 8, v10, v11);
        v77 = *(unsigned int *)(v2 + 3872);
      }
      *(_QWORD *)(*(_QWORD *)(v2 + 3864) + 8 * v77) = v129;
      ++*(_DWORD *)(v2 + 3872);
      goto LABEL_36;
    }
    v132 = *(__int64 **)(v2 + 240);
    v134 = v9;
    v133 = v2;
    while ( 1 )
    {
      v131 = *v132;
      v15 = *(_QWORD *)(*(_QWORD *)(*v132 + 8) + 8 * (7LL - *(unsigned int *)(*(_QWORD *)(*v132 + 8) + 8LL)));
      if ( v15 )
      {
        v16 = 8LL * *(unsigned int *)(v15 + 8);
        v17 = (_BYTE **)(v15 - v16);
        if ( v15 - v16 != v15 )
          break;
      }
LABEL_32:
      sub_399A150(v133, v134, v131);
      if ( v130 == ++v132 )
      {
        v9 = v134;
        v2 = v133;
        goto LABEL_34;
      }
    }
    while ( 1 )
    {
      v18 = *v17;
      if ( **v17 == 25 )
        break;
LABEL_31:
      if ( (_BYTE **)v15 == ++v17 )
        goto LABEL_32;
    }
    v19 = (unsigned int)v138;
    v20 = (unsigned __int64)v136;
    if ( (_DWORD)v138 )
    {
      v21 = (unsigned int)(v138 - 1);
      v22 = 1;
      v23 = ((((unsigned __int64)(((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)) << 32) - 1) >> 22)
          ^ (((unsigned __int64)(((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)) << 32) - 1);
      v24 = ((v23 - 1 - (v23 << 13)) >> 8) ^ (v23 - 1 - (v23 << 13));
      v25 = 0;
      v26 = (((((9 * v24) >> 15) ^ (9 * v24)) - 1 - ((((9 * v24) >> 15) ^ (9 * v24)) << 27)) >> 31)
          ^ ((((9 * v24) >> 15) ^ (9 * v24)) - 1 - ((((9 * v24) >> 15) ^ (9 * v24)) << 27));
      for ( j = v26 & (v138 - 1); ; j = v21 & v58 )
      {
        v28 = &v136[2 * j];
        v29 = (_BYTE *)*v28;
        if ( v18 == (_BYTE *)*v28 )
          break;
        if ( v29 == (_BYTE *)-8LL )
          goto LABEL_70;
LABEL_15:
        if ( v29 == (_BYTE *)-16LL && v28[1] == -16 && !v25 )
          v25 = &v136[2 * j];
LABEL_71:
        v58 = v22 + j;
        ++v22;
      }
      if ( !v28[1] )
        goto LABEL_31;
      if ( v29 != (_BYTE *)-8LL )
        goto LABEL_15;
LABEL_70:
      if ( v28[1] != -8 )
        goto LABEL_71;
      if ( !v25 )
        v25 = &v136[2 * j];
      ++v135;
      v34 = v137 + 1;
      if ( 4 * ((int)v137 + 1) < (unsigned int)(3 * v138) )
      {
        if ( (int)v138 - HIDWORD(v137) - v34 <= (unsigned int)v138 >> 3 )
        {
          v126 = v18;
          v92 = (((v21 >> 1) | v21 | (((v21 >> 1) | v21) >> 2)) >> 4) | (v21 >> 1) | v21 | (((v21 >> 1) | v21) >> 2);
          v93 = ((((v92 >> 8) | v92) >> 16) | (v92 >> 8) | v92) + 1;
          if ( (unsigned int)v93 < 0x40 )
            LODWORD(v93) = 64;
          LODWORD(v138) = v93;
          v94 = (_QWORD *)sub_22077B0(16LL * (unsigned int)v93);
          v18 = v126;
          v136 = v94;
          if ( v20 )
          {
            v137 = 0;
            v95 = (__int64 *)(v20 + 16 * v19);
            for ( k = &v94[2 * (unsigned int)v138]; k != v94; v94 += 2 )
            {
              if ( v94 )
              {
                *v94 = -8;
                v94[1] = -8;
              }
            }
            v127 = v15;
            v97 = (__int64 *)v20;
            v122 = v18;
            while ( 1 )
            {
              v98 = *v97;
              if ( *v97 == -8 )
              {
                if ( v97[1] != -8 )
                  goto LABEL_157;
              }
              else if ( v98 != -16 || v97[1] != -16 )
              {
LABEL_157:
                if ( !(_DWORD)v138 )
                {
                  MEMORY[0] = *v97;
                  BUG();
                }
                v99 = v97[1];
                v100 = 1;
                v101 = 0;
                v102 = ((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4);
                v103 = (((v102 | ((unsigned __int64)(((unsigned int)v98 >> 9) ^ ((unsigned int)v98 >> 4)) << 32))
                       - 1
                       - (v102 << 32)) >> 22)
                     ^ ((v102 | ((unsigned __int64)(((unsigned int)v98 >> 9) ^ ((unsigned int)v98 >> 4)) << 32))
                      - 1
                      - (v102 << 32));
                v104 = ((9 * (((v103 - 1 - (v103 << 13)) >> 8) ^ (v103 - 1 - (v103 << 13)))) >> 15)
                     ^ (9 * (((v103 - 1 - (v103 << 13)) >> 8) ^ (v103 - 1 - (v103 << 13))));
                for ( m = (v138 - 1) & (((v104 - 1 - (v104 << 27)) >> 31) ^ (v104 - 1 - ((_DWORD)v104 << 27)));
                      ;
                      m = (v138 - 1) & v118 )
                {
                  v106 = &v136[2 * m];
                  v107 = *v106;
                  if ( v98 == *v106 && v106[1] == v99 )
                    break;
                  if ( v107 == -8 )
                  {
                    if ( v106[1] == -8 )
                    {
                      if ( v101 )
                        v106 = v101;
                      break;
                    }
                  }
                  else if ( v107 == -16 && v106[1] == -16 && !v101 )
                  {
                    v101 = &v136[2 * m];
                  }
                  v118 = v100 + m;
                  ++v100;
                }
                *v106 = v98;
                v106[1] = v97[1];
                LODWORD(v137) = v137 + 1;
              }
              v97 += 2;
              if ( v95 == v97 )
              {
                v15 = v127;
                j___libc_free_0(v20);
                v108 = v136;
                v109 = v138;
                v18 = v122;
                v34 = v137 + 1;
                goto LABEL_169;
              }
            }
          }
          v137 = 0;
          v109 = v138;
          v117 = v138;
          v108 = &v94[2 * (unsigned int)v138];
          if ( v94 == v108 )
          {
            v34 = 1;
          }
          else
          {
            do
            {
              if ( v94 )
              {
                *v94 = -8;
                v94[1] = -8;
              }
              v94 += 2;
            }
            while ( v108 != v94 );
            v108 = v136;
            v109 = v117;
            v34 = v137 + 1;
          }
LABEL_169:
          if ( !v109 )
          {
LABEL_229:
            LODWORD(v137) = v137 + 1;
            BUG();
          }
          v110 = v109 - 1;
          v111 = 1;
          v112 = 0;
          for ( n = (v109 - 1) & v26; ; n = v110 & v115 )
          {
            v25 = &v108[2 * n];
            v114 = (_BYTE *)*v25;
            if ( v18 == (_BYTE *)*v25 && !v25[1] )
              break;
            if ( v114 == (_BYTE *)-8LL )
            {
              if ( v25[1] == -8 )
              {
                if ( v112 )
                  v25 = v112;
                goto LABEL_28;
              }
            }
            else if ( v114 == (_BYTE *)-16LL && v25[1] == -16 && !v112 )
            {
              v112 = &v108[2 * n];
            }
            v115 = v111 + n;
            ++v111;
          }
        }
        goto LABEL_28;
      }
    }
    else
    {
      ++v135;
    }
    v124 = v18;
    v59 = ((((((((unsigned int)(2 * v138 - 1) | ((unsigned __int64)(unsigned int)(2 * v138 - 1) >> 1)) >> 2)
             | (unsigned int)(2 * v138 - 1)
             | ((unsigned __int64)(unsigned int)(2 * v138 - 1) >> 1)) >> 4)
           | (((unsigned int)(2 * v138 - 1) | ((unsigned __int64)(unsigned int)(2 * v138 - 1) >> 1)) >> 2)
           | (unsigned int)(2 * v138 - 1)
           | ((unsigned __int64)(unsigned int)(2 * v138 - 1) >> 1)) >> 8)
         | (((((unsigned int)(2 * v138 - 1) | ((unsigned __int64)(unsigned int)(2 * v138 - 1) >> 1)) >> 2)
           | (unsigned int)(2 * v138 - 1)
           | ((unsigned __int64)(unsigned int)(2 * v138 - 1) >> 1)) >> 4)
         | (((unsigned int)(2 * v138 - 1) | ((unsigned __int64)(unsigned int)(2 * v138 - 1) >> 1)) >> 2)
         | (unsigned int)(2 * v138 - 1)
         | ((unsigned __int64)(unsigned int)(2 * v138 - 1) >> 1)) >> 16;
    v60 = (v59
         | (((((((unsigned int)(2 * v138 - 1) | ((unsigned __int64)(unsigned int)(2 * v138 - 1) >> 1)) >> 2)
             | (unsigned int)(2 * v138 - 1)
             | ((unsigned __int64)(unsigned int)(2 * v138 - 1) >> 1)) >> 4)
           | (((unsigned int)(2 * v138 - 1) | ((unsigned __int64)(unsigned int)(2 * v138 - 1) >> 1)) >> 2)
           | (unsigned int)(2 * v138 - 1)
           | ((unsigned __int64)(unsigned int)(2 * v138 - 1) >> 1)) >> 8)
         | (((((unsigned int)(2 * v138 - 1) | ((unsigned __int64)(unsigned int)(2 * v138 - 1) >> 1)) >> 2)
           | (unsigned int)(2 * v138 - 1)
           | ((unsigned __int64)(unsigned int)(2 * v138 - 1) >> 1)) >> 4)
         | (((unsigned int)(2 * v138 - 1) | ((unsigned __int64)(unsigned int)(2 * v138 - 1) >> 1)) >> 2)
         | (unsigned int)(2 * v138 - 1)
         | ((unsigned __int64)(unsigned int)(2 * v138 - 1) >> 1))
        + 1;
    if ( (unsigned int)v60 < 0x40 )
      LODWORD(v60) = 64;
    LODWORD(v138) = v60;
    v61 = (_QWORD *)sub_22077B0(16LL * (unsigned int)v60);
    v18 = v124;
    v136 = v61;
    if ( v20 )
    {
      v137 = 0;
      v62 = (__int64 *)(v20 + 16 * v19);
      for ( ii = &v61[2 * (unsigned int)v138]; ii != v61; v61 += 2 )
      {
        if ( v61 )
        {
          *v61 = -8;
          v61[1] = -8;
        }
      }
      v31 = (__int64 *)v20;
      if ( (__int64 *)v20 != v62 )
      {
        v125 = v15;
        while ( 1 )
        {
          v64 = *v31;
          if ( *v31 == -8 )
          {
            if ( v31[1] != -8 )
              goto LABEL_84;
            v31 += 2;
            if ( v62 == v31 )
              goto LABEL_22;
          }
          else if ( v64 == -16 && v31[1] == -16 )
          {
            v31 += 2;
            if ( v62 == v31 )
              goto LABEL_22;
          }
          else
          {
LABEL_84:
            if ( !(_DWORD)v138 )
            {
              MEMORY[0] = *v31;
              BUG();
            }
            v65 = v31[1];
            v66 = 1;
            v67 = 0;
            v68 = ((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4);
            v69 = (((v68 | ((unsigned __int64)(((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4)) << 32))
                  - 1
                  - (v68 << 32)) >> 22)
                ^ ((v68 | ((unsigned __int64)(((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4)) << 32))
                 - 1
                 - (v68 << 32));
            v70 = ((9 * (((v69 - 1 - (v69 << 13)) >> 8) ^ (v69 - 1 - (v69 << 13)))) >> 15)
                ^ (9 * (((v69 - 1 - (v69 << 13)) >> 8) ^ (v69 - 1 - (v69 << 13))));
            for ( jj = (v138 - 1) & (((v70 - 1 - (v70 << 27)) >> 31) ^ (v70 - 1 - ((_DWORD)v70 << 27)));
                  ;
                  jj = (v138 - 1) & v82 )
            {
              v72 = &v136[2 * jj];
              v73 = *v72;
              if ( v64 == *v72 && v72[1] == v65 )
                break;
              if ( v73 == -8 )
              {
                if ( v72[1] == -8 )
                {
                  if ( v67 )
                    v72 = v67;
                  break;
                }
              }
              else if ( v73 == -16 && v72[1] == -16 && !v67 )
              {
                v67 = &v136[2 * jj];
              }
              v82 = v66 + jj;
              ++v66;
            }
            *v72 = v64;
            v30 = v31[1];
            v31 += 2;
            v72[1] = v30;
            LODWORD(v137) = v137 + 1;
            if ( v62 == v31 )
            {
LABEL_22:
              v15 = v125;
              break;
            }
          }
        }
      }
      v123 = v18;
      j___libc_free_0(v20);
      v32 = v136;
      v33 = v138;
      v18 = v123;
      v34 = v137 + 1;
    }
    else
    {
      v137 = 0;
      v33 = v138;
      v116 = v138;
      v32 = &v61[2 * (unsigned int)v138];
      if ( v61 == v32 )
      {
        v34 = 1;
      }
      else
      {
        do
        {
          if ( v61 )
          {
            *v61 = -8;
            v61[1] = -8;
          }
          v61 += 2;
        }
        while ( v32 != v61 );
        v32 = v136;
        v33 = v116;
        v34 = v137 + 1;
      }
    }
    if ( !v33 )
      goto LABEL_229;
    v35 = 1;
    v36 = ((((unsigned __int64)(((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)) << 32) - 1) >> 22)
        ^ (((unsigned __int64)(((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)) << 32) - 1);
    v37 = ((9 * (((v36 - 1 - (v36 << 13)) >> 8) ^ (v36 - 1 - (v36 << 13)))) >> 15)
        ^ (9 * (((v36 - 1 - (v36 << 13)) >> 8) ^ (v36 - 1 - (v36 << 13))));
    v38 = v37 - 1 - (v37 << 27);
    v39 = v33 - 1;
    v40 = 0;
    for ( kk = (v33 - 1) & ((v38 >> 31) ^ v38); ; kk = v39 & v81 )
    {
      v25 = &v32[2 * kk];
      v42 = (_BYTE *)*v25;
      if ( v18 == (_BYTE *)*v25 && !v25[1] )
        break;
      if ( v42 == (_BYTE *)-8LL )
      {
        if ( v25[1] == -8 )
        {
          if ( v40 )
            v25 = v40;
          break;
        }
      }
      else if ( v42 == (_BYTE *)-16LL && v25[1] == -16 && !v40 )
      {
        v40 = &v32[2 * kk];
      }
      v81 = v35 + kk;
      ++v35;
    }
LABEL_28:
    LODWORD(v137) = v34;
    if ( *v25 != -8 || v25[1] != -8 )
      --HIDWORD(v137);
    *v25 = v18;
    v25[1] = 0;
    sub_3989E20(v133, v134, (__int64)v18, 0, *(_QWORD *)&v18[-8 * *((unsigned int *)v18 + 2)]);
    goto LABEL_31;
  }
  *(_QWORD *)(v2 + 32) = 0;
  *(_QWORD *)(v2 + 4008) = 0;
}
