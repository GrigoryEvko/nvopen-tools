// Function: sub_1C29FB0
// Address: 0x1c29fb0
//
void __fastcall sub_1C29FB0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  __int64 v6; // r8
  unsigned int v7; // ebx
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // r10
  void *v11; // r13
  int v12; // eax
  unsigned int v13; // ebx
  __int64 v14; // rax
  unsigned __int64 v15; // r12
  __int64 v17; // rbx
  __int64 v18; // r8
  __int64 v19; // r13
  char v20; // al
  __int64 v21; // r10
  __int64 v22; // rax
  __int64 *v23; // rbx
  __int64 *v24; // r11
  __int64 *v25; // r15
  __int64 v26; // r11
  unsigned int v27; // edi
  _QWORD *v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // r12
  char v33; // al
  unsigned int v34; // ecx
  int v35; // eax
  _QWORD *v36; // rdx
  __int64 v37; // rdi
  char v38; // al
  __int64 v39; // rax
  __int64 v40; // rax
  int v41; // esi
  int v42; // edi
  int v43; // eax
  int v44; // edx
  int v45; // eax
  __int64 v46; // rdi
  int v47; // eax
  int v48; // edx
  __int64 v49; // rsi
  int v50; // edi
  unsigned int v51; // eax
  __int64 v52; // rcx
  int v53; // edx
  int v54; // edx
  __int64 v55; // rsi
  int v56; // edi
  unsigned int v57; // eax
  __int64 v58; // rcx
  int v59; // r10d
  _QWORD *v60; // r8
  unsigned int v61; // r13d
  int v62; // r10d
  __int64 v63; // rsi
  __int64 v64; // rdx
  int v65; // eax
  int v67; // r15d
  int v68; // esi
  _QWORD *v69; // r10
  unsigned int v70; // eax
  _QWORD *v71; // r8
  __int64 v72; // rdi
  unsigned int v73; // r15d
  unsigned int v74; // edi
  int v75; // r15d
  __int64 v76; // rdx
  unsigned __int64 v77; // rsi
  unsigned __int64 v78; // rsi
  __int64 v81; // rcx
  unsigned int v82; // edx
  _QWORD *v83; // r11
  __int64 v84; // rsi
  int v85; // eax
  _QWORD *v86; // rdi
  __int64 v87; // rbx
  size_t v88; // r12
  __int64 v89; // rcx
  __int64 v90; // rsi
  __int64 i; // rax
  __int64 v92; // rcx
  unsigned int v93; // edx
  __int64 v94; // rsi
  int v95; // r12d
  __int64 *v96; // rdx
  int v97; // eax
  int v98; // ecx
  int v99; // r13d
  _QWORD *v100; // r9
  int v101; // r9d
  int v102; // r9d
  __int64 v103; // r10
  unsigned int v104; // eax
  __int64 v105; // r8
  int v106; // edi
  __int64 *v107; // rsi
  int v108; // r8d
  int v109; // r8d
  __int64 v110; // r9
  int v111; // esi
  unsigned int v112; // ebx
  __int64 *v113; // rax
  __int64 v114; // rdi
  __int64 v115; // rax
  void *v116; // [rsp+8h] [rbp-C8h]
  int *v117; // [rsp+18h] [rbp-B8h]
  __int64 v118; // [rsp+20h] [rbp-B0h]
  __int64 v119; // [rsp+28h] [rbp-A8h]
  __int64 v120; // [rsp+28h] [rbp-A8h]
  __int64 v121; // [rsp+28h] [rbp-A8h]
  __int64 v122; // [rsp+28h] [rbp-A8h]
  __int64 v123; // [rsp+30h] [rbp-A0h]
  int v124; // [rsp+38h] [rbp-98h]
  _QWORD *v125; // [rsp+38h] [rbp-98h]
  int v126; // [rsp+38h] [rbp-98h]
  int v127; // [rsp+38h] [rbp-98h]
  _QWORD *v128; // [rsp+38h] [rbp-98h]
  int v129; // [rsp+38h] [rbp-98h]
  int v131; // [rsp+40h] [rbp-90h]
  int v132; // [rsp+4Ch] [rbp-84h]
  int v133; // [rsp+50h] [rbp-80h]
  int v134; // [rsp+58h] [rbp-78h]
  int v135; // [rsp+5Ch] [rbp-74h]
  int v136; // [rsp+5Ch] [rbp-74h]
  __int64 v137; // [rsp+60h] [rbp-70h] BYREF
  __int64 v138; // [rsp+68h] [rbp-68h]
  __int64 v139; // [rsp+70h] [rbp-60h]
  unsigned int v140; // [rsp+78h] [rbp-58h]
  __int64 v141; // [rsp+80h] [rbp-50h] BYREF
  __int64 v142; // [rsp+88h] [rbp-48h]
  __int64 v143; // [rsp+90h] [rbp-40h]
  __int64 v144; // [rsp+98h] [rbp-38h]

  v4 = a1 + 112;
  v5 = *(_DWORD *)(a1 + 136);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 112);
    goto LABEL_158;
  }
  v6 = *(_QWORD *)(a1 + 120);
  v7 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v8 = (v5 - 1) & v7;
  v9 = (__int64 *)(v6 + 16LL * v8);
  v10 = *v9;
  if ( *v9 == a2 )
  {
LABEL_3:
    v117 = (int *)v9[1];
    goto LABEL_4;
  }
  v95 = 1;
  v96 = 0;
  while ( v10 != -8 )
  {
    if ( !v96 && v10 == -16 )
      v96 = v9;
    v8 = (v5 - 1) & (v95 + v8);
    v9 = (__int64 *)(v6 + 16LL * v8);
    v10 = *v9;
    if ( *v9 == a2 )
      goto LABEL_3;
    ++v95;
  }
  if ( !v96 )
    v96 = v9;
  v97 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  v98 = v97 + 1;
  if ( 4 * (v97 + 1) >= 3 * v5 )
  {
LABEL_158:
    sub_1C29D90(v4, 2 * v5);
    v101 = *(_DWORD *)(a1 + 136);
    if ( v101 )
    {
      v102 = v101 - 1;
      v103 = *(_QWORD *)(a1 + 120);
      v98 = *(_DWORD *)(a1 + 128) + 1;
      v104 = v102 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v96 = (__int64 *)(v103 + 16LL * v104);
      v105 = *v96;
      if ( *v96 != a2 )
      {
        v106 = 1;
        v107 = 0;
        while ( v105 != -8 )
        {
          if ( !v107 && v105 == -16 )
            v107 = v96;
          v104 = v102 & (v106 + v104);
          v96 = (__int64 *)(v103 + 16LL * v104);
          v105 = *v96;
          if ( *v96 == a2 )
            goto LABEL_148;
          ++v106;
        }
        if ( v107 )
          v96 = v107;
      }
      goto LABEL_148;
    }
    goto LABEL_205;
  }
  if ( v5 - *(_DWORD *)(a1 + 132) - v98 <= v5 >> 3 )
  {
    sub_1C29D90(v4, v5);
    v108 = *(_DWORD *)(a1 + 136);
    if ( v108 )
    {
      v109 = v108 - 1;
      v110 = *(_QWORD *)(a1 + 120);
      v111 = 1;
      v112 = v109 & v7;
      v98 = *(_DWORD *)(a1 + 128) + 1;
      v113 = 0;
      v96 = (__int64 *)(v110 + 16LL * v112);
      v114 = *v96;
      if ( *v96 != a2 )
      {
        while ( v114 != -8 )
        {
          if ( v114 == -16 && !v113 )
            v113 = v96;
          v112 = v109 & (v111 + v112);
          v96 = (__int64 *)(v110 + 16LL * v112);
          v114 = *v96;
          if ( *v96 == a2 )
            goto LABEL_148;
          ++v111;
        }
        if ( v113 )
          v96 = v113;
      }
      goto LABEL_148;
    }
LABEL_205:
    ++*(_DWORD *)(a1 + 128);
    BUG();
  }
LABEL_148:
  *(_DWORD *)(a1 + 128) = v98;
  if ( *v96 != -8 )
    --*(_DWORD *)(a1 + 132);
  v96[1] = 0;
  v117 = 0;
  *v96 = a2;
LABEL_4:
  v11 = 0;
  v134 = v117[2];
  v12 = v117[10];
  v132 = v117[3];
  v135 = v12;
  if ( v12 )
  {
    v87 = (unsigned int)(v12 + 63) >> 6;
    v88 = 8 * v87;
    v11 = (void *)malloc(8 * v87);
    if ( !v11 )
    {
      if ( v88 || (v115 = malloc(1u)) == 0 )
        sub_16BD1C0("Allocation failed", 1u);
      else
        v11 = (void *)v115;
    }
    memcpy(v11, *((const void **)v117 + 3), v88);
    v89 = (unsigned int)(v117[16] + 63) >> 6;
    if ( (unsigned int)v89 > (unsigned int)v87 )
      v89 = v87;
    if ( (_DWORD)v89 )
    {
      v90 = *((_QWORD *)v117 + 6);
      for ( i = 0; i != v89; ++i )
        *((_QWORD *)v11 + i) &= ~*(_QWORD *)(v90 + 8 * i);
    }
  }
  v137 = 0;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v143 = 0;
  v144 = 0;
  if ( v135 )
  {
    v13 = (unsigned int)(v135 - 1) >> 6;
    v14 = 0;
    v15 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v135;
    while ( 1 )
    {
      _RDX = *((_QWORD *)v11 + v14);
      if ( v13 == (_DWORD)v14 )
        _RDX = v15 & *((_QWORD *)v11 + v14);
      if ( _RDX )
        break;
      if ( ++v14 == v13 + 1 )
        goto LABEL_11;
    }
    __asm { tzcnt   rdx, rdx }
    v67 = _RDX + ((_DWORD)v14 << 6);
    if ( v67 != -1 )
    {
      v68 = 0;
      v69 = (_QWORD *)(*(_QWORD *)(a1 + 88) + 8LL * v67);
LABEL_100:
      ++v141;
LABEL_101:
      v125 = v69;
      sub_1353F00((__int64)&v141, 2 * v68);
      if ( !(_DWORD)v144 )
      {
LABEL_208:
        LODWORD(v143) = v143 + 1;
        BUG();
      }
      v69 = v125;
      v81 = *v125;
      v82 = (v144 - 1) & (((unsigned int)*v125 >> 9) ^ ((unsigned int)*v125 >> 4));
      v83 = (_QWORD *)(v142 + 8LL * v82);
      v84 = *v83;
      v85 = v143 + 1;
      if ( *v125 == *v83 )
      {
        while ( 1 )
        {
LABEL_114:
          LODWORD(v143) = v85;
          if ( *v83 != -8 )
            --HIDWORD(v143);
          *v83 = *v69;
          do
          {
LABEL_87:
            v73 = v67 + 1;
            if ( v135 == v73 )
              goto LABEL_11;
            v74 = v73 >> 6;
            if ( v13 < v73 >> 6 )
              goto LABEL_11;
            v75 = v73 & 0x3F;
            v76 = v74;
            v77 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v75);
            if ( v75 == 0 )
              v77 = 0;
            v78 = ~v77;
            while ( 1 )
            {
              _RAX = *((_QWORD *)v11 + v76);
              if ( v74 == (_DWORD)v76 )
                _RAX = v78 & *((_QWORD *)v11 + v76);
              if ( v13 == (_DWORD)v76 )
                _RAX &= v15;
              if ( _RAX )
                break;
              if ( v13 < (unsigned int)++v76 )
                goto LABEL_11;
            }
            __asm { tzcnt   rax, rax }
            v67 = _RAX + ((_DWORD)v76 << 6);
            if ( v67 == -1 )
              goto LABEL_11;
            v68 = v144;
            v69 = (_QWORD *)(*(_QWORD *)(a1 + 88) + 8LL * v67);
            if ( !(_DWORD)v144 )
              goto LABEL_100;
            v70 = (v144 - 1) & (((unsigned int)*v69 >> 9) ^ ((unsigned int)*v69 >> 4));
            v71 = (_QWORD *)(v142 + 8LL * v70);
            v72 = *v71;
          }
          while ( *v69 == *v71 );
          v127 = 1;
          v83 = 0;
          while ( v72 != -8 )
          {
            if ( v72 != -16 || v83 )
              v71 = v83;
            v70 = (v144 - 1) & (v127 + v70);
            v72 = *(_QWORD *)(v142 + 8LL * v70);
            if ( *v69 == v72 )
              goto LABEL_87;
            ++v127;
            v83 = v71;
            v71 = (_QWORD *)(v142 + 8LL * v70);
          }
          if ( !v83 )
            v83 = v71;
          ++v141;
          v85 = v143 + 1;
          if ( 4 * ((int)v143 + 1) >= (unsigned int)(3 * v144) )
            goto LABEL_101;
          if ( (int)v144 - (v85 + HIDWORD(v143)) <= (unsigned int)v144 >> 3 )
          {
            v128 = v69;
            sub_1353F00((__int64)&v141, v144);
            if ( !(_DWORD)v144 )
              goto LABEL_208;
            v69 = v128;
            v92 = *v128;
            v93 = (v144 - 1) & (((unsigned int)*v128 >> 9) ^ ((unsigned int)*v128 >> 4));
            v83 = (_QWORD *)(v142 + 8LL * v93);
            v94 = *v83;
            v85 = v143 + 1;
            if ( *v83 != *v128 )
              break;
          }
        }
        v129 = 1;
        v86 = 0;
        while ( v94 != -8 )
        {
          if ( !v86 && v94 == -16 )
            v86 = v83;
          v93 = (v144 - 1) & (v129 + v93);
          v83 = (_QWORD *)(v142 + 8LL * v93);
          v94 = *v83;
          if ( v92 == *v83 )
            goto LABEL_114;
          ++v129;
        }
      }
      else
      {
        v126 = 1;
        v86 = 0;
        while ( v84 != -8 )
        {
          if ( !v86 && v84 == -16 )
            v86 = v83;
          v82 = (v144 - 1) & (v126 + v82);
          v83 = (_QWORD *)(v142 + 8LL * v82);
          v84 = *v83;
          if ( v81 == *v83 )
            goto LABEL_114;
          ++v126;
        }
      }
      if ( v86 )
        v83 = v86;
      goto LABEL_114;
    }
  }
LABEL_11:
  v17 = a2;
  sub_1C29700(a1, a2, (__int64)&v137, (__int64)&v141);
  v18 = *(_QWORD *)(a2 + 48);
  v131 = 0;
  v124 = 0;
  v123 = v17 + 40;
  if ( v17 + 40 != v18 )
  {
    v116 = v11;
    v19 = v18;
    while ( 1 )
    {
      if ( !v19 )
        BUG();
      v20 = *(_BYTE *)(v19 - 8);
      v21 = v19 - 24;
      if ( v20 == 77 )
        goto LABEL_42;
      if ( v20 == 78 )
      {
        v64 = *(_QWORD *)(v19 - 48);
        if ( !*(_BYTE *)(v64 + 16) && (*(_BYTE *)(v64 + 33) & 0x20) != 0 )
        {
          if ( (unsigned int)(*(_DWORD *)(v64 + 36) - 35) <= 3 )
            goto LABEL_83;
          if ( (*(_BYTE *)(v64 + 33) & 0x20) != 0 )
          {
            v65 = *(_DWORD *)(v64 + 36);
            if ( v65 == 4 )
            {
LABEL_83:
              v136 = 0;
              v133 = 0;
              goto LABEL_34;
            }
            v136 = 0;
            v133 = 0;
            if ( (unsigned int)(v65 - 116) <= 1 )
              goto LABEL_34;
          }
        }
      }
      v22 = 3LL * (*(_DWORD *)(v19 - 4) & 0xFFFFFFF);
      if ( (*(_BYTE *)(v19 - 1) & 0x40) != 0 )
      {
        v23 = *(__int64 **)(v19 - 32);
        v24 = &v23[v22];
      }
      else
      {
        v24 = (__int64 *)(v19 - 24);
        v23 = (__int64 *)(v21 - v22 * 8);
      }
      v136 = 0;
      v133 = 0;
      if ( v23 != v24 )
      {
        v25 = v24;
        v118 = v19;
        v26 = v19 - 24;
        while ( 1 )
        {
          v32 = *v23;
          v33 = *(_BYTE *)(*(_QWORD *)*v23 + 8LL);
          if ( v33 != 13 && v33 != 16 )
            goto LABEL_26;
          v47 = *(_DWORD *)(a1 + 80);
          if ( v47 )
          {
            v48 = v47 - 1;
            v49 = *(_QWORD *)(a1 + 64);
            v50 = 1;
            v51 = (v47 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
            v52 = *(_QWORD *)(v49 + 16LL * v51);
            if ( v32 != v52 )
            {
              while ( v52 != -8 )
              {
                v51 = v48 & (v50 + v51);
                v52 = *(_QWORD *)(v49 + 16LL * v51);
                if ( v32 == v52 )
                  goto LABEL_26;
                ++v50;
              }
              goto LABEL_23;
            }
LABEL_26:
            if ( !v140 )
            {
              ++v137;
              goto LABEL_28;
            }
            v27 = (v140 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
            v28 = (_QWORD *)(v138 + 16LL * v27);
            v29 = *v28;
            if ( *v28 == v32 )
            {
LABEL_21:
              if ( v26 == v28[1] )
              {
                v119 = v26;
                v30 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
                v31 = sub_3952EB0(v32, v30);
                v133 += v31;
                v26 = v119;
                v136 += HIDWORD(v31);
              }
              goto LABEL_23;
            }
            v59 = 1;
            v36 = 0;
            while ( v29 != -8 )
            {
              if ( !v36 && v29 == -16 )
                v36 = v28;
              v27 = (v140 - 1) & (v59 + v27);
              v28 = (_QWORD *)(v138 + 16LL * v27);
              v29 = *v28;
              if ( v32 == *v28 )
                goto LABEL_21;
              ++v59;
            }
            if ( !v36 )
              v36 = v28;
            ++v137;
            v35 = v139 + 1;
            if ( 4 * ((int)v139 + 1) < 3 * v140 )
            {
              if ( v140 - HIDWORD(v139) - v35 <= v140 >> 3 )
              {
                v122 = v26;
                sub_1C29540((__int64)&v137, v140);
                if ( !v140 )
                {
LABEL_207:
                  LODWORD(v139) = v139 + 1;
                  BUG();
                }
                v60 = 0;
                v61 = (v140 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
                v26 = v122;
                v62 = 1;
                v35 = v139 + 1;
                v36 = (_QWORD *)(v138 + 16LL * v61);
                v63 = *v36;
                if ( *v36 != v32 )
                {
                  while ( v63 != -8 )
                  {
                    if ( !v60 && v63 == -16 )
                      v60 = v36;
                    v61 = (v140 - 1) & (v62 + v61);
                    v36 = (_QWORD *)(v138 + 16LL * v61);
                    v63 = *v36;
                    if ( v32 == *v36 )
                      goto LABEL_30;
                    ++v62;
                  }
                  if ( v60 )
                    v36 = v60;
                }
              }
              goto LABEL_30;
            }
LABEL_28:
            v120 = v26;
            sub_1C29540((__int64)&v137, 2 * v140);
            if ( !v140 )
              goto LABEL_207;
            v26 = v120;
            v34 = (v140 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
            v35 = v139 + 1;
            v36 = (_QWORD *)(v138 + 16LL * v34);
            v37 = *v36;
            if ( *v36 != v32 )
            {
              v99 = 1;
              v100 = 0;
              while ( v37 != -8 )
              {
                if ( v37 == -16 && !v100 )
                  v100 = v36;
                v34 = (v140 - 1) & (v99 + v34);
                v36 = (_QWORD *)(v138 + 16LL * v34);
                v37 = *v36;
                if ( v32 == *v36 )
                  goto LABEL_30;
                ++v99;
              }
              if ( v100 )
                v36 = v100;
            }
LABEL_30:
            LODWORD(v139) = v35;
            if ( *v36 != -8 )
              --HIDWORD(v139);
            v23 += 3;
            *v36 = v32;
            v36[1] = 0;
            if ( v25 == v23 )
            {
LABEL_33:
              v19 = v118;
              v21 = v26;
              break;
            }
          }
          else
          {
LABEL_23:
            v23 += 3;
            if ( v25 == v23 )
              goto LABEL_33;
          }
        }
      }
LABEL_34:
      v38 = *(_BYTE *)(*(_QWORD *)(v19 - 24) + 8LL);
      if ( v38 == 13 || v38 == 16 )
      {
        v53 = *(_DWORD *)(a1 + 80);
        if ( !v53 )
          goto LABEL_37;
        v54 = v53 - 1;
        v55 = *(_QWORD *)(a1 + 64);
        v56 = 1;
        v57 = v54 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v58 = *(_QWORD *)(v55 + 16LL * v57);
        if ( v21 != v58 )
        {
          while ( v58 != -8 )
          {
            v57 = v54 & (v56 + v57);
            v58 = *(_QWORD *)(v55 + 16LL * v57);
            if ( v21 == v58 )
              goto LABEL_36;
            ++v56;
          }
          goto LABEL_37;
        }
      }
LABEL_36:
      v121 = v21;
      v39 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
      v40 = sub_3952EB0(v121, v39);
      v134 += v40;
      v132 += HIDWORD(v40);
LABEL_37:
      v41 = v124;
      v42 = v131;
      if ( v124 < v134 )
        v41 = v134;
      if ( v131 < v132 )
        v42 = v132;
      v124 = v41;
      v131 = v42;
      v134 -= v133;
      v132 -= v136;
LABEL_42:
      v19 = *(_QWORD *)(v19 + 8);
      if ( v123 == v19 )
      {
        v11 = v116;
        break;
      }
    }
  }
  v43 = v124;
  if ( *v117 >= v124 )
    v43 = *v117;
  v44 = v43;
  v45 = v131;
  if ( v117[1] >= v131 )
    v45 = v117[1];
  *v117 = v44;
  v117[1] = v45;
  if ( *(_DWORD *)(a1 + 28) >= v45 )
    v45 = *(_DWORD *)(a1 + 28);
  if ( *(_DWORD *)(a1 + 24) >= v44 )
    v44 = *(_DWORD *)(a1 + 24);
  v46 = v142;
  *(_DWORD *)(a1 + 28) = v45;
  *(_DWORD *)(a1 + 24) = v44;
  j___libc_free_0(v46);
  j___libc_free_0(v138);
  _libc_free((unsigned __int64)v11);
}
