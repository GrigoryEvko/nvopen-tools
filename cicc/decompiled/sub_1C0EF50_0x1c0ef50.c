// Function: sub_1C0EF50
// Address: 0x1c0ef50
//
__int64 __fastcall sub_1C0EF50(size_t a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v10; // rsi
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 v13; // r9
  __int64 v14; // rax
  _DWORD *v15; // rax
  unsigned int v16; // ebx
  __int64 v17; // rax
  size_t v18; // rdx
  void *v19; // r8
  __int64 **v20; // r14
  __int64 **v21; // rbx
  __int64 v22; // rdi
  __int64 v23; // rax
  int v25; // esi
  int v27; // ebx
  __int64 v28; // r12
  __int64 *v29; // rax
  unsigned int v30; // esi
  __int64 v31; // r10
  unsigned int v32; // edx
  int *v33; // rax
  int v34; // edi
  int v35; // edx
  int v36; // ecx
  __int64 v37; // rdi
  unsigned int v38; // edx
  __int64 *v39; // rsi
  unsigned int v40; // ebx
  unsigned int v41; // r10d
  unsigned int v42; // esi
  int v43; // ebx
  unsigned __int64 v44; // rdi
  __int64 v45; // rdx
  unsigned __int64 v46; // rdi
  int v50; // edx
  __int64 **v51; // rbx
  __int64 **v52; // r11
  int v53; // r8d
  int v54; // eax
  __int64 **v55; // r12
  size_t v56; // r11
  __int64 v57; // r14
  __int64 v58; // r13
  __int64 v59; // rdi
  __int64 v60; // rsi
  __int64 *v61; // rax
  int v62; // r8d
  unsigned int v63; // ecx
  __int64 *v64; // rdx
  __int64 v65; // r9
  __int64 v66; // rdx
  int v67; // r15d
  unsigned int v68; // ecx
  __int64 *v69; // rdx
  __int64 v70; // r9
  __int64 v71; // rax
  int v72; // eax
  __int64 v73; // r10
  __int64 v74; // r10
  int j; // r15d
  int v76; // r9d
  int *v77; // rcx
  int v78; // eax
  int v79; // edx
  int v80; // edi
  __int64 **v81; // r12
  __int64 v82; // rsi
  __int64 v83; // rdi
  int v84; // eax
  __int64 v85; // rax
  int v86; // esi
  int v87; // r8d
  __int64 *v88; // rcx
  unsigned int v89; // edx
  __int64 *v90; // rdi
  __int64 v91; // r9
  int v92; // edx
  int *v93; // rdi
  __int64 v94; // rax
  unsigned int v95; // ecx
  __int64 v96; // rax
  int v97; // ecx
  int v98; // r8d
  unsigned int v99; // r10d
  size_t v100; // r9
  unsigned int v101; // r10d
  __int64 i; // rax
  __int64 v103; // rax
  size_t v104; // r9
  const void **v105; // rdx
  void *v106; // rcx
  unsigned int v107; // eax
  int v108; // r10d
  __int64 v109; // r15
  __int64 *v110; // r15
  __int64 *v111; // r10
  int v112; // eax
  int v113; // edx
  __int64 v114; // rcx
  __int64 *v115; // r10
  int v116; // eax
  int v117; // edx
  int v118; // r10d
  __int64 v119; // rax
  void *v120; // rax
  unsigned int v121; // r10d
  __int64 v122; // r9
  __int64 *v123; // rax
  __int64 *v124; // r15
  int v125; // ecx
  int *v126; // r9
  __int64 v127; // rax
  __int64 v128; // rax
  __int64 v129; // [rsp+0h] [rbp-F0h]
  unsigned int v130; // [rsp+8h] [rbp-E8h]
  unsigned int v131; // [rsp+8h] [rbp-E8h]
  void *v132; // [rsp+8h] [rbp-E8h]
  const void **v133; // [rsp+10h] [rbp-E0h]
  unsigned int v134; // [rsp+10h] [rbp-E0h]
  __int64 v135; // [rsp+10h] [rbp-E0h]
  size_t v136; // [rsp+10h] [rbp-E0h]
  size_t v137; // [rsp+10h] [rbp-E0h]
  size_t v138; // [rsp+18h] [rbp-D8h]
  void *v139; // [rsp+18h] [rbp-D8h]
  size_t v140; // [rsp+18h] [rbp-D8h]
  int v141; // [rsp+18h] [rbp-D8h]
  size_t v142; // [rsp+18h] [rbp-D8h]
  size_t v143; // [rsp+18h] [rbp-D8h]
  size_t v144; // [rsp+18h] [rbp-D8h]
  __int64 v145; // [rsp+20h] [rbp-D0h]
  int n; // [rsp+28h] [rbp-C8h]
  char na; // [rsp+28h] [rbp-C8h]
  size_t nb; // [rsp+28h] [rbp-C8h]
  unsigned int nc; // [rsp+28h] [rbp-C8h]
  int nd; // [rsp+28h] [rbp-C8h]
  size_t ne; // [rsp+28h] [rbp-C8h]
  size_t nf; // [rsp+28h] [rbp-C8h]
  size_t ng; // [rsp+28h] [rbp-C8h]
  size_t nh; // [rsp+28h] [rbp-C8h]
  unsigned int v156; // [rsp+30h] [rbp-C0h]
  int v157; // [rsp+30h] [rbp-C0h]
  int v158; // [rsp+30h] [rbp-C0h]
  int v159; // [rsp+30h] [rbp-C0h]
  size_t v160; // [rsp+30h] [rbp-C0h]
  __int64 v162; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v163; // [rsp+48h] [rbp-A8h] BYREF
  void *dest; // [rsp+50h] [rbp-A0h] BYREF
  size_t v165; // [rsp+58h] [rbp-98h]
  unsigned int v166; // [rsp+60h] [rbp-90h]
  __int64 v167; // [rsp+70h] [rbp-80h] BYREF
  __int64 v168; // [rsp+78h] [rbp-78h]
  __int64 v169; // [rsp+80h] [rbp-70h]
  __int64 v170; // [rsp+88h] [rbp-68h]
  _QWORD v171[4]; // [rsp+90h] [rbp-60h] BYREF
  char v172; // [rsp+B0h] [rbp-40h]

  if ( a4 )
  {
    v7 = *(_QWORD *)(a1 + 104);
    v8 = *(unsigned int *)(v7 + 64);
    if ( (_DWORD)v8 )
    {
      v10 = *(_QWORD *)(v7 + 48);
      v11 = (v8 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
      v12 = (__int64 *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( a4 == *v12 )
      {
LABEL_4:
        if ( v12 != (__int64 *)(v10 + 16 * v8) )
        {
          v14 = *(_QWORD *)(sub_1C0A150(v7, a4) + 72);
          v167 = 0;
          v168 = 0;
          v145 = v14;
          v15 = *(_DWORD **)(a1 + 104);
          v169 = 0;
          v170 = 0;
          LODWORD(v15) = *v15;
          dest = 0;
          v166 = (unsigned int)v15;
          v165 = 0;
          v16 = (unsigned int)((_DWORD)v15 + 63) >> 6;
          v17 = malloc(8LL * v16);
          v18 = 8LL * v16;
          v19 = (void *)v17;
          if ( !v17 )
          {
            if ( 8LL * v16 || (v128 = malloc(1u), v18 = 0, v19 = 0, !v128) )
            {
              v144 = (size_t)v19;
              nh = v18;
              sub_16BD1C0("Allocation failed", 1u);
              v18 = nh;
              v19 = (void *)v144;
            }
            else
            {
              v19 = (void *)v128;
            }
          }
          dest = v19;
          v165 = v16;
          if ( v16 )
            v19 = memset(v19, 0, v18);
          v20 = *(__int64 ***)(a3 + 8);
          v21 = &v20[*(unsigned int *)(a3 + 24)];
          if ( !*(_DWORD *)(a3 + 16) || v21 == v20 )
            goto LABEL_9;
          while ( *v20 + 1 == 0 || *v20 + 2 == 0 )
          {
            if ( ++v20 == v21 )
              goto LABEL_9;
          }
          if ( v21 == v20 )
            goto LABEL_9;
          na = 1;
          v81 = v20;
          while ( 1 )
          {
            v82 = **v81;
            v162 = v82;
            if ( a4 != v82 )
            {
              v83 = *(_QWORD *)(a1 + 112);
              if ( !v83 || !sub_15CC8F0(v83, v82, a4) )
              {
                sub_1C0B2E0((__int64)v171, a5, &v162);
                if ( v172 )
                {
                  v84 = sub_1C04FF0(*(_QWORD *)(a1 + 104), v162);
                  if ( v84 )
                  {
                    a2 |= v84;
                    sub_1C08B70(a5);
LABEL_106:
                    v19 = dest;
LABEL_9:
                    if ( !v166 )
                      goto LABEL_35;
                    v22 = (v166 - 1) >> 6;
                    v23 = 0;
                    while ( 1 )
                    {
                      _RBX = *((_QWORD *)v19 + v23);
                      v25 = v23;
                      if ( v23 == v22 )
                        break;
                      if ( _RBX )
                        goto LABEL_15;
                      if ( (_DWORD)v22 + 1 == ++v23 )
                        goto LABEL_35;
                    }
                    _RBX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v166;
                    if ( !_RBX )
                      goto LABEL_35;
LABEL_15:
                    __asm { tzcnt   rax, rbx }
                    v27 = (v25 << 6) + _RAX;
                    if ( v27 == -1 )
                      goto LABEL_35;
                    while ( 2 )
                    {
                      v28 = *(_QWORD *)(a1 + 104);
                      LODWORD(v163) = v27;
                      v29 = 0;
                      if ( *(_DWORD *)v28 > v27 )
                      {
                        v30 = *(_DWORD *)(v28 + 96);
                        if ( v30 )
                        {
                          v31 = *(_QWORD *)(v28 + 80);
                          v32 = (v30 - 1) & (37 * v27);
                          v33 = (int *)(v31 + 16LL * v32);
                          v34 = *v33;
                          if ( *v33 == v27 )
                          {
                            v29 = (__int64 *)*((_QWORD *)v33 + 1);
                            goto LABEL_20;
                          }
                          v76 = 1;
                          v77 = 0;
                          while ( v34 != 0x7FFFFFFF )
                          {
                            if ( v34 != 0x80000000 || v77 )
                              v33 = v77;
                            v125 = v76 + 1;
                            v32 = (v30 - 1) & (v76 + v32);
                            v126 = (int *)(v31 + 16LL * v32);
                            v34 = *v126;
                            if ( *v126 == v27 )
                            {
                              v29 = (__int64 *)*((_QWORD *)v126 + 1);
                              goto LABEL_20;
                            }
                            v76 = v125;
                            v77 = v33;
                            v33 = (int *)(v31 + 16LL * v32);
                          }
                          if ( !v77 )
                            v77 = v33;
                          v78 = *(_DWORD *)(v28 + 88);
                          ++*(_QWORD *)(v28 + 72);
                          v79 = v78 + 1;
                          if ( 4 * (v78 + 1) < 3 * v30 )
                          {
                            v80 = v27;
                            if ( v30 - *(_DWORD *)(v28 + 92) - v79 <= v30 >> 3 )
                              goto LABEL_87;
                            goto LABEL_82;
                          }
                        }
                        else
                        {
                          ++*(_QWORD *)(v28 + 72);
                        }
                        v30 *= 2;
LABEL_87:
                        sub_1C0A790(v28 + 72, v30);
                        sub_1C09960(v28 + 72, (int *)&v163, v171);
                        v77 = (int *)v171[0];
                        v80 = v163;
                        v79 = *(_DWORD *)(v28 + 88) + 1;
LABEL_82:
                        *(_DWORD *)(v28 + 88) = v79;
                        if ( *v77 != 0x7FFFFFFF )
                          --*(_DWORD *)(v28 + 92);
                        *v77 = v80;
                        v29 = 0;
                        *((_QWORD *)v77 + 1) = 0;
                      }
LABEL_20:
                      if ( v145 )
                      {
                        v35 = *(_DWORD *)(v145 + 24);
                        if ( v35 )
                        {
                          v36 = v35 - 1;
                          v37 = *(_QWORD *)(v145 + 8);
                          v38 = (v35 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
                          v39 = *(__int64 **)(v37 + 8LL * v38);
                          if ( v39 == v29 )
                          {
LABEL_23:
                            v40 = v27 + 1;
                            v19 = dest;
                            if ( v166 == v40 )
                              goto LABEL_35;
                            v41 = v40 >> 6;
                            v42 = (v166 - 1) >> 6;
                            if ( v40 >> 6 > v42 )
                              goto LABEL_35;
                            v43 = v40 & 0x3F;
                            v44 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v43);
                            if ( v43 == 0 )
                              v44 = 0;
                            v45 = v41;
                            v46 = ~v44;
                            while ( 1 )
                            {
                              _RAX = *((_QWORD *)dest + v45);
                              if ( v41 == (_DWORD)v45 )
                                _RAX = v46 & *((_QWORD *)dest + v45);
                              if ( (_DWORD)v45 == v42 )
                                _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v166;
                              if ( _RAX )
                                break;
                              if ( v42 < (unsigned int)++v45 )
                                goto LABEL_35;
                            }
                            __asm { tzcnt   rax, rax }
                            v27 = _RAX + ((_DWORD)v45 << 6);
                            if ( v27 == -1 )
                              goto LABEL_35;
                            continue;
                          }
                          v53 = 1;
                          while ( v39 != (__int64 *)-8LL )
                          {
                            v38 = v36 & (v53 + v38);
                            v39 = *(__int64 **)(v37 + 8LL * v38);
                            if ( v29 == v39 )
                              goto LABEL_23;
                            ++v53;
                          }
                        }
                      }
                      break;
                    }
                    v163 = *v29;
                    if ( a4 != v163 )
                    {
                      sub_1C0B2E0((__int64)v171, a5, &v163);
                      if ( v172 )
                      {
                        v54 = sub_1C04FF0(*(_QWORD *)(a1 + 104), v163);
                        if ( v54 )
                        {
                          a2 |= v54;
                          sub_1C08B70(a5);
                          v19 = dest;
LABEL_35:
                          _libc_free((unsigned __int64)v19);
                          j___libc_free_0(v168);
                          return a2;
                        }
                      }
                    }
                    goto LABEL_23;
                  }
                  v85 = *(_QWORD *)(sub_1C0A150(*(_QWORD *)(a1 + 104), v162) + 72);
                  v163 = v85;
                  if ( v85 )
                    break;
                }
              }
            }
LABEL_102:
            if ( ++v81 != v21 )
            {
              while ( *v81 == (__int64 *)-16LL || *v81 == (__int64 *)-8LL )
              {
                if ( v21 == ++v81 )
                  goto LABEL_106;
              }
              if ( v21 != v81 )
                continue;
            }
            goto LABEL_106;
          }
          v86 = v170;
          if ( (_DWORD)v170 )
          {
            v87 = 1;
            v88 = 0;
            v89 = (v170 - 1) & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
            v90 = (__int64 *)(v168 + 8LL * v89);
            v91 = *v90;
            if ( v85 == *v90 )
              goto LABEL_102;
            while ( v91 != -8 )
            {
              if ( !v88 && v91 == -16 )
                v88 = v90;
              v89 = (v170 - 1) & (v87 + v89);
              v90 = (__int64 *)(v168 + 8LL * v89);
              v91 = *v90;
              if ( v85 == *v90 )
                goto LABEL_102;
              ++v87;
            }
            if ( v88 )
              v90 = v88;
            ++v167;
            v92 = v169 + 1;
            if ( 4 * ((int)v169 + 1) < (unsigned int)(3 * v170) )
            {
              if ( (int)v170 - HIDWORD(v169) - v92 > (unsigned int)v170 >> 3 )
              {
LABEL_120:
                LODWORD(v169) = v92;
                if ( *v90 != -8 )
                  --HIDWORD(v169);
                *v90 = v85;
                v93 = *(int **)(a1 + 104);
                if ( na )
                {
                  v94 = sub_1C0A4B0(v93, v163);
                  if ( (void **)v94 != &dest )
                  {
                    v166 = *(_DWORD *)(v94 + 16);
                    v95 = (v166 + 63) >> 6;
                    if ( v166 > v165 << 6 )
                    {
                      v133 = (const void **)v94;
                      nb = v95;
                      v138 = 8LL * v95;
                      v103 = malloc(v138);
                      v104 = v138;
                      v105 = v133;
                      v106 = (void *)v103;
                      if ( !v103 )
                      {
                        if ( v138 || (v127 = malloc(1u), v105 = v133, v104 = 0, v106 = 0, !v127) )
                        {
                          v132 = v106;
                          v137 = v104;
                          v143 = (size_t)v105;
                          sub_16BD1C0("Allocation failed", 1u);
                          v105 = (const void **)v143;
                          v104 = v137;
                          v106 = v132;
                        }
                        else
                        {
                          v106 = (void *)v127;
                        }
                      }
                      v139 = v106;
                      memcpy(v106, *v105, v104);
                      _libc_free((unsigned __int64)dest);
                      dest = v139;
                      v165 = nb;
                    }
                    else
                    {
                      if ( v166 )
                        memcpy(dest, *(const void **)v94, 8LL * v95);
                      sub_13A4C60((__int64)&dest, 0);
                    }
                  }
                }
                else
                {
                  v96 = sub_1C0A4B0(v93, v163);
                  v99 = *(_DWORD *)(v96 + 16);
                  v100 = v96;
                  if ( v166 < v99 )
                  {
                    nc = v165;
                    if ( v99 <= v165 << 6 )
                      goto LABEL_137;
                    v129 = v96;
                    v130 = *(_DWORD *)(v96 + 16);
                    v119 = 2 * v165;
                    if ( (v99 + 63) >> 6 >= 2 * v165 )
                      v119 = (v99 + 63) >> 6;
                    v142 = v119;
                    v135 = 8 * v119;
                    v120 = realloc((unsigned __int64)dest, 8 * v119, 8 * (int)v119, v97, v98, v100);
                    v121 = v130;
                    v122 = v129;
                    if ( !v120 )
                    {
                      if ( v135 )
                      {
                        sub_16BD1C0("Allocation failed", 1u);
                        v120 = 0;
                        v121 = v130;
                        v122 = v129;
                      }
                      else
                      {
                        v120 = (void *)malloc(1u);
                        v122 = v129;
                        v121 = v130;
                        if ( !v120 )
                        {
                          sub_16BD1C0("Allocation failed", 1u);
                          v122 = v129;
                          v121 = v130;
                          v120 = 0;
                        }
                      }
                    }
                    dest = v120;
                    v131 = v121;
                    v136 = v122;
                    v165 = v142;
                    sub_13A4C60((__int64)&dest, 0);
                    v100 = v136;
                    v99 = v131;
                    if ( v165 != nc )
                    {
                      memset((char *)dest + 8 * nc, 0, 8 * (v165 - nc));
                      v99 = v131;
                      v100 = v136;
                    }
                    v107 = v166;
                    if ( v99 > v166 )
                    {
LABEL_137:
                      v134 = v99;
                      v140 = v100;
                      sub_13A4C60((__int64)&dest, 0);
                      v107 = v166;
                      v100 = v140;
                      v99 = v134;
                    }
                    v166 = v99;
                    if ( v107 > v99 )
                    {
                      ng = v100;
                      sub_13A4C60((__int64)&dest, 0);
                      v100 = ng;
                    }
                    v99 = *(_DWORD *)(v100 + 16);
                  }
                  v101 = (v99 + 63) >> 6;
                  if ( v101 )
                  {
                    for ( i = 0; i != v101; ++i )
                      *((_QWORD *)dest + i) |= *(_QWORD *)(*(_QWORD *)v100 + 8 * i);
                  }
                }
                na = 0;
                goto LABEL_102;
              }
LABEL_143:
              sub_1C0EDA0((__int64)&v167, v86);
              sub_1C09CD0((__int64)&v167, &v163, v171);
              v90 = (__int64 *)v171[0];
              v85 = v163;
              v92 = v169 + 1;
              goto LABEL_120;
            }
          }
          else
          {
            ++v167;
          }
          v86 = 2 * v170;
          goto LABEL_143;
        }
      }
      else
      {
        v50 = 1;
        while ( v13 != -8 )
        {
          v118 = v50 + 1;
          v11 = (v8 - 1) & (v50 + v11);
          v12 = (__int64 *)(v10 + 16LL * v11);
          v13 = *v12;
          if ( a4 == *v12 )
            goto LABEL_4;
          v50 = v118;
        }
      }
    }
  }
  v51 = *(__int64 ***)(a3 + 8);
  v52 = &v51[*(unsigned int *)(a3 + 24)];
  if ( *(_DWORD *)(a3 + 16) && v51 != v52 )
  {
    while ( *v51 == (__int64 *)-8LL || *v51 == (__int64 *)-16LL )
    {
      if ( ++v51 == v52 )
        return a2;
    }
    if ( v51 != v52 )
    {
      v55 = v52;
      v56 = a1;
      while ( 1 )
      {
        v57 = *(_QWORD *)(v56 + 104);
        v58 = **v51;
        v167 = v58;
        v59 = *(_QWORD *)(v57 + 48);
        v60 = *(unsigned int *)(v57 + 64);
        v61 = (__int64 *)(v59 + 16 * v60);
        if ( (_DWORD)v60 )
          break;
LABEL_75:
        v67 = 7;
LABEL_58:
        v167 = v58;
        if ( !(_DWORD)v60 )
          goto LABEL_72;
        v62 = v60 - 1;
LABEL_60:
        v68 = v62 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
        v69 = (__int64 *)(v59 + 16LL * v68);
        v70 = *v69;
        if ( v58 == *v69 )
        {
          if ( v69 != v61 )
          {
            v71 = v69[1];
            goto LABEL_63;
          }
LABEL_72:
          v72 = 7;
          goto LABEL_64;
        }
        v156 = v62 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
        v73 = *v69;
        for ( n = 1; ; n = v141 )
        {
          if ( v73 == -8 )
            goto LABEL_72;
          v141 = n + 1;
          v156 = v62 & (n + v156);
          v73 = *(_QWORD *)(v59 + 16LL * v156);
          ne = v59 + 16LL * v156;
          if ( v58 == v73 )
            break;
        }
        v159 = 1;
        v115 = 0;
        if ( (__int64 *)ne == v61 )
          goto LABEL_72;
        while ( v70 != -8 )
        {
          if ( v70 != -16 || v115 )
            v69 = v115;
          v68 = v62 & (v159 + v68);
          v123 = (__int64 *)(v59 + 16LL * v68);
          v70 = *v123;
          if ( v58 == *v123 )
          {
            v71 = v123[1];
            goto LABEL_63;
          }
          ++v159;
          v115 = v69;
          v69 = (__int64 *)(v59 + 16LL * v68);
        }
        v116 = *(_DWORD *)(v57 + 56);
        if ( !v115 )
          v115 = v69;
        ++*(_QWORD *)(v57 + 40);
        v117 = v116 + 1;
        if ( 4 * (v116 + 1) >= (unsigned int)(3 * v60) )
        {
          nf = v56;
          LODWORD(v60) = 2 * v60;
        }
        else
        {
          if ( (int)v60 - *(_DWORD *)(v57 + 60) - v117 > (unsigned int)v60 >> 3 )
            goto LABEL_162;
          nf = v56;
        }
        sub_1C04E30(v57 + 40, v60);
        sub_1C09800(v57 + 40, &v167, v171);
        v115 = (__int64 *)v171[0];
        v58 = v167;
        v56 = nf;
        v117 = *(_DWORD *)(v57 + 56) + 1;
LABEL_162:
        *(_DWORD *)(v57 + 56) = v117;
        if ( *v115 != -8 )
          --*(_DWORD *)(v57 + 60);
        *v115 = v58;
        v71 = 0;
        v115[1] = 0;
LABEL_63:
        v72 = *(_DWORD *)(v71 + 16);
LABEL_64:
        a2 |= v72 | v67;
        do
        {
          if ( ++v51 == v55 )
            return a2;
        }
        while ( *v51 == (__int64 *)-16LL || *v51 == (__int64 *)-8LL );
        if ( v51 == v55 )
          return a2;
      }
      v62 = v60 - 1;
      v63 = (v60 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
      v64 = (__int64 *)(v59 + 16LL * v63);
      v65 = *v64;
      if ( v58 == *v64 )
      {
        if ( v61 != v64 )
        {
          v66 = v64[1];
          goto LABEL_57;
        }
        v110 = (__int64 *)(v59 + 16 * v60);
LABEL_204:
        v61 = v110;
        v167 = v58;
        v67 = 7;
        goto LABEL_60;
      }
      v157 = (v60 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
      v74 = *v64;
      for ( j = 1; ; j = nd )
      {
        if ( v74 == -8 )
          goto LABEL_75;
        v108 = j + 1;
        v109 = v62 & (unsigned int)(v157 + j);
        nd = v108;
        v157 = v109;
        v110 = (__int64 *)(v59 + 16 * v109);
        v74 = *v110;
        if ( v58 == *v110 )
          break;
      }
      if ( v61 == v110 )
        goto LABEL_204;
      v158 = 1;
      v111 = 0;
      while ( v65 != -8 )
      {
        if ( v111 || v65 != -16 )
          v64 = v111;
        v63 = v62 & (v158 + v63);
        v124 = (__int64 *)(v59 + 16LL * v63);
        v65 = *v124;
        if ( v58 == *v124 )
        {
          v66 = v124[1];
          goto LABEL_57;
        }
        ++v158;
        v111 = v64;
        v64 = (__int64 *)(v59 + 16LL * v63);
      }
      v112 = *(_DWORD *)(v57 + 56);
      if ( !v111 )
        v111 = v64;
      ++*(_QWORD *)(v57 + 40);
      v113 = v112 + 1;
      if ( 4 * (v112 + 1) >= (unsigned int)(3 * v60) )
      {
        v160 = v56;
        LODWORD(v60) = 2 * v60;
      }
      else
      {
        v114 = v58;
        if ( (int)v60 - *(_DWORD *)(v57 + 60) - v113 > (unsigned int)v60 >> 3 )
          goto LABEL_152;
        v160 = v56;
      }
      sub_1C04E30(v57 + 40, v60);
      sub_1C09800(v57 + 40, &v167, v171);
      v111 = (__int64 *)v171[0];
      v114 = v167;
      v56 = v160;
      v113 = *(_DWORD *)(v57 + 56) + 1;
LABEL_152:
      *(_DWORD *)(v57 + 56) = v113;
      if ( *v111 != -8 )
        --*(_DWORD *)(v57 + 60);
      *v111 = v114;
      v66 = 0;
      v111[1] = 0;
      v57 = *(_QWORD *)(v56 + 104);
      v59 = *(_QWORD *)(v57 + 48);
      v60 = *(unsigned int *)(v57 + 64);
      v61 = (__int64 *)(v59 + 16 * v60);
LABEL_57:
      v67 = *(_DWORD *)(v66 + 12);
      goto LABEL_58;
    }
  }
  return a2;
}
