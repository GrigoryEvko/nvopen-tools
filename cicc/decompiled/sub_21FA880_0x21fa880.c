// Function: sub_21FA880
// Address: 0x21fa880
//
__int64 __fastcall sub_21FA880(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  int v4; // eax
  _QWORD *v5; // rdi
  unsigned int v6; // esi
  _QWORD *v7; // rax
  __int64 v8; // r8
  unsigned int v9; // ecx
  _QWORD *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rax
  unsigned int v15; // r14d
  __int64 v16; // r11
  __int64 *v17; // r12
  __int64 *v18; // r15
  __int64 *v19; // rcx
  __int64 **v20; // r9
  unsigned int v21; // edi
  __int64 **v22; // rax
  __int64 *v23; // rsi
  unsigned int v24; // edx
  __int64 *v25; // rax
  __int64 v26; // rdi
  unsigned int v27; // edx
  __int64 v28; // r13
  __int64 *v29; // r8
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  _QWORD *v32; // rax
  __int64 v33; // rdx
  __int64 *v34; // rsi
  _QWORD *i; // rdx
  __int64 *j; // rax
  __int64 v37; // rdx
  int v38; // ecx
  int v39; // edi
  __int64 v40; // r10
  int v41; // r14d
  __int64 *v42; // r13
  unsigned int v43; // r9d
  __int64 *v44; // rcx
  __int64 v45; // r11
  int v46; // ecx
  int v47; // edx
  int v48; // esi
  unsigned int v49; // ecx
  __int64 **v50; // r15
  __int64 *v51; // rax
  __int64 v52; // rax
  int v54; // r11d
  __int64 **v55; // r10
  int v56; // eax
  int v57; // eax
  int v58; // edi
  int v59; // r10d
  __int64 v60; // rsi
  unsigned int v61; // eax
  int v62; // r11d
  __int64 *v63; // r10
  int v64; // edx
  __int64 v65; // rdx
  unsigned int v66; // esi
  __int64 v67; // rdi
  unsigned int v68; // r12d
  unsigned int v69; // ecx
  __int64 *v70; // rax
  __int64 v71; // rdx
  int v72; // esi
  unsigned int v73; // ecx
  __int64 v74; // r8
  int v75; // r13d
  __int64 *v76; // r10
  int v77; // r13d
  unsigned int v78; // ecx
  __int64 v79; // r8
  __int64 v80; // rsi
  _QWORD *v81; // rsi
  _QWORD *v82; // rdx
  unsigned int v83; // edx
  unsigned int v84; // eax
  __int64 v85; // rax
  unsigned __int64 v86; // rax
  unsigned __int64 v87; // rax
  int v88; // r13d
  __int64 v89; // r12
  __int64 v90; // rax
  __int64 v91; // rax
  _QWORD *v92; // rax
  _QWORD *v93; // rdx
  __int64 *v94; // r13
  int v95; // r10d
  __int64 *v96; // r9
  int v97; // edi
  int v98; // ecx
  int v99; // r10d
  int v100; // esi
  int v101; // esi
  __int64 v102; // r9
  unsigned int v103; // edx
  __int64 v104; // r12
  int v105; // r8d
  __int64 *v106; // rdi
  int v107; // edx
  int v108; // edx
  __int64 v109; // r9
  int v110; // edi
  unsigned int v111; // r12d
  __int64 *v112; // rsi
  __int64 v113; // r8
  int v114; // r10d
  _QWORD *v115; // r13
  int v116; // edi
  int v117; // edx
  int v118; // r8d
  __int64 v119; // r10
  unsigned int v120; // ecx
  __int64 v121; // r12
  int v122; // edi
  _QWORD *v123; // rsi
  int v124; // edi
  int v125; // edi
  _QWORD *v126; // r11
  unsigned int v127; // r12d
  int v128; // ecx
  __int64 v129; // rsi
  _QWORD *v130; // rax
  __int64 v131; // [rsp+18h] [rbp-B8h]
  __int64 v132; // [rsp+20h] [rbp-B0h]
  __int64 *v133; // [rsp+20h] [rbp-B0h]
  __int64 v134; // [rsp+20h] [rbp-B0h]
  __int64 v135; // [rsp+20h] [rbp-B0h]
  int v136; // [rsp+28h] [rbp-A8h]
  __int64 v137; // [rsp+28h] [rbp-A8h]
  __int64 v138; // [rsp+28h] [rbp-A8h]
  __int64 v139; // [rsp+28h] [rbp-A8h]
  __int64 v140; // [rsp+28h] [rbp-A8h]
  __int64 v141; // [rsp+28h] [rbp-A8h]
  __int64 v142; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v143; // [rsp+38h] [rbp-98h]
  __int64 v144; // [rsp+40h] [rbp-90h]
  unsigned int v145; // [rsp+48h] [rbp-88h]
  _BYTE *v146; // [rsp+50h] [rbp-80h] BYREF
  __int64 v147; // [rsp+58h] [rbp-78h]
  _BYTE v148[112]; // [rsp+60h] [rbp-70h] BYREF

  v2 = a1;
  v131 = a1 + 392;
  v4 = *(_DWORD *)(a1 + 408);
  ++*(_QWORD *)(a1 + 392);
  v5 = *(_QWORD **)(a1 + 400);
  v6 = *(_DWORD *)(v2 + 416);
  if ( !v4 )
  {
    if ( !*(_DWORD *)(v2 + 412) )
      goto LABEL_8;
    if ( v6 > 0x40 )
    {
      v138 = v2;
      j___libc_free_0(v5);
      v2 = v138;
      v6 = 0;
      v5 = 0;
      *(_QWORD *)(v138 + 400) = 0;
      *(_QWORD *)(v138 + 408) = 0;
      *(_DWORD *)(v138 + 416) = 0;
      goto LABEL_8;
    }
    goto LABEL_4;
  }
  v83 = 4 * v4;
  if ( (unsigned int)(4 * v4) < 0x40 )
    v83 = 64;
  if ( v83 >= v6 )
  {
LABEL_4:
    v7 = &v5[2 * v6];
    if ( v7 != v5 )
    {
      do
      {
        *v5 = -8;
        v5 += 2;
      }
      while ( v7 != v5 );
      v5 = *(_QWORD **)(v2 + 400);
      v6 = *(_DWORD *)(v2 + 416);
    }
    *(_QWORD *)(v2 + 408) = 0;
    goto LABEL_8;
  }
  v84 = v4 - 1;
  if ( !v84 )
  {
    v89 = 2048;
    v88 = 128;
LABEL_117:
    v137 = v2;
    j___libc_free_0(v5);
    *(_DWORD *)(v137 + 416) = v88;
    v90 = sub_22077B0(v89);
    v2 = v137;
    v5 = (_QWORD *)v90;
    *(_QWORD *)(v137 + 400) = v90;
    v91 = *(unsigned int *)(v137 + 416);
    *(_QWORD *)(v137 + 408) = 0;
    v6 = v91;
    v92 = &v5[2 * v91];
    if ( v5 != v92 )
    {
      v93 = v5;
      do
      {
        if ( v93 )
          *v93 = -8;
        v93 += 2;
      }
      while ( v92 != v93 );
    }
    goto LABEL_8;
  }
  _BitScanReverse(&v84, v84);
  v85 = (unsigned int)(1 << (33 - (v84 ^ 0x1F)));
  if ( (int)v85 < 64 )
    v85 = 64;
  if ( (_DWORD)v85 != v6 )
  {
    v86 = (4 * (int)v85 / 3u + 1) | ((unsigned __int64)(4 * (int)v85 / 3u + 1) >> 1);
    v87 = ((v86 | (v86 >> 2)) >> 4) | v86 | (v86 >> 2) | ((((v86 | (v86 >> 2)) >> 4) | v86 | (v86 >> 2)) >> 8);
    v88 = (v87 | (v87 >> 16)) + 1;
    v89 = 16 * ((v87 | (v87 >> 16)) + 1);
    goto LABEL_117;
  }
  *(_QWORD *)(v2 + 408) = 0;
  v130 = &v5[2 * v85];
  do
  {
    if ( v5 )
      *v5 = -8;
    v5 += 2;
  }
  while ( v130 != v5 );
  v5 = *(_QWORD **)(v2 + 400);
  v6 = *(_DWORD *)(v2 + 416);
LABEL_8:
  v142 = 0;
  *(_QWORD *)(v2 + 424) = 0x100000001LL;
  v146 = v148;
  v143 = 0;
  v144 = 0;
  v145 = 0;
  v147 = 0x800000000LL;
  if ( !v6 )
  {
    ++*(_QWORD *)(v2 + 392);
    goto LABEL_184;
  }
  LODWORD(v8) = v6 - 1;
  v9 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = &v5[2 * v9];
  v11 = *v10;
  if ( *v10 == a2 )
    goto LABEL_10;
  v114 = 1;
  v115 = 0;
  while ( v11 != -8 )
  {
    if ( !v115 && v11 == -16 )
      v115 = v10;
    v9 = v8 & (v114 + v9);
    v10 = &v5[2 * v9];
    v11 = *v10;
    if ( *v10 == a2 )
      goto LABEL_10;
    ++v114;
  }
  v116 = *(_DWORD *)(v2 + 408);
  if ( v115 )
    v10 = v115;
  ++*(_QWORD *)(v2 + 392);
  v117 = v116 + 1;
  if ( 4 * (v116 + 1) >= 3 * v6 )
  {
LABEL_184:
    v140 = v2;
    sub_21FA480(v131, 2 * v6);
    v2 = v140;
    v118 = *(_DWORD *)(v140 + 416);
    if ( v118 )
    {
      LODWORD(v8) = v118 - 1;
      v119 = *(_QWORD *)(v140 + 400);
      v120 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v117 = *(_DWORD *)(v140 + 408) + 1;
      v10 = (_QWORD *)(v119 + 16LL * v120);
      v121 = *v10;
      if ( *v10 != a2 )
      {
        v122 = 1;
        v123 = 0;
        while ( v121 != -8 )
        {
          if ( v121 == -16 && !v123 )
            v123 = v10;
          v120 = v8 & (v122 + v120);
          v10 = (_QWORD *)(v119 + 16LL * v120);
          v121 = *v10;
          if ( *v10 == a2 )
            goto LABEL_170;
          ++v122;
        }
        if ( v123 )
          v10 = v123;
      }
      goto LABEL_170;
    }
    goto LABEL_238;
  }
  if ( v6 - (v117 + *(_DWORD *)(v2 + 412)) <= v6 >> 3 )
  {
    v141 = v2;
    sub_21FA480(v131, v6);
    v2 = v141;
    v124 = *(_DWORD *)(v141 + 416);
    if ( v124 )
    {
      v125 = v124 - 1;
      v8 = *(_QWORD *)(v141 + 400);
      v126 = 0;
      v127 = v125 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v117 = *(_DWORD *)(v141 + 408) + 1;
      v128 = 1;
      v10 = (_QWORD *)(v8 + 16LL * v127);
      v129 = *v10;
      if ( *v10 != a2 )
      {
        while ( v129 != -8 )
        {
          if ( v129 == -16 && !v126 )
            v126 = v10;
          v127 = v125 & (v128 + v127);
          v10 = (_QWORD *)(v8 + 16LL * v127);
          v129 = *v10;
          if ( *v10 == a2 )
            goto LABEL_170;
          ++v128;
        }
        if ( v126 )
          v10 = v126;
      }
      goto LABEL_170;
    }
LABEL_238:
    ++*(_DWORD *)(v2 + 408);
    BUG();
  }
LABEL_170:
  *(_DWORD *)(v2 + 408) = v117;
  if ( *v10 != -8 )
    --*(_DWORD *)(v2 + 412);
  *v10 = a2;
  *((_DWORD *)v10 + 2) = 0;
LABEL_10:
  *((_DWORD *)v10 + 2) = -1;
  v12 = (unsigned int)v147;
  if ( (unsigned int)v147 >= HIDWORD(v147) )
  {
    v139 = v2;
    sub_16CD150((__int64)&v146, v148, 0, 8, v8, v2);
    v12 = (unsigned int)v147;
    v2 = v139;
  }
  v136 = 1;
  *(_QWORD *)&v146[8 * v12] = a2;
  v13 = v2;
  LODWORD(v147) = v147 + 1;
  v14 = (unsigned int)v147;
  if ( !(_DWORD)v147 )
    goto LABEL_49;
  do
  {
    v15 = 0;
    v16 = *(_QWORD *)&v146[8 * v14 - 8];
    v17 = *(__int64 **)(v16 + 88);
    if ( v17 == *(__int64 **)(v16 + 96) )
      goto LABEL_83;
    v132 = *(_QWORD *)&v146[8 * v14 - 8];
    v18 = *(__int64 **)(v16 + 96);
    do
    {
      v28 = *(unsigned int *)(v13 + 416);
      v29 = *(__int64 **)(v13 + 400);
      if ( !(_DWORD)v28 )
      {
        ++*(_QWORD *)(v13 + 392);
        goto LABEL_25;
      }
      v19 = *v17;
      LODWORD(v20) = v28 - 1;
      v21 = (v28 - 1) & (((unsigned int)*v17 >> 9) ^ ((unsigned int)*v17 >> 4));
      v22 = (__int64 **)&v29[2 * v21];
      v23 = *v22;
      if ( (__int64 *)*v17 != *v22 )
      {
        v54 = 1;
        v55 = 0;
        while ( v23 != (__int64 *)-8LL )
        {
          if ( v23 == (__int64 *)-16LL && !v55 )
            v55 = v22;
          v21 = (unsigned int)v20 & (v54 + v21);
          v22 = (__int64 **)&v29[2 * v21];
          v23 = *v22;
          if ( (__int64 *)v19 == *v22 )
            goto LABEL_16;
          ++v54;
        }
        v50 = v55;
        if ( !v55 )
          v50 = v22;
        v56 = *(_DWORD *)(v13 + 408);
        ++*(_QWORD *)(v13 + 392);
        v47 = v56 + 1;
        if ( 4 * (v56 + 1) < (unsigned int)(3 * v28) )
        {
          if ( (int)v28 - (v47 + *(_DWORD *)(v13 + 412)) > (unsigned int)v28 >> 3 )
            goto LABEL_42;
          sub_21FA480(v131, v28);
          v57 = *(_DWORD *)(v13 + 416);
          if ( v57 )
          {
            v58 = v57 - 1;
            v59 = 1;
            v20 = 0;
            v60 = *(_QWORD *)(v13 + 400);
            v61 = (v57 - 1) & (((unsigned int)*v17 >> 9) ^ ((unsigned int)*v17 >> 4));
            v47 = *(_DWORD *)(v13 + 408) + 1;
            v50 = (__int64 **)(v60 + 16LL * v61);
            v29 = *v50;
            if ( (__int64 *)*v17 == *v50 )
              goto LABEL_42;
            while ( v29 != (__int64 *)-8LL )
            {
              if ( !v20 && v29 == (__int64 *)-16LL )
                v20 = v50;
              v61 = v58 & (v59 + v61);
              v50 = (__int64 **)(v60 + 16LL * v61);
              v29 = *v50;
              if ( (__int64 *)*v17 == *v50 )
                goto LABEL_42;
              ++v59;
            }
            goto LABEL_70;
          }
LABEL_242:
          ++*(_DWORD *)(v13 + 408);
          BUG();
        }
LABEL_25:
        v133 = v29;
        v30 = (((((((unsigned int)(2 * v28 - 1) | ((unsigned __int64)(unsigned int)(2 * v28 - 1) >> 1)) >> 2)
                | (unsigned int)(2 * v28 - 1)
                | ((unsigned __int64)(unsigned int)(2 * v28 - 1) >> 1)) >> 4)
              | (((unsigned int)(2 * v28 - 1) | ((unsigned __int64)(unsigned int)(2 * v28 - 1) >> 1)) >> 2)
              | (unsigned int)(2 * v28 - 1)
              | ((unsigned __int64)(unsigned int)(2 * v28 - 1) >> 1)) >> 8)
            | (((((unsigned int)(2 * v28 - 1) | ((unsigned __int64)(unsigned int)(2 * v28 - 1) >> 1)) >> 2)
              | (unsigned int)(2 * v28 - 1)
              | ((unsigned __int64)(unsigned int)(2 * v28 - 1) >> 1)) >> 4)
            | (((unsigned int)(2 * v28 - 1) | ((unsigned __int64)(unsigned int)(2 * v28 - 1) >> 1)) >> 2)
            | (unsigned int)(2 * v28 - 1)
            | ((unsigned __int64)(unsigned int)(2 * v28 - 1) >> 1);
        v31 = ((v30 >> 16) | v30) + 1;
        if ( (unsigned int)v31 < 0x40 )
          LODWORD(v31) = 64;
        *(_DWORD *)(v13 + 416) = v31;
        v32 = (_QWORD *)sub_22077B0(16LL * (unsigned int)v31);
        *(_QWORD *)(v13 + 400) = v32;
        if ( v133 )
        {
          v33 = *(unsigned int *)(v13 + 416);
          *(_QWORD *)(v13 + 408) = 0;
          v34 = &v133[2 * v28];
          for ( i = &v32[2 * v33]; i != v32; v32 += 2 )
          {
            if ( v32 )
              *v32 = -8;
          }
          for ( j = v133; v34 != j; j += 2 )
          {
            v37 = *j;
            if ( *j != -16 && v37 != -8 )
            {
              v38 = *(_DWORD *)(v13 + 416);
              if ( !v38 )
              {
                MEMORY[0] = *j;
                BUG();
              }
              v39 = v38 - 1;
              v40 = *(_QWORD *)(v13 + 400);
              v41 = 1;
              v42 = 0;
              v43 = (v38 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
              v44 = (__int64 *)(v40 + 16LL * v43);
              v45 = *v44;
              if ( v37 != *v44 )
              {
                while ( v45 != -8 )
                {
                  if ( v45 == -16 && !v42 )
                    v42 = v44;
                  v43 = v39 & (v41 + v43);
                  v44 = (__int64 *)(v40 + 16LL * v43);
                  v45 = *v44;
                  if ( v37 == *v44 )
                    goto LABEL_37;
                  ++v41;
                }
                if ( v42 )
                  v44 = v42;
              }
LABEL_37:
              *v44 = v37;
              *((_DWORD *)v44 + 2) = *((_DWORD *)j + 2);
              ++*(_DWORD *)(v13 + 408);
            }
          }
          j___libc_free_0(v133);
          v32 = *(_QWORD **)(v13 + 400);
          v46 = *(_DWORD *)(v13 + 416);
          v47 = *(_DWORD *)(v13 + 408) + 1;
        }
        else
        {
          v80 = *(unsigned int *)(v13 + 416);
          *(_QWORD *)(v13 + 408) = 0;
          v46 = v80;
          v81 = &v32[2 * v80];
          if ( v32 != v81 )
          {
            v82 = v32;
            do
            {
              if ( v82 )
                *v82 = -8;
              v82 += 2;
            }
            while ( v81 != v82 );
          }
          v47 = 1;
        }
        if ( v46 )
        {
          v48 = v46 - 1;
          v49 = (v46 - 1) & (((unsigned int)*v17 >> 9) ^ ((unsigned int)*v17 >> 4));
          v50 = (__int64 **)&v32[2 * v49];
          v29 = *v50;
          if ( (__int64 *)*v17 == *v50 )
          {
LABEL_42:
            *(_DWORD *)(v13 + 408) = v47;
            if ( *v50 != (__int64 *)-8LL )
              --*(_DWORD *)(v13 + 412);
            v51 = (__int64 *)*v17;
            *((_DWORD *)v50 + 2) = 0;
            *v50 = v51;
            goto LABEL_45;
          }
          v99 = 1;
          v20 = 0;
          while ( v29 != (__int64 *)-8LL )
          {
            if ( !v20 && v29 == (__int64 *)-16LL )
              v20 = v50;
            v49 = v48 & (v99 + v49);
            v50 = (__int64 **)&v32[2 * v49];
            v29 = *v50;
            if ( (__int64 *)*v17 == *v50 )
              goto LABEL_42;
            ++v99;
          }
LABEL_70:
          if ( v20 )
            v50 = v20;
          goto LABEL_42;
        }
        goto LABEL_242;
      }
LABEL_16:
      if ( !*((_DWORD *)v22 + 2) )
      {
        v50 = v22;
LABEL_45:
        v52 = (unsigned int)v147;
        if ( (unsigned int)v147 >= HIDWORD(v147) )
        {
          sub_16CD150((__int64)&v146, v148, 0, 8, (int)v29, (int)v20);
          v52 = (unsigned int)v147;
        }
        *(_QWORD *)&v146[8 * v52] = *v17;
        LODWORD(v147) = v147 + 1;
        *((_DWORD *)v50 + 2) = -1;
        v14 = (unsigned int)v147;
        goto LABEL_48;
      }
      if ( v145 )
      {
        v24 = (v145 - 1) & (((unsigned int)*v17 >> 9) ^ ((unsigned int)*v17 >> 4));
        v25 = (__int64 *)(v143 + 16LL * v24);
        v26 = *v25;
        if ( v19 == *v25 )
        {
          v27 = *((_DWORD *)v25 + 2) + 1;
          goto LABEL_20;
        }
        v62 = 1;
        v63 = 0;
        while ( v26 != -8 )
        {
          if ( v63 || v26 != -16 )
            v25 = v63;
          v24 = (v145 - 1) & (v62 + v24);
          v94 = (__int64 *)(v143 + 16LL * v24);
          v26 = *v94;
          if ( v19 == *v94 )
          {
            v27 = *((_DWORD *)v94 + 2) + 1;
            v25 = v94;
            goto LABEL_20;
          }
          ++v62;
          v63 = v25;
          v25 = (__int64 *)(v143 + 16LL * v24);
        }
        if ( v63 )
          v25 = v63;
        ++v142;
        v64 = v144 + 1;
        if ( 4 * ((int)v144 + 1) < 3 * v145 )
        {
          if ( v145 - HIDWORD(v144) - v64 <= v145 >> 3 )
          {
            sub_1DA35E0((__int64)&v142, v145);
            if ( !v145 )
            {
LABEL_241:
              LODWORD(v144) = v144 + 1;
              BUG();
            }
            v76 = 0;
            v77 = 1;
            v64 = v144 + 1;
            v78 = (v145 - 1) & (((unsigned int)*v17 >> 9) ^ ((unsigned int)*v17 >> 4));
            v25 = (__int64 *)(v143 + 16LL * v78);
            v79 = *v25;
            if ( *v25 != *v17 )
            {
              while ( v79 != -8 )
              {
                if ( v79 != -16 || v76 )
                  v25 = v76;
                v78 = (v145 - 1) & (v77 + v78);
                v79 = *(_QWORD *)(v143 + 16LL * v78);
                if ( *v17 == v79 )
                {
                  v25 = (__int64 *)(v143 + 16LL * v78);
                  goto LABEL_79;
                }
                ++v77;
                v76 = v25;
                v25 = (__int64 *)(v143 + 16LL * v78);
              }
LABEL_93:
              if ( v76 )
                v25 = v76;
              goto LABEL_79;
            }
          }
          goto LABEL_79;
        }
      }
      else
      {
        ++v142;
      }
      sub_1DA35E0((__int64)&v142, 2 * v145);
      if ( !v145 )
        goto LABEL_241;
      v64 = v144 + 1;
      v73 = (v145 - 1) & (((unsigned int)*v17 >> 9) ^ ((unsigned int)*v17 >> 4));
      v25 = (__int64 *)(v143 + 16LL * v73);
      v74 = *v25;
      if ( *v25 != *v17 )
      {
        v75 = 1;
        v76 = 0;
        while ( v74 != -8 )
        {
          if ( v74 == -16 && !v76 )
            v76 = v25;
          v73 = (v145 - 1) & (v75 + v73);
          v25 = (__int64 *)(v143 + 16LL * v73);
          v74 = *v25;
          if ( *v17 == *v25 )
            goto LABEL_79;
          ++v75;
        }
        goto LABEL_93;
      }
LABEL_79:
      LODWORD(v144) = v64;
      if ( *v25 != -8 )
        --HIDWORD(v144);
      v65 = *v17;
      *((_DWORD *)v25 + 2) = 0;
      *v25 = v65;
      v27 = 1;
LABEL_20:
      *((_DWORD *)v25 + 2) = v27;
      if ( *(_DWORD *)(v13 + 428) >= v27 )
        v27 = *(_DWORD *)(v13 + 428);
      ++v17;
      ++v15;
      *(_DWORD *)(v13 + 428) = v27;
    }
    while ( v18 != v17 );
    v16 = v132;
LABEL_83:
    if ( *(_DWORD *)(v13 + 424) >= v15 )
      v15 = *(_DWORD *)(v13 + 424);
    v66 = *(_DWORD *)(v13 + 416);
    *(_DWORD *)(v13 + 424) = v15;
    if ( !v66 )
    {
      ++*(_QWORD *)(v13 + 392);
LABEL_149:
      v134 = v16;
      sub_21FA480(v131, 2 * v66);
      v100 = *(_DWORD *)(v13 + 416);
      if ( v100 )
      {
        v16 = v134;
        v101 = v100 - 1;
        v102 = *(_QWORD *)(v13 + 400);
        v98 = *(_DWORD *)(v13 + 408) + 1;
        v103 = v101 & (((unsigned int)v134 >> 9) ^ ((unsigned int)v134 >> 4));
        v70 = (__int64 *)(v102 + 16LL * v103);
        v104 = *v70;
        if ( v134 != *v70 )
        {
          v105 = 1;
          v106 = 0;
          while ( v104 != -8 )
          {
            if ( !v106 && v104 == -16 )
              v106 = v70;
            v103 = v101 & (v105 + v103);
            v70 = (__int64 *)(v102 + 16LL * v103);
            v104 = *v70;
            if ( v134 == *v70 )
              goto LABEL_138;
            ++v105;
          }
          if ( v106 )
            v70 = v106;
        }
        goto LABEL_138;
      }
LABEL_239:
      ++*(_DWORD *)(v13 + 408);
      BUG();
    }
    v67 = *(_QWORD *)(v13 + 400);
    v68 = ((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4);
    v69 = (v66 - 1) & v68;
    v70 = (__int64 *)(v67 + 16LL * v69);
    v71 = *v70;
    if ( v16 == *v70 )
      goto LABEL_87;
    v95 = 1;
    v96 = 0;
    while ( v71 != -8 )
    {
      if ( v96 || v71 != -16 )
        v70 = v96;
      v69 = (v66 - 1) & (v95 + v69);
      v71 = *(_QWORD *)(v67 + 16LL * v69);
      if ( v16 == v71 )
      {
        v70 = (__int64 *)(v67 + 16LL * v69);
        goto LABEL_87;
      }
      ++v95;
      v96 = v70;
      v70 = (__int64 *)(v67 + 16LL * v69);
    }
    v97 = *(_DWORD *)(v13 + 408);
    if ( v96 )
      v70 = v96;
    ++*(_QWORD *)(v13 + 392);
    v98 = v97 + 1;
    if ( 4 * (v97 + 1) >= 3 * v66 )
      goto LABEL_149;
    if ( v66 - *(_DWORD *)(v13 + 412) - v98 <= v66 >> 3 )
    {
      v135 = v16;
      sub_21FA480(v131, v66);
      v107 = *(_DWORD *)(v13 + 416);
      if ( v107 )
      {
        v108 = v107 - 1;
        v109 = *(_QWORD *)(v13 + 400);
        v110 = 1;
        v111 = v108 & v68;
        v16 = v135;
        v98 = *(_DWORD *)(v13 + 408) + 1;
        v112 = 0;
        v70 = (__int64 *)(v109 + 16LL * v111);
        v113 = *v70;
        if ( v135 != *v70 )
        {
          while ( v113 != -8 )
          {
            if ( !v112 && v113 == -16 )
              v112 = v70;
            v111 = v108 & (v110 + v111);
            v70 = (__int64 *)(v109 + 16LL * v111);
            v113 = *v70;
            if ( v135 == *v70 )
              goto LABEL_138;
            ++v110;
          }
          if ( v112 )
            v70 = v112;
        }
        goto LABEL_138;
      }
      goto LABEL_239;
    }
LABEL_138:
    *(_DWORD *)(v13 + 408) = v98;
    if ( *v70 != -8 )
      --*(_DWORD *)(v13 + 412);
    *v70 = v16;
    *((_DWORD *)v70 + 2) = 0;
LABEL_87:
    v72 = v136++;
    *((_DWORD *)v70 + 2) = v72;
    v14 = (unsigned int)(v147 - 1);
    LODWORD(v147) = v147 - 1;
LABEL_48:
    ;
  }
  while ( v14 );
LABEL_49:
  if ( v146 != v148 )
    _libc_free((unsigned __int64)v146);
  return j___libc_free_0(v143);
}
