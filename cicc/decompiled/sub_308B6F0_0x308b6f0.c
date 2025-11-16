// Function: sub_308B6F0
// Address: 0x308b6f0
//
__int64 __fastcall sub_308B6F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v8; // eax
  _QWORD *v9; // rdi
  __int64 v10; // rsi
  _QWORD *v11; // rax
  __int64 v12; // r8
  int v13; // r11d
  unsigned int v14; // ecx
  _QWORD *v15; // rdx
  __int64 *v16; // rax
  __int64 v17; // r10
  _DWORD *v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v21; // rax
  unsigned int v22; // r12d
  __int64 v23; // rax
  __int64 *v24; // rbx
  __int64 *v25; // r15
  __int64 v26; // rdi
  __int64 v27; // r10
  __int64 v28; // rcx
  int v29; // r13d
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 v32; // r8
  _DWORD *v33; // r13
  int v34; // r11d
  __int64 *v35; // rcx
  unsigned int v36; // edx
  __int64 *v37; // rax
  __int64 v38; // r8
  unsigned int *v39; // rdx
  unsigned int v40; // eax
  unsigned int v41; // esi
  int v42; // eax
  int v43; // esi
  unsigned int v44; // eax
  int v45; // edx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rbx
  unsigned __int64 v49; // rdx
  int v51; // eax
  int v52; // eax
  int v53; // r11d
  __int64 v54; // r10
  __int64 v55; // rdi
  unsigned int v56; // eax
  int v57; // edi
  __int64 v58; // rax
  unsigned int v59; // esi
  __int64 v60; // r8
  unsigned int v61; // edi
  _QWORD *v62; // rax
  __int64 v63; // rcx
  int *v64; // rax
  int v65; // ebx
  unsigned int v66; // eax
  __int64 v67; // rdx
  int v68; // r11d
  __int64 *v69; // r10
  int v70; // r11d
  unsigned int v71; // eax
  __int64 v72; // rdx
  unsigned int v73; // edx
  unsigned int v74; // eax
  int v75; // r12d
  unsigned __int64 v76; // rdx
  unsigned __int64 v77; // rax
  __int64 v78; // rax
  _QWORD *v79; // rax
  _QWORD *v80; // rdx
  int v81; // r11d
  _QWORD *v82; // rdx
  int v83; // eax
  int v84; // ecx
  int v85; // r11d
  int v86; // esi
  int v87; // esi
  __int64 v88; // r9
  unsigned int v89; // eax
  __int64 v90; // r11
  int v91; // r8d
  _QWORD *v92; // rdi
  int v93; // r8d
  __int64 v94; // r10
  unsigned int v95; // ecx
  int v96; // edx
  int v97; // edi
  __int64 *v98; // rsi
  int v99; // eax
  int v100; // eax
  __int64 v101; // r9
  _QWORD *v102; // rsi
  unsigned int v103; // ebx
  int v104; // edi
  __int64 v105; // r8
  int v106; // ecx
  int v107; // edi
  int v108; // edi
  unsigned int v109; // r12d
  int v110; // ecx
  __int64 v111; // rsi
  _QWORD *v112; // rax
  __int64 v113; // [rsp+8h] [rbp-B8h]
  __int64 v114; // [rsp+10h] [rbp-B0h]
  int v115; // [rsp+1Ch] [rbp-A4h]
  __int64 v116; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v117; // [rsp+28h] [rbp-98h]
  __int64 v118; // [rsp+30h] [rbp-90h]
  unsigned int v119; // [rsp+38h] [rbp-88h]
  _BYTE *v120; // [rsp+40h] [rbp-80h] BYREF
  __int64 v121; // [rsp+48h] [rbp-78h]
  _BYTE v122[112]; // [rsp+50h] [rbp-70h] BYREF

  v113 = a1 + 360;
  v8 = *(_DWORD *)(a1 + 376);
  ++*(_QWORD *)(a1 + 360);
  v9 = *(_QWORD **)(a1 + 368);
  v10 = *(unsigned int *)(a1 + 384);
  if ( !v8 )
  {
    if ( !*(_DWORD *)(a1 + 380) )
      goto LABEL_8;
    if ( (unsigned int)v10 > 0x40 )
    {
      sub_C7D6A0((__int64)v9, 16 * v10, 8);
      *(_QWORD *)(a1 + 368) = 0;
      *(_QWORD *)(a1 + 392) = 0x100000001LL;
      v120 = v122;
      *(_QWORD *)(a1 + 376) = 0;
      *(_DWORD *)(a1 + 384) = 0;
      v116 = 0;
      v117 = 0;
      v118 = 0;
      v119 = 0;
      v121 = 0x800000000LL;
      goto LABEL_127;
    }
    goto LABEL_4;
  }
  v73 = 4 * v8;
  if ( (unsigned int)(4 * v8) < 0x40 )
    v73 = 64;
  if ( (unsigned int)v10 <= v73 )
  {
LABEL_4:
    v11 = &v9[2 * (unsigned int)v10];
    if ( v9 != v11 )
    {
      do
      {
        *v9 = -4096;
        v9 += 2;
      }
      while ( v11 != v9 );
      v9 = *(_QWORD **)(a1 + 368);
      LODWORD(v10) = *(_DWORD *)(a1 + 384);
    }
    *(_QWORD *)(a1 + 376) = 0;
    goto LABEL_8;
  }
  v74 = v8 - 1;
  if ( !v74 )
  {
    v75 = 64;
LABEL_96:
    sub_C7D6A0((__int64)v9, 16 * v10, 8);
    v76 = ((((((((4 * v75 / 3u + 1) | ((unsigned __int64)(4 * v75 / 3u + 1) >> 1)) >> 2)
             | (4 * v75 / 3u + 1)
             | ((unsigned __int64)(4 * v75 / 3u + 1) >> 1)) >> 4)
           | (((4 * v75 / 3u + 1) | ((unsigned __int64)(4 * v75 / 3u + 1) >> 1)) >> 2)
           | (4 * v75 / 3u + 1)
           | ((unsigned __int64)(4 * v75 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v75 / 3u + 1) | ((unsigned __int64)(4 * v75 / 3u + 1) >> 1)) >> 2)
           | (4 * v75 / 3u + 1)
           | ((unsigned __int64)(4 * v75 / 3u + 1) >> 1)) >> 4)
         | (((4 * v75 / 3u + 1) | ((unsigned __int64)(4 * v75 / 3u + 1) >> 1)) >> 2)
         | (4 * v75 / 3u + 1)
         | ((unsigned __int64)(4 * v75 / 3u + 1) >> 1)) >> 16;
    v77 = (v76
         | (((((((4 * v75 / 3u + 1) | ((unsigned __int64)(4 * v75 / 3u + 1) >> 1)) >> 2)
             | (4 * v75 / 3u + 1)
             | ((unsigned __int64)(4 * v75 / 3u + 1) >> 1)) >> 4)
           | (((4 * v75 / 3u + 1) | ((unsigned __int64)(4 * v75 / 3u + 1) >> 1)) >> 2)
           | (4 * v75 / 3u + 1)
           | ((unsigned __int64)(4 * v75 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v75 / 3u + 1) | ((unsigned __int64)(4 * v75 / 3u + 1) >> 1)) >> 2)
           | (4 * v75 / 3u + 1)
           | ((unsigned __int64)(4 * v75 / 3u + 1) >> 1)) >> 4)
         | (((4 * v75 / 3u + 1) | ((unsigned __int64)(4 * v75 / 3u + 1) >> 1)) >> 2)
         | (4 * v75 / 3u + 1)
         | ((unsigned __int64)(4 * v75 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 384) = v77;
    v78 = sub_C7D670(16 * v77, 8);
    *(_QWORD *)(a1 + 376) = 0;
    v9 = (_QWORD *)v78;
    *(_QWORD *)(a1 + 368) = v78;
    v10 = *(unsigned int *)(a1 + 384);
    v79 = (_QWORD *)(v78 + 16 * v10);
    if ( v9 != v79 )
    {
      v80 = v9;
      do
      {
        if ( v80 )
          *v80 = -4096;
        v80 += 2;
      }
      while ( v79 != v80 );
    }
    goto LABEL_8;
  }
  _BitScanReverse(&v74, v74);
  v75 = 1 << (33 - (v74 ^ 0x1F));
  if ( v75 < 64 )
    v75 = 64;
  if ( (_DWORD)v10 != v75 )
    goto LABEL_96;
  *(_QWORD *)(a1 + 376) = 0;
  v112 = &v9[2 * (unsigned int)v10];
  do
  {
    if ( v9 )
      *v9 = -4096;
    v9 += 2;
  }
  while ( v112 != v9 );
  v9 = *(_QWORD **)(a1 + 368);
  LODWORD(v10) = *(_DWORD *)(a1 + 384);
LABEL_8:
  v116 = 0;
  *(_QWORD *)(a1 + 392) = 0x100000001LL;
  v120 = v122;
  v117 = 0;
  v118 = 0;
  v119 = 0;
  v121 = 0x800000000LL;
  if ( !(_DWORD)v10 )
  {
LABEL_127:
    ++*(_QWORD *)(a1 + 360);
    LODWORD(v10) = 0;
    goto LABEL_128;
  }
  v12 = (unsigned int)(v10 - 1);
  v13 = 1;
  v14 = v12 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v15 = &v9[2 * v14];
  v16 = 0;
  v17 = *v15;
  if ( *v15 == a2 )
  {
LABEL_10:
    v18 = v15 + 1;
    goto LABEL_11;
  }
  while ( v17 != -4096 )
  {
    if ( v17 == -8192 && !v16 )
      v16 = v15;
    a6 = (unsigned int)(v13 + 1);
    v14 = v12 & (v13 + v14);
    v15 = &v9[2 * v14];
    v17 = *v15;
    if ( *v15 == a2 )
      goto LABEL_10;
    ++v13;
  }
  v106 = *(_DWORD *)(a1 + 376);
  if ( !v16 )
    v16 = v15;
  ++*(_QWORD *)(a1 + 360);
  v96 = v106 + 1;
  if ( 4 * (v106 + 1) >= (unsigned int)(3 * v10) )
  {
LABEL_128:
    sub_308B2C0(v113, 2 * v10);
    v93 = *(_DWORD *)(a1 + 384);
    if ( v93 )
    {
      v12 = (unsigned int)(v93 - 1);
      v94 = *(_QWORD *)(a1 + 368);
      v95 = v12 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v96 = *(_DWORD *)(a1 + 376) + 1;
      v16 = (__int64 *)(v94 + 16LL * v95);
      a6 = *v16;
      if ( *v16 != a2 )
      {
        v97 = 1;
        v98 = 0;
        while ( a6 != -4096 )
        {
          if ( a6 == -8192 && !v98 )
            v98 = v16;
          v95 = v12 & (v97 + v95);
          v16 = (__int64 *)(v94 + 16LL * v95);
          a6 = *v16;
          if ( *v16 == a2 )
            goto LABEL_151;
          ++v97;
        }
        if ( v98 )
          v16 = v98;
      }
      goto LABEL_151;
    }
    goto LABEL_202;
  }
  if ( (int)v10 - (v96 + *(_DWORD *)(a1 + 380)) <= (unsigned int)v10 >> 3 )
  {
    sub_308B2C0(v113, v10);
    v107 = *(_DWORD *)(a1 + 384);
    if ( v107 )
    {
      v108 = v107 - 1;
      v12 = *(_QWORD *)(a1 + 368);
      a6 = 0;
      v109 = v108 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v96 = *(_DWORD *)(a1 + 376) + 1;
      v110 = 1;
      v16 = (__int64 *)(v12 + 16LL * v109);
      v111 = *v16;
      if ( *v16 != a2 )
      {
        while ( v111 != -4096 )
        {
          if ( v111 == -8192 && !a6 )
            a6 = (__int64)v16;
          v109 = v108 & (v110 + v109);
          v16 = (__int64 *)(v12 + 16LL * v109);
          v111 = *v16;
          if ( *v16 == a2 )
            goto LABEL_151;
          ++v110;
        }
        if ( a6 )
          v16 = (__int64 *)a6;
      }
      goto LABEL_151;
    }
LABEL_202:
    ++*(_DWORD *)(a1 + 376);
    BUG();
  }
LABEL_151:
  *(_DWORD *)(a1 + 376) = v96;
  if ( *v16 != -4096 )
    --*(_DWORD *)(a1 + 380);
  *v16 = a2;
  v18 = v16 + 1;
  *v18 = 0;
LABEL_11:
  *v18 = -1;
  v19 = (unsigned int)v121;
  v20 = (unsigned int)v121 + 1LL;
  if ( v20 > HIDWORD(v121) )
  {
    sub_C8D5F0((__int64)&v120, v122, v20, 8u, v12, a6);
    v19 = (unsigned int)v121;
  }
  v115 = 1;
  *(_QWORD *)&v120[8 * v19] = a2;
  v21 = (unsigned int)(v121 + 1);
  LODWORD(v121) = v121 + 1;
LABEL_14:
  if ( !v21 )
    goto LABEL_35;
  do
  {
    v22 = 0;
    v23 = *(_QWORD *)&v120[8 * v21 - 8];
    v24 = *(__int64 **)(v23 + 112);
    v114 = v23;
    v25 = &v24[*(unsigned int *)(v23 + 120)];
    if ( v25 == v24 )
    {
LABEL_67:
      if ( *(_DWORD *)(a1 + 392) >= v22 )
        v22 = *(_DWORD *)(a1 + 392);
      v59 = *(_DWORD *)(a1 + 384);
      *(_DWORD *)(a1 + 392) = v22;
      if ( v59 )
      {
        v60 = *(_QWORD *)(a1 + 368);
        v61 = (v59 - 1) & (((unsigned int)v114 >> 9) ^ ((unsigned int)v114 >> 4));
        v62 = (_QWORD *)(v60 + 16LL * v61);
        v63 = *v62;
        if ( v114 == *v62 )
        {
LABEL_71:
          v64 = (int *)(v62 + 1);
LABEL_72:
          v65 = v115++;
          *v64 = v65;
          v21 = (unsigned int)(v121 - 1);
          LODWORD(v121) = v121 - 1;
          goto LABEL_14;
        }
        v81 = 1;
        v82 = 0;
        while ( v63 != -4096 )
        {
          if ( !v82 && v63 == -8192 )
            v82 = v62;
          v61 = (v59 - 1) & (v81 + v61);
          v62 = (_QWORD *)(v60 + 16LL * v61);
          v63 = *v62;
          if ( v114 == *v62 )
            goto LABEL_71;
          ++v81;
        }
        if ( !v82 )
          v82 = v62;
        v83 = *(_DWORD *)(a1 + 376);
        ++*(_QWORD *)(a1 + 360);
        v84 = v83 + 1;
        if ( 4 * (v83 + 1) < 3 * v59 )
        {
          if ( v59 - *(_DWORD *)(a1 + 380) - v84 <= v59 >> 3 )
          {
            sub_308B2C0(v113, v59);
            v99 = *(_DWORD *)(a1 + 384);
            if ( !v99 )
            {
LABEL_201:
              ++*(_DWORD *)(a1 + 376);
              BUG();
            }
            v100 = v99 - 1;
            v101 = *(_QWORD *)(a1 + 368);
            v102 = 0;
            v103 = v100 & (((unsigned int)v114 >> 9) ^ ((unsigned int)v114 >> 4));
            v104 = 1;
            v84 = *(_DWORD *)(a1 + 376) + 1;
            v82 = (_QWORD *)(v101 + 16LL * v103);
            v105 = *v82;
            if ( v114 != *v82 )
            {
              while ( v105 != -4096 )
              {
                if ( v105 == -8192 && !v102 )
                  v102 = v82;
                v103 = v100 & (v104 + v103);
                v82 = (_QWORD *)(v101 + 16LL * v103);
                v105 = *v82;
                if ( v114 == *v82 )
                  goto LABEL_108;
                ++v104;
              }
              if ( v102 )
                v82 = v102;
            }
          }
          goto LABEL_108;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 360);
      }
      sub_308B2C0(v113, 2 * v59);
      v86 = *(_DWORD *)(a1 + 384);
      if ( !v86 )
        goto LABEL_201;
      v87 = v86 - 1;
      v88 = *(_QWORD *)(a1 + 368);
      v84 = *(_DWORD *)(a1 + 376) + 1;
      v89 = v87 & (((unsigned int)v114 >> 9) ^ ((unsigned int)v114 >> 4));
      v82 = (_QWORD *)(v88 + 16LL * v89);
      v90 = *v82;
      if ( v114 != *v82 )
      {
        v91 = 1;
        v92 = 0;
        while ( v90 != -4096 )
        {
          if ( !v92 && v90 == -8192 )
            v92 = v82;
          v89 = v87 & (v91 + v89);
          v82 = (_QWORD *)(v88 + 16LL * v89);
          v90 = *v82;
          if ( v114 == *v82 )
            goto LABEL_108;
          ++v91;
        }
        if ( v92 )
          v82 = v92;
      }
LABEL_108:
      *(_DWORD *)(a1 + 376) = v84;
      if ( *v82 != -4096 )
        --*(_DWORD *)(a1 + 380);
      *((_DWORD *)v82 + 2) = 0;
      *v82 = v114;
      v64 = (int *)(v82 + 1);
      goto LABEL_72;
    }
    while ( 1 )
    {
      v41 = *(_DWORD *)(a1 + 384);
      if ( !v41 )
      {
        ++*(_QWORD *)(a1 + 360);
LABEL_27:
        sub_308B2C0(v113, 2 * v41);
        v42 = *(_DWORD *)(a1 + 384);
        if ( v42 )
        {
          v43 = v42 - 1;
          v32 = *(_QWORD *)(a1 + 368);
          v44 = (v42 - 1) & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
          v45 = *(_DWORD *)(a1 + 376) + 1;
          v28 = v32 + 16LL * v44;
          v30 = *(_QWORD *)v28;
          if ( *v24 != *(_QWORD *)v28 )
          {
            v85 = 1;
            v54 = 0;
            while ( v30 != -4096 )
            {
              if ( !v54 && v30 == -8192 )
                v54 = v28;
              v44 = v43 & (v85 + v44);
              v28 = v32 + 16LL * v44;
              v30 = *(_QWORD *)v28;
              if ( *v24 == *(_QWORD *)v28 )
                goto LABEL_29;
              ++v85;
            }
            goto LABEL_51;
          }
          goto LABEL_29;
        }
LABEL_204:
        ++*(_DWORD *)(a1 + 376);
        BUG();
      }
      v26 = *v24;
      v27 = *(_QWORD *)(a1 + 368);
      v28 = 0;
      v29 = 1;
      v30 = (v41 - 1) & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
      v31 = v27 + 16 * v30;
      v32 = *(_QWORD *)v31;
      if ( *v24 != *(_QWORD *)v31 )
        break;
LABEL_18:
      v33 = (_DWORD *)(v31 + 8);
      if ( !*(_DWORD *)(v31 + 8) )
        goto LABEL_32;
      if ( !v119 )
      {
        ++v116;
        goto LABEL_74;
      }
      v34 = 1;
      v35 = 0;
      v36 = (v119 - 1) & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
      v37 = (__int64 *)(v117 + 16LL * v36);
      v38 = *v37;
      if ( v26 != *v37 )
      {
        while ( v38 != -4096 )
        {
          if ( v38 == -8192 && !v35 )
            v35 = v37;
          v36 = (v119 - 1) & (v34 + v36);
          v37 = (__int64 *)(v117 + 16LL * v36);
          v38 = *v37;
          if ( v26 == *v37 )
            goto LABEL_21;
          ++v34;
        }
        if ( !v35 )
          v35 = v37;
        ++v116;
        v57 = v118 + 1;
        if ( 4 * ((int)v118 + 1) >= 3 * v119 )
        {
LABEL_74:
          sub_2EB73C0((__int64)&v116, 2 * v119);
          if ( !v119 )
            goto LABEL_203;
          v57 = v118 + 1;
          v66 = (v119 - 1) & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
          v35 = (__int64 *)(v117 + 16LL * v66);
          v67 = *v35;
          if ( *v35 != *v24 )
          {
            v68 = 1;
            v69 = 0;
            while ( v67 != -4096 )
            {
              if ( !v69 && v67 == -8192 )
                v69 = v35;
              v66 = (v119 - 1) & (v68 + v66);
              v35 = (__int64 *)(v117 + 16LL * v66);
              v67 = *v35;
              if ( *v24 == *v35 )
                goto LABEL_64;
              ++v68;
            }
LABEL_78:
            if ( v69 )
              v35 = v69;
          }
        }
        else if ( v119 - HIDWORD(v118) - v57 <= v119 >> 3 )
        {
          sub_2EB73C0((__int64)&v116, v119);
          if ( !v119 )
          {
LABEL_203:
            LODWORD(v118) = v118 + 1;
            BUG();
          }
          v69 = 0;
          v70 = 1;
          v57 = v118 + 1;
          v71 = (v119 - 1) & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
          v35 = (__int64 *)(v117 + 16LL * v71);
          v72 = *v35;
          if ( *v24 != *v35 )
          {
            while ( v72 != -4096 )
            {
              if ( v72 == -8192 && !v69 )
                v69 = v35;
              v71 = (v119 - 1) & (v70 + v71);
              v35 = (__int64 *)(v117 + 16LL * v71);
              v72 = *v35;
              if ( *v24 == *v35 )
                goto LABEL_64;
              ++v70;
            }
            goto LABEL_78;
          }
        }
LABEL_64:
        LODWORD(v118) = v57;
        if ( *v35 != -4096 )
          --HIDWORD(v118);
        v58 = *v24;
        v39 = (unsigned int *)(v35 + 1);
        *((_DWORD *)v35 + 2) = 0;
        *v35 = v58;
        v40 = 1;
        goto LABEL_22;
      }
LABEL_21:
      v39 = (unsigned int *)(v37 + 1);
      v40 = *((_DWORD *)v37 + 2) + 1;
LABEL_22:
      *v39 = v40;
      if ( *(_DWORD *)(a1 + 396) >= v40 )
        v40 = *(_DWORD *)(a1 + 396);
      ++v24;
      ++v22;
      *(_DWORD *)(a1 + 396) = v40;
      if ( v24 == v25 )
        goto LABEL_67;
    }
    while ( v32 != -4096 )
    {
      if ( v32 == -8192 && !v28 )
        v28 = v31;
      v30 = (v41 - 1) & (v29 + (_DWORD)v30);
      v31 = v27 + 16LL * (unsigned int)v30;
      v32 = *(_QWORD *)v31;
      if ( v26 == *(_QWORD *)v31 )
        goto LABEL_18;
      ++v29;
    }
    if ( !v28 )
      v28 = v31;
    v51 = *(_DWORD *)(a1 + 376);
    ++*(_QWORD *)(a1 + 360);
    v45 = v51 + 1;
    if ( 4 * (v51 + 1) >= 3 * v41 )
      goto LABEL_27;
    if ( v41 - *(_DWORD *)(a1 + 380) - v45 > v41 >> 3 )
      goto LABEL_29;
    sub_308B2C0(v113, v41);
    v52 = *(_DWORD *)(a1 + 384);
    if ( !v52 )
      goto LABEL_204;
    v32 = (unsigned int)(v52 - 1);
    v53 = 1;
    v54 = 0;
    v55 = *(_QWORD *)(a1 + 368);
    v56 = v32 & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
    v45 = *(_DWORD *)(a1 + 376) + 1;
    v28 = v55 + 16LL * v56;
    v30 = *(_QWORD *)v28;
    if ( *v24 != *(_QWORD *)v28 )
    {
      while ( v30 != -4096 )
      {
        if ( !v54 && v30 == -8192 )
          v54 = v28;
        v56 = v32 & (v53 + v56);
        v28 = v55 + 16LL * v56;
        v30 = *(_QWORD *)v28;
        if ( *v24 == *(_QWORD *)v28 )
          goto LABEL_29;
        ++v53;
      }
LABEL_51:
      if ( v54 )
        v28 = v54;
    }
LABEL_29:
    *(_DWORD *)(a1 + 376) = v45;
    if ( *(_QWORD *)v28 != -4096 )
      --*(_DWORD *)(a1 + 380);
    v46 = *v24;
    *(_DWORD *)(v28 + 8) = 0;
    v33 = (_DWORD *)(v28 + 8);
    *(_QWORD *)v28 = v46;
LABEL_32:
    v47 = (unsigned int)v121;
    v48 = *v24;
    v49 = (unsigned int)v121 + 1LL;
    if ( v49 > HIDWORD(v121) )
    {
      sub_C8D5F0((__int64)&v120, v122, v49, 8u, v32, v30);
      v47 = (unsigned int)v121;
    }
    *(_QWORD *)&v120[8 * v47] = v48;
    LODWORD(v121) = v121 + 1;
    *v33 = -1;
    v21 = (unsigned int)v121;
  }
  while ( (_DWORD)v121 );
LABEL_35:
  if ( v120 != v122 )
    _libc_free((unsigned __int64)v120);
  return sub_C7D6A0(v117, 16LL * v119, 8);
}
