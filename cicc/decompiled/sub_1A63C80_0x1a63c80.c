// Function: sub_1A63C80
// Address: 0x1a63c80
//
_QWORD **__fastcall sub_1A63C80(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD **result; // rax
  unsigned int v6; // esi
  _QWORD *v7; // r11
  unsigned __int64 v8; // r8
  __int64 *v9; // r14
  unsigned int v10; // r12d
  unsigned int v11; // edi
  __int64 *v12; // rax
  _QWORD *v13; // rcx
  __int64 v14; // r12
  unsigned int v15; // r13d
  unsigned int v16; // ecx
  __int64 v17; // r9
  int v18; // r10d
  __int64 *v19; // rdx
  int v20; // eax
  int v21; // ecx
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // rax
  __int64 v27; // rdx
  __int64 *v28; // r10
  _QWORD *j; // rdx
  __int64 *v30; // rax
  __int64 v31; // rcx
  int v32; // edi
  int v33; // edi
  __int64 v34; // r9
  unsigned int v35; // esi
  _QWORD *v36; // rdx
  __int64 v37; // r8
  int v38; // esi
  int v39; // esi
  unsigned int v40; // r13d
  __int64 v41; // rdi
  int v42; // r9d
  __int64 *v43; // r8
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  _QWORD *v46; // rax
  __int64 v47; // rdx
  __int64 *v48; // r10
  _QWORD *i; // rdx
  __int64 *v50; // rax
  __int64 v51; // rcx
  int v52; // edi
  int v53; // edi
  __int64 v54; // r9
  unsigned int v55; // esi
  _QWORD *v56; // rdx
  __int64 v57; // r8
  int v58; // esi
  int v59; // esi
  int v60; // r9d
  unsigned int v61; // r13d
  __int64 v62; // rdi
  __int64 v63; // rcx
  _QWORD *v64; // rcx
  _QWORD *v65; // rdx
  __int64 v66; // rcx
  _QWORD *v67; // rcx
  _QWORD *v68; // rdx
  int v69; // r9d
  __int64 *v70; // rdx
  int v71; // eax
  int v72; // ecx
  int v73; // eax
  int v74; // edi
  __int64 v75; // rsi
  unsigned int v76; // eax
  __int64 v77; // r8
  int v78; // r10d
  __int64 *v79; // r9
  int v80; // eax
  int v81; // eax
  __int64 v82; // rdi
  int v83; // r9d
  unsigned int v84; // r12d
  __int64 *v85; // r8
  __int64 v86; // rsi
  int v87; // [rsp+14h] [rbp-4Ch]
  int v88; // [rsp+14h] [rbp-4Ch]
  _QWORD *v89; // [rsp+18h] [rbp-48h]
  _QWORD *v90; // [rsp+18h] [rbp-48h]
  _QWORD *v91; // [rsp+18h] [rbp-48h]
  _QWORD *v92; // [rsp+18h] [rbp-48h]
  _QWORD *v93; // [rsp+18h] [rbp-48h]
  _QWORD *v94; // [rsp+18h] [rbp-48h]
  _QWORD *v95; // [rsp+18h] [rbp-48h]
  _QWORD *v96; // [rsp+18h] [rbp-48h]
  _QWORD *v97; // [rsp+18h] [rbp-48h]
  _QWORD **v98; // [rsp+20h] [rbp-40h]
  _QWORD **v99; // [rsp+28h] [rbp-38h]

  result = (_QWORD **)a1[1];
  v98 = result;
  v99 = (_QWORD **)*a1;
  if ( (_QWORD **)*a1 == result )
    return result;
  do
  {
    v6 = *(_DWORD *)(a2 + 24);
    v7 = *v99;
    if ( !v6 )
    {
      ++*(_QWORD *)a2;
      goto LABEL_101;
    }
    v8 = v6 - 1;
    v9 = *(__int64 **)(a2 + 8);
    v10 = ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4);
    v11 = v8 & v10;
    v12 = &v9[2 * ((unsigned int)v8 & v10)];
    v13 = (_QWORD *)*v12;
    if ( v7 == (_QWORD *)*v12 )
      goto LABEL_4;
    v69 = 1;
    v70 = 0;
    while ( 1 )
    {
      if ( v13 == (_QWORD *)-8LL )
      {
        if ( !v70 )
          v70 = v12;
        v71 = *(_DWORD *)(a2 + 16);
        ++*(_QWORD *)a2;
        v72 = v71 + 1;
        if ( 4 * (v71 + 1) < 3 * v6 )
        {
          if ( v6 - *(_DWORD *)(a2 + 20) - v72 > v6 >> 3 )
          {
LABEL_87:
            *(_DWORD *)(a2 + 16) = v72;
            if ( *v70 != -8 )
              --*(_DWORD *)(a2 + 20);
            *v70 = (__int64)v7;
            v70[1] = 0;
            goto LABEL_16;
          }
          v97 = v7;
          sub_1A63AC0(a2, v6);
          v80 = *(_DWORD *)(a2 + 24);
          if ( v80 )
          {
            v81 = v80 - 1;
            v82 = *(_QWORD *)(a2 + 8);
            v83 = 1;
            v84 = v81 & v10;
            v7 = v97;
            v85 = 0;
            v72 = *(_DWORD *)(a2 + 16) + 1;
            v70 = (__int64 *)(v82 + 16LL * v84);
            v86 = *v70;
            if ( v97 != (_QWORD *)*v70 )
            {
              while ( v86 != -8 )
              {
                if ( !v85 && v86 == -16 )
                  v85 = v70;
                v84 = v81 & (v83 + v84);
                v70 = (__int64 *)(v82 + 16LL * v84);
                v86 = *v70;
                if ( v97 == (_QWORD *)*v70 )
                  goto LABEL_87;
                ++v83;
              }
              if ( v85 )
                v70 = v85;
            }
            goto LABEL_87;
          }
LABEL_147:
          ++*(_DWORD *)(a2 + 16);
          BUG();
        }
LABEL_101:
        v96 = v7;
        sub_1A63AC0(a2, 2 * v6);
        v73 = *(_DWORD *)(a2 + 24);
        if ( v73 )
        {
          v7 = v96;
          v74 = v73 - 1;
          v75 = *(_QWORD *)(a2 + 8);
          v72 = *(_DWORD *)(a2 + 16) + 1;
          v76 = (v73 - 1) & (((unsigned int)v96 >> 9) ^ ((unsigned int)v96 >> 4));
          v70 = (__int64 *)(v75 + 16LL * v76);
          v77 = *v70;
          if ( v96 != (_QWORD *)*v70 )
          {
            v78 = 1;
            v79 = 0;
            while ( v77 != -8 )
            {
              if ( !v79 && v77 == -16 )
                v79 = v70;
              v76 = v74 & (v78 + v76);
              v70 = (__int64 *)(v75 + 16LL * v76);
              v77 = *v70;
              if ( v96 == (_QWORD *)*v70 )
                goto LABEL_87;
              ++v78;
            }
            if ( v79 )
              v70 = v79;
          }
          goto LABEL_87;
        }
        goto LABEL_147;
      }
      if ( v13 != (_QWORD *)-16LL || v70 )
        v12 = v70;
      v11 = v8 & (v69 + v11);
      v13 = (_QWORD *)v9[2 * v11];
      if ( v7 == v13 )
        break;
      ++v69;
      v70 = v12;
      v12 = &v9[2 * v11];
    }
    v12 = &v9[2 * v11];
    do
    {
LABEL_4:
      v14 = v12[1];
      if ( !v14 )
        goto LABEL_16;
      if ( a3 == *(_QWORD *)(v14 + 40) )
        goto LABEL_17;
      v15 = ((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4);
      v16 = v15 & v8;
      v12 = &v9[2 * (v15 & (unsigned int)v8)];
      v17 = *v12;
    }
    while ( *v12 == v14 );
    v18 = 1;
    v19 = 0;
    while ( v17 != -8 )
    {
      if ( !v19 && v17 == -16 )
        v19 = v12;
      v16 = v8 & (v18 + v16);
      v12 = &v9[2 * v16];
      v17 = *v12;
      if ( *v12 == v14 )
        goto LABEL_4;
      ++v18;
    }
    if ( !v19 )
      v19 = v12;
    v20 = *(_DWORD *)(a2 + 16);
    ++*(_QWORD *)a2;
    v21 = v20 + 1;
    if ( 4 * (v20 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a2 + 20) - v21 > v6 >> 3 )
        goto LABEL_13;
      v92 = v7;
      v44 = ((v8 | (v8 >> 1)) >> 2) | v8 | (v8 >> 1);
      v45 = ((((((v44 >> 4) | v44) >> 8) | (v44 >> 4) | v44) >> 16) | (((v44 >> 4) | v44) >> 8) | (v44 >> 4) | v44) + 1;
      if ( (unsigned int)v45 < 0x40 )
        LODWORD(v45) = 64;
      *(_DWORD *)(a2 + 24) = v45;
      v46 = (_QWORD *)sub_22077B0(16LL * (unsigned int)v45);
      v7 = v92;
      *(_QWORD *)(a2 + 8) = v46;
      if ( v9 )
      {
        v47 = *(unsigned int *)(a2 + 24);
        *(_QWORD *)(a2 + 16) = 0;
        v48 = &v9[2 * v6];
        for ( i = &v46[2 * v47]; i != v46; v46 += 2 )
        {
          if ( v46 )
            *v46 = -8;
        }
        v50 = v9;
        do
        {
          v51 = *v50;
          if ( *v50 != -16 && v51 != -8 )
          {
            v52 = *(_DWORD *)(a2 + 24);
            if ( !v52 )
            {
              MEMORY[0] = *v50;
              BUG();
            }
            v53 = v52 - 1;
            v54 = *(_QWORD *)(a2 + 8);
            v55 = v53 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
            v56 = (_QWORD *)(v54 + 16LL * v55);
            v57 = *v56;
            if ( *v56 != v51 )
            {
              v88 = 1;
              v95 = 0;
              while ( v57 != -8 )
              {
                if ( v57 == -16 )
                {
                  if ( v95 )
                    v56 = v95;
                  v95 = v56;
                }
                v55 = v53 & (v88 + v55);
                v56 = (_QWORD *)(v54 + 16LL * v55);
                v57 = *v56;
                if ( v51 == *v56 )
                  goto LABEL_53;
                ++v88;
              }
              if ( v95 )
                v56 = v95;
            }
LABEL_53:
            *v56 = v51;
            v56[1] = v50[1];
            ++*(_DWORD *)(a2 + 16);
          }
          v50 += 2;
        }
        while ( v48 != v50 );
        v93 = v7;
        j___libc_free_0(v9);
        v46 = *(_QWORD **)(a2 + 8);
        v58 = *(_DWORD *)(a2 + 24);
        v7 = v93;
        v21 = *(_DWORD *)(a2 + 16) + 1;
      }
      else
      {
        v66 = *(unsigned int *)(a2 + 24);
        *(_QWORD *)(a2 + 16) = 0;
        v58 = v66;
        v67 = &v46[2 * v66];
        if ( v46 != v67 )
        {
          v68 = v46;
          do
          {
            if ( v68 )
              *v68 = -8;
            v68 += 2;
          }
          while ( v67 != v68 );
        }
        v21 = 1;
      }
      if ( v58 )
      {
        v59 = v58 - 1;
        v60 = 1;
        v43 = 0;
        v61 = v59 & v15;
        v19 = &v46[2 * v61];
        v62 = *v19;
        if ( *v19 != v14 )
        {
          while ( v62 != -8 )
          {
            if ( !v43 && v62 == -16 )
              v43 = v19;
            v61 = v59 & (v60 + v61);
            v19 = &v46[2 * v61];
            v62 = *v19;
            if ( *v19 == v14 )
              goto LABEL_13;
            ++v60;
          }
          goto LABEL_38;
        }
        goto LABEL_13;
      }
LABEL_148:
      ++*(_DWORD *)(a2 + 16);
      BUG();
    }
    v90 = v7;
    v23 = (2 * v6 - 1) | ((unsigned __int64)(2 * v6 - 1) >> 1);
    v24 = (((v23 >> 2) | v23) >> 4) | (v23 >> 2) | v23;
    v25 = ((((v24 >> 8) | v24) >> 16) | (v24 >> 8) | v24) + 1;
    if ( (unsigned int)v25 < 0x40 )
      LODWORD(v25) = 64;
    *(_DWORD *)(a2 + 24) = v25;
    v26 = (_QWORD *)sub_22077B0(16LL * (unsigned int)v25);
    v7 = v90;
    *(_QWORD *)(a2 + 8) = v26;
    if ( v9 )
    {
      v27 = *(unsigned int *)(a2 + 24);
      *(_QWORD *)(a2 + 16) = 0;
      v28 = &v9[2 * v6];
      for ( j = &v26[2 * v27]; j != v26; v26 += 2 )
      {
        if ( v26 )
          *v26 = -8;
      }
      v30 = v9;
      do
      {
        v31 = *v30;
        if ( *v30 != -8 && v31 != -16 )
        {
          v32 = *(_DWORD *)(a2 + 24);
          if ( !v32 )
          {
            MEMORY[0] = *v30;
            BUG();
          }
          v33 = v32 - 1;
          v34 = *(_QWORD *)(a2 + 8);
          v35 = v33 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
          v36 = (_QWORD *)(v34 + 16LL * v35);
          v37 = *v36;
          if ( *v36 != v31 )
          {
            v87 = 1;
            v94 = 0;
            while ( v37 != -8 )
            {
              if ( v37 != -16 || v94 )
                v36 = v94;
              v35 = v33 & (v87 + v35);
              v37 = *(_QWORD *)(v34 + 16LL * v35);
              if ( v31 == v37 )
              {
                v36 = (_QWORD *)(v34 + 16LL * v35);
                goto LABEL_31;
              }
              v94 = v36;
              v36 = (_QWORD *)(v34 + 16LL * v35);
              ++v87;
            }
            if ( v94 )
              v36 = v94;
          }
LABEL_31:
          *v36 = v31;
          v36[1] = v30[1];
          ++*(_DWORD *)(a2 + 16);
        }
        v30 += 2;
      }
      while ( v28 != v30 );
      v91 = v7;
      j___libc_free_0(v9);
      v26 = *(_QWORD **)(a2 + 8);
      v38 = *(_DWORD *)(a2 + 24);
      v7 = v91;
      v21 = *(_DWORD *)(a2 + 16) + 1;
    }
    else
    {
      v63 = *(unsigned int *)(a2 + 24);
      *(_QWORD *)(a2 + 16) = 0;
      v38 = v63;
      v64 = &v26[2 * v63];
      if ( v26 != v64 )
      {
        v65 = v26;
        do
        {
          if ( v65 )
            *v65 = -8;
          v65 += 2;
        }
        while ( v64 != v65 );
      }
      v21 = 1;
    }
    if ( !v38 )
      goto LABEL_148;
    v39 = v38 - 1;
    v40 = v39 & v15;
    v19 = &v26[2 * v40];
    v41 = *v19;
    if ( *v19 != v14 )
    {
      v42 = 1;
      v43 = 0;
      while ( v41 != -8 )
      {
        if ( !v43 && v41 == -16 )
          v43 = v19;
        v40 = v39 & (v42 + v40);
        v19 = &v26[2 * v40];
        v41 = *v19;
        if ( *v19 == v14 )
          goto LABEL_13;
        ++v42;
      }
LABEL_38:
      if ( v43 )
        v19 = v43;
    }
LABEL_13:
    *(_DWORD *)(a2 + 16) = v21;
    if ( *v19 != -8 )
      --*(_DWORD *)(a2 + 20);
    *v19 = v14;
    v19[1] = 0;
LABEL_16:
    v89 = v7;
    v22 = sub_157EBA0(a3);
    v7 = v89;
    v14 = v22;
LABEL_17:
    sub_15F22F0(v7, v14);
    result = ++v99;
  }
  while ( v98 != v99 );
  return result;
}
