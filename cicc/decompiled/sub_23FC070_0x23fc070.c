// Function: sub_23FC070
// Address: 0x23fc070
//
__int64 __fastcall sub_23FC070(__int64 *a1, __int64 a2)
{
  __int64 v3; // r11
  __int64 *v4; // rbx
  __int64 *v5; // r10
  unsigned __int64 v6; // rax
  unsigned int v7; // edi
  _QWORD *v8; // rsi
  __int64 v9; // rcx
  unsigned int v10; // edx
  __int64 v11; // r13
  __int64 v12; // r15
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rax
  _QWORD *v15; // rax
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 *v18; // r11
  _QWORD *i; // rdx
  __int64 *v20; // rax
  __int64 v21; // r14
  int v22; // ecx
  int v23; // ecx
  __int64 v24; // r8
  unsigned int v25; // edx
  _QWORD *v26; // rsi
  __int64 v27; // rdi
  int v28; // ecx
  int v29; // esi
  int v30; // ecx
  unsigned int v31; // edx
  _QWORD *v32; // r8
  __int64 v33; // rdi
  __int64 *v34; // rbx
  __int64 result; // rax
  __int64 *k; // r13
  __int64 v37; // rdi
  int v38; // r9d
  int v39; // esi
  unsigned __int64 v40; // rax
  unsigned __int64 v41; // rax
  __int64 v42; // rax
  _QWORD *v43; // rax
  __int64 v44; // rdx
  _QWORD *j; // rdx
  __int64 *v46; // rax
  __int64 v47; // rsi
  int v48; // ecx
  int v49; // ecx
  __int64 v50; // r9
  unsigned int v51; // edx
  _QWORD *v52; // rdi
  __int64 v53; // r8
  int v54; // ecx
  int v55; // ecx
  int v56; // r9d
  _QWORD *v57; // rdi
  unsigned int v58; // r14d
  __int64 v59; // rdx
  _QWORD *v60; // rsi
  _QWORD *v61; // rdx
  _QWORD *v62; // rsi
  _QWORD *v63; // rdx
  int v64; // r14d
  _QWORD *v65; // r9
  int v66; // [rsp+4h] [rbp-5Ch]
  __int64 *v68; // [rsp+10h] [rbp-50h]
  __int64 *v69; // [rsp+10h] [rbp-50h]
  __int64 *v70; // [rsp+10h] [rbp-50h]
  int v71; // [rsp+10h] [rbp-50h]
  _QWORD *v72; // [rsp+10h] [rbp-50h]
  unsigned int v73; // [rsp+18h] [rbp-48h]
  __int64 *v74; // [rsp+18h] [rbp-48h]
  unsigned int v75; // [rsp+18h] [rbp-48h]
  __int64 v76; // [rsp+18h] [rbp-48h]
  _QWORD *v77; // [rsp+18h] [rbp-48h]
  __int64 v78; // [rsp+20h] [rbp-40h]
  __int64 v79; // [rsp+28h] [rbp-38h]

  v3 = *a1;
  v78 = *a1 + 96LL * *((unsigned int *)a1 + 2);
  if ( *a1 == v78 )
    goto LABEL_30;
  do
  {
    v4 = *(__int64 **)(v3 + 16);
    v5 = &v4[*(unsigned int *)(v3 + 24)];
    if ( v5 == v4 )
      goto LABEL_29;
    v79 = v3;
    do
    {
      while ( 1 )
      {
        v10 = *(_DWORD *)(a2 + 24);
        v11 = *v4;
        v12 = *(_QWORD *)(a2 + 8);
        if ( !v10 )
        {
          ++*(_QWORD *)a2;
LABEL_8:
          v68 = v5;
          v73 = v10;
          v13 = ((((((((2 * v10 - 1) | ((unsigned __int64)(2 * v10 - 1) >> 1)) >> 2)
                   | (2 * v10 - 1)
                   | ((unsigned __int64)(2 * v10 - 1) >> 1)) >> 4)
                 | (((2 * v10 - 1) | ((unsigned __int64)(2 * v10 - 1) >> 1)) >> 2)
                 | (2 * v10 - 1)
                 | ((unsigned __int64)(2 * v10 - 1) >> 1)) >> 8)
               | (((((2 * v10 - 1) | ((unsigned __int64)(2 * v10 - 1) >> 1)) >> 2)
                 | (2 * v10 - 1)
                 | ((unsigned __int64)(2 * v10 - 1) >> 1)) >> 4)
               | (((2 * v10 - 1) | ((unsigned __int64)(2 * v10 - 1) >> 1)) >> 2)
               | (2 * v10 - 1)
               | ((unsigned __int64)(2 * v10 - 1) >> 1)) >> 16;
          v14 = (v13
               | (((((((2 * v10 - 1) | ((unsigned __int64)(2 * v10 - 1) >> 1)) >> 2)
                   | (2 * v10 - 1)
                   | ((unsigned __int64)(2 * v10 - 1) >> 1)) >> 4)
                 | (((2 * v10 - 1) | ((unsigned __int64)(2 * v10 - 1) >> 1)) >> 2)
                 | (2 * v10 - 1)
                 | ((unsigned __int64)(2 * v10 - 1) >> 1)) >> 8)
               | (((((2 * v10 - 1) | ((unsigned __int64)(2 * v10 - 1) >> 1)) >> 2)
                 | (2 * v10 - 1)
                 | ((unsigned __int64)(2 * v10 - 1) >> 1)) >> 4)
               | (((2 * v10 - 1) | ((unsigned __int64)(2 * v10 - 1) >> 1)) >> 2)
               | (2 * v10 - 1)
               | ((unsigned __int64)(2 * v10 - 1) >> 1))
              + 1;
          if ( (unsigned int)v14 < 0x40 )
            LODWORD(v14) = 64;
          *(_DWORD *)(a2 + 24) = v14;
          v15 = (_QWORD *)sub_C7D670(8LL * (unsigned int)v14, 8);
          v5 = v68;
          *(_QWORD *)(a2 + 8) = v15;
          if ( v12 )
          {
            v16 = 8LL * v73;
            v17 = *(unsigned int *)(a2 + 24);
            *(_QWORD *)(a2 + 16) = 0;
            v18 = (__int64 *)(v12 + v16);
            for ( i = &v15[v17]; i != v15; ++v15 )
            {
              if ( v15 )
                *v15 = -4096;
            }
            v20 = (__int64 *)v12;
            if ( (__int64 *)v12 != v18 )
            {
              do
              {
                v21 = *v20;
                if ( *v20 != -8192 && v21 != -4096 )
                {
                  v22 = *(_DWORD *)(a2 + 24);
                  if ( !v22 )
                  {
                    MEMORY[0] = *v20;
                    BUG();
                  }
                  v23 = v22 - 1;
                  v24 = *(_QWORD *)(a2 + 8);
                  v25 = v23 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
                  v26 = (_QWORD *)(v24 + 8LL * v25);
                  v27 = *v26;
                  if ( v21 != *v26 )
                  {
                    v71 = 1;
                    v77 = 0;
                    while ( v27 != -4096 )
                    {
                      if ( !v77 )
                      {
                        if ( v27 != -8192 )
                          v26 = 0;
                        v77 = v26;
                      }
                      v25 = v23 & (v71 + v25);
                      v26 = (_QWORD *)(v24 + 8LL * v25);
                      v27 = *v26;
                      if ( v21 == *v26 )
                        goto LABEL_20;
                      ++v71;
                    }
                    if ( v77 )
                      v26 = v77;
                  }
LABEL_20:
                  *v26 = v21;
                  ++*(_DWORD *)(a2 + 16);
                }
                ++v20;
              }
              while ( v18 != v20 );
            }
            v74 = v5;
            sub_C7D6A0(v12, v16, 8);
            v15 = *(_QWORD **)(a2 + 8);
            v28 = *(_DWORD *)(a2 + 24);
            v5 = v74;
            v29 = *(_DWORD *)(a2 + 16) + 1;
          }
          else
          {
            *(_QWORD *)(a2 + 16) = 0;
            v60 = &v15[*(unsigned int *)(a2 + 24)];
            v28 = *(_DWORD *)(a2 + 24);
            if ( v15 != v60 )
            {
              v61 = v15;
              do
              {
                if ( v61 )
                  *v61 = -4096;
                ++v61;
              }
              while ( v60 != v61 );
            }
            v29 = 1;
          }
          if ( !v28 )
            goto LABEL_116;
          v30 = v28 - 1;
          v31 = v30 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v32 = &v15[v31];
          v33 = *v32;
          if ( v11 != *v32 )
          {
            v64 = 1;
            v65 = 0;
            while ( v33 != -4096 )
            {
              if ( !v65 && v33 == -8192 )
                v65 = v32;
              v31 = v30 & (v64 + v31);
              v32 = &v15[v31];
              v33 = *v32;
              if ( v11 == *v32 )
                goto LABEL_25;
              ++v64;
            }
            if ( v65 )
              v32 = v65;
          }
          goto LABEL_25;
        }
        v6 = v10 - 1;
        v7 = v6 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v8 = (_QWORD *)(v12 + 8LL * v7);
        v9 = *v8;
        if ( v11 != *v8 )
          break;
LABEL_5:
        if ( v5 == ++v4 )
          goto LABEL_28;
      }
      v38 = 1;
      v32 = 0;
      while ( v9 != -4096 )
      {
        if ( v32 || v9 != -8192 )
          v8 = v32;
        v7 = v6 & (v38 + v7);
        v9 = *(_QWORD *)(v12 + 8LL * v7);
        if ( v11 == v9 )
          goto LABEL_5;
        ++v38;
        v32 = v8;
        v8 = (_QWORD *)(v12 + 8LL * v7);
      }
      if ( !v32 )
        v32 = v8;
      v39 = *(_DWORD *)(a2 + 16);
      ++*(_QWORD *)a2;
      v29 = v39 + 1;
      if ( 4 * v29 >= 3 * v10 )
        goto LABEL_8;
      if ( v10 - *(_DWORD *)(a2 + 20) - v29 <= v10 >> 3 )
      {
        v69 = v5;
        v75 = v10;
        v40 = (((v6 >> 1) | v6) >> 2) | (v6 >> 1) | v6;
        v41 = (((v40 >> 4) | v40) >> 8) | (v40 >> 4) | v40;
        v42 = ((v41 >> 16) | v41) + 1;
        if ( (unsigned int)v42 < 0x40 )
          LODWORD(v42) = 64;
        *(_DWORD *)(a2 + 24) = v42;
        v43 = (_QWORD *)sub_C7D670(8LL * (unsigned int)v42, 8);
        v5 = v69;
        *(_QWORD *)(a2 + 8) = v43;
        if ( v12 )
        {
          v44 = *(unsigned int *)(a2 + 24);
          *(_QWORD *)(a2 + 16) = 0;
          v76 = 8LL * v75;
          for ( j = &v43[v44]; j != v43; ++v43 )
          {
            if ( v43 )
              *v43 = -4096;
          }
          v46 = (__int64 *)v12;
          do
          {
            v47 = *v46;
            if ( *v46 != -8192 && v47 != -4096 )
            {
              v48 = *(_DWORD *)(a2 + 24);
              if ( !v48 )
              {
                MEMORY[0] = *v46;
                BUG();
              }
              v49 = v48 - 1;
              v50 = *(_QWORD *)(a2 + 8);
              v51 = v49 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
              v52 = (_QWORD *)(v50 + 8LL * v51);
              v53 = *v52;
              if ( *v52 != v47 )
              {
                v66 = 1;
                v72 = 0;
                while ( v53 != -4096 )
                {
                  if ( v53 == -8192 )
                  {
                    if ( v72 )
                      v52 = v72;
                    v72 = v52;
                  }
                  v51 = v49 & (v66 + v51);
                  v52 = (_QWORD *)(v50 + 8LL * v51);
                  v53 = *v52;
                  if ( v47 == *v52 )
                    goto LABEL_51;
                  ++v66;
                }
                if ( v72 )
                  v52 = v72;
              }
LABEL_51:
              *v52 = v47;
              ++*(_DWORD *)(a2 + 16);
            }
            ++v46;
          }
          while ( (__int64 *)(v12 + v76) != v46 );
          v70 = v5;
          sub_C7D6A0(v12, v76, 8);
          v43 = *(_QWORD **)(a2 + 8);
          v54 = *(_DWORD *)(a2 + 24);
          v5 = v70;
          v29 = *(_DWORD *)(a2 + 16) + 1;
        }
        else
        {
          *(_QWORD *)(a2 + 16) = 0;
          v62 = &v43[*(unsigned int *)(a2 + 24)];
          v54 = *(_DWORD *)(a2 + 24);
          if ( v43 != v62 )
          {
            v63 = v43;
            do
            {
              if ( v63 )
                *v63 = -4096;
              ++v63;
            }
            while ( v62 != v63 );
          }
          v29 = 1;
        }
        if ( !v54 )
        {
LABEL_116:
          ++*(_DWORD *)(a2 + 16);
          BUG();
        }
        v55 = v54 - 1;
        v56 = 1;
        v57 = 0;
        v58 = v55 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v32 = &v43[v58];
        v59 = *v32;
        if ( v11 != *v32 )
        {
          while ( v59 != -4096 )
          {
            if ( v59 == -8192 && !v57 )
              v57 = v32;
            v58 = v55 & (v56 + v58);
            v32 = &v43[v58];
            v59 = *v32;
            if ( v11 == *v32 )
              goto LABEL_25;
            ++v56;
          }
          if ( v57 )
            v32 = v57;
        }
      }
LABEL_25:
      *(_DWORD *)(a2 + 16) = v29;
      if ( *v32 != -4096 )
        --*(_DWORD *)(a2 + 20);
      ++v4;
      *v32 = v11;
    }
    while ( v5 != v4 );
LABEL_28:
    v3 = v79;
LABEL_29:
    v3 += 96;
  }
  while ( v78 != v3 );
LABEL_30:
  v34 = (__int64 *)a1[98];
  result = *((unsigned int *)a1 + 198);
  for ( k = &v34[result]; k != v34; result = sub_23FC070(v37, a2) )
    v37 = *v34++;
  return result;
}
