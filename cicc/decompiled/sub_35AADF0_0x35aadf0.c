// Function: sub_35AADF0
// Address: 0x35aadf0
//
__int64 __fastcall sub_35AADF0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // eax
  __int64 v9; // rcx
  _QWORD *v10; // rax
  __int64 v11; // rbx
  __int64 *v12; // rsi
  __int64 v13; // rdi
  _QWORD *v14; // rdx
  __int64 v16; // rax
  unsigned int v17; // esi
  __int64 v18; // r9
  __int64 *v19; // r11
  int v20; // r13d
  unsigned int v21; // edx
  __int64 *v22; // rdi
  __int64 v23; // r8
  _QWORD *v24; // rbx
  _QWORD *v25; // r13
  __int64 v26; // r8
  unsigned int v27; // eax
  _QWORD *v28; // rdi
  __int64 v29; // rcx
  unsigned int v30; // esi
  int v31; // eax
  int v32; // esi
  __int64 v33; // r9
  unsigned int v34; // eax
  _QWORD *v35; // r10
  __int64 v36; // rcx
  int v37; // edx
  int v38; // eax
  __int64 v39; // rbx
  __int64 v40; // rax
  int v41; // r11d
  int v42; // eax
  int v43; // eax
  int v44; // r11d
  __int64 v45; // r9
  int v46; // esi
  unsigned int v47; // eax
  _QWORD *v48; // rcx
  __int64 v49; // rdi
  int v50; // r8d
  __int64 v51; // r10
  unsigned int v52; // edx
  __int64 v53; // rsi
  int v54; // edi
  __int64 *v55; // rcx
  int v56; // r8d
  __int64 v57; // r10
  int v58; // edi
  unsigned int v59; // edx
  __int64 v60; // rsi
  int v61; // r11d
  _QWORD *v62; // r8

  v7 = *(_DWORD *)(a1 + 16);
  if ( !v7 )
  {
    v9 = *(unsigned int *)(a1 + 40);
    v10 = *(_QWORD **)(a1 + 32);
    v11 = *a2;
    v12 = &v10[v9];
    v13 = (8 * v9) >> 3;
    if ( !((8 * v9) >> 5) )
      goto LABEL_12;
    v14 = &v10[4 * ((8 * v9) >> 5)];
    while ( 1 )
    {
      if ( *v10 == v11 )
        goto LABEL_9;
      if ( v10[1] == v11 )
        break;
      if ( v10[2] == v11 )
      {
        a5 = 0;
        if ( v12 == v10 + 2 )
          goto LABEL_15;
        return 0;
      }
      if ( v10[3] == v11 )
      {
        a5 = 0;
        if ( v12 == v10 + 3 )
          goto LABEL_15;
        return 0;
      }
      v10 += 4;
      if ( v14 == v10 )
      {
        v13 = v12 - v10;
LABEL_12:
        if ( v13 != 2 )
        {
          if ( v13 != 3 )
          {
            if ( v13 != 1 )
              goto LABEL_15;
LABEL_36:
            if ( *v10 == v11 )
              goto LABEL_9;
LABEL_15:
            if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
            {
              sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v9 + 1, 8u, a5, a6);
              v12 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
            }
            *v12 = v11;
            v16 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
            *(_DWORD *)(a1 + 40) = v16;
            if ( (unsigned int)v16 <= 0x10 )
              return 1;
            v24 = *(_QWORD **)(a1 + 32);
            v25 = &v24[v16];
            while ( 2 )
            {
              v30 = *(_DWORD *)(a1 + 24);
              if ( !v30 )
              {
                ++*(_QWORD *)a1;
                goto LABEL_27;
              }
              v26 = *(_QWORD *)(a1 + 8);
              v27 = (v30 - 1) & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
              v28 = (_QWORD *)(v26 + 8LL * v27);
              v29 = *v28;
              if ( *v24 != *v28 )
              {
                v41 = 1;
                v35 = 0;
                while ( v29 != -4096 )
                {
                  if ( v35 || v29 != -8192 )
                    v28 = v35;
                  v27 = (v30 - 1) & (v41 + v27);
                  v29 = *(_QWORD *)(v26 + 8LL * v27);
                  if ( *v24 == v29 )
                    goto LABEL_24;
                  ++v41;
                  v35 = v28;
                  v28 = (_QWORD *)(v26 + 8LL * v27);
                }
                v42 = *(_DWORD *)(a1 + 16);
                if ( !v35 )
                  v35 = v28;
                ++*(_QWORD *)a1;
                v37 = v42 + 1;
                if ( 4 * (v42 + 1) >= 3 * v30 )
                {
LABEL_27:
                  sub_2E36C70(a1, 2 * v30);
                  v31 = *(_DWORD *)(a1 + 24);
                  if ( !v31 )
                    goto LABEL_112;
                  v32 = v31 - 1;
                  v33 = *(_QWORD *)(a1 + 8);
                  v34 = (v31 - 1) & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
                  v35 = (_QWORD *)(v33 + 8LL * v34);
                  v36 = *v35;
                  v37 = *(_DWORD *)(a1 + 16) + 1;
                  if ( *v24 != *v35 )
                  {
                    v61 = 1;
                    v62 = 0;
                    while ( v36 != -4096 )
                    {
                      if ( v36 == -8192 && !v62 )
                        v62 = v35;
                      v34 = v32 & (v61 + v34);
                      v35 = (_QWORD *)(v33 + 8LL * v34);
                      v36 = *v35;
                      if ( *v24 == *v35 )
                        goto LABEL_29;
                      ++v61;
                    }
                    if ( v62 )
                      v35 = v62;
                  }
                }
                else if ( v30 - *(_DWORD *)(a1 + 20) - v37 <= v30 >> 3 )
                {
                  sub_2E36C70(a1, v30);
                  v43 = *(_DWORD *)(a1 + 24);
                  if ( !v43 )
                    goto LABEL_112;
                  v44 = v43 - 1;
                  v45 = *(_QWORD *)(a1 + 8);
                  v46 = 1;
                  v47 = (v43 - 1) & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
                  v35 = (_QWORD *)(v45 + 8LL * v47);
                  v37 = *(_DWORD *)(a1 + 16) + 1;
                  v48 = 0;
                  v49 = *v35;
                  if ( *v24 != *v35 )
                  {
                    while ( v49 != -4096 )
                    {
                      if ( !v48 && v49 == -8192 )
                        v48 = v35;
                      v47 = v44 & (v46 + v47);
                      v35 = (_QWORD *)(v45 + 8LL * v47);
                      v49 = *v35;
                      if ( *v24 == *v35 )
                        goto LABEL_29;
                      ++v46;
                    }
                    if ( v48 )
                      v35 = v48;
                  }
                }
LABEL_29:
                *(_DWORD *)(a1 + 16) = v37;
                if ( *v35 != -4096 )
                  --*(_DWORD *)(a1 + 20);
                *v35 = *v24;
              }
LABEL_24:
              if ( v25 == ++v24 )
                return 1;
              continue;
            }
          }
          if ( *v10 == v11 )
            goto LABEL_9;
          ++v10;
        }
        if ( *v10 != v11 )
        {
          ++v10;
          goto LABEL_36;
        }
LABEL_9:
        a5 = 0;
        if ( v12 == v10 )
          goto LABEL_15;
        return 0;
      }
    }
    a5 = 0;
    if ( v12 == v10 + 1 )
      goto LABEL_15;
    return 0;
  }
  v17 = *(_DWORD *)(a1 + 24);
  if ( !v17 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_71;
  }
  v18 = *(_QWORD *)(a1 + 8);
  v19 = 0;
  v20 = 1;
  v21 = (v17 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v22 = (__int64 *)(v18 + 8LL * v21);
  v23 = *v22;
  if ( *a2 != *v22 )
  {
    while ( v23 != -4096 )
    {
      if ( v19 || v23 != -8192 )
        v22 = v19;
      v21 = (v17 - 1) & (v20 + v21);
      v23 = *(_QWORD *)(v18 + 8LL * v21);
      if ( *a2 == v23 )
        return 0;
      ++v20;
      v19 = v22;
      v22 = (__int64 *)(v18 + 8LL * v21);
    }
    if ( !v19 )
      v19 = v22;
    v38 = v7 + 1;
    ++*(_QWORD *)a1;
    if ( 4 * v38 < 3 * v17 )
    {
      if ( v17 - *(_DWORD *)(a1 + 20) - v38 > v17 >> 3 )
      {
LABEL_53:
        *(_DWORD *)(a1 + 16) = v38;
        if ( *v19 != -4096 )
          --*(_DWORD *)(a1 + 20);
        v39 = *a2;
        *v19 = v39;
        v40 = *(unsigned int *)(a1 + 40);
        if ( v40 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
        {
          sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v40 + 1, 8u, v23, v18);
          v40 = *(unsigned int *)(a1 + 40);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v40) = v39;
        ++*(_DWORD *)(a1 + 40);
        return 1;
      }
      sub_2E36C70(a1, v17);
      v56 = *(_DWORD *)(a1 + 24);
      if ( v56 )
      {
        v18 = *a2;
        v23 = (unsigned int)(v56 - 1);
        v57 = *(_QWORD *)(a1 + 8);
        v55 = 0;
        v58 = 1;
        v59 = v23 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
        v19 = (__int64 *)(v57 + 8LL * v59);
        v60 = *v19;
        v38 = *(_DWORD *)(a1 + 16) + 1;
        if ( *a2 == *v19 )
          goto LABEL_53;
        while ( v60 != -4096 )
        {
          if ( !v55 && v60 == -8192 )
            v55 = v19;
          v59 = v23 & (v58 + v59);
          v19 = (__int64 *)(v57 + 8LL * v59);
          v60 = *v19;
          if ( v18 == *v19 )
            goto LABEL_53;
          ++v58;
        }
        goto LABEL_75;
      }
      goto LABEL_112;
    }
LABEL_71:
    sub_2E36C70(a1, 2 * v17);
    v50 = *(_DWORD *)(a1 + 24);
    if ( v50 )
    {
      v18 = *a2;
      v23 = (unsigned int)(v50 - 1);
      v51 = *(_QWORD *)(a1 + 8);
      v52 = v23 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v19 = (__int64 *)(v51 + 8LL * v52);
      v53 = *v19;
      v38 = *(_DWORD *)(a1 + 16) + 1;
      if ( *a2 == *v19 )
        goto LABEL_53;
      v54 = 1;
      v55 = 0;
      while ( v53 != -4096 )
      {
        if ( v53 == -8192 && !v55 )
          v55 = v19;
        v52 = v23 & (v54 + v52);
        v19 = (__int64 *)(v51 + 8LL * v52);
        v53 = *v19;
        if ( v18 == *v19 )
          goto LABEL_53;
        ++v54;
      }
LABEL_75:
      if ( v55 )
        v19 = v55;
      goto LABEL_53;
    }
LABEL_112:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  return 0;
}
