// Function: sub_3762BC0
// Address: 0x3762bc0
//
__int64 __fastcall sub_3762BC0(__int64 a1, __int64 *a2)
{
  int v3; // eax
  _QWORD *v5; // rdi
  __int64 *v7; // rsi
  __int64 v8; // r8
  __int64 *v9; // r9
  __int64 result; // rax
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned int v13; // esi
  __int64 v14; // r9
  _QWORD *v15; // r11
  int v16; // r13d
  unsigned int v17; // edx
  _QWORD *v18; // r8
  __int64 v19; // rdi
  _QWORD *v20; // r12
  _QWORD *v21; // r13
  __int64 v22; // r8
  unsigned int v23; // eax
  _QWORD *v24; // rdi
  __int64 v25; // rcx
  unsigned int v26; // esi
  int v27; // eax
  int v28; // ecx
  __int64 v29; // r8
  unsigned int v30; // eax
  _QWORD *v31; // r10
  __int64 v32; // rdi
  int v33; // edx
  int v34; // eax
  __int64 v35; // r12
  __int64 v36; // rax
  int v37; // r11d
  int v38; // eax
  int v39; // eax
  int v40; // ecx
  __int64 v41; // r8
  _QWORD *v42; // r9
  int v43; // r11d
  unsigned int v44; // eax
  __int64 v45; // rdi
  int v46; // eax
  int v47; // ecx
  unsigned int v48; // edx
  __int64 v49; // rdi
  int v50; // r10d
  int v51; // eax
  int v52; // ecx
  int v53; // r10d
  unsigned int v54; // edx
  __int64 v55; // rdi
  int v56; // r11d

  v3 = *(_DWORD *)(a1 + 16);
  if ( !v3 )
  {
    v5 = *(_QWORD **)(a1 + 32);
    v7 = &v5[*(unsigned int *)(a1 + 40)];
    v9 = sub_3759330(v5, (__int64)v7, a2);
    result = 0;
    if ( v7 != v9 )
      return result;
    v11 = *a2;
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v8 + 1, 8u, v8, (__int64)v9);
      v7 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
    }
    *v7 = v11;
    v12 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
    *(_DWORD *)(a1 + 40) = v12;
    if ( (unsigned int)v12 <= 0x10 )
      return 1;
    v20 = *(_QWORD **)(a1 + 32);
    v21 = &v20[v12];
    while ( 1 )
    {
      v26 = *(_DWORD *)(a1 + 24);
      if ( !v26 )
        break;
      v22 = *(_QWORD *)(a1 + 8);
      v23 = (v26 - 1) & (((unsigned int)*v20 >> 9) ^ ((unsigned int)*v20 >> 4));
      v24 = (_QWORD *)(v22 + 8LL * v23);
      v25 = *v24;
      if ( *v20 != *v24 )
      {
        v37 = 1;
        v31 = 0;
        while ( v25 != -4096 )
        {
          if ( v31 || v25 != -8192 )
            v24 = v31;
          v23 = (v26 - 1) & (v37 + v23);
          v25 = *(_QWORD *)(v22 + 8LL * v23);
          if ( *v20 == v25 )
            goto LABEL_12;
          ++v37;
          v31 = v24;
          v24 = (_QWORD *)(v22 + 8LL * v23);
        }
        v38 = *(_DWORD *)(a1 + 16);
        if ( !v31 )
          v31 = v24;
        ++*(_QWORD *)a1;
        v33 = v38 + 1;
        if ( 4 * (v38 + 1) < 3 * v26 )
        {
          if ( v26 - *(_DWORD *)(a1 + 20) - v33 <= v26 >> 3 )
          {
            sub_32B3220(a1, v26);
            v39 = *(_DWORD *)(a1 + 24);
            if ( !v39 )
              goto LABEL_85;
            v40 = v39 - 1;
            v41 = *(_QWORD *)(a1 + 8);
            v42 = 0;
            v43 = 1;
            v44 = (v39 - 1) & (((unsigned int)*v20 >> 9) ^ ((unsigned int)*v20 >> 4));
            v31 = (_QWORD *)(v41 + 8LL * v44);
            v45 = *v31;
            v33 = *(_DWORD *)(a1 + 16) + 1;
            if ( *v20 != *v31 )
            {
              while ( v45 != -4096 )
              {
                if ( !v42 && v45 == -8192 )
                  v42 = v31;
                v44 = v40 & (v43 + v44);
                v31 = (_QWORD *)(v41 + 8LL * v44);
                v45 = *v31;
                if ( *v20 == *v31 )
                  goto LABEL_17;
                ++v43;
              }
LABEL_43:
              if ( v42 )
                v31 = v42;
            }
          }
LABEL_17:
          *(_DWORD *)(a1 + 16) = v33;
          if ( *v31 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v31 = *v20;
          goto LABEL_12;
        }
LABEL_15:
        sub_32B3220(a1, 2 * v26);
        v27 = *(_DWORD *)(a1 + 24);
        if ( !v27 )
          goto LABEL_85;
        v28 = v27 - 1;
        v29 = *(_QWORD *)(a1 + 8);
        v30 = (v27 - 1) & (((unsigned int)*v20 >> 9) ^ ((unsigned int)*v20 >> 4));
        v31 = (_QWORD *)(v29 + 8LL * v30);
        v32 = *v31;
        v33 = *(_DWORD *)(a1 + 16) + 1;
        if ( *v20 != *v31 )
        {
          v56 = 1;
          v42 = 0;
          while ( v32 != -4096 )
          {
            if ( v32 == -8192 && !v42 )
              v42 = v31;
            v30 = v28 & (v56 + v30);
            v31 = (_QWORD *)(v29 + 8LL * v30);
            v32 = *v31;
            if ( *v20 == *v31 )
              goto LABEL_17;
            ++v56;
          }
          goto LABEL_43;
        }
        goto LABEL_17;
      }
LABEL_12:
      if ( v21 == ++v20 )
        return 1;
    }
    ++*(_QWORD *)a1;
    goto LABEL_15;
  }
  v13 = *(_DWORD *)(a1 + 24);
  if ( !v13 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_47;
  }
  v14 = *(_QWORD *)(a1 + 8);
  v15 = 0;
  v16 = 1;
  v17 = (v13 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v18 = (_QWORD *)(v14 + 8LL * v17);
  v19 = *v18;
  if ( *v18 == *a2 )
    return 0;
  while ( v19 != -4096 )
  {
    if ( v15 || v19 != -8192 )
      v18 = v15;
    v17 = (v13 - 1) & (v16 + v17);
    v19 = *(_QWORD *)(v14 + 8LL * v17);
    if ( *a2 == v19 )
      return 0;
    ++v16;
    v15 = v18;
    v18 = (_QWORD *)(v14 + 8LL * v17);
  }
  if ( !v15 )
    v15 = v18;
  v34 = v3 + 1;
  ++*(_QWORD *)a1;
  if ( 4 * v34 >= 3 * v13 )
  {
LABEL_47:
    sub_32B3220(a1, 2 * v13);
    v46 = *(_DWORD *)(a1 + 24);
    if ( v46 )
    {
      v47 = v46 - 1;
      v18 = *(_QWORD **)(a1 + 8);
      v48 = (v46 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v15 = &v18[v48];
      v49 = *v15;
      v34 = *(_DWORD *)(a1 + 16) + 1;
      if ( *a2 == *v15 )
        goto LABEL_29;
      v50 = 1;
      v14 = 0;
      while ( v49 != -4096 )
      {
        if ( v49 == -8192 && !v14 )
          v14 = (__int64)v15;
        v48 = v47 & (v50 + v48);
        v15 = &v18[v48];
        v49 = *v15;
        if ( *a2 == *v15 )
          goto LABEL_29;
        ++v50;
      }
LABEL_51:
      if ( v14 )
        v15 = (_QWORD *)v14;
      goto LABEL_29;
    }
LABEL_85:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v13 - *(_DWORD *)(a1 + 20) - v34 <= v13 >> 3 )
  {
    sub_32B3220(a1, v13);
    v51 = *(_DWORD *)(a1 + 24);
    if ( v51 )
    {
      v52 = v51 - 1;
      v18 = *(_QWORD **)(a1 + 8);
      v14 = 0;
      v53 = 1;
      v54 = (v51 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v15 = &v18[v54];
      v55 = *v15;
      v34 = *(_DWORD *)(a1 + 16) + 1;
      if ( *a2 == *v15 )
        goto LABEL_29;
      while ( v55 != -4096 )
      {
        if ( !v14 && v55 == -8192 )
          v14 = (__int64)v15;
        v54 = v52 & (v53 + v54);
        v15 = &v18[v54];
        v55 = *v15;
        if ( *a2 == *v15 )
          goto LABEL_29;
        ++v53;
      }
      goto LABEL_51;
    }
    goto LABEL_85;
  }
LABEL_29:
  *(_DWORD *)(a1 + 16) = v34;
  if ( *v15 != -4096 )
    --*(_DWORD *)(a1 + 20);
  v35 = *a2;
  *v15 = v35;
  v36 = *(unsigned int *)(a1 + 40);
  if ( v36 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v36 + 1, 8u, (__int64)v18, v14);
    v36 = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v36) = v35;
  ++*(_DWORD *)(a1 + 40);
  return 1;
}
