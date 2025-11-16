// Function: sub_9C0E00
// Address: 0x9c0e00
//
unsigned __int64 __fastcall sub_9C0E00(__int64 a1, __int64 *a2)
{
  unsigned __int64 result; // rax
  __int64 v5; // rcx
  __int64 v6; // rbx
  _QWORD *v7; // rsi
  __int64 v8; // rdi
  _QWORD *v9; // rdx
  unsigned __int64 *v10; // rbx
  unsigned __int64 *v11; // r13
  __int64 v12; // r8
  unsigned __int64 *v13; // rdi
  unsigned __int64 v14; // rcx
  unsigned int v15; // esi
  int v16; // eax
  int v17; // esi
  __int64 v18; // r9
  unsigned int v19; // eax
  unsigned __int64 *v20; // r10
  unsigned __int64 v21; // rcx
  int v22; // edx
  unsigned int v23; // esi
  __int64 v24; // r9
  _QWORD *v25; // r11
  int v26; // r13d
  unsigned int v27; // edx
  _QWORD *v28; // rdi
  __int64 v29; // r8
  int v30; // eax
  __int64 v31; // rbx
  int v32; // r11d
  int v33; // eax
  int v34; // eax
  int v35; // r11d
  __int64 v36; // r9
  int v37; // esi
  unsigned int v38; // eax
  unsigned __int64 *v39; // rcx
  unsigned __int64 v40; // rdi
  int v41; // r8d
  int v42; // r8d
  __int64 v43; // r10
  unsigned int v44; // edx
  __int64 v45; // rsi
  int v46; // edi
  _QWORD *v47; // rcx
  int v48; // r8d
  int v49; // r8d
  __int64 v50; // r10
  int v51; // edi
  unsigned int v52; // edx
  __int64 v53; // rsi
  int v54; // r11d
  unsigned __int64 *v55; // r8

  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    v23 = *(_DWORD *)(a1 + 24);
    if ( v23 )
    {
      v24 = *(_QWORD *)(a1 + 8);
      v25 = 0;
      v26 = 1;
      v27 = (v23 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v28 = (_QWORD *)(v24 + 8LL * v27);
      v29 = *v28;
      if ( *a2 == *v28 )
        return result;
      while ( v29 != -4096 )
      {
        if ( v25 || v29 != -8192 )
          v28 = v25;
        v27 = (v23 - 1) & (v26 + v27);
        v29 = *(_QWORD *)(v24 + 8LL * v27);
        if ( *a2 == v29 )
          return result;
        ++v26;
        v25 = v28;
        v28 = (_QWORD *)(v24 + 8LL * v27);
      }
      if ( !v25 )
        v25 = v28;
      v30 = result + 1;
      ++*(_QWORD *)a1;
      if ( 4 * v30 < 3 * v23 )
      {
        if ( v23 - *(_DWORD *)(a1 + 20) - v30 > v23 >> 3 )
        {
LABEL_35:
          *(_DWORD *)(a1 + 16) = v30;
          if ( *v25 != -4096 )
            --*(_DWORD *)(a1 + 20);
          v31 = *a2;
          *v25 = v31;
          result = *(unsigned int *)(a1 + 40);
          if ( result + 1 > *(unsigned int *)(a1 + 44) )
          {
            sub_C8D5F0(a1 + 32, a1 + 48, result + 1, 8);
            result = *(unsigned int *)(a1 + 40);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v31;
          ++*(_DWORD *)(a1 + 40);
          return result;
        }
        sub_9C0C30(a1, v23);
        v48 = *(_DWORD *)(a1 + 24);
        if ( v48 )
        {
          v49 = v48 - 1;
          v50 = *(_QWORD *)(a1 + 8);
          v47 = 0;
          v51 = 1;
          v52 = v49 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
          v25 = (_QWORD *)(v50 + 8LL * v52);
          v53 = *v25;
          v30 = *(_DWORD *)(a1 + 16) + 1;
          if ( *v25 == *a2 )
            goto LABEL_35;
          while ( v53 != -4096 )
          {
            if ( !v47 && v53 == -8192 )
              v47 = v25;
            v52 = v49 & (v51 + v52);
            v25 = (_QWORD *)(v50 + 8LL * v52);
            v53 = *v25;
            if ( *a2 == *v25 )
              goto LABEL_35;
            ++v51;
          }
          goto LABEL_69;
        }
        goto LABEL_111;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_9C0C30(a1, 2 * v23);
    v41 = *(_DWORD *)(a1 + 24);
    if ( v41 )
    {
      v42 = v41 - 1;
      v43 = *(_QWORD *)(a1 + 8);
      v44 = v42 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v25 = (_QWORD *)(v43 + 8LL * v44);
      v45 = *v25;
      v30 = *(_DWORD *)(a1 + 16) + 1;
      if ( *a2 == *v25 )
        goto LABEL_35;
      v46 = 1;
      v47 = 0;
      while ( v45 != -4096 )
      {
        if ( v45 == -8192 && !v47 )
          v47 = v25;
        v44 = v42 & (v46 + v44);
        v25 = (_QWORD *)(v43 + 8LL * v44);
        v45 = *v25;
        if ( *a2 == *v25 )
          goto LABEL_35;
        ++v46;
      }
LABEL_69:
      if ( v47 )
        v25 = v47;
      goto LABEL_35;
    }
LABEL_111:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  v5 = *(unsigned int *)(a1 + 40);
  result = *(_QWORD *)(a1 + 32);
  v6 = *a2;
  v7 = (_QWORD *)(result + 8 * v5);
  v8 = (8 * v5) >> 3;
  if ( !((8 * v5) >> 5) )
    goto LABEL_12;
  v9 = (_QWORD *)(result + 32 * ((8 * v5) >> 5));
  do
  {
    if ( *(_QWORD *)result == v6 )
      goto LABEL_9;
    if ( *(_QWORD *)(result + 8) == v6 )
    {
      result += 8LL;
      if ( v7 == (_QWORD *)result )
        goto LABEL_15;
      return result;
    }
    if ( *(_QWORD *)(result + 16) == v6 )
    {
      result += 16LL;
      if ( v7 == (_QWORD *)result )
        goto LABEL_15;
      return result;
    }
    if ( *(_QWORD *)(result + 24) == v6 )
    {
      result += 24LL;
      if ( v7 == (_QWORD *)result )
        goto LABEL_15;
      return result;
    }
    result += 32LL;
  }
  while ( v9 != (_QWORD *)result );
  v8 = (__int64)((__int64)v7 - result) >> 3;
LABEL_12:
  if ( v8 == 2 )
  {
LABEL_42:
    if ( *(_QWORD *)result != v6 )
    {
      result += 8LL;
      goto LABEL_44;
    }
LABEL_9:
    if ( v7 == (_QWORD *)result )
      goto LABEL_15;
    return result;
  }
  if ( v8 == 3 )
  {
    if ( *(_QWORD *)result == v6 )
      goto LABEL_9;
    result += 8LL;
    goto LABEL_42;
  }
  if ( v8 != 1 )
    goto LABEL_15;
LABEL_44:
  if ( *(_QWORD *)result == v6 )
    goto LABEL_9;
LABEL_15:
  if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, a1 + 48, v5 + 1, 8);
    v7 = (_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
  }
  *v7 = v6;
  result = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
  *(_DWORD *)(a1 + 40) = result;
  if ( (unsigned int)result > 4 )
  {
    v10 = *(unsigned __int64 **)(a1 + 32);
    v11 = &v10[result];
    while ( 1 )
    {
      v15 = *(_DWORD *)(a1 + 24);
      if ( !v15 )
        break;
      v12 = *(_QWORD *)(a1 + 8);
      result = (v15 - 1) & (((unsigned int)*v10 >> 9) ^ ((unsigned int)*v10 >> 4));
      v13 = (unsigned __int64 *)(v12 + 8 * result);
      v14 = *v13;
      if ( *v10 != *v13 )
      {
        v32 = 1;
        v20 = 0;
        while ( v14 != -4096 )
        {
          if ( v20 || v14 != -8192 )
            v13 = v20;
          result = (v15 - 1) & (v32 + (_DWORD)result);
          v14 = *(_QWORD *)(v12 + 8LL * (unsigned int)result);
          if ( *v10 == v14 )
            goto LABEL_20;
          ++v32;
          v20 = v13;
          v13 = (unsigned __int64 *)(v12 + 8LL * (unsigned int)result);
        }
        v33 = *(_DWORD *)(a1 + 16);
        if ( !v20 )
          v20 = v13;
        ++*(_QWORD *)a1;
        v22 = v33 + 1;
        if ( 4 * (v33 + 1) < 3 * v15 )
        {
          if ( v15 - *(_DWORD *)(a1 + 20) - v22 <= v15 >> 3 )
          {
            sub_9C0C30(a1, v15);
            v34 = *(_DWORD *)(a1 + 24);
            if ( !v34 )
            {
LABEL_110:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v35 = v34 - 1;
            v36 = *(_QWORD *)(a1 + 8);
            v37 = 1;
            v38 = (v34 - 1) & (((unsigned int)*v10 >> 9) ^ ((unsigned int)*v10 >> 4));
            v20 = (unsigned __int64 *)(v36 + 8LL * v38);
            v22 = *(_DWORD *)(a1 + 16) + 1;
            v39 = 0;
            v40 = *v20;
            if ( *v10 != *v20 )
            {
              while ( v40 != -4096 )
              {
                if ( !v39 && v40 == -8192 )
                  v39 = v20;
                v38 = v35 & (v37 + v38);
                v20 = (unsigned __int64 *)(v36 + 8LL * v38);
                v40 = *v20;
                if ( *v10 == *v20 )
                  goto LABEL_25;
                ++v37;
              }
              if ( v39 )
                v20 = v39;
            }
          }
          goto LABEL_25;
        }
LABEL_23:
        sub_9C0C30(a1, 2 * v15);
        v16 = *(_DWORD *)(a1 + 24);
        if ( !v16 )
          goto LABEL_110;
        v17 = v16 - 1;
        v18 = *(_QWORD *)(a1 + 8);
        v19 = (v16 - 1) & (((unsigned int)*v10 >> 9) ^ ((unsigned int)*v10 >> 4));
        v20 = (unsigned __int64 *)(v18 + 8LL * v19);
        v21 = *v20;
        v22 = *(_DWORD *)(a1 + 16) + 1;
        if ( *v10 != *v20 )
        {
          v54 = 1;
          v55 = 0;
          while ( v21 != -4096 )
          {
            if ( v21 == -8192 && !v55 )
              v55 = v20;
            v19 = v17 & (v54 + v19);
            v20 = (unsigned __int64 *)(v18 + 8LL * v19);
            v21 = *v20;
            if ( *v10 == *v20 )
              goto LABEL_25;
            ++v54;
          }
          if ( v55 )
            v20 = v55;
        }
LABEL_25:
        *(_DWORD *)(a1 + 16) = v22;
        if ( *v20 != -4096 )
          --*(_DWORD *)(a1 + 20);
        result = *v10;
        *v20 = *v10;
      }
LABEL_20:
      if ( v11 == ++v10 )
        return result;
    }
    ++*(_QWORD *)a1;
    goto LABEL_23;
  }
  return result;
}
