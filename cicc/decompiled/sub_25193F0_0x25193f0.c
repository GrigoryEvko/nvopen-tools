// Function: sub_25193F0
// Address: 0x25193f0
//
unsigned __int64 __fastcall sub_25193F0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 result; // rax
  __int64 v9; // rcx
  __int64 v10; // rbx
  __int64 *v11; // rsi
  __int64 v12; // rdi
  _QWORD *v13; // rdx
  unsigned __int64 *v14; // rbx
  unsigned __int64 *v15; // r13
  __int64 v16; // r8
  unsigned __int64 *v17; // rdi
  unsigned __int64 v18; // rcx
  unsigned int v19; // esi
  int v20; // eax
  int v21; // esi
  __int64 v22; // r9
  unsigned int v23; // eax
  unsigned __int64 *v24; // r10
  unsigned __int64 v25; // rcx
  int v26; // edx
  unsigned int v27; // esi
  __int64 v28; // r9
  __int64 *v29; // r11
  int v30; // r13d
  unsigned int v31; // edx
  __int64 *v32; // rdi
  __int64 v33; // r8
  int v34; // eax
  __int64 v35; // rbx
  int v36; // r11d
  int v37; // eax
  int v38; // eax
  int v39; // r11d
  __int64 v40; // r9
  int v41; // esi
  unsigned int v42; // eax
  unsigned __int64 *v43; // rcx
  unsigned __int64 v44; // rdi
  int v45; // r11d
  unsigned __int64 *v46; // r8
  __int64 *v47; // [rsp+8h] [rbp-28h] BYREF

  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    v27 = *(_DWORD *)(a1 + 24);
    if ( v27 )
    {
      v28 = *(_QWORD *)(a1 + 8);
      v29 = 0;
      v30 = 1;
      v31 = (v27 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v32 = (__int64 *)(v28 + 8LL * v31);
      v33 = *v32;
      if ( *v32 == *a2 )
        return result;
      while ( v33 != -4096 )
      {
        if ( v29 || v33 != -8192 )
          v32 = v29;
        v31 = (v27 - 1) & (v30 + v31);
        v33 = *(_QWORD *)(v28 + 8LL * v31);
        if ( *a2 == v33 )
          return result;
        ++v30;
        v29 = v32;
        v32 = (__int64 *)(v28 + 8LL * v31);
      }
      if ( !v29 )
        v29 = v32;
      v34 = result + 1;
      ++*(_QWORD *)a1;
      v47 = v29;
      if ( 4 * v34 < 3 * v27 )
      {
        if ( v27 - *(_DWORD *)(a1 + 20) - v34 > v27 >> 3 )
        {
LABEL_35:
          *(_DWORD *)(a1 + 16) = v34;
          if ( *v29 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v29 = *a2;
          result = *(unsigned int *)(a1 + 40);
          v35 = *a2;
          if ( result + 1 > *(unsigned int *)(a1 + 44) )
          {
            sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), result + 1, 8u, v33, v28);
            result = *(unsigned int *)(a1 + 40);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v35;
          ++*(_DWORD *)(a1 + 40);
          return result;
        }
LABEL_66:
        sub_BD14B0(a1, v27);
        sub_2511C60(a1, a2, &v47);
        v29 = v47;
        v34 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_35;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
      v47 = 0;
    }
    v27 *= 2;
    goto LABEL_66;
  }
  v9 = *(unsigned int *)(a1 + 40);
  result = *(_QWORD *)(a1 + 32);
  v10 = *a2;
  v11 = (__int64 *)(result + 8 * v9);
  v12 = (8 * v9) >> 3;
  if ( !((8 * v9) >> 5) )
    goto LABEL_12;
  v13 = (_QWORD *)(result + 32 * ((8 * v9) >> 5));
  do
  {
    if ( *(_QWORD *)result == v10 )
      goto LABEL_9;
    if ( *(_QWORD *)(result + 8) == v10 )
    {
      result += 8LL;
      if ( v11 == (__int64 *)result )
        goto LABEL_15;
      return result;
    }
    if ( *(_QWORD *)(result + 16) == v10 )
    {
      result += 16LL;
      if ( v11 == (__int64 *)result )
        goto LABEL_15;
      return result;
    }
    if ( *(_QWORD *)(result + 24) == v10 )
    {
      result += 24LL;
      if ( v11 == (__int64 *)result )
        goto LABEL_15;
      return result;
    }
    result += 32LL;
  }
  while ( (_QWORD *)result != v13 );
  v12 = (__int64)((__int64)v11 - result) >> 3;
LABEL_12:
  if ( v12 == 2 )
  {
LABEL_42:
    if ( *(_QWORD *)result != v10 )
    {
      result += 8LL;
      goto LABEL_44;
    }
LABEL_9:
    if ( v11 == (__int64 *)result )
      goto LABEL_15;
    return result;
  }
  if ( v12 == 3 )
  {
    if ( *(_QWORD *)result == v10 )
      goto LABEL_9;
    result += 8LL;
    goto LABEL_42;
  }
  if ( v12 != 1 )
    goto LABEL_15;
LABEL_44:
  if ( *(_QWORD *)result == v10 )
    goto LABEL_9;
LABEL_15:
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v9 + 1, 8u, a5, a6);
    v11 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
  }
  *v11 = v10;
  result = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
  *(_DWORD *)(a1 + 40) = result;
  if ( (unsigned int)result > 8 )
  {
    v14 = *(unsigned __int64 **)(a1 + 32);
    v15 = &v14[result];
    while ( 1 )
    {
      v19 = *(_DWORD *)(a1 + 24);
      if ( !v19 )
        break;
      v16 = *(_QWORD *)(a1 + 8);
      result = (v19 - 1) & (((unsigned int)*v14 >> 9) ^ ((unsigned int)*v14 >> 4));
      v17 = (unsigned __int64 *)(v16 + 8 * result);
      v18 = *v17;
      if ( *v14 != *v17 )
      {
        v36 = 1;
        v24 = 0;
        while ( v18 != -4096 )
        {
          if ( v24 || v18 != -8192 )
            v17 = v24;
          result = (v19 - 1) & (v36 + (_DWORD)result);
          v18 = *(_QWORD *)(v16 + 8LL * (unsigned int)result);
          if ( *v14 == v18 )
            goto LABEL_20;
          ++v36;
          v24 = v17;
          v17 = (unsigned __int64 *)(v16 + 8LL * (unsigned int)result);
        }
        v37 = *(_DWORD *)(a1 + 16);
        if ( !v24 )
          v24 = v17;
        ++*(_QWORD *)a1;
        v26 = v37 + 1;
        if ( 4 * (v37 + 1) < 3 * v19 )
        {
          if ( v19 - *(_DWORD *)(a1 + 20) - v26 <= v19 >> 3 )
          {
            sub_BD14B0(a1, v19);
            v38 = *(_DWORD *)(a1 + 24);
            if ( !v38 )
            {
LABEL_92:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v39 = v38 - 1;
            v40 = *(_QWORD *)(a1 + 8);
            v41 = 1;
            v42 = (v38 - 1) & (((unsigned int)*v14 >> 9) ^ ((unsigned int)*v14 >> 4));
            v24 = (unsigned __int64 *)(v40 + 8LL * v42);
            v26 = *(_DWORD *)(a1 + 16) + 1;
            v43 = 0;
            v44 = *v24;
            if ( *v24 != *v14 )
            {
              while ( v44 != -4096 )
              {
                if ( v44 == -8192 && !v43 )
                  v43 = v24;
                v42 = v39 & (v41 + v42);
                v24 = (unsigned __int64 *)(v40 + 8LL * v42);
                v44 = *v24;
                if ( *v14 == *v24 )
                  goto LABEL_25;
                ++v41;
              }
              if ( v43 )
                v24 = v43;
            }
          }
          goto LABEL_25;
        }
LABEL_23:
        sub_BD14B0(a1, 2 * v19);
        v20 = *(_DWORD *)(a1 + 24);
        if ( !v20 )
          goto LABEL_92;
        v21 = v20 - 1;
        v22 = *(_QWORD *)(a1 + 8);
        v23 = (v20 - 1) & (((unsigned int)*v14 >> 9) ^ ((unsigned int)*v14 >> 4));
        v24 = (unsigned __int64 *)(v22 + 8LL * v23);
        v25 = *v24;
        v26 = *(_DWORD *)(a1 + 16) + 1;
        if ( *v14 != *v24 )
        {
          v45 = 1;
          v46 = 0;
          while ( v25 != -4096 )
          {
            if ( v25 == -8192 && !v46 )
              v46 = v24;
            v23 = v21 & (v45 + v23);
            v24 = (unsigned __int64 *)(v22 + 8LL * v23);
            v25 = *v24;
            if ( *v14 == *v24 )
              goto LABEL_25;
            ++v45;
          }
          if ( v46 )
            v24 = v46;
        }
LABEL_25:
        *(_DWORD *)(a1 + 16) = v26;
        if ( *v24 != -4096 )
          --*(_DWORD *)(a1 + 20);
        result = *v14;
        *v24 = *v14;
      }
LABEL_20:
      if ( v15 == ++v14 )
        return result;
    }
    ++*(_QWORD *)a1;
    goto LABEL_23;
  }
  return result;
}
