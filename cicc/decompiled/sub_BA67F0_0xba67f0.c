// Function: sub_BA67F0
// Address: 0xba67f0
//
__int64 __fastcall sub_BA67F0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v5; // rcx
  __int64 v6; // rbx
  __int64 *v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 *v10; // rbx
  __int64 *v11; // r13
  __int64 v12; // r8
  __int64 *v13; // rdi
  __int64 v14; // rcx
  unsigned int v15; // esi
  int v16; // eax
  int v17; // ecx
  __int64 v18; // r8
  unsigned int v19; // eax
  __int64 *v20; // r10
  __int64 v21; // rdx
  int v22; // eax
  int v23; // edx
  unsigned int v24; // esi
  __int64 v25; // r9
  __int64 *v26; // r11
  int v27; // r13d
  unsigned int v28; // edx
  __int64 *v29; // rdi
  __int64 v30; // r8
  int v31; // eax
  __int64 v32; // rbx
  int v33; // r11d
  int v34; // eax
  int v35; // eax
  int v36; // r9d
  __int64 *v37; // rdi
  _QWORD v38[5]; // [rsp+8h] [rbp-28h] BYREF

  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    v24 = *(_DWORD *)(a1 + 24);
    if ( v24 )
    {
      v25 = *(_QWORD *)(a1 + 8);
      v26 = 0;
      v27 = 1;
      v28 = (v24 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v29 = (__int64 *)(v25 + 8LL * v28);
      v30 = *v29;
      if ( *v29 == *a2 )
        return result;
      while ( v30 != -4096 )
      {
        if ( v30 != -8192 || v26 )
          v29 = v26;
        v28 = (v24 - 1) & (v27 + v28);
        v30 = *(_QWORD *)(v25 + 8LL * v28);
        if ( *a2 == v30 )
          return result;
        ++v27;
        v26 = v29;
        v29 = (__int64 *)(v25 + 8LL * v28);
      }
      if ( !v26 )
        v26 = v29;
      v31 = result + 1;
      ++*(_QWORD *)a1;
      v38[0] = v26;
      if ( 4 * v31 < 3 * v24 )
      {
        if ( v24 - *(_DWORD *)(a1 + 20) - v31 > v24 >> 3 )
        {
LABEL_36:
          *(_DWORD *)(a1 + 16) = v31;
          if ( *v26 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v26 = *a2;
          result = *(unsigned int *)(a1 + 40);
          v32 = *a2;
          if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
          {
            sub_C8D5F0(a1 + 32, a1 + 48, result + 1, 8);
            result = *(unsigned int *)(a1 + 40);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v32;
          ++*(_DWORD *)(a1 + 40);
          return result;
        }
LABEL_62:
        sub_9C0C30(a1, v24);
        sub_B95BD0(a1, a2, v38);
        v26 = (__int64 *)v38[0];
        v31 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_36;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
      v38[0] = 0;
    }
    v24 *= 2;
    goto LABEL_62;
  }
  v5 = *(unsigned int *)(a1 + 40);
  result = *(_QWORD *)(a1 + 32);
  v6 = *a2;
  v7 = (__int64 *)(result + 8 * v5);
  v8 = (8 * v5) >> 3;
  if ( !((8 * v5) >> 5) )
    goto LABEL_12;
  v9 = result + 32 * ((8 * v5) >> 5);
  do
  {
    if ( *(_QWORD *)result == v6 )
      goto LABEL_9;
    if ( *(_QWORD *)(result + 8) == v6 )
    {
      result += 8;
      if ( v7 == (__int64 *)result )
        goto LABEL_15;
      return result;
    }
    if ( *(_QWORD *)(result + 16) == v6 )
    {
      result += 16;
      if ( v7 == (__int64 *)result )
        goto LABEL_15;
      return result;
    }
    if ( *(_QWORD *)(result + 24) == v6 )
    {
      result += 24;
      if ( v7 == (__int64 *)result )
        goto LABEL_15;
      return result;
    }
    result += 32;
  }
  while ( result != v9 );
  v8 = ((__int64)v7 - result) >> 3;
LABEL_12:
  if ( v8 == 2 )
  {
LABEL_43:
    if ( *(_QWORD *)result != v6 )
    {
      result += 8;
      goto LABEL_45;
    }
LABEL_9:
    if ( v7 == (__int64 *)result )
      goto LABEL_15;
    return result;
  }
  if ( v8 == 3 )
  {
    if ( *(_QWORD *)result == v6 )
      goto LABEL_9;
    result += 8;
    goto LABEL_43;
  }
  if ( v8 != 1 )
    goto LABEL_15;
LABEL_45:
  if ( *(_QWORD *)result == v6 )
    goto LABEL_9;
LABEL_15:
  if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, a1 + 48, v5 + 1, 8);
    v7 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
  }
  *v7 = v6;
  result = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
  *(_DWORD *)(a1 + 40) = result;
  if ( (unsigned int)result > 4 )
  {
    v10 = *(__int64 **)(a1 + 32);
    v11 = &v10[result];
    while ( 1 )
    {
      v15 = *(_DWORD *)(a1 + 24);
      if ( !v15 )
        break;
      v12 = *(_QWORD *)(a1 + 8);
      result = (v15 - 1) & (((unsigned int)*v10 >> 9) ^ ((unsigned int)*v10 >> 4));
      v13 = (__int64 *)(v12 + 8 * result);
      v14 = *v13;
      if ( *v10 != *v13 )
      {
        v33 = 1;
        v20 = 0;
        while ( v14 != -4096 )
        {
          if ( v20 || v14 != -8192 )
            v13 = v20;
          result = (v15 - 1) & (v33 + (_DWORD)result);
          v14 = *(_QWORD *)(v12 + 8LL * (unsigned int)result);
          if ( *v10 == v14 )
            goto LABEL_20;
          ++v33;
          v20 = v13;
          v13 = (__int64 *)(v12 + 8LL * (unsigned int)result);
        }
        v34 = *(_DWORD *)(a1 + 16);
        if ( !v20 )
          v20 = v13;
        ++*(_QWORD *)a1;
        v23 = v34 + 1;
        v38[0] = v20;
        if ( 4 * (v34 + 1) < 3 * v15 )
        {
          if ( v15 - *(_DWORD *)(a1 + 20) - v23 <= v15 >> 3 )
          {
            sub_9C0C30(a1, v15);
            sub_B95BD0(a1, v10, v38);
            v20 = (__int64 *)v38[0];
            v23 = *(_DWORD *)(a1 + 16) + 1;
          }
LABEL_26:
          *(_DWORD *)(a1 + 16) = v23;
          if ( *v20 != -4096 )
            --*(_DWORD *)(a1 + 20);
          result = *v10;
          *v20 = *v10;
          goto LABEL_20;
        }
LABEL_23:
        sub_9C0C30(a1, 2 * v15);
        v16 = *(_DWORD *)(a1 + 24);
        if ( v16 )
        {
          v17 = v16 - 1;
          v18 = *(_QWORD *)(a1 + 8);
          v19 = (v16 - 1) & (((unsigned int)*v10 >> 9) ^ ((unsigned int)*v10 >> 4));
          v20 = (__int64 *)(v18 + 8LL * v19);
          v21 = *v20;
          if ( *v10 != *v20 )
          {
            v36 = 1;
            v37 = 0;
            while ( v21 != -4096 )
            {
              if ( v21 == -8192 && !v37 )
                v37 = v20;
              v19 = v17 & (v36 + v19);
              v20 = (__int64 *)(v18 + 8LL * v19);
              v21 = *v20;
              if ( *v10 == *v20 )
                goto LABEL_25;
              ++v36;
            }
            if ( v37 )
              v20 = v37;
          }
LABEL_25:
          v22 = *(_DWORD *)(a1 + 16);
          v38[0] = v20;
          v23 = v22 + 1;
        }
        else
        {
          v35 = *(_DWORD *)(a1 + 16);
          v38[0] = 0;
          v20 = 0;
          v23 = v35 + 1;
        }
        goto LABEL_26;
      }
LABEL_20:
      if ( v11 == ++v10 )
        return result;
    }
    ++*(_QWORD *)a1;
    v38[0] = 0;
    goto LABEL_23;
  }
  return result;
}
