// Function: sub_2ADDD60
// Address: 0x2addd60
//
__int64 __fastcall sub_2ADDD60(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v9; // rcx
  __int64 v10; // rbx
  __int64 *v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 *v14; // rbx
  __int64 *v15; // r14
  __int64 v16; // r8
  __int64 *v17; // rdi
  __int64 v18; // rcx
  unsigned int v19; // esi
  __int64 *v20; // r10
  int v21; // edx
  unsigned int v22; // esi
  __int64 v23; // r9
  __int64 *v24; // r11
  int v25; // r13d
  unsigned int v26; // edx
  __int64 *v27; // rdi
  __int64 v28; // r8
  int v29; // eax
  int v30; // r11d
  int v31; // eax
  _QWORD v32[7]; // [rsp+8h] [rbp-38h] BYREF

  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    v22 = *(_DWORD *)(a1 + 24);
    if ( v22 )
    {
      v23 = *(_QWORD *)(a1 + 8);
      v24 = 0;
      v25 = 1;
      v26 = (v22 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v27 = (__int64 *)(v23 + 8LL * v26);
      v28 = *v27;
      if ( *v27 == *a2 )
        return result;
      while ( v28 != -4096 )
      {
        if ( v24 || v28 != -8192 )
          v27 = v24;
        v26 = (v22 - 1) & (v25 + v26);
        v28 = *(_QWORD *)(v23 + 8LL * v26);
        if ( *a2 == v28 )
          return result;
        ++v25;
        v24 = v27;
        v27 = (__int64 *)(v23 + 8LL * v26);
      }
      if ( !v24 )
        v24 = v27;
      v29 = result + 1;
      ++*(_QWORD *)a1;
      v32[0] = v24;
      if ( 4 * v29 < 3 * v22 )
      {
        if ( v22 - *(_DWORD *)(a1 + 20) - v29 > v22 >> 3 )
        {
LABEL_35:
          *(_DWORD *)(a1 + 16) = v29;
          if ( *v24 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v24 = *a2;
          return sub_9C95B0(a1 + 32, *a2);
        }
LABEL_59:
        sub_CF4090(a1, v22);
        sub_23FDF60(a1, a2, v32);
        v24 = (__int64 *)v32[0];
        v29 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_35;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
      v32[0] = 0;
    }
    v22 *= 2;
    goto LABEL_59;
  }
  v9 = *(unsigned int *)(a1 + 40);
  result = *(_QWORD *)(a1 + 32);
  v10 = *a2;
  v11 = (__int64 *)(result + 8 * v9);
  v12 = (8 * v9) >> 3;
  if ( !((8 * v9) >> 5) )
    goto LABEL_12;
  v13 = result + 32 * ((8 * v9) >> 5);
  do
  {
    if ( *(_QWORD *)result == v10 )
      goto LABEL_9;
    if ( *(_QWORD *)(result + 8) == v10 )
    {
      result += 8;
      if ( v11 == (__int64 *)result )
        goto LABEL_15;
      return result;
    }
    if ( *(_QWORD *)(result + 16) == v10 )
    {
      result += 16;
      if ( v11 == (__int64 *)result )
        goto LABEL_15;
      return result;
    }
    if ( *(_QWORD *)(result + 24) == v10 )
    {
      result += 24;
      if ( v11 == (__int64 *)result )
        goto LABEL_15;
      return result;
    }
    result += 32;
  }
  while ( v13 != result );
  v12 = ((__int64)v11 - result) >> 3;
LABEL_12:
  if ( v12 == 2 )
  {
LABEL_40:
    if ( *(_QWORD *)result != v10 )
    {
      result += 8;
      goto LABEL_42;
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
    result += 8;
    goto LABEL_40;
  }
  if ( v12 != 1 )
    goto LABEL_15;
LABEL_42:
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
    v14 = *(__int64 **)(a1 + 32);
    v15 = &v14[result];
    while ( 1 )
    {
      v19 = *(_DWORD *)(a1 + 24);
      if ( !v19 )
        break;
      v16 = *(_QWORD *)(a1 + 8);
      result = (v19 - 1) & (((unsigned int)*v14 >> 9) ^ ((unsigned int)*v14 >> 4));
      v17 = (__int64 *)(v16 + 8 * result);
      v18 = *v17;
      if ( *v14 != *v17 )
      {
        v30 = 1;
        v20 = 0;
        while ( v18 != -4096 )
        {
          if ( v20 || v18 != -8192 )
            v17 = v20;
          result = (v19 - 1) & (v30 + (_DWORD)result);
          v18 = *(_QWORD *)(v16 + 8LL * (unsigned int)result);
          if ( *v14 == v18 )
            goto LABEL_20;
          ++v30;
          v20 = v17;
          v17 = (__int64 *)(v16 + 8LL * (unsigned int)result);
        }
        v31 = *(_DWORD *)(a1 + 16);
        if ( !v20 )
          v20 = v17;
        ++*(_QWORD *)a1;
        v21 = v31 + 1;
        v32[0] = v20;
        if ( 4 * (v31 + 1) < 3 * v19 )
        {
          if ( v19 - *(_DWORD *)(a1 + 20) - v21 <= v19 >> 3 )
          {
LABEL_24:
            sub_CF4090(a1, v19);
            sub_23FDF60(a1, v14, v32);
            v20 = (__int64 *)v32[0];
            v21 = *(_DWORD *)(a1 + 16) + 1;
          }
          *(_DWORD *)(a1 + 16) = v21;
          if ( *v20 != -4096 )
            --*(_DWORD *)(a1 + 20);
          result = *v14;
          *v20 = *v14;
          goto LABEL_20;
        }
LABEL_23:
        v19 *= 2;
        goto LABEL_24;
      }
LABEL_20:
      if ( v15 == ++v14 )
        return result;
    }
    ++*(_QWORD *)a1;
    v32[0] = 0;
    goto LABEL_23;
  }
  return result;
}
