// Function: sub_D6CE50
// Address: 0xd6ce50
//
__int64 __fastcall sub_D6CE50(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rcx
  __int64 result; // rax
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
  __int64 v22; // r9
  unsigned int v23; // esi
  int v24; // eax
  __int64 *v25; // rdx
  int v26; // eax
  __int64 v27; // r8
  __int64 v28; // rbx
  int v29; // r11d
  int v30; // eax
  __int64 *v31; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v32[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( *(_DWORD *)(a1 + 16) )
  {
    result = sub_D6B750(a1, a2, &v31);
    if ( (_BYTE)result )
      return result;
    v23 = *(_DWORD *)(a1 + 24);
    v24 = *(_DWORD *)(a1 + 16);
    v25 = v31;
    ++*(_QWORD *)a1;
    v26 = v24 + 1;
    v27 = 2 * v23;
    v32[0] = v25;
    if ( 4 * v26 >= 3 * v23 )
    {
      v23 *= 2;
    }
    else if ( v23 - *(_DWORD *)(a1 + 20) - v26 > v23 >> 3 )
    {
LABEL_31:
      *(_DWORD *)(a1 + 16) = v26;
      if ( *v25 != -4096 )
        --*(_DWORD *)(a1 + 20);
      *v25 = *a2;
      result = *(unsigned int *)(a1 + 40);
      v28 = *a2;
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
      {
        sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), result + 1, 8u, v27, v22);
        result = *(unsigned int *)(a1 + 40);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v28;
      ++*(_DWORD *)(a1 + 40);
      return result;
    }
    sub_D6CC80(a1, v23);
    sub_D6B750(a1, a2, v32);
    v25 = (__int64 *)v32[0];
    v26 = *(_DWORD *)(a1 + 16) + 1;
    goto LABEL_31;
  }
  v8 = *(unsigned int *)(a1 + 40);
  result = *(_QWORD *)(a1 + 32);
  v10 = *a2;
  v11 = (__int64 *)(result + 8 * v8);
  v12 = (8 * v8) >> 3;
  if ( !((8 * v8) >> 5) )
    goto LABEL_12;
  v13 = result + 32 * ((8 * v8) >> 5);
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
LABEL_38:
    if ( *(_QWORD *)result != v10 )
    {
      result += 8;
      goto LABEL_40;
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
    goto LABEL_38;
  }
  if ( v12 != 1 )
    goto LABEL_15;
LABEL_40:
  if ( *(_QWORD *)result == v10 )
    goto LABEL_9;
LABEL_15:
  if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v8 + 1, 8u, a5, a6);
    v11 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
  }
  *v11 = v10;
  result = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
  *(_DWORD *)(a1 + 40) = result;
  if ( (unsigned int)result > 4 )
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
        v29 = 1;
        v20 = 0;
        while ( v18 != -4096 )
        {
          if ( v20 || v18 != -8192 )
            v17 = v20;
          result = (v19 - 1) & (v29 + (_DWORD)result);
          v18 = *(_QWORD *)(v16 + 8LL * (unsigned int)result);
          if ( *v14 == v18 )
            goto LABEL_20;
          ++v29;
          v20 = v17;
          v17 = (__int64 *)(v16 + 8LL * (unsigned int)result);
        }
        v30 = *(_DWORD *)(a1 + 16);
        if ( !v20 )
          v20 = v17;
        ++*(_QWORD *)a1;
        v21 = v30 + 1;
        v32[0] = v20;
        if ( 4 * (v30 + 1) < 3 * v19 )
        {
          if ( v19 - *(_DWORD *)(a1 + 20) - v21 <= v19 >> 3 )
          {
LABEL_24:
            sub_D6CC80(a1, v19);
            sub_D6B750(a1, v14, v32);
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
