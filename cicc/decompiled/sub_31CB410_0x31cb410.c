// Function: sub_31CB410
// Address: 0x31cb410
//
__int64 __fastcall sub_31CB410(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v6; // r12
  unsigned int v7; // esi
  __int64 v8; // r8
  __int64 v9; // rdi
  int v10; // r10d
  __int64 *v11; // r9
  unsigned int v12; // ecx
  __int64 *v13; // rdx
  __int64 result; // rax
  int v15; // eax
  int v16; // edx
  __int64 i; // r12
  __int64 v18; // rbx
  __int64 v19; // rax
  int v20; // eax
  int v21; // ecx
  __int64 v22; // rdi
  __int64 v23; // rsi
  int v24; // r10d
  int v25; // eax
  __int64 v26; // rsi
  unsigned int v27; // r15d
  __int64 *v28; // rdi
  __int64 v29; // rcx

  v6 = *a1;
  v7 = *(_DWORD *)(*a1 + 24);
  if ( !v7 )
  {
    ++*(_QWORD *)v6;
    goto LABEL_21;
  }
  v8 = v7 - 1;
  v9 = *(_QWORD *)(v6 + 8);
  v10 = 1;
  v11 = 0;
  v12 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (__int64 *)(v9 + 8LL * v12);
  result = *v13;
  if ( a2 == *v13 )
    return result;
  while ( result != -4096 )
  {
    if ( v11 || result != -8192 )
      v13 = v11;
    v12 = v8 & (v10 + v12);
    result = *(_QWORD *)(v9 + 8LL * v12);
    if ( a2 == result )
      return result;
    ++v10;
    v11 = v13;
    v13 = (__int64 *)(v9 + 8LL * v12);
  }
  v15 = *(_DWORD *)(v6 + 16);
  if ( !v11 )
    v11 = v13;
  ++*(_QWORD *)v6;
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v7 )
  {
LABEL_21:
    sub_31CB240(v6, 2 * v7);
    v20 = *(_DWORD *)(v6 + 24);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(v6 + 8);
      result = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v11 = (__int64 *)(v22 + 8 * result);
      v23 = *v11;
      v16 = *(_DWORD *)(v6 + 16) + 1;
      if ( a2 != *v11 )
      {
        v24 = 1;
        v8 = 0;
        while ( v23 != -4096 )
        {
          if ( !v8 && v23 == -8192 )
            v8 = (__int64)v11;
          result = v21 & (unsigned int)(v24 + result);
          v11 = (__int64 *)(v22 + 8LL * (unsigned int)result);
          v23 = *v11;
          if ( a2 == *v11 )
            goto LABEL_13;
          ++v24;
        }
        if ( v8 )
          v11 = (__int64 *)v8;
      }
      goto LABEL_13;
    }
    goto LABEL_45;
  }
  result = v7 - *(_DWORD *)(v6 + 20) - v16;
  if ( (unsigned int)result <= v7 >> 3 )
  {
    sub_31CB240(v6, v7);
    v25 = *(_DWORD *)(v6 + 24);
    if ( v25 )
    {
      result = (unsigned int)(v25 - 1);
      v26 = *(_QWORD *)(v6 + 8);
      v8 = 1;
      v27 = result & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v11 = (__int64 *)(v26 + 8LL * v27);
      v16 = *(_DWORD *)(v6 + 16) + 1;
      v28 = 0;
      v29 = *v11;
      if ( a2 != *v11 )
      {
        while ( v29 != -4096 )
        {
          if ( v29 == -8192 && !v28 )
            v28 = v11;
          v27 = result & (v8 + v27);
          v11 = (__int64 *)(v26 + 8LL * v27);
          v29 = *v11;
          if ( a2 == *v11 )
            goto LABEL_13;
          v8 = (unsigned int)(v8 + 1);
        }
        if ( v28 )
          v11 = v28;
      }
      goto LABEL_13;
    }
LABEL_45:
    ++*(_DWORD *)(v6 + 16);
    BUG();
  }
LABEL_13:
  *(_DWORD *)(v6 + 16) = v16;
  if ( *v11 != -4096 )
    --*(_DWORD *)(v6 + 20);
  *v11 = a2;
  for ( i = *(_QWORD *)(a2 + 16); i; i = *(_QWORD *)(i + 8) )
  {
    v18 = a1[1];
    v19 = *(unsigned int *)(v18 + 8);
    if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(v18 + 12) )
    {
      sub_C8D5F0(a1[1], (const void *)(v18 + 16), v19 + 1, 0x10u, v8, (__int64)v11);
      v19 = *(unsigned int *)(v18 + 8);
    }
    result = *(_QWORD *)v18 + 16 * v19;
    *(_QWORD *)result = i;
    *(_QWORD *)(result + 8) = a3;
    ++*(_DWORD *)(v18 + 8);
  }
  return result;
}
