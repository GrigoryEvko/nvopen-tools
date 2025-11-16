// Function: sub_1F49190
// Address: 0x1f49190
//
__int64 *__fastcall sub_1F49190(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v7; // r12
  unsigned int v8; // esi
  __int64 v9; // rdi
  unsigned int v10; // ecx
  __int64 *result; // rax
  __int64 v12; // rdx
  int v13; // r10d
  __int64 *v14; // r9
  int v15; // ecx
  int v16; // ecx
  int v17; // eax
  int v18; // esi
  __int64 v19; // r8
  unsigned int v20; // edx
  __int64 v21; // rdi
  int v22; // r10d
  __int64 *v23; // r9
  int v24; // eax
  int v25; // edx
  __int64 v26; // rdi
  __int64 *v27; // r8
  unsigned int v28; // r14d
  int v29; // r9d
  __int64 v30; // rsi

  v7 = *(_QWORD *)(a1 + 216);
  v8 = *(_DWORD *)(v7 + 24);
  if ( !v8 )
  {
    ++*(_QWORD *)v7;
    goto LABEL_14;
  }
  v9 = *(_QWORD *)(v7 + 8);
  v10 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = (__int64 *)(v9 + 24LL * v10);
  v12 = *result;
  if ( *result == a2 )
    goto LABEL_3;
  v13 = 1;
  v14 = 0;
  while ( v12 != -4 )
  {
    if ( !v14 && v12 == -8 )
      v14 = result;
    v10 = (v8 - 1) & (v13 + v10);
    result = (__int64 *)(v9 + 24LL * v10);
    v12 = *result;
    if ( *result == a2 )
      goto LABEL_3;
    ++v13;
  }
  v15 = *(_DWORD *)(v7 + 16);
  if ( v14 )
    result = v14;
  ++*(_QWORD *)v7;
  v16 = v15 + 1;
  if ( 4 * v16 >= 3 * v8 )
  {
LABEL_14:
    sub_1F48FC0(v7, 2 * v8);
    v17 = *(_DWORD *)(v7 + 24);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(v7 + 8);
      v20 = (v17 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = *(_DWORD *)(v7 + 16) + 1;
      result = (__int64 *)(v19 + 24LL * v20);
      v21 = *result;
      if ( *result != a2 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != -4 )
        {
          if ( !v23 && v21 == -8 )
            v23 = result;
          v20 = v18 & (v22 + v20);
          result = (__int64 *)(v19 + 24LL * v20);
          v21 = *result;
          if ( *result == a2 )
            goto LABEL_10;
          ++v22;
        }
        if ( v23 )
          result = v23;
      }
      goto LABEL_10;
    }
    goto LABEL_42;
  }
  if ( v8 - *(_DWORD *)(v7 + 20) - v16 <= v8 >> 3 )
  {
    sub_1F48FC0(v7, v8);
    v24 = *(_DWORD *)(v7 + 24);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(v7 + 8);
      v27 = 0;
      v28 = (v24 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v29 = 1;
      v16 = *(_DWORD *)(v7 + 16) + 1;
      result = (__int64 *)(v26 + 24LL * v28);
      v30 = *result;
      if ( *result != a2 )
      {
        while ( v30 != -4 )
        {
          if ( !v27 && v30 == -8 )
            v27 = result;
          v28 = v25 & (v29 + v28);
          result = (__int64 *)(v26 + 24LL * v28);
          v30 = *result;
          if ( *result == a2 )
            goto LABEL_10;
          ++v29;
        }
        if ( v27 )
          result = v27;
      }
      goto LABEL_10;
    }
LABEL_42:
    ++*(_DWORD *)(v7 + 16);
    BUG();
  }
LABEL_10:
  *(_DWORD *)(v7 + 16) = v16;
  if ( *result != -4 )
    --*(_DWORD *)(v7 + 20);
  *result = a2;
  result[1] = 0;
  *((_BYTE *)result + 16) = 0;
LABEL_3:
  result[1] = a3;
  *((_BYTE *)result + 16) = a4;
  return result;
}
