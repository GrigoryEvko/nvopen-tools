// Function: sub_16730E0
// Address: 0x16730e0
//
__int64 *__fastcall sub_16730E0(__int64 a1)
{
  __int64 *v1; // r13
  __int64 *v2; // r12
  __int64 v3; // rbx
  unsigned int v4; // esi
  __int64 v5; // r8
  unsigned int v6; // edx
  __int64 *result; // rax
  __int64 v8; // rdi
  int v9; // r11d
  __int64 *v10; // r10
  int v11; // ecx
  int v12; // ecx
  __int64 v13; // rdx
  int v14; // eax
  int v15; // esi
  __int64 v16; // r9
  unsigned int v17; // edx
  __int64 v18; // r8
  int v19; // r11d
  __int64 *v20; // r10
  int v21; // eax
  int v22; // esi
  __int64 v23; // r9
  int v24; // r11d
  unsigned int v25; // edx
  __int64 v26; // r8

  sub_1623BA0(**(_QWORD **)a1, **(_DWORD **)(a1 + 8), **(_QWORD **)(a1 + 16));
  v1 = *(__int64 **)(a1 + 16);
  v2 = *(__int64 **)(a1 + 32);
  v3 = *(_QWORD *)(a1 + 24);
  v4 = *(_DWORD *)(v3 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)v3;
    goto LABEL_14;
  }
  v5 = *(_QWORD *)(v3 + 8);
  v6 = (v4 - 1) & (((unsigned int)*v2 >> 9) ^ ((unsigned int)*v2 >> 4));
  result = (__int64 *)(v5 + 24LL * v6);
  v8 = *result;
  if ( *v2 == *result )
    goto LABEL_3;
  v9 = 1;
  v10 = 0;
  while ( v8 != -8 )
  {
    if ( !v10 && v8 == -16 )
      v10 = result;
    v6 = (v4 - 1) & (v9 + v6);
    result = (__int64 *)(v5 + 24LL * v6);
    v8 = *result;
    if ( *v2 == *result )
      goto LABEL_3;
    ++v9;
  }
  v11 = *(_DWORD *)(v3 + 16);
  if ( v10 )
    result = v10;
  ++*(_QWORD *)v3;
  v12 = v11 + 1;
  if ( 4 * v12 >= 3 * v4 )
  {
LABEL_14:
    sub_1672CC0(v3, 2 * v4);
    v14 = *(_DWORD *)(v3 + 24);
    if ( v14 )
    {
      v15 = v14 - 1;
      v16 = *(_QWORD *)(v3 + 8);
      v12 = *(_DWORD *)(v3 + 16) + 1;
      v17 = (v14 - 1) & (((unsigned int)*v2 >> 9) ^ ((unsigned int)*v2 >> 4));
      result = (__int64 *)(v16 + 24LL * v17);
      v18 = *result;
      if ( *result == *v2 )
        goto LABEL_10;
      v19 = 1;
      v20 = 0;
      while ( v18 != -8 )
      {
        if ( !v20 && v18 == -16 )
          v20 = result;
        v17 = v15 & (v19 + v17);
        result = (__int64 *)(v16 + 24LL * v17);
        v18 = *result;
        if ( *v2 == *result )
          goto LABEL_10;
        ++v19;
      }
LABEL_18:
      if ( v20 )
        result = v20;
      goto LABEL_10;
    }
LABEL_39:
    ++*(_DWORD *)(v3 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(v3 + 20) - v12 <= v4 >> 3 )
  {
    sub_1672CC0(v3, v4);
    v21 = *(_DWORD *)(v3 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(v3 + 8);
      v20 = 0;
      v24 = 1;
      v12 = *(_DWORD *)(v3 + 16) + 1;
      v25 = (v21 - 1) & (((unsigned int)*v2 >> 9) ^ ((unsigned int)*v2 >> 4));
      result = (__int64 *)(v23 + 24LL * v25);
      v26 = *result;
      if ( *v2 == *result )
        goto LABEL_10;
      while ( v26 != -8 )
      {
        if ( !v20 && v26 == -16 )
          v20 = result;
        v25 = v22 & (v24 + v25);
        result = (__int64 *)(v23 + 24LL * v25);
        v26 = *result;
        if ( *v2 == *result )
          goto LABEL_10;
        ++v24;
      }
      goto LABEL_18;
    }
    goto LABEL_39;
  }
LABEL_10:
  *(_DWORD *)(v3 + 16) = v12;
  if ( *result != -8 )
    --*(_DWORD *)(v3 + 20);
  v13 = *v2;
  result[1] = 0;
  *((_DWORD *)result + 4) = 0;
  *result = v13;
LABEL_3:
  result[1] = *v1;
  return result;
}
