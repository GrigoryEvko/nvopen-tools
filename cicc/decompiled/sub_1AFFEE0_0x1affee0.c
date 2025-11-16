// Function: sub_1AFFEE0
// Address: 0x1affee0
//
_QWORD *__fastcall sub_1AFFEE0(__int64 a1, __int64 *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  unsigned int v6; // edx
  _QWORD *result; // rax
  __int64 v8; // rdi
  int v9; // r11d
  _QWORD *v10; // r10
  int v11; // ecx
  int v12; // ecx
  __int64 v13; // rdx
  int v14; // eax
  int v15; // esi
  __int64 v16; // r9
  unsigned int v17; // edx
  __int64 v18; // r8
  int v19; // r11d
  _QWORD *v20; // r10
  int v21; // eax
  int v22; // esi
  __int64 v23; // r9
  int v24; // r11d
  unsigned int v25; // edx
  __int64 v26; // r8

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_14;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  result = (_QWORD *)(v5 + 16LL * v6);
  v8 = *result;
  if ( *a2 == *result )
    return result;
  v9 = 1;
  v10 = 0;
  while ( v8 != -8 )
  {
    if ( !v10 && v8 == -16 )
      v10 = result;
    v6 = (v4 - 1) & (v9 + v6);
    result = (_QWORD *)(v5 + 16LL * v6);
    v8 = *result;
    if ( *a2 == *result )
      return result;
    ++v9;
  }
  v11 = *(_DWORD *)(a1 + 16);
  if ( v10 )
    result = v10;
  ++*(_QWORD *)a1;
  v12 = v11 + 1;
  if ( 4 * v12 >= 3 * v4 )
  {
LABEL_14:
    sub_1447B20(a1, 2 * v4);
    v14 = *(_DWORD *)(a1 + 24);
    if ( v14 )
    {
      v15 = v14 - 1;
      v16 = *(_QWORD *)(a1 + 8);
      v12 = *(_DWORD *)(a1 + 16) + 1;
      v17 = (v14 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      result = (_QWORD *)(v16 + 16LL * v17);
      v18 = *result;
      if ( *result == *a2 )
        goto LABEL_10;
      v19 = 1;
      v20 = 0;
      while ( v18 != -8 )
      {
        if ( !v20 && v18 == -16 )
          v20 = result;
        v17 = v15 & (v19 + v17);
        result = (_QWORD *)(v16 + 16LL * v17);
        v18 = *result;
        if ( *a2 == *result )
          goto LABEL_10;
        ++v19;
      }
LABEL_18:
      if ( v20 )
        result = v20;
      goto LABEL_10;
    }
LABEL_39:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v12 <= v4 >> 3 )
  {
    sub_1447B20(a1, v4);
    v21 = *(_DWORD *)(a1 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 8);
      v20 = 0;
      v24 = 1;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      v25 = (v21 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      result = (_QWORD *)(v23 + 16LL * v25);
      v26 = *result;
      if ( *a2 == *result )
        goto LABEL_10;
      while ( v26 != -8 )
      {
        if ( !v20 && v26 == -16 )
          v20 = result;
        v25 = v22 & (v24 + v25);
        result = (_QWORD *)(v23 + 16LL * v25);
        v26 = *result;
        if ( *a2 == *result )
          goto LABEL_10;
        ++v24;
      }
      goto LABEL_18;
    }
    goto LABEL_39;
  }
LABEL_10:
  *(_DWORD *)(a1 + 16) = v12;
  if ( *result != -8 )
    --*(_DWORD *)(a1 + 20);
  v13 = *a2;
  result[1] = 0;
  *result = v13;
  return result;
}
