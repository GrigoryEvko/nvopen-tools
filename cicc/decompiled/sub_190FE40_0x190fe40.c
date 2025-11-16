// Function: sub_190FE40
// Address: 0x190fe40
//
int *__fastcall sub_190FE40(__int64 a1, int *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  unsigned int v6; // ecx
  int *result; // rax
  int v8; // edi
  int v9; // r11d
  int *v10; // r10
  int v11; // ecx
  int v12; // ecx
  int v13; // edx
  int v14; // eax
  int v15; // esi
  __int64 v16; // r9
  unsigned int v17; // edx
  int v18; // r8d
  int v19; // r11d
  int *v20; // r10
  int v21; // eax
  int v22; // edx
  __int64 v23; // r9
  int v24; // r11d
  unsigned int v25; // edi
  int v26; // r8d

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_14;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (v4 - 1) & (37 * *a2);
  result = (int *)(v5 + 40LL * v6);
  v8 = *result;
  if ( *a2 == *result )
    return result;
  v9 = 1;
  v10 = 0;
  while ( v8 != -1 )
  {
    if ( !v10 && v8 == -2 )
      v10 = result;
    v6 = (v4 - 1) & (v9 + v6);
    result = (int *)(v5 + 40LL * v6);
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
    sub_190FC70(a1, 2 * v4);
    v14 = *(_DWORD *)(a1 + 24);
    if ( v14 )
    {
      v15 = v14 - 1;
      v16 = *(_QWORD *)(a1 + 8);
      v12 = *(_DWORD *)(a1 + 16) + 1;
      v17 = (v14 - 1) & (37 * *a2);
      result = (int *)(v16 + 40LL * v17);
      v18 = *result;
      if ( *result == *a2 )
        goto LABEL_10;
      v19 = 1;
      v20 = 0;
      while ( v18 != -1 )
      {
        if ( !v20 && v18 == -2 )
          v20 = result;
        v17 = v15 & (v19 + v17);
        result = (int *)(v16 + 40LL * v17);
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
    sub_190FC70(a1, v4);
    v21 = *(_DWORD *)(a1 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 8);
      v20 = 0;
      v24 = 1;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      v25 = (v21 - 1) & (37 * *a2);
      result = (int *)(v23 + 40LL * v25);
      v26 = *result;
      if ( *a2 == *result )
        goto LABEL_10;
      while ( v26 != -1 )
      {
        if ( !v20 && v26 == -2 )
          v20 = result;
        v25 = v22 & (v24 + v25);
        result = (int *)(v23 + 40LL * v25);
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
  if ( *result != -1 )
    --*(_DWORD *)(a1 + 20);
  v13 = *a2;
  *((_QWORD *)result + 1) = 0;
  *((_QWORD *)result + 2) = 0;
  *result = v13;
  *((_QWORD *)result + 3) = 0;
  *((_QWORD *)result + 4) = 0;
  return result;
}
