// Function: sub_18C5D50
// Address: 0x18c5d50
//
char *__fastcall sub_18C5D50(__int64 a1, _QWORD *a2)
{
  unsigned int v4; // esi
  char *result; // rax
  __int64 v6; // r8
  char **v7; // r10
  int v8; // r11d
  unsigned int v9; // edx
  char **v10; // rdi
  char *v11; // rcx
  int v12; // eax
  int v13; // edx
  _BYTE *v14; // rsi
  int v15; // eax
  int v16; // ecx
  __int64 v17; // r8
  unsigned int v18; // eax
  char *v19; // rdi
  int v20; // r11d
  char **v21; // r9
  int v22; // eax
  int v23; // ecx
  __int64 v24; // r8
  int v25; // r11d
  unsigned int v26; // eax
  char *v27; // rdi

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_21;
  }
  result = (char *)*a2;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 0;
  v8 = 1;
  v9 = (v4 - 1) & (37 * *a2);
  v10 = (char **)(v6 + 8LL * v9);
  v11 = *v10;
  if ( *v10 == (char *)*a2 )
    return result;
  while ( v11 != (char *)-1LL )
  {
    if ( v11 != (char *)-2LL || v7 )
      v10 = v7;
    v9 = (v4 - 1) & (v8 + v9);
    v11 = *(char **)(v6 + 8LL * v9);
    if ( result == v11 )
      return result;
    ++v8;
    v7 = v10;
    v10 = (char **)(v6 + 8LL * v9);
  }
  v12 = *(_DWORD *)(a1 + 16);
  if ( !v7 )
    v7 = v10;
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v4 )
  {
LABEL_21:
    sub_142F750(a1, 2 * v4);
    v15 = *(_DWORD *)(a1 + 24);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 8);
      v18 = (v15 - 1) & (37 * *a2);
      v7 = (char **)(v17 + 8LL * v18);
      v19 = *v7;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v7 == (char *)*a2 )
        goto LABEL_13;
      v20 = 1;
      v21 = 0;
      while ( v19 != (char *)-1LL )
      {
        if ( v19 == (char *)-2LL && !v21 )
          v21 = v7;
        v18 = v16 & (v20 + v18);
        v7 = (char **)(v17 + 8LL * v18);
        v19 = *v7;
        if ( (char *)*a2 == *v7 )
          goto LABEL_13;
        ++v20;
      }
LABEL_25:
      if ( v21 )
        v7 = v21;
      goto LABEL_13;
    }
LABEL_42:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
    sub_142F750(a1, v4);
    v22 = *(_DWORD *)(a1 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 8);
      v21 = 0;
      v25 = 1;
      v26 = (v22 - 1) & (37 * *a2);
      v7 = (char **)(v24 + 8LL * v26);
      v27 = *v7;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v7 == (char *)*a2 )
        goto LABEL_13;
      while ( v27 != (char *)-1LL )
      {
        if ( !v21 && v27 == (char *)-2LL )
          v21 = v7;
        v26 = v23 & (v25 + v26);
        v7 = (char **)(v24 + 8LL * v26);
        v27 = *v7;
        if ( (char *)*a2 == *v7 )
          goto LABEL_13;
        ++v25;
      }
      goto LABEL_25;
    }
    goto LABEL_42;
  }
LABEL_13:
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v7 != (char *)-1LL )
    --*(_DWORD *)(a1 + 20);
  result = (char *)*a2;
  *v7 = (char *)*a2;
  v14 = *(_BYTE **)(a1 + 40);
  if ( v14 == *(_BYTE **)(a1 + 48) )
    return sub_9CA200(a1 + 32, v14, a2);
  if ( v14 )
  {
    *(_QWORD *)v14 = result;
    v14 = *(_BYTE **)(a1 + 40);
  }
  *(_QWORD *)(a1 + 40) = v14 + 8;
  return result;
}
