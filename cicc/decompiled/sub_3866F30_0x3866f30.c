// Function: sub_3866F30
// Address: 0x3866f30
//
unsigned __int64 __fastcall sub_3866F30(__int64 a1, unsigned __int64 *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  unsigned __int64 *v6; // r10
  int v7; // r11d
  unsigned __int64 result; // rax
  unsigned __int64 *v9; // rdi
  unsigned __int64 v10; // rcx
  int v11; // eax
  int v12; // edx
  char *v13; // rsi
  int v14; // eax
  int v15; // ecx
  __int64 v16; // r8
  unsigned int v17; // eax
  unsigned __int64 v18; // rdi
  int v19; // r11d
  unsigned __int64 *v20; // r9
  int v21; // eax
  int v22; // ecx
  __int64 v23; // r8
  int v24; // r11d
  unsigned int v25; // eax
  unsigned __int64 v26; // rdi

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_21;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 0;
  v7 = 1;
  result = (v4 - 1) & ((unsigned int)*a2 ^ (unsigned int)(*a2 >> 9));
  v9 = (unsigned __int64 *)(v5 + 8 * result);
  v10 = *v9;
  if ( *v9 == *a2 )
    return result;
  while ( v10 != -4 )
  {
    if ( v10 != -16 || v6 )
      v9 = v6;
    result = (v4 - 1) & (v7 + (_DWORD)result);
    v10 = *(_QWORD *)(v5 + 8LL * (unsigned int)result);
    if ( *a2 == v10 )
      return result;
    ++v7;
    v6 = v9;
    v9 = (unsigned __int64 *)(v5 + 8LL * (unsigned int)result);
  }
  v11 = *(_DWORD *)(a1 + 16);
  if ( !v6 )
    v6 = v9;
  ++*(_QWORD *)a1;
  v12 = v11 + 1;
  if ( 4 * (v11 + 1) >= 3 * v4 )
  {
LABEL_21:
    sub_3866D80(a1, 2 * v4);
    v14 = *(_DWORD *)(a1 + 24);
    if ( v14 )
    {
      v15 = v14 - 1;
      v16 = *(_QWORD *)(a1 + 8);
      v17 = (v14 - 1) & (*a2 ^ (*a2 >> 9));
      v6 = (unsigned __int64 *)(v16 + 8LL * v17);
      v18 = *v6;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v6 == *a2 )
        goto LABEL_13;
      v19 = 1;
      v20 = 0;
      while ( v18 != -4 )
      {
        if ( v18 == -16 && !v20 )
          v20 = v6;
        v17 = v15 & (v19 + v17);
        v6 = (unsigned __int64 *)(v16 + 8LL * v17);
        v18 = *v6;
        if ( *a2 == *v6 )
          goto LABEL_13;
        ++v19;
      }
LABEL_25:
      if ( v20 )
        v6 = v20;
      goto LABEL_13;
    }
LABEL_42:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v12 <= v4 >> 3 )
  {
    sub_3866D80(a1, v4);
    v21 = *(_DWORD *)(a1 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 8);
      v20 = 0;
      v24 = 1;
      v25 = (v21 - 1) & (*a2 ^ (*a2 >> 9));
      v6 = (unsigned __int64 *)(v23 + 8LL * v25);
      v26 = *v6;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v6 == *a2 )
        goto LABEL_13;
      while ( v26 != -4 )
      {
        if ( !v20 && v26 == -16 )
          v20 = v6;
        v25 = v22 & (v24 + v25);
        v6 = (unsigned __int64 *)(v23 + 8LL * v25);
        v26 = *v6;
        if ( *a2 == *v6 )
          goto LABEL_13;
        ++v24;
      }
      goto LABEL_25;
    }
    goto LABEL_42;
  }
LABEL_13:
  *(_DWORD *)(a1 + 16) = v12;
  if ( *v6 != -4 )
    --*(_DWORD *)(a1 + 20);
  result = *a2;
  *v6 = *a2;
  v13 = *(char **)(a1 + 40);
  if ( v13 == *(char **)(a1 + 48) )
    return sub_3863360((unsigned __int64 *)(a1 + 32), v13, a2);
  if ( v13 )
  {
    result = *a2;
    *(_QWORD *)v13 = *a2;
    v13 = *(char **)(a1 + 40);
  }
  *(_QWORD *)(a1 + 40) = v13 + 8;
  return result;
}
