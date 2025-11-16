// Function: sub_215C810
// Address: 0x215c810
//
_QWORD *__fastcall sub_215C810(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdi
  unsigned int v7; // esi
  __int64 v8; // r8
  unsigned int v9; // ecx
  _QWORD *result; // rax
  __int64 v11; // rdx
  int v12; // r11d
  _QWORD *v13; // r10
  int v14; // ecx
  int v15; // ecx
  int v16; // eax
  int v17; // esi
  __int64 v18; // r8
  unsigned int v19; // edx
  __int64 v20; // rdi
  int v21; // r10d
  _QWORD *v22; // r9
  int v23; // eax
  int v24; // edx
  __int64 v25; // rdi
  _QWORD *v26; // r8
  unsigned int v27; // r14d
  int v28; // r9d
  __int64 v29; // rsi

  v6 = a1 + 632;
  v7 = *(_DWORD *)(a1 + 656);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 632);
    goto LABEL_14;
  }
  v8 = *(_QWORD *)(a1 + 640);
  v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = (_QWORD *)(v8 + 16LL * v9);
  v11 = *result;
  if ( *result == a2 )
    goto LABEL_3;
  v12 = 1;
  v13 = 0;
  while ( v11 != -8 )
  {
    if ( !v13 && v11 == -16 )
      v13 = result;
    v9 = (v7 - 1) & (v12 + v9);
    result = (_QWORD *)(v8 + 16LL * v9);
    v11 = *result;
    if ( *result == a2 )
      goto LABEL_3;
    ++v12;
  }
  v14 = *(_DWORD *)(a1 + 648);
  if ( v13 )
    result = v13;
  ++*(_QWORD *)(a1 + 632);
  v15 = v14 + 1;
  if ( 4 * v15 >= 3 * v7 )
  {
LABEL_14:
    sub_215C650(v6, 2 * v7);
    v16 = *(_DWORD *)(a1 + 656);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 640);
      v19 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(a1 + 648) + 1;
      result = (_QWORD *)(v18 + 16LL * v19);
      v20 = *result;
      if ( *result != a2 )
      {
        v21 = 1;
        v22 = 0;
        while ( v20 != -8 )
        {
          if ( !v22 && v20 == -16 )
            v22 = result;
          v19 = v17 & (v21 + v19);
          result = (_QWORD *)(v18 + 16LL * v19);
          v20 = *result;
          if ( *result == a2 )
            goto LABEL_10;
          ++v21;
        }
        if ( v22 )
          result = v22;
      }
      goto LABEL_10;
    }
    goto LABEL_42;
  }
  if ( v7 - *(_DWORD *)(a1 + 652) - v15 <= v7 >> 3 )
  {
    sub_215C650(v6, v7);
    v23 = *(_DWORD *)(a1 + 656);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 640);
      v26 = 0;
      v27 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v28 = 1;
      v15 = *(_DWORD *)(a1 + 648) + 1;
      result = (_QWORD *)(v25 + 16LL * v27);
      v29 = *result;
      if ( *result != a2 )
      {
        while ( v29 != -8 )
        {
          if ( !v26 && v29 == -16 )
            v26 = result;
          v27 = v24 & (v28 + v27);
          result = (_QWORD *)(v25 + 16LL * v27);
          v29 = *result;
          if ( *result == a2 )
            goto LABEL_10;
          ++v28;
        }
        if ( v26 )
          result = v26;
      }
      goto LABEL_10;
    }
LABEL_42:
    ++*(_DWORD *)(a1 + 648);
    BUG();
  }
LABEL_10:
  *(_DWORD *)(a1 + 648) = v15;
  if ( *result != -8 )
    --*(_DWORD *)(a1 + 652);
  *result = a2;
  result[1] = 0;
LABEL_3:
  result[1] = a3;
  return result;
}
