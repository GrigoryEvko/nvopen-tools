// Function: sub_B85490
// Address: 0xb85490
//
_QWORD *__fastcall sub_B85490(__int64 a1, __int64 *a2)
{
  unsigned int v3; // esi
  __int64 v4; // r9
  int v5; // r11d
  unsigned int v6; // ecx
  _QWORD *v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // r8
  _QWORD *result; // rax
  int v11; // ecx
  int v12; // ecx
  __int64 v13; // rdx
  _QWORD *v14; // rdx
  int v15; // eax
  int v16; // esi
  __int64 v17; // r9
  unsigned int v18; // edx
  __int64 v19; // r8
  int v20; // r11d
  _QWORD *v21; // r10
  int v22; // eax
  int v23; // esi
  __int64 v24; // r9
  int v25; // r11d
  unsigned int v26; // edx
  __int64 v27; // r8

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_18;
  }
  v4 = *(_QWORD *)(a1 + 8);
  v5 = 1;
  v6 = (v3 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v7 = (_QWORD *)(v4 + 104LL * v6);
  v8 = 0;
  v9 = *v7;
  if ( *a2 == *v7 )
    return v7 + 1;
  while ( v9 != -4096 )
  {
    if ( !v8 && v9 == -8192 )
      v8 = v7;
    v6 = (v3 - 1) & (v5 + v6);
    v7 = (_QWORD *)(v4 + 104LL * v6);
    v9 = *v7;
    if ( *a2 == *v7 )
      return v7 + 1;
    ++v5;
  }
  v11 = *(_DWORD *)(a1 + 16);
  if ( !v8 )
    v8 = v7;
  ++*(_QWORD *)a1;
  v12 = v11 + 1;
  if ( 4 * v12 >= 3 * v3 )
  {
LABEL_18:
    sub_B85250(a1, 2 * v3);
    v15 = *(_DWORD *)(a1 + 24);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 8);
      v18 = (v15 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v8 = (_QWORD *)(v17 + 104LL * v18);
      v19 = *v8;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v8 == *a2 )
        goto LABEL_14;
      v20 = 1;
      v21 = 0;
      while ( v19 != -4096 )
      {
        if ( !v21 && v19 == -8192 )
          v21 = v8;
        v18 = v16 & (v20 + v18);
        v8 = (_QWORD *)(v17 + 104LL * v18);
        v19 = *v8;
        if ( *a2 == *v8 )
          goto LABEL_14;
        ++v20;
      }
LABEL_22:
      if ( v21 )
        v8 = v21;
      goto LABEL_14;
    }
LABEL_38:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v3 - *(_DWORD *)(a1 + 20) - v12 <= v3 >> 3 )
  {
    sub_B85250(a1, v3);
    v22 = *(_DWORD *)(a1 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 8);
      v21 = 0;
      v25 = 1;
      v26 = (v22 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v8 = (_QWORD *)(v24 + 104LL * v26);
      v27 = *v8;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      if ( *a2 == *v8 )
        goto LABEL_14;
      while ( v27 != -4096 )
      {
        if ( !v21 && v27 == -8192 )
          v21 = v8;
        v26 = v23 & (v25 + v26);
        v8 = (_QWORD *)(v24 + 104LL * v26);
        v27 = *v8;
        if ( *a2 == *v8 )
          goto LABEL_14;
        ++v25;
      }
      goto LABEL_22;
    }
    goto LABEL_38;
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v12;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 20);
  v13 = *a2;
  v8[1] = 0;
  v8[3] = 8;
  *v8 = v13;
  v14 = v8 + 5;
  result = v8 + 1;
  result[1] = v14;
  *((_DWORD *)result + 6) = 0;
  *((_BYTE *)result + 28) = 1;
  return result;
}
