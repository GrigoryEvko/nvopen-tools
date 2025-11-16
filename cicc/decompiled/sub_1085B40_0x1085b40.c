// Function: sub_1085B40
// Address: 0x1085b40
//
__int64 __fastcall sub_1085B40(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned int v5; // esi
  int v6; // r14d
  __int64 v7; // r8
  _QWORD *v8; // rdx
  unsigned int v9; // ecx
  _QWORD *v10; // rax
  __int64 v11; // r10
  __int64 *v12; // r13
  __int64 result; // rax
  int v14; // eax
  int v15; // ecx
  size_t *v16; // rsi
  size_t v17; // rdx
  __int64 v18; // rsi
  int v19; // eax
  int v20; // esi
  __int64 v21; // r8
  unsigned int v22; // eax
  __int64 v23; // rdi
  int v24; // r10d
  _QWORD *v25; // r9
  int v26; // eax
  int v27; // eax
  int v28; // r9d
  _QWORD *v29; // r8
  __int64 v30; // rdi
  unsigned int v31; // r13d
  __int64 v32; // rsi

  v3 = a1 + 176;
  v5 = *(_DWORD *)(a1 + 200);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 176);
    goto LABEL_23;
  }
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 184);
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (_QWORD *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( *v10 == a2 )
  {
LABEL_3:
    v12 = v10 + 1;
    result = v10[1];
    if ( result )
      return result;
    goto LABEL_18;
  }
  while ( v11 != -4096 )
  {
    if ( !v8 && v11 == -8192 )
      v8 = v10;
    v9 = (v5 - 1) & (v6 + v9);
    v10 = (_QWORD *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == a2 )
      goto LABEL_3;
    ++v6;
  }
  if ( !v8 )
    v8 = v10;
  v14 = *(_DWORD *)(a1 + 192);
  ++*(_QWORD *)(a1 + 176);
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v5 )
  {
LABEL_23:
    sub_1085690(v3, 2 * v5);
    v19 = *(_DWORD *)(a1 + 200);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 184);
      v22 = (v19 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(a1 + 192) + 1;
      v8 = (_QWORD *)(v21 + 16LL * v22);
      v23 = *v8;
      if ( *v8 != a2 )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != -4096 )
        {
          if ( !v25 && v23 == -8192 )
            v25 = v8;
          v22 = v20 & (v24 + v22);
          v8 = (_QWORD *)(v21 + 16LL * v22);
          v23 = *v8;
          if ( *v8 == a2 )
            goto LABEL_15;
          ++v24;
        }
        if ( v25 )
          v8 = v25;
      }
      goto LABEL_15;
    }
    goto LABEL_46;
  }
  if ( v5 - *(_DWORD *)(a1 + 196) - v15 <= v5 >> 3 )
  {
    sub_1085690(v3, v5);
    v26 = *(_DWORD *)(a1 + 200);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = 1;
      v29 = 0;
      v30 = *(_QWORD *)(a1 + 184);
      v31 = v27 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(a1 + 192) + 1;
      v8 = (_QWORD *)(v30 + 16LL * v31);
      v32 = *v8;
      if ( *v8 != a2 )
      {
        while ( v32 != -4096 )
        {
          if ( !v29 && v32 == -8192 )
            v29 = v8;
          v31 = v27 & (v28 + v31);
          v8 = (_QWORD *)(v30 + 16LL * v31);
          v32 = *v8;
          if ( *v8 == a2 )
            goto LABEL_15;
          ++v28;
        }
        if ( v29 )
          v8 = v29;
      }
      goto LABEL_15;
    }
LABEL_46:
    ++*(_DWORD *)(a1 + 192);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 192) = v15;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 196);
  *v8 = a2;
  v12 = v8 + 1;
  v8[1] = 0;
LABEL_18:
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v16 = *(size_t **)(a2 - 8);
    v17 = *v16;
    v18 = (__int64)(v16 + 3);
  }
  else
  {
    v17 = 0;
    v18 = 0;
  }
  result = sub_1084C60((_QWORD *)a1, v18, v17);
  *v12 = result;
  return result;
}
