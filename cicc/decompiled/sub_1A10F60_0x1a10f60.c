// Function: sub_1A10F60
// Address: 0x1a10f60
//
_QWORD *__fastcall sub_1A10F60(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  unsigned int v4; // esi
  int v6; // r14d
  __int64 v7; // rdi
  _QWORD *v8; // rdx
  unsigned int v9; // ecx
  _QWORD *v10; // rax
  __int64 v11; // r10
  _QWORD *result; // rax
  int v13; // eax
  int v14; // ecx
  unsigned __int8 v15; // cl
  int v16; // eax
  int v17; // esi
  __int64 v18; // r8
  unsigned int v19; // eax
  __int64 v20; // rdi
  int v21; // r10d
  _QWORD *v22; // r9
  int v23; // eax
  int v24; // eax
  __int64 v25; // rdi
  _QWORD *v26; // r8
  unsigned int v27; // r12d
  int v28; // r9d
  __int64 v29; // rsi

  v2 = a1 + 120;
  v4 = *(_DWORD *)(a1 + 144);
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 120);
    goto LABEL_21;
  }
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 128);
  v8 = 0;
  v9 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (_QWORD *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
    return v10 + 1;
  while ( v11 != -8 )
  {
    if ( !v8 && v11 == -16 )
      v8 = v10;
    v9 = (v4 - 1) & (v6 + v9);
    v10 = (_QWORD *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
      return v10 + 1;
    ++v6;
  }
  if ( !v8 )
    v8 = v10;
  v13 = *(_DWORD *)(a1 + 136);
  ++*(_QWORD *)(a1 + 120);
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v4 )
  {
LABEL_21:
    sub_1A0FE70(v2, 2 * v4);
    v16 = *(_DWORD *)(a1 + 144);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 128);
      v19 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_DWORD *)(a1 + 136) + 1;
      v8 = (_QWORD *)(v18 + 16LL * v19);
      v20 = *v8;
      if ( a2 != *v8 )
      {
        v21 = 1;
        v22 = 0;
        while ( v20 != -8 )
        {
          if ( !v22 && v20 == -16 )
            v22 = v8;
          v19 = v17 & (v21 + v19);
          v8 = (_QWORD *)(v18 + 16LL * v19);
          v20 = *v8;
          if ( a2 == *v8 )
            goto LABEL_15;
          ++v21;
        }
        if ( v22 )
          v8 = v22;
      }
      goto LABEL_15;
    }
    goto LABEL_44;
  }
  if ( v4 - *(_DWORD *)(a1 + 140) - v14 <= v4 >> 3 )
  {
    sub_1A0FE70(v2, v4);
    v23 = *(_DWORD *)(a1 + 144);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 128);
      v26 = 0;
      v27 = v24 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v28 = 1;
      v14 = *(_DWORD *)(a1 + 136) + 1;
      v8 = (_QWORD *)(v25 + 16LL * v27);
      v29 = *v8;
      if ( a2 != *v8 )
      {
        while ( v29 != -8 )
        {
          if ( !v26 && v29 == -16 )
            v26 = v8;
          v27 = v24 & (v28 + v27);
          v8 = (_QWORD *)(v25 + 16LL * v27);
          v29 = *v8;
          if ( a2 == *v8 )
            goto LABEL_15;
          ++v28;
        }
        if ( v26 )
          v8 = v26;
      }
      goto LABEL_15;
    }
LABEL_44:
    ++*(_DWORD *)(a1 + 136);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 136) = v14;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 140);
  *v8 = a2;
  result = v8 + 1;
  v8[1] = 0;
  v15 = *(_BYTE *)(a2 + 16);
  if ( v15 != 9 && v15 <= 0x10u )
    v8[1] = a2 | 2;
  return result;
}
