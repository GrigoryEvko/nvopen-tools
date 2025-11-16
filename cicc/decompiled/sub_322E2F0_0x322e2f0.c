// Function: sub_322E2F0
// Address: 0x322e2f0
//
_QWORD *__fastcall sub_322E2F0(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rax
  unsigned int v5; // esi
  __int64 v6; // r12
  _QWORD *v7; // r8
  __int64 v8; // rdi
  int v9; // r11d
  _QWORD *v10; // rdx
  unsigned int v11; // r9d
  _QWORD *result; // rax
  __int64 v13; // rcx
  int v14; // eax
  __int64 v15; // rcx
  int v16; // eax
  int v17; // esi
  unsigned int v18; // eax
  __int64 v19; // rdi
  int v20; // r10d
  _QWORD *v21; // r9
  int v22; // eax
  int v23; // eax
  __int64 v24; // rdi
  unsigned int v25; // r13d
  int v26; // r9d
  __int64 v27; // rsi

  v4 = *(_QWORD **)a2;
  if ( !*(_QWORD *)a2 )
  {
    if ( (*(_BYTE *)(a2 + 9) & 0x70) != 0x20 || *(char *)(a2 + 8) < 0 )
      BUG();
    *(_BYTE *)(a2 + 8) |= 8u;
    v4 = sub_E807D0(*(_QWORD *)(a2 + 24));
    *(_QWORD *)a2 = v4;
  }
  v5 = *(_DWORD *)(a1 + 3632);
  v6 = v4[1];
  v7 = (_QWORD *)(a1 + 3608);
  v8 = *(_QWORD *)(a1 + 3616);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 3608);
    goto LABEL_23;
  }
  v9 = 1;
  v10 = 0;
  v11 = (v5 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  result = (_QWORD *)(v8 + 16LL * v11);
  v13 = *result;
  if ( v6 == *result )
    return result;
  while ( v13 != -4096 )
  {
    if ( v13 != -8192 || v10 )
      result = v10;
    v11 = (v5 - 1) & (v9 + v11);
    v13 = *(_QWORD *)(v8 + 16LL * v11);
    if ( v6 == v13 )
      return result;
    ++v9;
    v10 = result;
    result = (_QWORD *)(v8 + 16LL * v11);
  }
  if ( !v10 )
    v10 = result;
  v14 = *(_DWORD *)(a1 + 3624);
  ++*(_QWORD *)(a1 + 3608);
  v15 = (unsigned int)(v14 + 1);
  if ( 4 * (int)v15 >= 3 * v5 )
  {
LABEL_23:
    sub_107F430(a1 + 3608, 2 * v5);
    v16 = *(_DWORD *)(a1 + 3632);
    if ( v16 )
    {
      v17 = v16 - 1;
      v7 = *(_QWORD **)(a1 + 3616);
      v18 = (v16 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v15 = (unsigned int)(*(_DWORD *)(a1 + 3624) + 1);
      v10 = &v7[2 * v18];
      v19 = *v10;
      if ( v6 != *v10 )
      {
        v20 = 1;
        v21 = 0;
        while ( v19 != -4096 )
        {
          if ( v19 == -8192 && !v21 )
            v21 = v10;
          v18 = v17 & (v20 + v18);
          v10 = &v7[2 * v18];
          v19 = *v10;
          if ( v6 == *v10 )
            goto LABEL_17;
          ++v20;
        }
        if ( v21 )
          v10 = v21;
      }
      goto LABEL_17;
    }
    goto LABEL_47;
  }
  if ( v5 - *(_DWORD *)(a1 + 3628) - (unsigned int)v15 <= v5 >> 3 )
  {
    sub_107F430(a1 + 3608, v5);
    v22 = *(_DWORD *)(a1 + 3632);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 3616);
      v7 = 0;
      v25 = v23 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v26 = 1;
      v15 = (unsigned int)(*(_DWORD *)(a1 + 3624) + 1);
      v10 = (_QWORD *)(v24 + 16LL * v25);
      v27 = *v10;
      if ( v6 != *v10 )
      {
        while ( v27 != -4096 )
        {
          if ( !v7 && v27 == -8192 )
            v7 = v10;
          v25 = v23 & (v26 + v25);
          v10 = (_QWORD *)(v24 + 16LL * v25);
          v27 = *v10;
          if ( v6 == *v10 )
            goto LABEL_17;
          ++v26;
        }
        if ( v7 )
          v10 = v7;
      }
      goto LABEL_17;
    }
LABEL_47:
    ++*(_DWORD *)(a1 + 3624);
    BUG();
  }
LABEL_17:
  *(_DWORD *)(a1 + 3624) = v15;
  if ( *v10 != -4096 )
    --*(_DWORD *)(a1 + 3628);
  *v10 = v6;
  v10[1] = a2;
  if ( *(_BYTE *)(a1 + 3769) )
    return (_QWORD *)sub_37291A0(a1 + 4840, a2, 0, v15, v7);
  result = (_QWORD *)sub_3220AA0(a1);
  if ( (unsigned __int16)result > 4u )
    return (_QWORD *)sub_37291A0(a1 + 4840, a2, 0, v15, v7);
  return result;
}
