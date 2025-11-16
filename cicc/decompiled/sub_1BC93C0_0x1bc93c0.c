// Function: sub_1BC93C0
// Address: 0x1bc93c0
//
__int64 __fastcall sub_1BC93C0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  __int64 v6; // r8
  unsigned int v7; // ecx
  _QWORD *v8; // rax
  __int64 v9; // r10
  __int64 result; // rax
  int v11; // r14d
  _QWORD *v12; // rdx
  int v13; // eax
  int v14; // ecx
  int v15; // eax
  int v16; // esi
  __int64 v17; // r8
  unsigned int v18; // eax
  __int64 v19; // rdi
  int v20; // r10d
  _QWORD *v21; // r9
  int v22; // eax
  int v23; // eax
  __int64 v24; // rdi
  int v25; // r9d
  unsigned int v26; // r13d
  _QWORD *v27; // r8
  __int64 v28; // rsi

  v4 = a1 + 40;
  v5 = *(_DWORD *)(a1 + 64);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 40);
    goto LABEL_17;
  }
  v6 = *(_QWORD *)(a1 + 48);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (_QWORD *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( *v8 == a2 )
  {
LABEL_3:
    result = v8[1];
    if ( result )
    {
      if ( *(_DWORD *)(result + 80) != *(_DWORD *)(a1 + 224) )
        return 0;
    }
    return result;
  }
  v11 = 1;
  v12 = 0;
  while ( v9 != -8 )
  {
    if ( !v12 && v9 == -16 )
      v12 = v8;
    v7 = (v5 - 1) & (v11 + v7);
    v8 = (_QWORD *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( *v8 == a2 )
      goto LABEL_3;
    ++v11;
  }
  if ( !v12 )
    v12 = v8;
  v13 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v5 )
  {
LABEL_17:
    sub_1BC8C30(v4, 2 * v5);
    v15 = *(_DWORD *)(a1 + 64);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 48);
      v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_DWORD *)(a1 + 56) + 1;
      v12 = (_QWORD *)(v17 + 16LL * v18);
      v19 = *v12;
      if ( *v12 != a2 )
      {
        v20 = 1;
        v21 = 0;
        while ( v19 != -8 )
        {
          if ( !v21 && v19 == -16 )
            v21 = v12;
          v18 = v16 & (v20 + v18);
          v12 = (_QWORD *)(v17 + 16LL * v18);
          v19 = *v12;
          if ( *v12 == a2 )
            goto LABEL_13;
          ++v20;
        }
        if ( v21 )
          v12 = v21;
      }
      goto LABEL_13;
    }
    goto LABEL_45;
  }
  if ( v5 - *(_DWORD *)(a1 + 60) - v14 <= v5 >> 3 )
  {
    sub_1BC8C30(v4, v5);
    v22 = *(_DWORD *)(a1 + 64);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 48);
      v25 = 1;
      v26 = v23 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v27 = 0;
      v14 = *(_DWORD *)(a1 + 56) + 1;
      v12 = (_QWORD *)(v24 + 16LL * v26);
      v28 = *v12;
      if ( *v12 != a2 )
      {
        while ( v28 != -8 )
        {
          if ( !v27 && v28 == -16 )
            v27 = v12;
          v26 = v23 & (v25 + v26);
          v12 = (_QWORD *)(v24 + 16LL * v26);
          v28 = *v12;
          if ( *v12 == a2 )
            goto LABEL_13;
          ++v25;
        }
        if ( v27 )
          v12 = v27;
      }
      goto LABEL_13;
    }
LABEL_45:
    ++*(_DWORD *)(a1 + 56);
    BUG();
  }
LABEL_13:
  *(_DWORD *)(a1 + 56) = v14;
  if ( *v12 != -8 )
    --*(_DWORD *)(a1 + 60);
  *v12 = a2;
  v12[1] = 0;
  return 0;
}
