// Function: sub_30E5660
// Address: 0x30e5660
//
__int64 __fastcall sub_30E5660(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  unsigned int v5; // esi
  __int64 v6; // rdi
  __int64 v7; // r8
  int v8; // r15d
  __int64 v9; // rcx
  __int64 v10; // r9
  unsigned int v11; // edx
  __int64 v12; // rax
  __int64 v13; // r11
  __int64 v14; // rdx
  int v16; // eax
  int v17; // edx
  int v18; // eax
  __int64 v19; // rdi
  unsigned int v20; // eax
  __int64 v21; // rsi
  int v22; // r10d
  int v23; // eax
  int v24; // eax
  __int64 v25; // rsi
  unsigned int v26; // r14d
  __int64 v27; // rdi

  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_DWORD *)(v4 + 208);
  v6 = v4 + 184;
  if ( !v5 )
  {
    ++*(_QWORD *)(v4 + 184);
    goto LABEL_19;
  }
  v7 = v5 - 1;
  v8 = 1;
  v9 = *(_QWORD *)(v4 + 192);
  v10 = 0;
  v11 = v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = v9 + 16LL * v11;
  v13 = *(_QWORD *)v12;
  if ( *(_QWORD *)v12 == a2 )
  {
LABEL_3:
    v14 = *(unsigned int *)(v12 + 8);
    return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64, __int64))a1)(
             *(_QWORD *)(a1 + 8),
             a2,
             v14,
             v9,
             v7,
             v10);
  }
  while ( v13 != -4096 )
  {
    if ( !v10 && v13 == -8192 )
      v10 = v12;
    v11 = v7 & (v8 + v11);
    v12 = v9 + 16LL * v11;
    v13 = *(_QWORD *)v12;
    if ( *(_QWORD *)v12 == a2 )
      goto LABEL_3;
    ++v8;
  }
  if ( !v10 )
    v10 = v12;
  v16 = *(_DWORD *)(v4 + 200);
  ++*(_QWORD *)(v4 + 184);
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v5 )
  {
LABEL_19:
    sub_30E3EE0(v6, 2 * v5);
    v18 = *(_DWORD *)(v4 + 208);
    if ( v18 )
    {
      v9 = (unsigned int)(v18 - 1);
      v19 = *(_QWORD *)(v4 + 192);
      v20 = v9 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v17 = *(_DWORD *)(v4 + 200) + 1;
      v10 = v19 + 16LL * v20;
      v21 = *(_QWORD *)v10;
      if ( *(_QWORD *)v10 != a2 )
      {
        v22 = 1;
        v7 = 0;
        while ( v21 != -4096 )
        {
          if ( !v7 && v21 == -8192 )
            v7 = v10;
          v20 = v9 & (v22 + v20);
          v10 = v19 + 16LL * v20;
          v21 = *(_QWORD *)v10;
          if ( *(_QWORD *)v10 == a2 )
            goto LABEL_15;
          ++v22;
        }
        if ( v7 )
          v10 = v7;
      }
      goto LABEL_15;
    }
    goto LABEL_42;
  }
  v9 = v5 >> 3;
  if ( v5 - *(_DWORD *)(v4 + 204) - v17 <= (unsigned int)v9 )
  {
    sub_30E3EE0(v6, v5);
    v23 = *(_DWORD *)(v4 + 208);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(v4 + 192);
      v7 = 1;
      v26 = v24 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v17 = *(_DWORD *)(v4 + 200) + 1;
      v27 = 0;
      v10 = v25 + 16LL * v26;
      v9 = *(_QWORD *)v10;
      if ( *(_QWORD *)v10 != a2 )
      {
        while ( v9 != -4096 )
        {
          if ( !v27 && v9 == -8192 )
            v27 = v10;
          v26 = v24 & (v7 + v26);
          v10 = v25 + 16LL * v26;
          v9 = *(_QWORD *)v10;
          if ( *(_QWORD *)v10 == a2 )
            goto LABEL_15;
          v7 = (unsigned int)(v7 + 1);
        }
        if ( v27 )
          v10 = v27;
      }
      goto LABEL_15;
    }
LABEL_42:
    ++*(_DWORD *)(v4 + 200);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(v4 + 200) = v17;
  if ( *(_QWORD *)v10 != -4096 )
    --*(_DWORD *)(v4 + 204);
  *(_QWORD *)v10 = a2;
  v14 = 0;
  *(_DWORD *)(v10 + 8) = 0;
  return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64, __int64))a1)(
           *(_QWORD *)(a1 + 8),
           a2,
           v14,
           v9,
           v7,
           v10);
}
