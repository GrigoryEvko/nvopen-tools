// Function: sub_2BFCD30
// Address: 0x2bfcd30
//
char __fastcall sub_2BFCD30(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  _QWORD *v5; // r13
  __int64 v6; // rsi
  _QWORD *v7; // r14
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  char v10; // al
  _QWORD *v11; // r13
  __int64 v12; // rax
  __int64 v13; // r12
  unsigned __int64 v14; // r9
  __int64 v15; // r12
  unsigned __int64 v16; // r15
  __int64 v17; // r14
  __int64 v18; // rbx
  void *v19; // rdi
  __int64 v20; // r8
  size_t v21; // rdx
  unsigned int v22; // r13d
  __int64 v23; // rdi
  __int64 v24; // rax
  int v26; // [rsp+4h] [rbp-3Ch]
  int v27; // [rsp+4h] [rbp-3Ch]
  unsigned __int64 v28; // [rsp+8h] [rbp-38h]
  unsigned __int64 v29; // [rsp+8h] [rbp-38h]

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v5 = (_QWORD *)(a1 + 16);
    v7 = (_QWORD *)(a1 + 304);
  }
  else
  {
    v4 = *(unsigned int *)(a1 + 24);
    v5 = *(_QWORD **)(a1 + 16);
    v6 = 9 * v4;
    if ( !(_DWORD)v4 )
      goto LABEL_41;
    v7 = &v5[v6];
    if ( &v5[v6] == v5 )
      goto LABEL_41;
  }
  do
  {
    if ( *v5 != -8192 && *v5 != -4096 )
    {
      v8 = v5[5];
      if ( (_QWORD *)v8 != v5 + 7 )
        _libc_free(v8);
      v9 = v5[1];
      if ( (_QWORD *)v9 != v5 + 3 )
        _libc_free(v9);
    }
    v5 += 9;
  }
  while ( v5 != v7 );
  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
  {
    v5 = *(_QWORD **)(a1 + 16);
    v6 = 9LL * *(unsigned int *)(a1 + 24);
LABEL_41:
    sub_C7D6A0((__int64)v5, v6 * 8, 8);
  }
  v10 = *(_BYTE *)(a1 + 8) | 1;
  *(_BYTE *)(a1 + 8) = v10;
  if ( (*(_BYTE *)(a2 + 8) & 1) == 0 && *(_DWORD *)(a2 + 24) > 4u )
  {
    *(_BYTE *)(a1 + 8) = v10 & 0xFE;
    if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    {
      v23 = 288;
      v22 = 4;
    }
    else
    {
      v22 = *(_DWORD *)(a2 + 24);
      v23 = 72LL * v22;
    }
    v24 = sub_C7D670(v23, 8);
    *(_DWORD *)(a1 + 24) = v22;
    *(_QWORD *)(a1 + 16) = v24;
  }
  v11 = (_QWORD *)(a1 + 16);
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8) & 0xFFFFFFFE | *(_DWORD *)(a1 + 8) & 1;
  *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
  LOBYTE(v12) = *(_BYTE *)(a1 + 8) & 1;
  if ( !(_BYTE)v12 )
    v11 = *(_QWORD **)(a1 + 16);
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v13 = a2 + 16;
    if ( !(_BYTE)v12 )
      goto LABEL_18;
  }
  else
  {
    v13 = *(_QWORD *)(a2 + 16);
    if ( !(_BYTE)v12 )
    {
LABEL_18:
      v14 = *(unsigned int *)(a1 + 24);
      if ( !*(_DWORD *)(a1 + 24) )
        return v12;
      goto LABEL_19;
    }
  }
  v14 = 4;
LABEL_19:
  v15 = v13 + 72;
  v16 = 0;
  do
  {
    if ( v11 )
    {
      v12 = *(_QWORD *)(v15 - 72);
      *v11 = v12;
    }
    else
    {
      v12 = MEMORY[0];
    }
    if ( v12 != -4096 )
    {
      v17 = v15 - 64;
      v18 = (__int64)(v11 + 1);
      LOBYTE(v12) = v12 != -8192;
      if ( (_BYTE)v12 )
      {
        do
        {
          v19 = (void *)(v18 + 16);
          *(_DWORD *)(v18 + 8) = 0;
          *(_QWORD *)v18 = v18 + 16;
          *(_DWORD *)(v18 + 12) = 2;
          v20 = *(unsigned int *)(v17 + 8);
          if ( (_DWORD)v20 && v17 != v18 )
          {
            v21 = 8LL * (unsigned int)v20;
            if ( (unsigned int)v20 <= 2
              || (v27 = *(_DWORD *)(v17 + 8),
                  v29 = v14,
                  LOBYTE(v12) = sub_C8D5F0(v18, (const void *)(v18 + 16), (unsigned int)v20, 8u, v20, v14),
                  v19 = *(void **)v18,
                  v14 = v29,
                  LODWORD(v20) = v27,
                  (v21 = 8LL * *(unsigned int *)(v17 + 8)) != 0) )
            {
              v26 = v20;
              v28 = v14;
              LOBYTE(v12) = (unsigned __int8)memcpy(v19, *(const void **)v17, v21);
              LODWORD(v20) = v26;
              v14 = v28;
            }
            *(_DWORD *)(v18 + 8) = v20;
          }
          v17 += 32;
          v18 += 32;
        }
        while ( v17 != v15 );
      }
    }
    ++v16;
    v11 += 9;
    v15 += 72;
  }
  while ( v16 < v14 );
  return v12;
}
