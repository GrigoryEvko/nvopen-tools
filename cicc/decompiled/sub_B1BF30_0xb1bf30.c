// Function: sub_B1BF30
// Address: 0xb1bf30
//
void __fastcall sub_B1BF30(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rdx
  _QWORD *v5; // r13
  _QWORD *v6; // rbx
  _QWORD *v7; // rdi
  _QWORD *v8; // rdi
  char v9; // al
  __int64 v10; // rbx
  char v11; // al
  __int64 v12; // r12
  unsigned __int64 v13; // r14
  __int64 v14; // r12
  unsigned __int64 i; // r13
  __int64 v16; // rax
  unsigned int v17; // ebx
  __int64 v18; // rdi
  __int64 v19; // rax

  v3 = a2;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v5 = (_QWORD *)(a1 + 16);
    v6 = (_QWORD *)(a1 + 304);
  }
  else
  {
    v4 = *(unsigned int *)(a1 + 24);
    v5 = *(_QWORD **)(a1 + 16);
    a2 = 72 * v4;
    if ( !(_DWORD)v4 )
      goto LABEL_37;
    v6 = &v5[(unsigned __int64)a2 / 8];
    if ( &v5[(unsigned __int64)a2 / 8] == v5 )
      goto LABEL_37;
  }
  do
  {
    if ( *v5 != -8192 && *v5 != -4096 )
    {
      v7 = (_QWORD *)v5[5];
      if ( v7 != v5 + 7 )
        _libc_free(v7, a2);
      v8 = (_QWORD *)v5[1];
      if ( v8 != v5 + 3 )
        _libc_free(v8, a2);
    }
    v5 += 9;
  }
  while ( v5 != v6 );
  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
  {
    v5 = *(_QWORD **)(a1 + 16);
    a2 = 72LL * *(unsigned int *)(a1 + 24);
LABEL_37:
    sub_C7D6A0(v5, a2, 8);
  }
  v9 = *(_BYTE *)(a1 + 8) | 1;
  *(_BYTE *)(a1 + 8) = v9;
  if ( (*(_BYTE *)(v3 + 8) & 1) == 0 && *(_DWORD *)(v3 + 24) > 4u )
  {
    *(_BYTE *)(a1 + 8) = v9 & 0xFE;
    if ( (*(_BYTE *)(v3 + 8) & 1) != 0 )
    {
      v18 = 288;
      v17 = 4;
    }
    else
    {
      v17 = *(_DWORD *)(v3 + 24);
      v18 = 72LL * v17;
    }
    v19 = sub_C7D670(v18, 8);
    *(_DWORD *)(a1 + 24) = v17;
    *(_QWORD *)(a1 + 16) = v19;
  }
  v10 = a1 + 16;
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(v3 + 8) & 0xFFFFFFFE | *(_DWORD *)(a1 + 8) & 1;
  *(_DWORD *)(a1 + 12) = *(_DWORD *)(v3 + 12);
  v11 = *(_BYTE *)(a1 + 8) & 1;
  if ( !v11 )
    v10 = *(_QWORD *)(a1 + 16);
  if ( (*(_BYTE *)(v3 + 8) & 1) != 0 )
  {
    v12 = v3 + 16;
    if ( !v11 )
      goto LABEL_18;
  }
  else
  {
    v12 = *(_QWORD *)(v3 + 16);
    if ( !v11 )
    {
LABEL_18:
      v13 = *(unsigned int *)(a1 + 24);
      if ( !v13 )
        return;
      goto LABEL_19;
    }
  }
  v13 = 4;
LABEL_19:
  v14 = v12 + 40;
  for ( i = 0; i < v13; ++i )
  {
    if ( v10 )
    {
      v16 = *(_QWORD *)(v14 - 40);
      *(_QWORD *)v10 = v16;
    }
    else
    {
      v16 = MEMORY[0];
    }
    if ( v16 != -4096 && v16 != -8192 )
    {
      *(_DWORD *)(v10 + 16) = 0;
      *(_QWORD *)(v10 + 8) = v10 + 24;
      *(_DWORD *)(v10 + 20) = 2;
      if ( *(_DWORD *)(v14 - 24) )
        sub_B18900(v10 + 8, v14 - 32);
      *(_DWORD *)(v10 + 48) = 0;
      *(_QWORD *)(v10 + 40) = v10 + 56;
      *(_DWORD *)(v10 + 52) = 2;
      if ( *(_DWORD *)(v14 + 8) )
        sub_B18900(v10 + 40, v14);
    }
    v10 += 72;
    v14 += 72;
  }
}
