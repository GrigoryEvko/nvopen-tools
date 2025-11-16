// Function: sub_2E647F0
// Address: 0x2e647f0
//
void __fastcall sub_2E647F0(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  __int64 v4; // rdx
  _QWORD *v5; // r12
  __int64 v6; // rsi
  _QWORD *v7; // r13
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  __int64 v10; // rdx
  _QWORD *v11; // r12
  __int64 v12; // rsi
  _QWORD *v13; // r13
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi

  v2 = a1 + 632;
  v3 = *(_QWORD *)(a1 + 616);
  if ( v3 != v2 )
    _libc_free(v3);
  if ( (*(_BYTE *)(a1 + 312) & 1) != 0 )
  {
    v5 = (_QWORD *)(a1 + 320);
    v7 = (_QWORD *)(a1 + 608);
  }
  else
  {
    v4 = *(unsigned int *)(a1 + 328);
    v5 = *(_QWORD **)(a1 + 320);
    v6 = 9 * v4;
    if ( !(_DWORD)v4 || (v7 = &v5[v6], &v5[v6] == v5) )
    {
LABEL_28:
      sub_C7D6A0((__int64)v5, v6 * 8, 8);
      if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
        goto LABEL_15;
LABEL_29:
      v11 = (_QWORD *)(a1 + 16);
      v13 = (_QWORD *)(a1 + 304);
      goto LABEL_17;
    }
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
  if ( (*(_BYTE *)(a1 + 312) & 1) == 0 )
  {
    v5 = *(_QWORD **)(a1 + 320);
    v6 = 9LL * *(unsigned int *)(a1 + 328);
    goto LABEL_28;
  }
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    goto LABEL_29;
LABEL_15:
  v10 = *(unsigned int *)(a1 + 24);
  v11 = *(_QWORD **)(a1 + 16);
  v12 = 9 * v10;
  if ( !(_DWORD)v10 )
    goto LABEL_26;
  v13 = &v11[v12];
  if ( &v11[v12] == v11 )
    goto LABEL_26;
  do
  {
LABEL_17:
    if ( *v11 != -4096 && *v11 != -8192 )
    {
      v14 = v11[5];
      if ( (_QWORD *)v14 != v11 + 7 )
        _libc_free(v14);
      v15 = v11[1];
      if ( (_QWORD *)v15 != v11 + 3 )
        _libc_free(v15);
    }
    v11 += 9;
  }
  while ( v11 != v13 );
  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
  {
    v11 = *(_QWORD **)(a1 + 16);
    v12 = 9LL * *(unsigned int *)(a1 + 24);
LABEL_26:
    sub_C7D6A0((__int64)v11, v12 * 8, 8);
  }
}
