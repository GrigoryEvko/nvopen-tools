// Function: sub_B1A8B0
// Address: 0xb1a8b0
//
__int64 __fastcall sub_B1A8B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // rdx
  _QWORD *v6; // r12
  _QWORD *v7; // r13
  _QWORD *v8; // rdi
  _QWORD *v9; // rdi
  __int64 v10; // rdx
  __int64 *v11; // r12
  __int64 *v12; // r13
  __int64 result; // rax
  __int64 *v14; // rdi
  __int64 *v15; // rdi

  v3 = a1 + 632;
  v4 = *(_QWORD *)(a1 + 616);
  if ( v4 != v3 )
    _libc_free(v4, a2);
  if ( (*(_BYTE *)(a1 + 312) & 1) != 0 )
  {
    v6 = (_QWORD *)(a1 + 320);
    v7 = (_QWORD *)(a1 + 608);
  }
  else
  {
    v5 = *(unsigned int *)(a1 + 328);
    v6 = *(_QWORD **)(a1 + 320);
    a2 = 72 * v5;
    if ( !(_DWORD)v5 || (v7 = &v6[(unsigned __int64)a2 / 8], &v6[(unsigned __int64)a2 / 8] == v6) )
    {
LABEL_28:
      sub_C7D6A0(v6, a2, 8);
      if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
        goto LABEL_15;
LABEL_29:
      v11 = (__int64 *)(a1 + 16);
      v12 = (__int64 *)(a1 + 304);
      goto LABEL_17;
    }
  }
  do
  {
    if ( *v6 != -8192 && *v6 != -4096 )
    {
      v8 = (_QWORD *)v6[5];
      if ( v8 != v6 + 7 )
        _libc_free(v8, a2);
      v9 = (_QWORD *)v6[1];
      if ( v9 != v6 + 3 )
        _libc_free(v9, a2);
    }
    v6 += 9;
  }
  while ( v6 != v7 );
  if ( (*(_BYTE *)(a1 + 312) & 1) == 0 )
  {
    v6 = *(_QWORD **)(a1 + 320);
    a2 = 72LL * *(unsigned int *)(a1 + 328);
    goto LABEL_28;
  }
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    goto LABEL_29;
LABEL_15:
  v10 = *(unsigned int *)(a1 + 24);
  v11 = *(__int64 **)(a1 + 16);
  a2 = 72 * v10;
  if ( !(_DWORD)v10 )
    return sub_C7D6A0(v11, a2, 8);
  v12 = &v11[(unsigned __int64)a2 / 8];
  if ( &v11[(unsigned __int64)a2 / 8] == v11 )
    return sub_C7D6A0(v11, a2, 8);
  do
  {
LABEL_17:
    result = *v11;
    if ( *v11 != -4096 && result != -8192 )
    {
      v14 = (__int64 *)v11[5];
      if ( v14 != v11 + 7 )
        _libc_free(v14, a2);
      v15 = (__int64 *)v11[1];
      result = (__int64)(v11 + 3);
      if ( v15 != v11 + 3 )
        result = _libc_free(v15, a2);
    }
    v11 += 9;
  }
  while ( v11 != v12 );
  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
  {
    v11 = *(__int64 **)(a1 + 16);
    a2 = 72LL * *(unsigned int *)(a1 + 24);
    return sub_C7D6A0(v11, a2, 8);
  }
  return result;
}
