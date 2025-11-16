// Function: sub_F0BE20
// Address: 0xf0be20
//
__int64 __fastcall sub_F0BE20(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  _QWORD *v4; // r12
  _QWORD *v5; // rbx
  _QWORD *v6; // rdi
  __int64 v7; // rsi
  _QWORD *v8; // r12
  __int64 v9; // rsi
  _QWORD *v10; // rbx
  _QWORD *v11; // rdi

  *(_QWORD *)a1 = &unk_49E4F30;
  if ( (*(_BYTE *)(a1 + 992) & 1) == 0 )
  {
    a2 = 2LL * *(unsigned int *)(a1 + 1008);
    sub_C7D6A0(*(_QWORD *)(a1 + 1000), a2 * 8, 8);
  }
  if ( (*(_BYTE *)(a1 + 400) & 1) != 0 )
  {
    v4 = (_QWORD *)(a1 + 408);
    v5 = (_QWORD *)(a1 + 984);
  }
  else
  {
    v3 = *(unsigned int *)(a1 + 416);
    v4 = *(_QWORD **)(a1 + 408);
    a2 = 9 * v3;
    if ( !(_DWORD)v3 )
      goto LABEL_24;
    v5 = &v4[a2];
    if ( &v4[a2] == v4 )
      goto LABEL_24;
  }
  do
  {
    if ( *v4 != -4096 && *v4 != -8192 )
    {
      v6 = (_QWORD *)v4[1];
      if ( v6 != v4 + 3 )
        _libc_free(v6, a2 * 8);
    }
    v4 += 9;
  }
  while ( v4 != v5 );
  if ( (*(_BYTE *)(a1 + 400) & 1) == 0 )
  {
    v4 = *(_QWORD **)(a1 + 408);
    a2 = 9LL * *(unsigned int *)(a1 + 416);
LABEL_24:
    sub_C7D6A0((__int64)v4, a2 * 8, 8);
  }
  if ( (*(_BYTE *)(a1 + 256) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 264), 16LL * *(unsigned int *)(a1 + 272), 8);
  v7 = *(unsigned int *)(a1 + 224);
  if ( (_DWORD)v7 )
  {
    v8 = *(_QWORD **)(a1 + 208);
    v9 = 4 * v7;
    v10 = &v8[v9];
    do
    {
      if ( *v8 != -8192 && *v8 != -4096 )
      {
        v11 = (_QWORD *)v8[1];
        if ( v11 != v8 + 3 )
          _libc_free(v11, v9 * 8);
      }
      v8 += 4;
    }
    while ( v10 != v8 );
    v7 = *(unsigned int *)(a1 + 224);
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 208), 32 * v7, 8);
}
