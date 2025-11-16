// Function: sub_233AAF0
// Address: 0x233aaf0
//
void __fastcall sub_233AAF0(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // r12
  _QWORD *v4; // r14
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  _QWORD *v8; // rbx
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  _QWORD *v11; // rbx
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r12
  unsigned __int64 v17; // rdi

  if ( !*(_BYTE *)(a1 + 500) )
    _libc_free(*(_QWORD *)(a1 + 480));
  v2 = *(unsigned int *)(a1 + 464);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 448);
    v4 = &v3[17 * v2];
    do
    {
      if ( *v3 != -8192 && *v3 != -4096 )
      {
        v5 = v3[13];
        while ( v5 )
        {
          sub_2307D90(*(_QWORD *)(v5 + 24));
          v6 = v5;
          v5 = *(_QWORD *)(v5 + 16);
          j_j___libc_free_0(v6);
        }
        v7 = v3[1];
        if ( (_QWORD *)v7 != v3 + 3 )
          _libc_free(v7);
      }
      v3 += 17;
    }
    while ( v4 != v3 );
    v2 = *(unsigned int *)(a1 + 464);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 448), 136 * v2, 8);
  v8 = *(_QWORD **)(a1 + 400);
  while ( v8 )
  {
    v9 = (unsigned __int64)v8;
    v8 = (_QWORD *)*v8;
    j_j___libc_free_0(v9);
  }
  memset(*(void **)(a1 + 384), 0, 8LL * *(_QWORD *)(a1 + 392));
  v10 = *(_QWORD *)(a1 + 384);
  *(_QWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  if ( v10 != a1 + 432 )
    j_j___libc_free_0(v10);
  v11 = *(_QWORD **)(a1 + 344);
  while ( v11 )
  {
    v12 = (unsigned __int64)v11;
    v11 = (_QWORD *)*v11;
    if ( !*(_BYTE *)(v12 + 44) )
      _libc_free(*(_QWORD *)(v12 + 24));
    j_j___libc_free_0(v12);
  }
  memset(*(void **)(a1 + 328), 0, 8LL * *(_QWORD *)(a1 + 336));
  v13 = *(_QWORD *)(a1 + 328);
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  if ( v13 != a1 + 376 )
    j_j___libc_free_0(v13);
  v14 = *(unsigned int *)(a1 + 320);
  if ( (_DWORD)v14 )
  {
    v15 = *(_QWORD *)(a1 + 304);
    v16 = v15 + 72 * v14;
    do
    {
      while ( *(_QWORD *)v15 == -4096 || *(_QWORD *)v15 == -8192 || *(_BYTE *)(v15 + 36) )
      {
        v15 += 72;
        if ( v16 == v15 )
          goto LABEL_31;
      }
      v17 = *(_QWORD *)(v15 + 16);
      v15 += 72;
      _libc_free(v17);
    }
    while ( v16 != v15 );
LABEL_31:
    v14 = *(unsigned int *)(a1 + 320);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 304), 72 * v14, 8);
  if ( !*(_BYTE *)(a1 + 36) )
    _libc_free(*(_QWORD *)(a1 + 16));
}
