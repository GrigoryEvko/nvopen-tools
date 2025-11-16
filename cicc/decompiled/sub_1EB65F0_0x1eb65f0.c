// Function: sub_1EB65F0
// Address: 0x1eb65f0
//
void *__fastcall sub_1EB65F0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  __int64 v14; // r13
  __int64 i; // rbx
  __int64 v16; // rdi

  *(_QWORD *)a1 = off_49FD9B0;
  _libc_free(*(_QWORD *)(a1 + 1072));
  v2 = *(_QWORD *)(a1 + 1024);
  if ( v2 != a1 + 1040 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 752);
  if ( v3 != a1 + 768 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 672);
  if ( v4 != a1 + 688 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 648);
  if ( v5 )
    j_j___libc_free_0(v5, *(_QWORD *)(a1 + 664) - v5);
  v6 = *(unsigned int *)(a1 + 640);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD *)(a1 + 624);
    v8 = v7 + 56 * v6;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v7 <= 0xFFFFFFFD )
        {
          v9 = *(_QWORD *)(v7 + 8);
          if ( v9 != v7 + 24 )
            break;
        }
        v7 += 56;
        if ( v8 == v7 )
          goto LABEL_15;
      }
      _libc_free(v9);
      v7 += 56;
    }
    while ( v8 != v7 );
  }
LABEL_15:
  j___libc_free_0(*(_QWORD *)(a1 + 624));
  _libc_free(*(_QWORD *)(a1 + 600));
  v10 = *(_QWORD *)(a1 + 392);
  if ( v10 != a1 + 408 )
    _libc_free(v10);
  v11 = *(_QWORD *)(a1 + 368);
  if ( v11 != a1 + 384 )
    _libc_free(v11);
  v12 = *(_QWORD *)(a1 + 352);
  if ( v12 )
    j_j___libc_free_0_0(v12);
  _libc_free(*(_QWORD *)(a1 + 328));
  v13 = *(_QWORD *)(a1 + 304);
  if ( v13 != a1 + 320 )
    _libc_free(v13);
  v14 = *(_QWORD *)(a1 + 264);
  if ( v14 )
  {
    for ( i = v14 + 24LL * *(_QWORD *)(v14 - 8); v14 != i; i -= 24 )
    {
      v16 = *(_QWORD *)(i - 8);
      if ( v16 )
        j_j___libc_free_0_0(v16);
    }
    j_j_j___libc_free_0_0(v14 - 8);
  }
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
