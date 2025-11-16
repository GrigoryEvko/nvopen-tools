// Function: sub_1DE4260
// Address: 0x1de4260
//
void *__fastcall sub_1DE4260(__int64 a1)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r13
  unsigned __int64 v4; // rdi
  unsigned __int64 *v5; // rbx
  unsigned __int64 v6; // r13
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __int64 v9; // r13
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rdi
  __int64 v13; // rsi
  unsigned __int64 v14; // rdi
  __int64 v15; // r13
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi

  *(_QWORD *)a1 = off_49FB1F8;
  j___libc_free_0(*(_QWORD *)(a1 + 896));
  sub_1DE3B40(a1 + 784);
  v2 = *(unsigned __int64 **)(a1 + 800);
  v3 = &v2[*(unsigned int *)(a1 + 808)];
  while ( v3 != v2 )
  {
    v4 = *v2++;
    _libc_free(v4);
  }
  v5 = *(unsigned __int64 **)(a1 + 848);
  v6 = (unsigned __int64)&v5[2 * *(unsigned int *)(a1 + 856)];
  if ( v5 != (unsigned __int64 *)v6 )
  {
    do
    {
      v7 = *v5;
      v5 += 2;
      _libc_free(v7);
    }
    while ( v5 != (unsigned __int64 *)v6 );
    v6 = *(_QWORD *)(a1 + 848);
  }
  if ( v6 != a1 + 864 )
    _libc_free(v6);
  v8 = *(_QWORD *)(a1 + 800);
  if ( v8 != a1 + 816 )
    _libc_free(v8);
  v9 = *(unsigned int *)(a1 + 776);
  if ( (_DWORD)v9 )
  {
    v10 = *(_QWORD *)(a1 + 760);
    v11 = v10 + 32 * v9;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v10 <= 0xFFFFFFFD )
        {
          v12 = *(_QWORD *)(v10 + 8);
          if ( v12 )
            break;
        }
        v10 += 32;
        if ( v11 == v10 )
          goto LABEL_16;
      }
      v13 = *(_QWORD *)(v10 + 24);
      v10 += 32;
      j_j___libc_free_0(v12, v13 - v12);
    }
    while ( v11 != v10 );
  }
LABEL_16:
  j___libc_free_0(*(_QWORD *)(a1 + 760));
  v14 = *(_QWORD *)(a1 + 672);
  if ( v14 != a1 + 688 )
    _libc_free(v14);
  v15 = *(_QWORD *)(a1 + 568);
  if ( v15 )
  {
    j___libc_free_0(*(_QWORD *)(v15 + 16));
    j_j___libc_free_0(v15, 40);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 528));
  v16 = *(_QWORD *)(a1 + 376);
  if ( v16 != a1 + 392 )
    _libc_free(v16);
  v17 = *(_QWORD *)(a1 + 232);
  if ( v17 != a1 + 248 )
    _libc_free(v17);
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
