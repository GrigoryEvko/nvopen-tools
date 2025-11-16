// Function: sub_1DF7910
// Address: 0x1df7910
//
void *__fastcall sub_1DF7910(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 *v3; // rbx
  __int64 v4; // rax
  unsigned __int64 *v5; // r13
  unsigned __int64 v6; // rdi
  unsigned __int64 *v7; // rbx
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi

  *(_QWORD *)a1 = off_49FB5D0;
  v2 = *(_QWORD *)(a1 + 488);
  if ( v2 != a1 + 504 )
    _libc_free(v2);
  v3 = *(unsigned __int64 **)(a1 + 400);
  v4 = *(unsigned int *)(a1 + 408);
  *(_QWORD *)(a1 + 376) = 0;
  v5 = &v3[v4];
  while ( v5 != v3 )
  {
    v6 = *v3++;
    _libc_free(v6);
  }
  v7 = *(unsigned __int64 **)(a1 + 448);
  v8 = (unsigned __int64)&v7[2 * *(unsigned int *)(a1 + 456)];
  if ( v7 != (unsigned __int64 *)v8 )
  {
    do
    {
      v9 = *v7;
      v7 += 2;
      _libc_free(v9);
    }
    while ( v7 != (unsigned __int64 *)v8 );
    v8 = *(_QWORD *)(a1 + 448);
  }
  if ( v8 != a1 + 464 )
    _libc_free(v8);
  v10 = *(_QWORD *)(a1 + 400);
  if ( v10 != a1 + 416 )
    _libc_free(v10);
  j___libc_free_0(*(_QWORD *)(a1 + 344));
  j___libc_free_0(*(_QWORD *)(a1 + 312));
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
