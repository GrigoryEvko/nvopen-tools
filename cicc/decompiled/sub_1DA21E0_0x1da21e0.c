// Function: sub_1DA21E0
// Address: 0x1da21e0
//
void *__fastcall sub_1DA21E0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rdi
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rdi

  *(_QWORD *)a1 = off_49FAB78;
  v2 = *(_QWORD *)(a1 + 456);
  if ( v2 != a1 + 472 )
    _libc_free(v2);
  sub_1DA2140(a1 + 400);
  v3 = *(_QWORD *)(a1 + 400);
  if ( v3 != a1 + 448 )
    j_j___libc_free_0(v3, 8LL * *(_QWORD *)(a1 + 408));
  v4 = *(_QWORD **)(a1 + 360);
  while ( v4 )
  {
    v5 = v4;
    v4 = (_QWORD *)*v4;
    v6 = v5[13];
    if ( (_QWORD *)v6 != v5 + 15 )
      _libc_free(v6);
    v7 = v5[7];
    if ( (_QWORD *)v7 != v5 + 9 )
      _libc_free(v7);
    j_j___libc_free_0(v5, 216);
  }
  memset(*(void **)(a1 + 344), 0, 8LL * *(_QWORD *)(a1 + 352));
  v8 = *(_QWORD *)(a1 + 344);
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  if ( v8 != a1 + 392 )
    j_j___libc_free_0(v8, 8LL * *(_QWORD *)(a1 + 352));
  sub_1DA2140(a1 + 288);
  v9 = *(_QWORD *)(a1 + 288);
  if ( v9 != a1 + 336 )
    j_j___libc_free_0(v9, 8LL * *(_QWORD *)(a1 + 296));
  _libc_free(*(_QWORD *)(a1 + 256));
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
