// Function: sub_1F0FBA0
// Address: 0x1f0fba0
//
void *__fastcall sub_1F0FBA0(__int64 a1)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 *i; // rax
  unsigned __int64 *v6; // rcx
  unsigned __int64 v7; // rsi
  unsigned __int64 *v8; // rbx
  unsigned __int64 *v9; // r13
  unsigned __int64 v10; // rdi
  unsigned __int64 *v11; // rbx
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi

  v2 = (unsigned __int64 *)(a1 + 336);
  *(_QWORD *)(a1 + 344) = a1 + 336;
  *(_QWORD *)a1 = &unk_49FE760;
  *(_QWORD *)(a1 + 336) = (a1 + 336) | *(_QWORD *)(a1 + 336) & 7LL;
  v3 = *(_QWORD *)(a1 + 536);
  if ( v3 != a1 + 552 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 392);
  if ( v4 != a1 + 408 )
    _libc_free(v4);
  j___libc_free_0(*(_QWORD *)(a1 + 368));
  for ( i = *(unsigned __int64 **)(a1 + 344); v2 != i; *v6 &= 7u )
  {
    v6 = i;
    i = (unsigned __int64 *)i[1];
    v7 = *v6 & 0xFFFFFFFFFFFFFFF8LL;
    *i = v7 | *i & 7;
    *(_QWORD *)(v7 + 8) = i;
    v6[1] = 0;
  }
  v8 = *(unsigned __int64 **)(a1 + 248);
  v9 = &v8[*(unsigned int *)(a1 + 256)];
  while ( v9 != v8 )
  {
    v10 = *v8++;
    _libc_free(v10);
  }
  v11 = *(unsigned __int64 **)(a1 + 296);
  v12 = (unsigned __int64)&v11[2 * *(unsigned int *)(a1 + 304)];
  if ( v11 != (unsigned __int64 *)v12 )
  {
    do
    {
      v13 = *v11;
      v11 += 2;
      _libc_free(v13);
    }
    while ( v11 != (unsigned __int64 *)v12 );
    v12 = *(_QWORD *)(a1 + 296);
  }
  if ( v12 != a1 + 312 )
    _libc_free(v12);
  v14 = *(_QWORD *)(a1 + 248);
  if ( v14 != a1 + 264 )
    _libc_free(v14);
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
