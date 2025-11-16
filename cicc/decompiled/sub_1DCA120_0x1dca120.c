// Function: sub_1DCA120
// Address: 0x1dca120
//
__int64 __fastcall sub_1DCA120(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // rdi
  __int64 v4; // rdi
  unsigned __int64 *v5; // rbx
  unsigned __int64 *v6; // r13
  unsigned __int64 v7; // rdi
  unsigned __int64 *v8; // rbx
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi

  v2 = *(_QWORD *)(a1 + 416);
  *(_QWORD *)a1 = &unk_49FAEA0;
  while ( v2 )
  {
    sub_1DC9780(*(_QWORD *)(v2 + 24));
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    j_j___libc_free_0(v3, 48);
  }
  sub_1DC9D90(a1 + 344);
  v4 = *(_QWORD *)(a1 + 344);
  if ( v4 != a1 + 392 )
    j_j___libc_free_0(v4, 8LL * *(_QWORD *)(a1 + 352));
  v5 = *(unsigned __int64 **)(a1 + 256);
  v6 = &v5[*(unsigned int *)(a1 + 264)];
  while ( v6 != v5 )
  {
    v7 = *v5++;
    _libc_free(v7);
  }
  v8 = *(unsigned __int64 **)(a1 + 304);
  v9 = (unsigned __int64)&v8[2 * *(unsigned int *)(a1 + 312)];
  if ( v8 != (unsigned __int64 *)v9 )
  {
    do
    {
      v10 = *v8;
      v8 += 2;
      _libc_free(v10);
    }
    while ( (unsigned __int64 *)v9 != v8 );
    v9 = *(_QWORD *)(a1 + 304);
  }
  if ( v9 != a1 + 320 )
    _libc_free(v9);
  v11 = *(_QWORD *)(a1 + 256);
  if ( v11 != a1 + 272 )
    _libc_free(v11);
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  sub_16366C0((_QWORD *)a1);
  return j_j___libc_free_0(a1, 448);
}
