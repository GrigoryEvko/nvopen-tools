// Function: sub_1E291F0
// Address: 0x1e291f0
//
__int64 __fastcall sub_1E291F0(__int64 a1)
{
  __int64 v2; // rdi
  unsigned __int64 *v3; // rbx
  unsigned __int64 *v4; // r13
  unsigned __int64 v5; // rdi
  unsigned __int64 *v6; // rbx
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  __int64 v10; // rdi

  v2 = a1 + 232;
  *(_QWORD *)(v2 - 232) = &unk_49FBD08;
  sub_1E28CF0(v2);
  v3 = *(unsigned __int64 **)(a1 + 304);
  v4 = &v3[*(unsigned int *)(a1 + 312)];
  while ( v4 != v3 )
  {
    v5 = *v3++;
    _libc_free(v5);
  }
  v6 = *(unsigned __int64 **)(a1 + 352);
  v7 = (unsigned __int64)&v6[2 * *(unsigned int *)(a1 + 360)];
  if ( v6 != (unsigned __int64 *)v7 )
  {
    do
    {
      v8 = *v6;
      v6 += 2;
      _libc_free(v8);
    }
    while ( (unsigned __int64 *)v7 != v6 );
    v7 = *(_QWORD *)(a1 + 352);
  }
  if ( v7 != a1 + 368 )
    _libc_free(v7);
  v9 = *(_QWORD *)(a1 + 304);
  if ( v9 != a1 + 320 )
    _libc_free(v9);
  v10 = *(_QWORD *)(a1 + 264);
  if ( v10 )
    j_j___libc_free_0(v10, *(_QWORD *)(a1 + 280) - v10);
  j___libc_free_0(*(_QWORD *)(a1 + 240));
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  sub_16366C0((_QWORD *)a1);
  return j_j___libc_free_0(a1, 392);
}
