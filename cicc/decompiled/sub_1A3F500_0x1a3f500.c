// Function: sub_1A3F500
// Address: 0x1a3f500
//
__int64 __fastcall sub_1A3F500(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rbx
  __int64 v4; // r12
  unsigned __int64 v5; // rdi

  *a1 = off_49F5308;
  j___libc_free_0(a1[61]);
  v2 = a1[26];
  if ( (_QWORD *)v2 != a1 + 28 )
    _libc_free(v2);
  v3 = a1[22];
  while ( v3 )
  {
    v4 = v3;
    sub_1A3F1B0(*(_QWORD **)(v3 + 24));
    v5 = *(_QWORD *)(v3 + 40);
    v3 = *(_QWORD *)(v3 + 16);
    if ( v5 != v4 + 56 )
      _libc_free(v5);
    j_j___libc_free_0(v4, 120);
  }
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 520);
}
