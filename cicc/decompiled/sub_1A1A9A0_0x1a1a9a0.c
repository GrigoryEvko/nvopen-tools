// Function: sub_1A1A9A0
// Address: 0x1a1a9a0
//
__int64 __fastcall sub_1A1A9A0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi

  *a1 = off_49F5260;
  v2 = a1[97];
  if ( (_QWORD *)v2 != a1 + 99 )
    _libc_free(v2);
  j___libc_free_0(a1[94]);
  v3 = a1[89];
  if ( (_QWORD *)v3 != a1 + 91 )
    _libc_free(v3);
  j___libc_free_0(a1[86]);
  v4 = a1[82];
  if ( v4 )
    j_j___libc_free_0(v4, a1[84] - v4);
  v5 = a1[64];
  if ( (_QWORD *)v5 != a1 + 66 )
    _libc_free(v5);
  j___libc_free_0(a1[61]);
  v6 = a1[50];
  if ( (_QWORD *)v6 != a1 + 52 )
    _libc_free(v6);
  j___libc_free_0(a1[47]);
  v7 = a1[28];
  if ( (_QWORD *)v7 != a1 + 30 )
    _libc_free(v7);
  j___libc_free_0(a1[25]);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 808);
}
