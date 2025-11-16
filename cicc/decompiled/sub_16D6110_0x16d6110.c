// Function: sub_16D6110
// Address: 0x16d6110
//
void __fastcall sub_16D6110(_QWORD *a1)
{
  _QWORD *v2; // rdi
  _QWORD *v3; // rdi
  unsigned __int64 v4; // rdi

  *a1 = &unk_49EF608;
  v2 = (_QWORD *)a1[28];
  if ( v2 != a1 + 30 )
    j_j___libc_free_0(v2, a1[30] + 1LL);
  v3 = (_QWORD *)a1[22];
  a1[21] = &unk_49E7488;
  if ( v3 != a1 + 24 )
    j_j___libc_free_0(v3, a1[24] + 1LL);
  v4 = a1[12];
  if ( v4 != a1[11] )
    _libc_free(v4);
}
