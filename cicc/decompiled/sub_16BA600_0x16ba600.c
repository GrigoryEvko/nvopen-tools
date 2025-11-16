// Function: sub_16BA600
// Address: 0x16ba600
//
void __fastcall sub_16BA600(_QWORD *a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  *a1 = &unk_49EF0D0;
  v2 = a1[21];
  if ( v2 )
    j_j___libc_free_0(v2, a1[23] - v2);
  v3 = a1[12];
  if ( v3 != a1[11] )
    _libc_free(v3);
}
