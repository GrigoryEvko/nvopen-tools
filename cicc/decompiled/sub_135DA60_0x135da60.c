// Function: sub_135DA60
// Address: 0x135da60
//
__int64 __fastcall sub_135DA60(_QWORD *a1)
{
  __int64 v1; // r13
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v1 = a1[20];
  *a1 = &unk_49E86C8;
  if ( v1 )
  {
    v3 = *(_QWORD *)(v1 + 904);
    if ( v3 != *(_QWORD *)(v1 + 896) )
      _libc_free(v3);
    v4 = *(_QWORD *)(v1 + 800);
    if ( v4 != *(_QWORD *)(v1 + 792) )
      _libc_free(v4);
    if ( (*(_BYTE *)(v1 + 72) & 1) == 0 )
      j___libc_free_0(*(_QWORD *)(v1 + 80));
    j_j___libc_free_0(v1, 1056);
  }
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 168);
}
