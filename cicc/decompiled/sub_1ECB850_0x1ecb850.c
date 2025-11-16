// Function: sub_1ECB850
// Address: 0x1ecb850
//
void *__fastcall sub_1ECB850(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rbx
  __int64 v4; // rdi
  __int64 v5; // rbx
  __int64 v6; // rdi

  *a1 = off_49FDDD0;
  v2 = a1[44];
  if ( v2 != a1[43] )
    _libc_free(v2);
  v3 = a1[38];
  while ( v3 )
  {
    sub_1ECB530(*(_QWORD *)(v3 + 24));
    v4 = v3;
    v3 = *(_QWORD *)(v3 + 16);
    j_j___libc_free_0(v4, 40);
  }
  v5 = a1[32];
  while ( v5 )
  {
    sub_1ECB530(*(_QWORD *)(v5 + 24));
    v6 = v5;
    v5 = *(_QWORD *)(v5 + 16);
    j_j___libc_free_0(v6, 40);
  }
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
