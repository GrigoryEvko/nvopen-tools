// Function: sub_1C43360
// Address: 0x1c43360
//
void *__fastcall sub_1C43360(_QWORD *a1)
{
  __int64 v2; // rbx
  __int64 v3; // rdi
  __int64 v4; // rdi

  v2 = a1[26];
  *a1 = off_49F7B30;
  while ( v2 )
  {
    sub_1C43190(*(_QWORD *)(v2 + 24));
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    j_j___libc_free_0(v3, 40);
  }
  v4 = a1[21];
  if ( v4 )
    j_j___libc_free_0(v4, a1[23] - v4);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
