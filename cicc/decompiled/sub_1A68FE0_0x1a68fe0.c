// Function: sub_1A68FE0
// Address: 0x1a68fe0
//
void *__fastcall sub_1A68FE0(_QWORD *a1)
{
  __int64 v2; // rdi
  _QWORD *v3; // rbx
  _QWORD *v4; // rdi

  *a1 = off_49F5720;
  v2 = a1[27];
  if ( v2 )
    j_j___libc_free_0(v2, a1[29] - v2);
  v3 = (_QWORD *)a1[24];
  while ( a1 + 24 != v3 )
  {
    v4 = v3;
    v3 = (_QWORD *)*v3;
    j_j___libc_free_0(v4, 64);
  }
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
