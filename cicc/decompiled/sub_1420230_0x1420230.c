// Function: sub_1420230
// Address: 0x1420230
//
void __fastcall sub_1420230(_QWORD *a1)
{
  _QWORD *v1; // rbx
  unsigned __int64 v2; // rdi

  v1 = a1 + 8;
  *a1 = &unk_49EB390;
  j___libc_free_0(a1[265]);
  v2 = a1[6];
  if ( (_QWORD *)v2 != v1 )
    _libc_free(v2);
}
