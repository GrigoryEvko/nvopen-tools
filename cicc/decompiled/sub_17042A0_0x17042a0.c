// Function: sub_17042A0
// Address: 0x17042a0
//
void *__fastcall sub_17042A0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_49EFFD8;
  j___libc_free_0(a1[279]);
  v2 = a1[20];
  if ( (_QWORD *)v2 != a1 + 22 )
    _libc_free(v2);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
