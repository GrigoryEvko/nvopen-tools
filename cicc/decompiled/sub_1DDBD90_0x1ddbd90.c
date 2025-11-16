// Function: sub_1DDBD90
// Address: 0x1ddbd90
//
void *__fastcall sub_1DDBD90(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_49FB100;
  v2 = a1[29];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
