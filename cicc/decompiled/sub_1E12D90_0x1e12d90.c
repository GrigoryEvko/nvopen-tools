// Function: sub_1E12D90
// Address: 0x1e12d90
//
void *__fastcall sub_1E12D90(_QWORD *a1)
{
  void (__fastcall *v1)(_QWORD *, _QWORD *, __int64); // rax

  *a1 = off_49FB920;
  v1 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[31];
  if ( v1 )
    v1(a1 + 29, a1 + 29, 3);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
