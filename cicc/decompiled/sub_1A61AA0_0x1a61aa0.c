// Function: sub_1A61AA0
// Address: 0x1a61aa0
//
void *__fastcall sub_1A61AA0(_QWORD *a1)
{
  void (__fastcall *v1)(_QWORD *, _QWORD *, __int64); // rax

  *a1 = off_49F5528;
  v1 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[25];
  if ( v1 )
    v1(a1 + 23, a1 + 23, 3);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
