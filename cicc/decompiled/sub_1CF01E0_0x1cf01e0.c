// Function: sub_1CF01E0
// Address: 0x1cf01e0
//
void *__fastcall sub_1CF01E0(_QWORD *a1)
{
  void (__fastcall *v1)(_QWORD *, _QWORD *, __int64); // rax

  *a1 = &unk_49F91A8;
  v1 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[22];
  if ( v1 )
    v1(a1 + 20, a1 + 20, 3);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
