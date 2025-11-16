// Function: sub_19CF6C0
// Address: 0x19cf6c0
//
void *__fastcall sub_19CF6C0(_QWORD *a1)
{
  void (__fastcall *v1)(_QWORD *, _QWORD *, __int64); // rax
  void (__fastcall *v2)(_QWORD *, _QWORD *, __int64); // rax
  void (__fastcall *v3)(_QWORD *, _QWORD *, __int64); // rax

  *a1 = off_49F49B0;
  v1 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[32];
  if ( v1 )
    v1(a1 + 30, a1 + 30, 3);
  v2 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[28];
  if ( v2 )
    v2(a1 + 26, a1 + 26, 3);
  v3 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[24];
  if ( v3 )
    v3(a1 + 22, a1 + 22, 3);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
