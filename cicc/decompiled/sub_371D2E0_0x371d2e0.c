// Function: sub_371D2E0
// Address: 0x371d2e0
//
void __fastcall sub_371D2E0(_QWORD *a1)
{
  void (__fastcall *v1)(_QWORD *, _QWORD *, __int64); // rax

  *a1 = &unk_4A3D0A0;
  v1 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[24];
  if ( v1 )
    v1(a1 + 22, a1 + 22, 3);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
