// Function: sub_36CDDB0
// Address: 0x36cddb0
//
void __fastcall sub_36CDDB0(_QWORD *a1)
{
  void (__fastcall *v1)(_QWORD *, _QWORD *, __int64); // rax

  *a1 = &unk_49DD7D8;
  v1 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[24];
  if ( v1 )
    v1(a1 + 22, a1 + 22, 3);
  sub_BB9280((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
