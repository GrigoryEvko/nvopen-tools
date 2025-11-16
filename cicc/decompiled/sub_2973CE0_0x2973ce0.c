// Function: sub_2973CE0
// Address: 0x2973ce0
//
void __fastcall sub_2973CE0(_QWORD *a1)
{
  void (__fastcall *v1)(_QWORD *, _QWORD *, __int64); // rax

  *a1 = off_4A22140;
  v1 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[27];
  if ( v1 )
    v1(a1 + 25, a1 + 25, 3);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
