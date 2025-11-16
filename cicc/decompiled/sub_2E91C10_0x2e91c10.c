// Function: sub_2E91C10
// Address: 0x2e91c10
//
__int64 __fastcall sub_2E91C10(_QWORD *a1)
{
  void (__fastcall *v1)(_QWORD *, _QWORD *, __int64); // rax

  *a1 = off_4A29080;
  v1 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[27];
  if ( v1 )
    v1(a1 + 25, a1 + 25, 3);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
