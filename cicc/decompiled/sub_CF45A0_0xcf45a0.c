// Function: sub_CF45A0
// Address: 0xcf45a0
//
__int64 __fastcall sub_CF45A0(_QWORD *a1)
{
  void (__fastcall *v1)(_QWORD *, _QWORD *, __int64); // rax

  *a1 = &unk_49DD7D8;
  v1 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[24];
  if ( v1 )
    v1(a1 + 22, a1 + 22, 3);
  return sub_BB9280((__int64)a1);
}
