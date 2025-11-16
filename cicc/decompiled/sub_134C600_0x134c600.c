// Function: sub_134C600
// Address: 0x134c600
//
__int64 __fastcall sub_134C600(_QWORD *a1)
{
  void (__fastcall *v1)(_QWORD *, _QWORD *, __int64); // rax

  *a1 = off_49E8078;
  v1 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[22];
  if ( v1 )
    v1(a1 + 20, a1 + 20, 3);
  return sub_16367B0(a1);
}
