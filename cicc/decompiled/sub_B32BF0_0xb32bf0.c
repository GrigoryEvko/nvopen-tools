// Function: sub_B32BF0
// Address: 0xb32bf0
//
void __fastcall sub_B32BF0(_QWORD *a1)
{
  void (__fastcall *v1)(_QWORD *, _QWORD *, __int64); // rax

  *a1 = &unk_49DA0D8;
  v1 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[3];
  if ( v1 )
    v1(a1 + 1, a1 + 1, 3);
  nullsub_61();
}
