// Function: sub_22FF640
// Address: 0x22ff640
//
void __fastcall sub_22FF640(unsigned __int64 a1)
{
  __int64 v2; // rdi

  v2 = a1 + 8;
  *(_QWORD *)(v2 - 8) = &unk_4A0AF70;
  sub_22C31B0(v2);
  j_j___libc_free_0(a1);
}
