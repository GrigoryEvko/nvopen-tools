// Function: sub_22FF750
// Address: 0x22ff750
//
void __fastcall sub_22FF750(unsigned __int64 a1, __int64 a2)
{
  __int64 v3; // rdi

  v3 = a1 + 8;
  *(_QWORD *)(v3 - 8) = &unk_4A0AE30;
  sub_DA11D0(v3, a2);
  j_j___libc_free_0(a1);
}
