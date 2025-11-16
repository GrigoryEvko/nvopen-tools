// Function: sub_22FF810
// Address: 0x22ff810
//
void __fastcall sub_22FF810(unsigned __int64 a1)
{
  _QWORD *v2; // rdi

  v2 = (_QWORD *)(a1 + 8);
  *(v2 - 1) = &unk_4A0AD68;
  sub_DFE7B0(v2);
  j_j___libc_free_0(a1);
}
