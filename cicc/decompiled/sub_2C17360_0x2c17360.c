// Function: sub_2C17360
// Address: 0x2c17360
//
void __fastcall sub_2C17360(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi

  *(a1 - 12) = &unk_4A24D28;
  *a1 = &unk_4A24D98;
  v2 = a1 + 9;
  *(a1 - 7) = &unk_4A24D60;
  v3 = a1[7];
  if ( (_QWORD *)v3 != v2 )
    j_j___libc_free_0(v3);
  sub_2C17120((__int64)(a1 - 12));
}
