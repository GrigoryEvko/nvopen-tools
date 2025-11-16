// Function: sub_2C180B0
// Address: 0x2c180b0
//
void __fastcall sub_2C180B0(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi

  *(a1 - 12) = &unk_4A24E58;
  *a1 = &unk_4A24EE0;
  v2 = a1 + 9;
  *(a1 - 7) = &unk_4A24EA8;
  v3 = a1[7];
  if ( (_QWORD *)v3 != v2 )
    j_j___libc_free_0(v3);
  *(a1 - 12) = &unk_4A23FE8;
  *(a1 - 7) = &unk_4A24030;
  *a1 = &unk_4A24068;
  sub_2C17120((__int64)(a1 - 12));
}
