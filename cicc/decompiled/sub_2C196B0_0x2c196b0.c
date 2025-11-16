// Function: sub_2C196B0
// Address: 0x2c196b0
//
void __fastcall sub_2C196B0(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi

  *a1 = &unk_4A23B70;
  a1[12] = &unk_4A23BF0;
  v2 = a1 + 23;
  a1[5] = &unk_4A23BB8;
  v3 = a1[21];
  if ( (_QWORD *)v3 != v2 )
    j_j___libc_free_0(v3);
  *a1 = &unk_4A23258;
  a1[5] = &unk_4A23290;
  a1[12] = &unk_4A232C8;
  sub_2C17120((__int64)a1);
}
