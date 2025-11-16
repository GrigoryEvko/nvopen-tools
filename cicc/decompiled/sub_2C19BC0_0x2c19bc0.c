// Function: sub_2C19BC0
// Address: 0x2c19bc0
//
void __fastcall sub_2C19BC0(_QWORD *a1)
{
  unsigned __int64 v1; // r12
  _QWORD *v3; // rax
  unsigned __int64 v4; // rdi

  v1 = (unsigned __int64)(a1 - 12);
  *(a1 - 12) = &unk_4A23B70;
  *a1 = &unk_4A23BF0;
  v3 = a1 + 11;
  *(a1 - 7) = &unk_4A23BB8;
  v4 = a1[9];
  if ( (_QWORD *)v4 != v3 )
    j_j___libc_free_0(v4);
  *(a1 - 12) = &unk_4A23258;
  *(a1 - 7) = &unk_4A23290;
  *a1 = &unk_4A232C8;
  sub_2C17120(v1);
  j_j___libc_free_0(v1);
}
