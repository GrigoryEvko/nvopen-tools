// Function: sub_2C19AA0
// Address: 0x2c19aa0
//
void __fastcall sub_2C19AA0(_QWORD *a1)
{
  unsigned __int64 v1; // r12
  _QWORD *v3; // rax
  unsigned __int64 v4; // rdi

  v1 = (unsigned __int64)(a1 - 12);
  *(a1 - 12) = &unk_4A24E58;
  *a1 = &unk_4A24EE0;
  v3 = a1 + 9;
  *(a1 - 7) = &unk_4A24EA8;
  v4 = a1[7];
  if ( (_QWORD *)v4 != v3 )
    j_j___libc_free_0(v4);
  *(a1 - 12) = &unk_4A23FE8;
  *(a1 - 7) = &unk_4A24030;
  *a1 = &unk_4A24068;
  sub_2C17120(v1);
  j_j___libc_free_0(v1);
}
