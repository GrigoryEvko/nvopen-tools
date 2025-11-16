// Function: sub_2C17300
// Address: 0x2c17300
//
void __fastcall sub_2C17300(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi

  *a1 = &unk_4A24D28;
  a1[12] = &unk_4A24D98;
  v2 = a1 + 21;
  a1[5] = &unk_4A24D60;
  v3 = a1[19];
  if ( (_QWORD *)v3 != v2 )
    j_j___libc_free_0(v3);
  sub_2C17120((__int64)a1);
}
