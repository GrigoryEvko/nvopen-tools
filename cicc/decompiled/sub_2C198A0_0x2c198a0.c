// Function: sub_2C198A0
// Address: 0x2c198a0
//
void __fastcall sub_2C198A0(_QWORD *a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 v2; // r8

  v1 = (unsigned __int64)(a1 - 5);
  v2 = a1[14];
  *(a1 - 5) = &unk_4A24D28;
  a1[7] = &unk_4A24D98;
  *a1 = &unk_4A24D60;
  if ( (_QWORD *)v2 != a1 + 16 )
    j_j___libc_free_0(v2);
  sub_2C17120(v1);
  j_j___libc_free_0(v1);
}
