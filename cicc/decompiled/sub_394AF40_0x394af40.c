// Function: sub_394AF40
// Address: 0x394af40
//
void __fastcall sub_394AF40(_QWORD *a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 v3; // rdi

  v1 = a1[96];
  *a1 = &unk_4A3F140;
  if ( v1 )
  {
    j___libc_free_0(*(_QWORD *)(v1 + 8));
    j_j___libc_free_0(v1);
  }
  v3 = a1[88];
  if ( (_QWORD *)v3 != a1 + 90 )
    j_j___libc_free_0(v3);
  j___libc_free_0(a1[55]);
}
