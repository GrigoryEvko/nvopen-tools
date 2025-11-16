// Function: sub_3568BD0
// Address: 0x3568bd0
//
void __fastcall sub_3568BD0(__int64 a1)
{
  unsigned __int64 v1; // r12

  sub_3568A00(a1 + 40);
  v1 = *(_QWORD *)(a1 + 32);
  if ( v1 )
  {
    sub_3568740(*(_QWORD *)(a1 + 32));
    j_j___libc_free_0(v1);
    *(_QWORD *)(a1 + 32) = 0;
  }
}
