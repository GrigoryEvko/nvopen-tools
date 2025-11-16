// Function: sub_22DC6F0
// Address: 0x22dc6f0
//
void __fastcall sub_22DC6F0(__int64 a1)
{
  unsigned __int64 v1; // r12

  sub_22DC520(a1 + 40);
  v1 = *(_QWORD *)(a1 + 32);
  if ( v1 )
  {
    sub_22DBFB0(*(_QWORD *)(a1 + 32));
    j_j___libc_free_0(v1);
    *(_QWORD *)(a1 + 32) = 0;
  }
}
