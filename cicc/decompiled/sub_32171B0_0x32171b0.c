// Function: sub_32171B0
// Address: 0x32171b0
//
__int64 __fastcall sub_32171B0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_4A35670;
  v2 = a1[4];
  if ( v2 )
    j_j___libc_free_0(v2);
  return sub_3252AD0(a1);
}
