// Function: sub_28E5AD0
// Address: 0x28e5ad0
//
__int64 __fastcall sub_28E5AD0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_4A21C58;
  v2 = a1[22];
  if ( v2 )
    j_j___libc_free_0(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
