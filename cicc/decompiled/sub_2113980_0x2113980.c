// Function: sub_2113980
// Address: 0x2113980
//
__int64 __fastcall sub_2113980(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi

  *a1 = &unk_4A01048;
  v2 = a1[9];
  if ( v2 )
    j_j___libc_free_0_0(v2);
  v3 = a1[6];
  if ( v3 )
    j_j___libc_free_0_0(v3);
  nullsub_751();
  return j_j___libc_free_0(a1, 96);
}
