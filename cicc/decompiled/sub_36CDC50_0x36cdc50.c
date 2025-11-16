// Function: sub_36CDC50
// Address: 0x36cdc50
//
__int64 __fastcall sub_36CDC50(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A3B1C0;
  v2 = a1[22];
  if ( v2 )
    j_j___libc_free_0(v2);
  return sub_BB9280((__int64)a1);
}
