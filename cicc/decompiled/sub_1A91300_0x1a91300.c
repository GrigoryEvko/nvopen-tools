// Function: sub_1A91300
// Address: 0x1a91300
//
__int64 __fastcall sub_1A91300(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = off_49F5D48;
  v2 = a1[20];
  if ( v2 )
    j_j___libc_free_0(v2, a1[22] - v2);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 224);
}
