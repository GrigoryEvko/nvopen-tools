// Function: sub_1CFC5B0
// Address: 0x1cfc5b0
//
__int64 __fastcall sub_1CFC5B0(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi

  *a1 = off_49F9420;
  j___libc_free_0(a1[87]);
  v2 = a1[83];
  if ( v2 )
    j_j___libc_free_0(v2, a1[85] - v2);
  v3 = a1[80];
  *a1 = &unk_49F9818;
  if ( v3 )
    j_j___libc_free_0(v3, a1[82] - v3);
  sub_1F012F0(a1);
  return j_j___libc_free_0(a1, 720);
}
