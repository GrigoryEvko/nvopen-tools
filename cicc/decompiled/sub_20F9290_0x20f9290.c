// Function: sub_20F9290
// Address: 0x20f9290
//
__int64 __fastcall sub_20F9290(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi

  *a1 = &unk_4A00AB0;
  v2 = a1[6];
  if ( v2 )
    j_j___libc_free_0(v2, a1[8] - v2);
  v3 = a1[3];
  if ( v3 )
    j_j___libc_free_0(v3, a1[5] - v3);
  return j_j___libc_free_0(a1, 80);
}
