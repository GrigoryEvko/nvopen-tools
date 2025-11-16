// Function: sub_14980B0
// Address: 0x14980b0
//
__int64 __fastcall sub_14980B0(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_49EC768;
  v2 = a1[20];
  if ( v2 )
    j_j___libc_free_0(v2, 16);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
