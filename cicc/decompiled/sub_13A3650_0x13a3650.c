// Function: sub_13A3650
// Address: 0x13a3650
//
__int64 __fastcall sub_13A3650(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_49E96B0;
  v2 = a1[20];
  if ( v2 )
    j_j___libc_free_0(v2, 48);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 168);
}
