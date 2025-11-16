// Function: sub_13A3410
// Address: 0x13a3410
//
__int64 __fastcall sub_13A3410(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_49E96B0;
  v2 = a1[20];
  if ( v2 )
    j_j___libc_free_0(v2, 48);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
