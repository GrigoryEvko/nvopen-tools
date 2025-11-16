// Function: sub_14221D0
// Address: 0x14221d0
//
__int64 __fastcall sub_14221D0(_QWORD *a1)
{
  __int64 v1; // r13

  v1 = a1[20];
  *a1 = &unk_49EB2E8;
  if ( v1 )
  {
    sub_1421ED0(v1);
    j_j___libc_free_0(v1, 344);
  }
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
