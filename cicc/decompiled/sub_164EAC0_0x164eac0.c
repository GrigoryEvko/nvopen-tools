// Function: sub_164EAC0
// Address: 0x164eac0
//
__int64 __fastcall sub_164EAC0(_QWORD *a1)
{
  __int64 v1; // r13

  v1 = a1[20];
  *a1 = off_49EE2D8;
  if ( v1 )
  {
    sub_164DC80(v1);
    j_j___libc_free_0(v1, 1672);
  }
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 176);
}
