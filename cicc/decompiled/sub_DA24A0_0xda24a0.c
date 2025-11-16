// Function: sub_DA24A0
// Address: 0xda24a0
//
__int64 __fastcall sub_DA24A0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13

  v2 = a1[22];
  *a1 = &unk_49DE938;
  if ( v2 )
  {
    sub_DA11D0(v2, a2);
    j_j___libc_free_0(v2, 1576);
  }
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
