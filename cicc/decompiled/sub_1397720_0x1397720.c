// Function: sub_1397720
// Address: 0x1397720
//
__int64 __fastcall sub_1397720(_QWORD *a1)
{
  __int64 v1; // r13

  v1 = a1[20];
  *a1 = &unk_49E8FA0;
  if ( v1 )
  {
    sub_1397630(v1);
    j_j___libc_free_0(v1, 72);
  }
  return sub_1636790(a1);
}
