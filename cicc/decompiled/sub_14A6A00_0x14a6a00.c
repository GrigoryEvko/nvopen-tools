// Function: sub_14A6A00
// Address: 0x14a6a00
//
__int64 __fastcall sub_14A6A00(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_49ECB20;
  v2 = a1[20];
  if ( v2 )
    j_j___libc_free_0(v2, 8);
  sub_16367B0(a1);
  return j_j___libc_free_0(a1, 168);
}
