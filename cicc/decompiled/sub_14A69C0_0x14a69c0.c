// Function: sub_14A69C0
// Address: 0x14a69c0
//
__int64 __fastcall sub_14A69C0(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_49ECB20;
  v2 = a1[20];
  if ( v2 )
    j_j___libc_free_0(v2, 8);
  return sub_16367B0(a1);
}
