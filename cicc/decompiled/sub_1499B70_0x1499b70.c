// Function: sub_1499B70
// Address: 0x1499b70
//
__int64 __fastcall sub_1499B70(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_49EC810;
  v2 = a1[20];
  if ( v2 )
    j_j___libc_free_0(v2, 8);
  sub_16367B0(a1);
  return j_j___libc_free_0(a1, 168);
}
