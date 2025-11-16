// Function: sub_FDB9D0
// Address: 0xfdb9d0
//
__int64 __fastcall sub_FDB9D0(_QWORD *a1)
{
  __int64 v1; // rax

  *a1 = &unk_49DB368;
  v1 = a1[3];
  if ( v1 != -4096 && v1 != 0 && v1 != -8192 )
    sub_BD60C0(a1 + 1);
  return j_j___libc_free_0(a1, 40);
}
