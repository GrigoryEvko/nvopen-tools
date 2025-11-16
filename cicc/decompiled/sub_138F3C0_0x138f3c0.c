// Function: sub_138F3C0
// Address: 0x138f3c0
//
__int64 __fastcall sub_138F3C0(_QWORD *a1)
{
  __int64 v1; // r13

  v1 = a1[20];
  *a1 = &unk_49E8EC8;
  if ( v1 )
  {
    sub_138F1A0(v1);
    j_j___libc_free_0(v1, 56);
  }
  sub_16367B0(a1);
  return j_j___libc_free_0(a1, 168);
}
