// Function: sub_1A01AF0
// Address: 0x1a01af0
//
__int64 __fastcall sub_1A01AF0(__int64 *a1)
{
  _QWORD *v2; // r12
  _QWORD *v3; // rbx
  __int64 v4; // rdi

  v2 = a1 + 38;
  v3 = a1 + 110;
  *a1 = (__int64)off_49F5068;
  do
  {
    v4 = v3[1];
    v3 -= 4;
    j___libc_free_0(v4);
  }
  while ( v3 != v2 );
  sub_1A019D0(a1 + 32);
  j___libc_free_0(a1[29]);
  j___libc_free_0(a1[25]);
  j___libc_free_0(a1[21]);
  *a1 = (__int64)&unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 920);
}
