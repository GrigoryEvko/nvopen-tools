// Function: sub_144ADD0
// Address: 0x144add0
//
__int64 __fastcall sub_144ADD0(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 *v3; // rbx
  unsigned __int64 v4; // r13
  __int64 v5; // rdi

  *a1 = &unk_49EBCC0;
  a1[20] = &unk_49EBD78;
  v2 = a1[71];
  if ( v2 )
  {
    v3 = (__int64 *)a1[76];
    v4 = a1[80] + 8LL;
    if ( v4 > (unsigned __int64)v3 )
    {
      do
      {
        v5 = *v3++;
        j_j___libc_free_0(v5, 512);
      }
      while ( v4 > (unsigned __int64)v3 );
      v2 = a1[71];
    }
    j_j___libc_free_0(v2, 8LL * a1[72]);
  }
  sub_160F3F0(a1 + 20);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 672);
}
