// Function: sub_14038A0
// Address: 0x14038a0
//
__int64 __fastcall sub_14038A0(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 *v3; // rbx
  unsigned __int64 v4; // r13
  __int64 v5; // rdi

  *(a1 - 20) = &unk_49EADF8;
  *a1 = &unk_49EAEB0;
  v2 = a1[51];
  if ( v2 )
  {
    v3 = (__int64 *)a1[56];
    v4 = a1[60] + 8LL;
    if ( v4 > (unsigned __int64)v3 )
    {
      do
      {
        v5 = *v3++;
        j_j___libc_free_0(v5, 512);
      }
      while ( v4 > (unsigned __int64)v3 );
      v2 = a1[51];
    }
    j_j___libc_free_0(v2, 8LL * a1[52]);
  }
  sub_160F3F0(a1);
  *(a1 - 20) = &unk_49EE078;
  return sub_16366C0(a1 - 20);
}
