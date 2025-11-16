// Function: sub_1403DC0
// Address: 0x1403dc0
//
__int64 __fastcall sub_1403DC0(_QWORD *a1)
{
  _QWORD *v1; // r14
  __int64 v3; // rdi
  __int64 *v4; // rbx
  unsigned __int64 v5; // r13
  __int64 v6; // rdi

  v1 = a1 - 20;
  *(a1 - 20) = &unk_49EADF8;
  *a1 = &unk_49EAEB0;
  v3 = a1[51];
  if ( v3 )
  {
    v4 = (__int64 *)a1[56];
    v5 = a1[60] + 8LL;
    if ( v5 > (unsigned __int64)v4 )
    {
      do
      {
        v6 = *v4++;
        j_j___libc_free_0(v6, 512);
      }
      while ( v5 > (unsigned __int64)v4 );
      v3 = a1[51];
    }
    j_j___libc_free_0(v3, 8LL * a1[52]);
  }
  sub_160F3F0(a1);
  *(a1 - 20) = &unk_49EE078;
  sub_16366C0(v1);
  return j_j___libc_free_0(v1, 672);
}
