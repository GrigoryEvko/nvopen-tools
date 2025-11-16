// Function: sub_D57940
// Address: 0xd57940
//
__int64 __fastcall sub_D57940(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v4; // rdi
  __int64 *v5; // rbx
  unsigned __int64 v6; // r13
  __int64 v7; // rdi

  v2 = (__int64)(a1 - 22);
  *(a1 - 22) = &unk_49DE1D0;
  *a1 = &unk_49DE288;
  v4 = a1[49];
  if ( v4 )
  {
    v5 = (__int64 *)a1[54];
    v6 = a1[58] + 8LL;
    if ( v6 > (unsigned __int64)v5 )
    {
      do
      {
        v7 = *v5++;
        j_j___libc_free_0(v7, 512);
      }
      while ( v6 > (unsigned __int64)v5 );
      v4 = a1[49];
    }
    a2 = 8LL * a1[50];
    j_j___libc_free_0(v4, a2);
  }
  sub_B81E70((__int64)a1, a2);
  *(a1 - 22) = &unk_49DAF80;
  sub_BB9100(v2);
  return j_j___libc_free_0(v2, 672);
}
