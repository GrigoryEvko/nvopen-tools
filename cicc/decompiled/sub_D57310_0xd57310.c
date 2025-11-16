// Function: sub_D57310
// Address: 0xd57310
//
__int64 __fastcall sub_D57310(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 *v4; // rbx
  unsigned __int64 v5; // r13
  __int64 v6; // rdi

  *(a1 - 22) = &unk_49DE1D0;
  *a1 = &unk_49DE288;
  v3 = a1[49];
  if ( v3 )
  {
    v4 = (__int64 *)a1[54];
    v5 = a1[58] + 8LL;
    if ( v5 > (unsigned __int64)v4 )
    {
      do
      {
        v6 = *v4++;
        j_j___libc_free_0(v6, 512);
      }
      while ( v5 > (unsigned __int64)v4 );
      v3 = a1[49];
    }
    a2 = 8LL * a1[50];
    j_j___libc_free_0(v3, a2);
  }
  sub_B81E70((__int64)a1, a2);
  *(a1 - 22) = &unk_49DAF80;
  return sub_BB9100((__int64)(a1 - 22));
}
