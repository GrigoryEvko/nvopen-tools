// Function: sub_22E32E0
// Address: 0x22e32e0
//
void __fastcall sub_22E32E0(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v2; // r14
  unsigned __int64 v4; // rdi
  unsigned __int64 *v5; // rbx
  unsigned __int64 v6; // r13
  unsigned __int64 v7; // rdi

  v2 = (unsigned __int64)(a1 - 22);
  *(a1 - 22) = &unk_4A0A190;
  *a1 = &unk_4A0A248;
  v4 = a1[49];
  if ( v4 )
  {
    v5 = (unsigned __int64 *)a1[54];
    v6 = a1[58] + 8LL;
    if ( v6 > (unsigned __int64)v5 )
    {
      do
      {
        v7 = *v5++;
        j_j___libc_free_0(v7);
      }
      while ( v6 > (unsigned __int64)v5 );
      v4 = a1[49];
    }
    a2 = 8LL * a1[50];
    j_j___libc_free_0(v4);
  }
  sub_B81E70((__int64)a1, a2);
  *(a1 - 22) = &unk_49DAF80;
  sub_BB9100(v2);
  j_j___libc_free_0(v2);
}
