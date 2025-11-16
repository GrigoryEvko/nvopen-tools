// Function: sub_22E2BA0
// Address: 0x22e2ba0
//
__int64 __fastcall sub_22E2BA0(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v3; // rdi
  unsigned __int64 *v4; // rbx
  unsigned __int64 v5; // r13
  unsigned __int64 v6; // rdi

  *a1 = &unk_4A0A190;
  a1[22] = &unk_4A0A248;
  v3 = a1[71];
  if ( v3 )
  {
    v4 = (unsigned __int64 *)a1[76];
    v5 = a1[80] + 8LL;
    if ( v5 > (unsigned __int64)v4 )
    {
      do
      {
        v6 = *v4++;
        j_j___libc_free_0(v6);
      }
      while ( v5 > (unsigned __int64)v4 );
      v3 = a1[71];
    }
    a2 = 8LL * a1[72];
    j_j___libc_free_0(v3);
  }
  sub_B81E70((__int64)(a1 + 22), a2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
