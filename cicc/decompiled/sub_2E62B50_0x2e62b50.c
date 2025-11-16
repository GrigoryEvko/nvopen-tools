// Function: sub_2E62B50
// Address: 0x2e62b50
//
void __fastcall sub_2E62B50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // r13
  __int64 *v6; // rax
  __int64 *v7; // rax

  v5 = (__int64 *)sub_2E5E6D0(a1, a3);
  v6 = (__int64 *)sub_2E5E6D0(a1, a2);
  v7 = sub_2E5E740(a1, v6, v5);
  if ( v7 )
  {
    sub_2E62640(a1, a4, (__int64)v7);
    nullsub_1600();
  }
}
