// Function: sub_F459D0
// Address: 0xf459d0
//
__int64 __fastcall sub_F459D0(__int64 a1, int a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v5; // rax

  v3 = 0;
  if ( a2 <= 1 )
  {
    v5 = sub_B92180(a1);
    v3 = v5;
    if ( v5 )
      sub_AE8440(a3, v5);
    sub_F45470(a1, a3);
  }
  return v3;
}
