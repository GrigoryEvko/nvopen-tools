// Function: sub_1603DC0
// Address: 0x1603dc0
//
__int64 __fastcall sub_1603DC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v7; // rbx
  __int64 v8; // r14
  __int64 v9; // rsi

  v7 = (__int64 *)(a2 + 24);
  v8 = a2 + 24 + 8LL * *(unsigned int *)(a2 + 16);
  if ( a2 + 24 != v8 )
  {
    do
    {
      v9 = *v7++;
      sub_16BD4C0(a5, v9);
    }
    while ( (__int64 *)v8 != v7 );
  }
  return sub_16BD750(a5, a3);
}
