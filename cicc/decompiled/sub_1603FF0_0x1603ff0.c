// Function: sub_1603FF0
// Address: 0x1603ff0
//
__int64 __fastcall sub_1603FF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rsi

  v4 = (__int64 *)(a2 + 24);
  v5 = a2 + 24 + 8LL * *(unsigned int *)(a2 + 16);
  if ( a2 + 24 != v5 )
  {
    do
    {
      v6 = *v4++;
      sub_16BD4C0(a3, v6);
    }
    while ( (__int64 *)v5 != v4 );
  }
  return sub_16BDDB0(a3);
}
