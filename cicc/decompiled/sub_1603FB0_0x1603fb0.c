// Function: sub_1603FB0
// Address: 0x1603fb0
//
__int64 __fastcall sub_1603FB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rbx
  __int64 result; // rax
  __int64 v5; // r13
  __int64 v7; // rsi

  v3 = (__int64 *)(a2 + 24);
  result = *(unsigned int *)(a2 + 16);
  v5 = a2 + 24 + 8 * result;
  if ( a2 + 24 != v5 )
  {
    do
    {
      v7 = *v3++;
      result = sub_16BD4C0(a3, v7);
    }
    while ( (__int64 *)v5 != v3 );
  }
  return result;
}
