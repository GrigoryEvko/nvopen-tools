// Function: sub_DF58F0
// Address: 0xdf58f0
//
char __fastcall sub_DF58F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  char result; // al

  v7 = a3;
  LOBYTE(a3) = a3 == 0;
  result = a3 | (a2 == 0);
  if ( !result )
    return sub_DF54F0(a2, v7, a3, a4, a5, a6);
  return result;
}
