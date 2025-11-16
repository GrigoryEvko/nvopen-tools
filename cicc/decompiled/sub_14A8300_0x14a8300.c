// Function: sub_14A8300
// Address: 0x14a8300
//
char __fastcall sub_14A8300(__int64 a1, __int64 a2, __int64 a3)
{
  char result; // al

  if ( a2 == a3 )
    return 1;
  result = a3 == 0 || a2 == 0;
  if ( !result )
    return sub_14A8040(a2, a3, 0);
  return result;
}
