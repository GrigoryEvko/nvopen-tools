// Function: sub_B2F070
// Address: 0xb2f070
//
char __fastcall sub_B2F070(__int64 a1, int a2)
{
  char result; // al

  if ( !a1 )
    return a2 != 0;
  result = sub_B2F060(a1);
  if ( !result )
    return a2 != 0;
  return result;
}
