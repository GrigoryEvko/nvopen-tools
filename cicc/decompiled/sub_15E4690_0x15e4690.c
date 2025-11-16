// Function: sub_15E4690
// Address: 0x15e4690
//
char __fastcall sub_15E4690(__int64 a1, int a2)
{
  char result; // al

  if ( !a1 )
    return a2 != 0;
  result = sub_15E4640(a1);
  if ( !result )
    return a2 != 0;
  return result;
}
