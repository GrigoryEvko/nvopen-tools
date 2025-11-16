// Function: sub_149A210
// Address: 0x149a210
//
char __fastcall sub_149A210(__int64 a1, __int64 a2, __int64 a3)
{
  char result; // al

  result = a3 == 0 || a2 == 0;
  if ( !result )
    return sub_1499CE0(a2, a3);
  return result;
}
