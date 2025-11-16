// Function: sub_14CF780
// Address: 0x14cf780
//
char __fastcall sub_14CF780(__int64 a1, unsigned int a2)
{
  char v2; // r13
  char result; // al

  v2 = sub_15FF7F0(a1);
  if ( v2 == (unsigned __int8)sub_15FF7F0(a2) || (unsigned __int8)sub_15FF7F0((unsigned int)a1) && a2 - 32 <= 1 )
    return 1;
  result = sub_15FF7F0(a2);
  if ( result )
    return (unsigned int)(a1 - 32) <= 1;
  return result;
}
