// Function: sub_11FAEC0
// Address: 0x11faec0
//
char __fastcall sub_11FAEC0(int a1, int a2)
{
  bool v2; // r13
  char result; // al

  v2 = sub_B532B0(a1);
  if ( v2 == sub_B532B0(a2) || sub_B532B0(a1) && (unsigned int)(a2 - 32) <= 1 )
    return 1;
  result = sub_B532B0(a2);
  if ( result )
    return (unsigned int)(a1 - 32) <= 1;
  return result;
}
