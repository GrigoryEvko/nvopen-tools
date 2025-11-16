// Function: sub_D84440
// Address: 0xd84440
//
char __fastcall sub_D84440(__int64 a1, unsigned __int64 a2)
{
  char result; // al

  result = *(_BYTE *)(a1 + 24);
  if ( result )
    return *(_QWORD *)(a1 + 16) <= a2;
  return result;
}
