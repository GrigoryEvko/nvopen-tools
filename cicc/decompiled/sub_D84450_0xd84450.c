// Function: sub_D84450
// Address: 0xd84450
//
char __fastcall sub_D84450(__int64 a1, unsigned __int64 a2)
{
  char result; // al

  result = *(_BYTE *)(a1 + 40);
  if ( result )
    return *(_QWORD *)(a1 + 32) >= a2;
  return result;
}
