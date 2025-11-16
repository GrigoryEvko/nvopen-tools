// Function: sub_1441D60
// Address: 0x1441d60
//
char __fastcall sub_1441D60(__int64 a1, unsigned __int64 a2)
{
  char result; // al

  if ( *(_BYTE *)(a1 + 40) )
    return *(_QWORD *)(a1 + 32) >= a2;
  sub_1441BF0(a1);
  result = *(_BYTE *)(a1 + 40);
  if ( result )
    return *(_QWORD *)(a1 + 32) >= a2;
  return result;
}
