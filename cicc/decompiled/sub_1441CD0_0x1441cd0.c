// Function: sub_1441CD0
// Address: 0x1441cd0
//
char __fastcall sub_1441CD0(__int64 a1, unsigned __int64 a2)
{
  char result; // al

  if ( *(_BYTE *)(a1 + 24) )
    return *(_QWORD *)(a1 + 16) <= a2;
  sub_1441BF0(a1);
  result = *(_BYTE *)(a1 + 24);
  if ( result )
    return *(_QWORD *)(a1 + 16) <= a2;
  return result;
}
