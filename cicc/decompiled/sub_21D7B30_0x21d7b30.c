// Function: sub_21D7B30
// Address: 0x21d7b30
//
char __fastcall sub_21D7B30(__int64 a1, _QWORD *a2, int a3)
{
  int v3; // eax
  char result; // al

  v3 = *(_DWORD *)(*(_QWORD *)(a1 + 81552) + 82308LL);
  if ( v3 != -1 )
    return v3 > 0;
  if ( !a3 )
    return 0;
  result = 1;
  if ( *(_DWORD *)(a2[1] + 816LL) )
    return sub_21D7A90(a1, a2);
  return result;
}
