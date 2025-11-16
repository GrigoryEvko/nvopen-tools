// Function: sub_1F4C150
// Address: 0x1f4c150
//
char __fastcall sub_1F4C150(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v3; // rsi
  char result; // al
  _WORD *v5; // rax

  if ( sub_1F4B690(a1) )
  {
    v2 = *(unsigned __int16 *)(*(_QWORD *)(a2 + 16) + 6LL);
    v3 = 0;
    if ( sub_1F4B690(a1) )
      v3 = a1 + 72;
    return sub_38D7610(v2, v3);
  }
  else
  {
    result = sub_1F4B670(a1);
    if ( result )
    {
      v5 = sub_1F4B8B0(a1, a2);
      return sub_38D7430(*(_QWORD *)(a1 + 176), v5);
    }
  }
  return result;
}
