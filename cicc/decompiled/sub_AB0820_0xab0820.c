// Function: sub_AB0820
// Address: 0xab0820
//
char __fastcall sub_AB0820(__int64 a1, __int64 a2)
{
  char result; // al

  if ( sub_AAF7D0(a1) || sub_AAF7D0(a2) || sub_AB0760(a1) && (unsigned __int8)sub_AB06D0(a2) )
    return 1;
  result = sub_AB06D0(a1);
  if ( result )
    return sub_AB0760(a2);
  return result;
}
