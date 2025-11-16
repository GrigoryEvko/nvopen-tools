// Function: sub_31C3110
// Address: 0x31c3110
//
char __fastcall sub_31C3110(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r12
  char result; // al

  v2 = *a2;
  if ( sub_318B670(*a2) )
  {
    if ( v2 )
      return sub_31C2D80(*a1, v2);
    else
      return sub_318B640(0);
  }
  else
  {
    result = sub_318B640(v2);
    if ( v2 && result )
      return sub_31C2D80(*a1 + 88, v2);
  }
  return result;
}
