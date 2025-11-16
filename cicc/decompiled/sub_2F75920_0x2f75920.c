// Function: sub_2F75920
// Address: 0x2f75920
//
char __fastcall sub_2F75920(__int64 a1, __int64 a2)
{
  char result; // al
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9

  if ( sub_2F753A0(a1) || (result = sub_2F753D0(a1)) != 0 )
  {
    if ( sub_2F753D0(a1) )
    {
      result = sub_2F753A0(a1);
      if ( !result )
        return sub_2F75570(a1, a2, v7, v8, v9, v10);
    }
    else
    {
      return sub_2F75730(a1, a2, v3, v4, v5, v6);
    }
  }
  return result;
}
