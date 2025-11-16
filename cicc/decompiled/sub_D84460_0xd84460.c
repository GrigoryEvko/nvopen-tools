// Function: sub_D84460
// Address: 0xd84460
//
char __fastcall sub_D84460(__int64 a1, __int64 a2)
{
  char result; // al
  unsigned __int64 v3; // [rsp-38h] [rbp-38h] BYREF
  char v4; // [rsp-28h] [rbp-28h]

  if ( !a2 )
    return 0;
  result = sub_B2D610(a2, 5);
  if ( !result )
  {
    if ( *(_QWORD *)(a1 + 8) )
    {
      sub_B2EE70((__int64)&v3, a2, 0);
      result = v4;
      if ( v4 )
        return sub_D84450(a1, v3);
    }
    else
    {
      return 0;
    }
  }
  return result;
}
