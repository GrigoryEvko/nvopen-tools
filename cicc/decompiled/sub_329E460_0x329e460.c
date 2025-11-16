// Function: sub_329E460
// Address: 0x329e460
//
char __fastcall sub_329E460(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // rax
  char result; // al

  if ( sub_3265990(a4 + 216, a1)
    || sub_3265990(a4 + 144, a1)
    || sub_3265990(a4 + 72, a1)
    || (result = sub_3265990(a4, a1)) != 0 )
  {
    v5 = *(_QWORD *)(a1 + 56);
    if ( v5 )
    {
      v6 = 1;
      while ( 1 )
      {
        while ( *(_DWORD *)(v5 + 8) != a2 )
        {
          v5 = *(_QWORD *)(v5 + 32);
          if ( !v5 )
            return v6 ^ 1;
        }
        if ( !v6 )
          return 0;
        v7 = *(_QWORD *)(v5 + 32);
        if ( !v7 )
          break;
        if ( a2 == *(_DWORD *)(v7 + 8) )
          return 0;
        v5 = *(_QWORD *)(v7 + 32);
        v6 = 0;
        if ( !v5 )
          return v6 ^ 1;
      }
      return 1;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
