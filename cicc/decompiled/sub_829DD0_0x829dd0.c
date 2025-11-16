// Function: sub_829DD0
// Address: 0x829dd0
//
__int64 __fastcall sub_829DD0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 result; // rax

  if ( a5 && (a5 == a1 || (unsigned int)sub_8D97D0(a5, a1, 0, a4, a5)) )
    return 1;
  if ( a4 )
  {
    if ( a4 != a2 )
    {
      if ( a2 )
      {
        if ( dword_4F07588 )
        {
          v7 = *(_QWORD *)(a2 + 32);
          if ( *(_QWORD *)(a4 + 32) == v7 )
          {
            if ( v7 )
              return 1;
          }
        }
        if ( sub_829CB0(a4, a1, 0) )
          return 1;
        return a2 && a3 && sub_829CB0(a2, a1, a3) != 0;
      }
      result = sub_829CB0(a4, a1, 0);
      if ( !result )
        return result;
    }
    return 1;
  }
  return a2 && a3 && sub_829CB0(a2, a1, a3) != 0;
}
