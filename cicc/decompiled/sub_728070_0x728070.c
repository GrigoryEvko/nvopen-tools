// Function: sub_728070
// Address: 0x728070
//
__int64 __fastcall sub_728070(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  int v4; // r13d
  __int64 v5; // rsi
  __int64 result; // rax
  _QWORD *v7; // rbx

  v3 = *(_QWORD *)(a2 + 104);
  if ( v3 )
  {
    if ( v3 == a1 )
    {
      return a2;
    }
    else
    {
      v4 = dword_4F077C4;
      while ( 1 )
      {
        if ( v4 == 2 && (unsigned __int8)(*(_BYTE *)(v3 + 140) - 9) <= 2u )
        {
          v5 = *(_QWORD *)(*(_QWORD *)(v3 + 168) + 152LL);
          if ( v5 )
          {
            if ( (*(_BYTE *)(v5 + 29) & 0x20) == 0 )
            {
              result = sub_728070(a1);
              if ( result )
                break;
            }
          }
        }
        v3 = *(_QWORD *)(v3 + 112);
        if ( !v3 )
          goto LABEL_12;
        if ( a1 == v3 )
          return a2;
      }
    }
  }
  else
  {
LABEL_12:
    v7 = *(_QWORD **)(a2 + 160);
    if ( v7 )
    {
      do
      {
        result = sub_728070(a1);
        if ( result )
          break;
        v7 = (_QWORD *)*v7;
      }
      while ( v7 );
    }
    else
    {
      return 0;
    }
  }
  return result;
}
