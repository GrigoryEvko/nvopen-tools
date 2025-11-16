// Function: sub_386EC30
// Address: 0x386ec30
//
__int64 __fastcall sub_386EC30(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rax
  _QWORD *v6; // rax

  if ( !a1 )
    return a2;
  if ( a2 )
  {
    if ( a1 != a2 )
    {
      v4 = (_QWORD *)a2;
      while ( 1 )
      {
        v4 = (_QWORD *)*v4;
        if ( (_QWORD *)a1 == v4 )
          break;
        if ( !v4 )
        {
          if ( a2 == a1 )
            return a2;
          v6 = (_QWORD *)a1;
          while ( 1 )
          {
            v6 = (_QWORD *)*v6;
            if ( (_QWORD *)a2 == v6 )
              break;
            if ( !v6 )
            {
              if ( sub_15CC8F0(a3, **(_QWORD **)(a1 + 32), **(_QWORD **)(a2 + 32)) )
                return a2;
              sub_15CC8F0(a3, **(_QWORD **)(a2 + 32), **(_QWORD **)(a1 + 32));
              return a1;
            }
          }
          return a1;
        }
      }
    }
    return a2;
  }
  return a1;
}
