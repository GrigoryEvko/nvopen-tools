// Function: sub_18518A0
// Address: 0x18518a0
//
__int64 __fastcall sub_18518A0(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v5; // rdx
  _QWORD *v7; // rax
  _QWORD *v8; // rcx
  unsigned __int64 v9; // rsi
  __int64 v10; // r9

  v5 = a3 % a2;
  v7 = *(_QWORD **)(a1 + 8 * v5);
  if ( v7 )
  {
    v8 = (_QWORD *)*v7;
    if ( *v7 )
    {
      v9 = v8[1];
      v10 = 0;
      while ( 1 )
      {
        if ( a3 == v9 )
        {
          v8 = (_QWORD *)*v8;
          ++v10;
          if ( !v8 )
            return v10;
        }
        else
        {
          if ( v10 )
            return v10;
          v8 = (_QWORD *)*v8;
          if ( !v8 )
            return v10;
        }
        v9 = v8[1];
        if ( v5 != v9 % a2 )
          return v10;
      }
    }
  }
  return 0;
}
