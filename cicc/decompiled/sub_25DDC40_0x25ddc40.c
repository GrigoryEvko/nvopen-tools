// Function: sub_25DDC40
// Address: 0x25ddc40
//
__int64 __fastcall sub_25DDC40(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // rax
  char v5; // al
  __int64 result; // rax
  __int64 v7; // rax
  char v8; // dl
  _BYTE **v9; // r12

  v2 = *(_QWORD *)(a1 + 16);
  if ( v2 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v3 = *(_QWORD *)(v2 + 24);
        if ( *(_BYTE *)v3 <= 0x1Cu )
          return 0;
        v4 = sub_B43CB0(*(_QWORD *)(v2 + 24));
        if ( sub_B2F070(v4, 0) )
          return 0;
        v5 = *(_BYTE *)v3;
        if ( *(_BYTE *)v3 <= 0x1Cu )
          return 0;
        if ( v5 != 61 )
          break;
LABEL_15:
        v2 = *(_QWORD *)(v2 + 8);
        if ( !v2 )
          return 1;
      }
      if ( v5 == 62 )
      {
        v7 = *(_QWORD *)(v3 - 64);
        if ( a1 != v7 )
          goto LABEL_15;
        if ( v7 )
          return 0;
        v2 = *(_QWORD *)(v2 + 8);
        if ( !v2 )
          return 1;
      }
      else
      {
        if ( v5 == 85 || v5 == 34 )
        {
          if ( a1 != *(_QWORD *)(v3 - 32) )
            return 0;
          goto LABEL_15;
        }
        if ( v5 != 79 && v5 != 63 )
        {
          if ( v5 == 84 )
          {
            sub_AE6EC0(a2, v3);
            if ( v8 )
            {
              result = sub_25DDC40(v3, a2);
              if ( !(_BYTE)result )
                return result;
            }
          }
          else
          {
            if ( v5 != 82 || sub_B532B0(*(_WORD *)(v3 + 2) & 0x3F) )
              return 0;
            v9 = (*(_BYTE *)(v3 + 7) & 0x40) != 0
               ? *(_BYTE ***)(v3 - 8)
               : (_BYTE **)(v3 - 32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF));
            if ( **v9 != 61 || *v9[4] != 20 )
              return 0;
          }
          goto LABEL_15;
        }
        if ( !(unsigned __int8)sub_25DDC40(v3, a2) )
          return 0;
        v2 = *(_QWORD *)(v2 + 8);
        if ( !v2 )
          return 1;
      }
    }
  }
  return 1;
}
