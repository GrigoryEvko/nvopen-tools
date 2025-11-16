// Function: sub_1749C90
// Address: 0x1749c90
//
_BOOL8 __fastcall sub_1749C90(__int64 a1, __int64 a2)
{
  _BOOL4 v2; // r12d
  __int64 v4; // rbx
  int v5; // r8d
  __int64 v6; // rax
  _QWORD *v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r13
  _QWORD *v12; // rax
  _QWORD *v13; // r13
  _QWORD *v14; // rbx

  if ( *(_BYTE *)(a1 + 16) <= 0x10u )
    return 1;
  v4 = a1;
  while ( 2 )
  {
    v2 = sub_1749B70(v4, a2);
    if ( v2 )
    {
      return 1;
    }
    else if ( (unsigned __int8)v5 > 0x17u )
    {
      v6 = *(_QWORD *)(v4 + 8);
      if ( v6 )
      {
        if ( !*(_QWORD *)(v6 + 8) )
        {
          switch ( v5 )
          {
            case '#':
            case '%':
            case '\'':
            case '2':
            case '3':
            case '4':
              if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
                v7 = *(_QWORD **)(v4 - 8);
              else
                v7 = (_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
              if ( !(unsigned __int8)sub_1749C90(*v7, a2) )
                return v2;
              if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
                v8 = *(_QWORD *)(v4 - 8);
              else
                v8 = v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
              v4 = *(_QWORD *)(v8 + 24);
              goto LABEL_16;
            case '<':
            case '=':
            case '>':
              return 1;
            case 'M':
              v11 = 3LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
              v12 = (_QWORD *)(v4 - v11 * 8);
              if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
                v12 = *(_QWORD **)(v4 - 8);
              v13 = &v12[v11];
              if ( v13 == v12 )
                return 1;
              v14 = v12;
              break;
            case 'O':
              if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
                v9 = *(_QWORD *)(v4 - 8);
              else
                v9 = v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
              if ( !(unsigned __int8)sub_1749C90(*(_QWORD *)(v9 + 24), a2) )
                return v2;
              if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
                v10 = *(_QWORD *)(v4 - 8);
              else
                v10 = v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
              v4 = *(_QWORD *)(v10 + 48);
LABEL_16:
              if ( *(_BYTE *)(v4 + 16) > 0x10u )
                continue;
              return 1;
            default:
              return v2;
          }
          while ( (unsigned __int8)sub_1749C90(*v14, a2) )
          {
            v14 += 3;
            if ( v13 == v14 )
              return 1;
          }
        }
      }
    }
    return v2;
  }
}
