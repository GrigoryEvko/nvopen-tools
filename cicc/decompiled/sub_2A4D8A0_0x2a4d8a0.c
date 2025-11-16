// Function: sub_2A4D8A0
// Address: 0x2a4d8a0
//
__int64 __fastcall sub_2A4D8A0(__int64 a1, char a2)
{
  __int64 v2; // rbx
  char v3; // r15
  __int64 v4; // r12
  char v5; // al
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax

  v2 = *(_QWORD *)(a1 + 16);
  if ( !v2 )
    return 1;
  v3 = 0;
  while ( 1 )
  {
    v4 = *(_QWORD *)(v2 + 24);
    v5 = *(_BYTE *)v4;
    if ( *(_BYTE *)v4 <= 0x1Cu )
      break;
    switch ( v5 )
    {
      case '=':
        if ( (*(_BYTE *)(v4 + 2) & 1) != 0 || *(_QWORD *)(v4 + 8) != *(_QWORD *)(a1 + 72) )
          return 0;
        break;
      case '>':
        v7 = *(_QWORD *)(v4 - 64);
        if ( !v7 )
          BUG();
        if ( a1 == v7 )
          return 0;
        v8 = *(_QWORD *)(a1 + 72);
        if ( *(_QWORD *)(v7 + 8) != v8 || a2 && (unsigned int)*(unsigned __int8 *)(v8 + 8) - 15 <= 1 && v3 )
          return 0;
        if ( (*(_BYTE *)(v4 + 2) & 1) != 0 )
          return 0;
        v3 = 1;
        break;
      case 'U':
        v9 = *(_QWORD *)(v4 - 32);
        if ( !v9 || *(_BYTE *)v9 || *(_QWORD *)(v9 + 24) != *(_QWORD *)(v4 + 80) || (*(_BYTE *)(v9 + 33) & 0x20) == 0 )
          return 0;
        if ( !sub_B46A10(*(_QWORD *)(v2 + 24)) && !sub_BD2BE0(v4) )
        {
          v10 = *(_QWORD *)(v4 - 32);
          if ( !v10 || *(_BYTE *)v10 || *(_QWORD *)(v10 + 24) != *(_QWORD *)(v4 + 80) )
            BUG();
          if ( *(_DWORD *)(v10 + 36) != 171 )
            return 0;
        }
        break;
      case 'N':
        if ( !(unsigned __int8)sub_98C600(*(_QWORD *)(v2 + 24)) )
          return 0;
        break;
      case '?':
        if ( !(unsigned __int8)sub_B4DCF0(*(_QWORD *)(v2 + 24)) || !(unsigned __int8)sub_98C600(v4) )
          return 0;
        break;
      default:
        if ( v5 != 79 || !(unsigned __int8)sub_98C5F0(*(_QWORD *)(v2 + 24)) )
          return 0;
        break;
    }
    v2 = *(_QWORD *)(v2 + 8);
    if ( !v2 )
      return 1;
  }
  return 0;
}
