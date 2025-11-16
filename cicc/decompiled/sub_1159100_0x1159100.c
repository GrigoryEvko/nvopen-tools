// Function: sub_1159100
// Address: 0x1159100
//
_BOOL8 __fastcall sub_1159100(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  int v3; // edx
  unsigned int v4; // ecx
  unsigned __int8 v5; // al
  _BOOL8 result; // rax
  __int64 *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rcx

  if ( *(_BYTE *)a2 <= 0x1Cu )
    return 0;
  switch ( *(_BYTE *)a2 )
  {
    case ')':
    case '+':
    case '-':
    case '/':
    case '2':
    case '5':
    case 'J':
    case 'K':
    case 'S':
      goto LABEL_6;
    case '*':
    case ',':
    case '.':
    case '0':
    case '1':
    case '3':
    case '4':
    case '6':
    case '7':
    case '8':
    case '9':
    case ':':
    case ';':
    case '<':
    case '=':
    case '>':
    case '?':
    case '@':
    case 'A':
    case 'B':
    case 'C':
    case 'D':
    case 'E':
    case 'F':
    case 'G':
    case 'H':
    case 'I':
    case 'L':
    case 'M':
    case 'N':
    case 'O':
    case 'P':
    case 'Q':
    case 'R':
      return 0;
    case 'T':
    case 'U':
    case 'V':
      v2 = *(_QWORD *)(a2 + 8);
      v3 = *(unsigned __int8 *)(v2 + 8);
      v4 = v3 - 17;
      v5 = *(_BYTE *)(v2 + 8);
      if ( (unsigned int)(v3 - 17) <= 1 )
        v5 = *(_BYTE *)(**(_QWORD **)(v2 + 16) + 8LL);
      if ( v5 <= 3u || v5 == 5 || (v5 & 0xFD) == 4 )
        goto LABEL_6;
      if ( (_BYTE)v3 == 15 )
      {
        if ( (*(_BYTE *)(v2 + 9) & 4) == 0 || !sub_BCB420(v2) )
          return 0;
        v7 = *(__int64 **)(v2 + 16);
        v2 = *v7;
        v3 = *(unsigned __int8 *)(*v7 + 8);
        v4 = v3 - 17;
      }
      else if ( (_BYTE)v3 == 16 )
      {
        do
        {
          v2 = *(_QWORD *)(v2 + 24);
          LOBYTE(v3) = *(_BYTE *)(v2 + 8);
        }
        while ( (_BYTE)v3 == 16 );
        v4 = (unsigned __int8)v3 - 17;
      }
      if ( v4 <= 1 )
        LOBYTE(v3) = *(_BYTE *)(**(_QWORD **)(v2 + 16) + 8LL);
      if ( (unsigned __int8)v3 > 3u && (_BYTE)v3 != 5 && (v3 & 0xFD) != 4 )
        return 0;
LABEL_6:
      result = (*(_BYTE *)(a2 + 1) & 2) != 0;
      if ( (*(_BYTE *)(a2 + 1) & 2) == 0 )
        return 0;
      if ( *(_BYTE *)a2 != 85 )
        return 0;
      v8 = *(_QWORD *)(a2 - 32);
      if ( !v8 )
        return 0;
      if ( *(_BYTE *)v8 )
        return 0;
      if ( *(_QWORD *)(v8 + 24) != *(_QWORD *)(a2 + 80) )
        return 0;
      if ( *(_DWORD *)(v8 + 36) != *(_DWORD *)a1 )
        return 0;
      v9 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      if ( *(_QWORD *)(a1 + 16) != *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 8) - v9)) )
        return 0;
      v10 = *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 24) - v9));
      if ( !v10 )
        return 0;
      **(_QWORD **)(a1 + 32) = v10;
      return result;
    default:
      return 0;
  }
}
