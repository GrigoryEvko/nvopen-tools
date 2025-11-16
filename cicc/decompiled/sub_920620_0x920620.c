// Function: sub_920620
// Address: 0x920620
//
__int64 __fastcall sub_920620(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rbx
  int v3; // edx
  unsigned int v4; // ecx
  unsigned __int8 v5; // al
  __int64 *v6; // rax

  if ( *(_BYTE *)a1 <= 0x1Cu )
    return 0;
  switch ( *(_BYTE *)a1 )
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
      return 1;
    case 'T':
    case 'U':
    case 'V':
      v2 = *(_QWORD *)(a1 + 8);
      v3 = *(unsigned __int8 *)(v2 + 8);
      v4 = v3 - 17;
      v5 = *(_BYTE *)(v2 + 8);
      if ( (unsigned int)(v3 - 17) <= 1 )
        v5 = *(_BYTE *)(**(_QWORD **)(v2 + 16) + 8LL);
      if ( v5 <= 3u || v5 == 5 || (v5 & 0xFD) == 4 )
        return 1;
      if ( (_BYTE)v3 == 15 )
      {
        if ( (*(_BYTE *)(v2 + 9) & 4) == 0 || !(unsigned __int8)sub_BCB420(*(_QWORD *)(a1 + 8)) )
          goto LABEL_20;
        v6 = *(__int64 **)(v2 + 16);
        v2 = *v6;
        v3 = *(unsigned __int8 *)(*v6 + 8);
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
      if ( (unsigned __int8)v3 <= 3u || (_BYTE)v3 == 5 || (v3 & 0xFD) == 4 )
        return 1;
LABEL_20:
      result = 0;
      break;
    default:
      return 0;
  }
  return result;
}
