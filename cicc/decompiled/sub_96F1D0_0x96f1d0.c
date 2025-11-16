// Function: sub_96F1D0
// Address: 0x96f1d0
//
__int64 __fastcall sub_96F1D0(__int64 a1)
{
  __int64 v1; // r12
  int v2; // edx
  unsigned int v3; // ecx
  unsigned __int8 v4; // al
  __int64 *v5; // rax

  if ( !a1 )
    return 0;
  if ( *(_BYTE *)a1 > 0x1Cu )
  {
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
        return a1;
      case 'T':
      case 'U':
      case 'V':
        v1 = *(_QWORD *)(a1 + 8);
        v2 = *(unsigned __int8 *)(v1 + 8);
        v3 = v2 - 17;
        v4 = *(_BYTE *)(v1 + 8);
        if ( (unsigned int)(v2 - 17) <= 1 )
          v4 = *(_BYTE *)(**(_QWORD **)(v1 + 16) + 8LL);
        if ( v4 <= 3u || v4 == 5 || (v4 & 0xFD) == 4 )
          return a1;
        if ( (_BYTE)v2 == 15 )
        {
          if ( (*(_BYTE *)(v1 + 9) & 4) == 0 || !(unsigned __int8)sub_BCB420(*(_QWORD *)(a1 + 8)) )
            return 0;
          v5 = *(__int64 **)(v1 + 16);
          v1 = *v5;
          v2 = *(unsigned __int8 *)(*v5 + 8);
          v3 = v2 - 17;
        }
        else if ( (_BYTE)v2 == 16 )
        {
          do
          {
            v1 = *(_QWORD *)(v1 + 24);
            LOBYTE(v2) = *(_BYTE *)(v1 + 8);
          }
          while ( (_BYTE)v2 == 16 );
          v3 = (unsigned __int8)v2 - 17;
        }
        if ( v3 <= 1 )
          LOBYTE(v2) = *(_BYTE *)(**(_QWORD **)(v1 + 16) + 8LL);
        if ( (unsigned __int8)v2 <= 3u || (_BYTE)v2 == 5 || (v2 & 0xFD) == 4 )
          return a1;
        break;
      default:
        return 0;
    }
  }
  return 0;
}
