// Function: sub_34E2730
// Address: 0x34e2730
//
char __fastcall sub_34E2730(unsigned __int8 *a1, __int64 a2)
{
  _BYTE *v2; // rax
  __int64 v3; // r13
  int v4; // edx
  unsigned int v5; // ecx
  unsigned __int8 v6; // al
  int v7; // esi

  LODWORD(v2) = *a1;
  if ( (unsigned __int8)v2 > 0x1Cu )
  {
    LODWORD(v2) = (_DWORD)v2 - 41;
    switch ( (int)v2 )
    {
      case 0:
      case 2:
      case 4:
      case 6:
      case 9:
      case 12:
      case 33:
      case 34:
      case 42:
        goto LABEL_19;
      case 43:
      case 44:
      case 45:
        v3 = *((_QWORD *)a1 + 1);
        v4 = *(unsigned __int8 *)(v3 + 8);
        v5 = v4 - 17;
        v6 = *(_BYTE *)(v3 + 8);
        if ( (unsigned int)(v4 - 17) <= 1 )
          v6 = *(_BYTE *)(**(_QWORD **)(v3 + 16) + 8LL);
        if ( v6 <= 3u )
          goto LABEL_19;
        if ( v6 == 5 )
          goto LABEL_19;
        LOBYTE(v2) = v6 & 0xFD;
        if ( (_BYTE)v2 == 4 )
          goto LABEL_19;
        if ( (_BYTE)v4 == 15 )
        {
          if ( (*(_BYTE *)(v3 + 9) & 4) == 0 )
            return (char)v2;
          LOBYTE(v2) = sub_BCB420(*((_QWORD *)a1 + 1));
          if ( !(_BYTE)v2 )
            return (char)v2;
          v2 = *(_BYTE **)(v3 + 16);
          v3 = *(_QWORD *)v2;
          v4 = *(unsigned __int8 *)(*(_QWORD *)v2 + 8LL);
          v5 = v4 - 17;
        }
        else if ( (_BYTE)v4 == 16 )
        {
          do
          {
            v3 = *(_QWORD *)(v3 + 24);
            LOBYTE(v4) = *(_BYTE *)(v3 + 8);
          }
          while ( (_BYTE)v4 == 16 );
          v5 = (unsigned __int8)v4 - 17;
        }
        if ( v5 <= 1 )
        {
          v2 = **(_BYTE ***)(v3 + 16);
          LOBYTE(v4) = v2[8];
        }
        if ( (unsigned __int8)v4 <= 3u || (_BYTE)v4 == 5 || (v4 & 0xFD) == 4 )
        {
LABEL_19:
          LOBYTE(v2) = sub_920620(a2);
          if ( (_BYTE)v2 )
          {
            v7 = *(_BYTE *)(a2 + 1) >> 1;
            if ( v7 == 127 )
              v7 = -1;
            LOBYTE(v2) = sub_B45150((__int64)a1, v7);
          }
        }
        break;
      default:
        return (char)v2;
    }
  }
  return (char)v2;
}
