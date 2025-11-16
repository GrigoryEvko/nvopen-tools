// Function: sub_F08660
// Address: 0xf08660
//
char __fastcall sub_F08660(unsigned __int8 *a1)
{
  _BYTE *v1; // rax
  __int64 v2; // rbx
  int v3; // edx
  unsigned int v4; // ecx
  unsigned __int8 v5; // al
  int v6; // eax

  LODWORD(v1) = *a1;
  if ( (unsigned __int8)v1 > 0x1Cu )
  {
    LODWORD(v1) = (_DWORD)v1 - 41;
    switch ( (int)v1 )
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
        goto LABEL_18;
      case 43:
      case 44:
      case 45:
        v2 = *((_QWORD *)a1 + 1);
        v3 = *(unsigned __int8 *)(v2 + 8);
        v4 = v3 - 17;
        v5 = *(_BYTE *)(v2 + 8);
        if ( (unsigned int)(v3 - 17) <= 1 )
          v5 = *(_BYTE *)(**(_QWORD **)(v2 + 16) + 8LL);
        if ( v5 <= 3u )
          goto LABEL_18;
        if ( v5 == 5 )
          goto LABEL_18;
        LOBYTE(v1) = v5 & 0xFD;
        if ( (_BYTE)v1 == 4 )
          goto LABEL_18;
        if ( (_BYTE)v3 == 15 )
        {
          if ( (*(_BYTE *)(v2 + 9) & 4) == 0 )
            break;
          LOBYTE(v1) = sub_BCB420(*((_QWORD *)a1 + 1));
          if ( !(_BYTE)v1 )
            break;
          v1 = *(_BYTE **)(v2 + 16);
          v2 = *(_QWORD *)v1;
          v3 = *(unsigned __int8 *)(*(_QWORD *)v1 + 8LL);
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
        {
          v1 = **(_BYTE ***)(v2 + 16);
          LOBYTE(v3) = v1[8];
        }
        if ( (unsigned __int8)v3 <= 3u || (_BYTE)v3 == 5 || (v3 & 0xFD) == 4 )
        {
LABEL_18:
          v6 = sub_B45210((__int64)a1);
          a1[1] &= 1u;
          LOBYTE(v1) = sub_B45150((__int64)a1, v6);
          return (char)v1;
        }
        break;
      default:
        break;
    }
  }
  a1[1] &= 1u;
  return (char)v1;
}
