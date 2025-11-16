// Function: sub_BB5300
// Address: 0xbb5300
//
bool __fastcall sub_BB5300(__int64 a1)
{
  char v1; // dl
  _BOOL4 v2; // eax
  char v3; // al
  bool v4; // dl
  __int64 v5; // rbx
  int v6; // edx
  unsigned int v7; // ecx
  unsigned __int8 v8; // al
  __int64 *v9; // rax
  bool v11; // [rsp+8h] [rbp-48h]
  bool v12; // [rsp+8h] [rbp-48h]
  __int64 v13; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-38h]
  __int64 v15; // [rsp+20h] [rbp-30h]
  unsigned int v16; // [rsp+28h] [rbp-28h]
  char v17; // [rsp+30h] [rbp-20h]

  v1 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 <= 0x1Cu )
  {
    switch ( *(_WORD *)(a1 + 2) )
    {
      case 0xD:
      case 0xF:
      case 0x11:
      case 0x19:
        goto LABEL_9;
      case 0x13:
      case 0x14:
      case 0x1A:
      case 0x1B:
      case 0x1D:
      case 0x35:
LABEL_3:
        LOBYTE(v2) = (*(_BYTE *)(a1 + 1) & 2) != 0;
        return v2;
      case 0x22:
LABEL_14:
        LOBYTE(v2) = 1;
        if ( !(*(_BYTE *)(a1 + 1) >> 1) )
        {
          sub_BB52D0((__int64)&v13, a1);
          LOBYTE(v2) = v17;
          if ( v17 )
          {
            v17 = 0;
            if ( v16 > 0x40 && v15 )
            {
              v11 = v2;
              j_j___libc_free_0_0(v15);
              LOBYTE(v2) = v11;
            }
            if ( v14 > 0x40 && v13 )
            {
              v12 = v2;
              j_j___libc_free_0_0(v13);
              LOBYTE(v2) = v12;
            }
          }
        }
        return v2;
      case 0x26:
LABEL_6:
        if ( v1 != 67 )
          goto LABEL_7;
LABEL_9:
        v3 = *(_BYTE *)(a1 + 1);
        v4 = (v3 & 4) != 0;
        v2 = (v3 & 2) != 0;
        if ( !v2 )
          LOBYTE(v2) = v4;
        break;
      default:
LABEL_7:
        LOBYTE(v2) = 0;
        break;
    }
  }
  else
  {
    switch ( v1 )
    {
      case '*':
      case ',':
      case '.':
      case '6':
        goto LABEL_9;
      case '0':
      case '1':
      case '7':
      case '8':
      case ':':
      case 'R':
        goto LABEL_3;
      case '?':
        goto LABEL_14;
      case 'C':
        goto LABEL_6;
      case 'D':
      case 'H':
        LOBYTE(v2) = 0;
        if ( ((v1 - 68) & 0xFB) == 0 )
          LOBYTE(v2) = sub_B44910(a1);
        break;
      default:
        switch ( v1 )
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
            goto LABEL_36;
          case 'T':
          case 'U':
          case 'V':
            v5 = *(_QWORD *)(a1 + 8);
            v6 = *(unsigned __int8 *)(v5 + 8);
            v7 = v6 - 17;
            v8 = *(_BYTE *)(v5 + 8);
            if ( (unsigned int)(v6 - 17) <= 1 )
              v8 = *(_BYTE *)(**(_QWORD **)(v5 + 16) + 8LL);
            if ( v8 <= 3u || v8 == 5 || (v8 & 0xFD) == 4 )
              goto LABEL_36;
            if ( (_BYTE)v6 == 15 )
            {
              if ( (*(_BYTE *)(v5 + 9) & 4) == 0 || !(unsigned __int8)sub_BCB420(v5) )
                goto LABEL_7;
              v9 = *(__int64 **)(v5 + 16);
              v5 = *v9;
              v6 = *(unsigned __int8 *)(*v9 + 8);
              v7 = v6 - 17;
            }
            else if ( (_BYTE)v6 == 16 )
            {
              do
              {
                v5 = *(_QWORD *)(v5 + 24);
                LOBYTE(v6) = *(_BYTE *)(v5 + 8);
              }
              while ( (_BYTE)v6 == 16 );
              v7 = (unsigned __int8)v6 - 17;
            }
            if ( v7 <= 1 )
              LOBYTE(v6) = *(_BYTE *)(**(_QWORD **)(v5 + 16) + 8LL);
            if ( (unsigned __int8)v6 <= 3u || (_BYTE)v6 == 5 || (v6 & 0xFD) == 4 )
            {
LABEL_36:
              LOBYTE(v2) = ((*(_BYTE *)(a1 + 1) >> 1) & 6) != 0;
              return v2;
            }
            break;
          default:
            goto LABEL_7;
        }
        goto LABEL_7;
    }
  }
  return v2;
}
