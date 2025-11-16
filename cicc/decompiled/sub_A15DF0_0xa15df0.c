// Function: sub_A15DF0
// Address: 0xa15df0
//
__int64 __fastcall sub_A15DF0(unsigned __int8 *a1)
{
  unsigned __int64 v2; // rax
  int v3; // edx
  __int64 v4; // rcx
  __int64 result; // rax
  unsigned __int16 v6; // ax
  int v7; // edi
  unsigned __int8 v8; // dl
  char v9; // dl
  __int64 v10; // rbx
  int v11; // ecx
  unsigned int v12; // esi
  unsigned __int8 v13; // dl
  __int64 *v14; // rax

  v2 = *a1;
  if ( (unsigned __int8)v2 > 0x1Cu )
  {
    v3 = (unsigned __int8)v2;
    if ( (unsigned __int8)v2 > 0x36u )
    {
      if ( (unsigned __int8)(v2 - 55) > 1u && (_BYTE)v2 != 58 )
        goto LABEL_16;
    }
    else
    {
      v4 = 0x40540000000000LL;
      if ( _bittest64(&v4, v2) )
        return (a1[1] >> 1) & 3;
      if ( (unsigned int)(unsigned __int8)v2 - 48 > 1 )
      {
LABEL_16:
        v7 = (unsigned __int8)v2 - 29;
        switch ( (char)v2 )
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
            goto LABEL_17;
          case 'T':
          case 'U':
          case 'V':
            v10 = *((_QWORD *)a1 + 1);
            v11 = *(unsigned __int8 *)(v10 + 8);
            v12 = v11 - 17;
            v13 = *(_BYTE *)(v10 + 8);
            if ( (unsigned int)(v11 - 17) <= 1 )
              v13 = *(_BYTE *)(**(_QWORD **)(v10 + 16) + 8LL);
            if ( v13 <= 3u || v13 == 5 || (v13 & 0xFD) == 4 )
              goto LABEL_17;
            if ( (_BYTE)v11 != 15 )
            {
              if ( (_BYTE)v11 == 16 )
              {
                do
                {
                  v10 = *(_QWORD *)(v10 + 24);
                  LOBYTE(v11) = *(_BYTE *)(v10 + 8);
                }
                while ( (_BYTE)v11 == 16 );
                v12 = (unsigned __int8)v11 - 17;
              }
LABEL_39:
              if ( v12 <= 1 )
                LOBYTE(v11) = *(_BYTE *)(**(_QWORD **)(v10 + 16) + 8LL);
              if ( (unsigned __int8)v11 <= 3u || (_BYTE)v11 == 5 || (v11 & 0xFD) == 4 )
              {
LABEL_17:
                v8 = a1[1];
                result = (unsigned __int8)(v8 << 6) & 0x80;
                v9 = v8 >> 1;
                if ( (v9 & 2) != 0 )
                  result |= 2uLL;
                if ( (v9 & 4) != 0 )
                  result |= 4uLL;
                if ( (v9 & 8) != 0 )
                  result |= 8uLL;
                if ( (v9 & 0x10) != 0 )
                  result |= 0x10uLL;
                if ( (v9 & 0x20) != 0 )
                  result |= 0x20uLL;
                if ( (v9 & 0x40) != 0 )
                  return result | 0x40;
                return result;
              }
              goto LABEL_44;
            }
            if ( (*(_BYTE *)(v10 + 9) & 4) == 0 )
              goto LABEL_47;
            if ( (unsigned __int8)sub_BCB420(*((_QWORD *)a1 + 1)) )
            {
              v14 = *(__int64 **)(v10 + 16);
              v10 = *v14;
              v11 = *(unsigned __int8 *)(*v14 + 8);
              v12 = v11 - 17;
              goto LABEL_39;
            }
LABEL_44:
            LOBYTE(v2) = *a1;
            if ( *a1 <= 0x1Cu )
            {
              if ( (_BYTE)v2 != 5 )
                return 0;
              v6 = *((_WORD *)a1 + 1);
              goto LABEL_13;
            }
            v3 = (unsigned __int8)v2;
LABEL_46:
            v7 = v3 - 29;
LABEL_47:
            if ( ((v7 - 39) & 0xFFFFFFFB) == 0 )
              return (unsigned __int8)sub_B44910(a1);
            if ( (_BYTE)v2 == 67 )
              return (a1[1] >> 1) & 3;
            if ( (_BYTE)v2 != 63 )
            {
              if ( (_BYTE)v2 == 82 )
                return (a1[1] & 2) != 0;
              return 0;
            }
            break;
          default:
            goto LABEL_46;
        }
        return (a1[1] >> 1) & 7;
      }
    }
    return (a1[1] & 2) != 0;
  }
  if ( (_BYTE)v2 != 5 )
    return 0;
  v6 = *((_WORD *)a1 + 1);
  if ( (v6 & 0xFFF7) == 0x11 || (v6 & 0xFFFD) == 0xD )
    return (a1[1] >> 1) & 3;
  if ( (unsigned int)v6 - 19 <= 1 || (unsigned __int16)(v6 - 26) <= 1u )
    return (a1[1] & 2) != 0;
LABEL_13:
  if ( v6 != 34 )
    return 0;
  return (a1[1] >> 1) & 7;
}
