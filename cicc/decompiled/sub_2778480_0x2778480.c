// Function: sub_2778480
// Address: 0x2778480
//
char __fastcall sub_2778480(__int64 a1)
{
  unsigned __int8 v1; // cl
  __int64 v2; // rax
  __int64 v3; // rsi
  __int64 v4; // rax
  unsigned __int8 v5; // cl
  __int64 v6; // rax
  unsigned int v7; // eax
  unsigned __int16 v8; // ax

  v1 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 == 85 )
  {
    v2 = *(_QWORD *)(a1 - 32);
    if ( v2 && !*(_BYTE *)v2 && (v3 = *(_QWORD *)(v2 + 24), *(_QWORD *)(a1 + 80) == v3) )
    {
      switch ( *(_DWORD *)(v2 + 36) )
      {
        case 0x66:
        case 0x67:
        case 0x68:
        case 0x69:
        case 0x6C:
        case 0x6F:
        case 0x70:
        case 0x72:
        case 0x73:
        case 0x88:
        case 0x8D:
          LOBYTE(v4) = (!((unsigned __int16)sub_B59EF0((unsigned __int8 *)a1, v3) >> 8)
                     || (v7 = sub_B59EF0((unsigned __int8 *)a1, v3), v3 = v7, LOWORD(v3) = BYTE1(v7), !BYTE1(v7))
                     || (_BYTE)v7 != 2)
                    && (!((unsigned __int16)sub_B59DB0((unsigned __int8 *)a1, v3) >> 8)
                     || (v8 = sub_B59DB0((unsigned __int8 *)a1, v3), !HIBYTE(v8))
                     || (_BYTE)v8 != 7);
          break;
        default:
          goto LABEL_5;
      }
    }
    else
    {
LABEL_5:
      LOBYTE(v4) = sub_B49E00(a1);
      if ( (_BYTE)v4 )
      {
        LOBYTE(v4) = 0;
        if ( *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) != 7 )
        {
          v6 = sub_B43CB0(a1);
          LODWORD(v4) = sub_B2D610(v6, 49) ^ 1;
        }
      }
    }
  }
  else
  {
    LOBYTE(v4) = v1 == 41 || (unsigned int)v1 - 67 <= 0xC;
    if ( !(_BYTE)v4 )
    {
      v5 = v1 - 42;
      if ( v5 <= 0x36u )
        return (0x5F13000003FFFFuLL >> v5) & 1;
    }
  }
  return v4;
}
