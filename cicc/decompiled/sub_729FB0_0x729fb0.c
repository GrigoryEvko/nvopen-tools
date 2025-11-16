// Function: sub_729FB0
// Address: 0x729fb0
//
void __fastcall sub_729FB0(__int64 a1, __int64 a2, __int64 a3)
{
  char *v3; // rax
  char *v4; // rsi
  __int64 v5; // rax
  char v6; // al
  __int64 v7; // rax

  if ( !*(_QWORD *)(a1 - 16) )
  {
    if ( (unsigned __int8)a2 > 0x30u )
    {
      if ( (_BYTE)a2 == 65 )
      {
        a2 = 65;
      }
      else
      {
        if ( (_BYTE)a2 != 75 )
          return;
        a2 = 75;
      }
LABEL_9:
      if ( dword_4F07588 )
        v3 = *(char **)(a3 + 376);
      else
        v3 = (char *)&unk_4F06D00;
      v4 = &v3[16 * a2];
      v5 = *((_QWORD *)v4 + 1);
      if ( a1 != v5 )
      {
        if ( v5 )
          *(_QWORD *)(v5 - 16) = a1;
        else
          *(_QWORD *)v4 = a1;
        *((_QWORD *)v4 + 1) = a1;
      }
    }
    else if ( (unsigned __int8)a2 > 1u )
    {
      a2 = (unsigned __int8)a2;
      switch ( (char)a2 )
      {
        case 2:
          if ( *(_BYTE *)(a1 + 173) == 12 || (*(_QWORD *)(a1 + 168) & 0xFF0000100000LL) == 0x10000000000LL )
            goto LABEL_9;
          goto LABEL_25;
        case 3:
        case 27:
        case 48:
          goto LABEL_9;
        case 6:
          v6 = *(_BYTE *)(a1 + 140);
          if ( (unsigned __int8)(v6 - 9) <= 2u )
            return;
          if ( v6 == 2 )
          {
            if ( (*(_BYTE *)(a1 + 161) & 8) != 0 )
              return;
          }
          else
          {
            if ( v6 == 14 )
              goto LABEL_9;
            if ( v6 == 7 && *(_QWORD *)(*(_QWORD *)(a1 + 168) + 8LL) )
              return;
          }
LABEL_25:
          if ( !*(_QWORD *)(a1 + 8) && (*(_BYTE *)(a1 + 89) & 4) == 0 )
          {
            v7 = *(_QWORD *)(a1 + 40);
            if ( !v7 || *(_BYTE *)(v7 + 28) != 3 )
              goto LABEL_9;
          }
          break;
        case 37:
          if ( (*(_BYTE *)(*(_QWORD *)(a1 + 40) + 177LL) & 0x20) != 0 || !*(_QWORD *)(a1 + 56) )
            goto LABEL_9;
          return;
        default:
          return;
      }
    }
  }
}
