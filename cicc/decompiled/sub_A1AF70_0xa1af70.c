// Function: sub_A1AF70
// Address: 0xa1af70
//
void __fastcall sub_A1AF70(__int64 a1, __int64 a2, unsigned int a3)
{
  char v3; // al
  unsigned int v4; // esi

  v3 = (*(_BYTE *)(a2 + 8) >> 1) & 7;
  switch ( v3 )
  {
    case 2:
      if ( *(_QWORD *)a2 )
        sub_A17DE0(a1, a3, *(_QWORD *)a2);
      break;
    case 4:
      if ( (unsigned __int8)(a3 - 97) > 0x19u )
      {
        if ( (unsigned __int8)(a3 - 65) <= 0x19u )
        {
          v4 = (char)a3 - 39;
          goto LABEL_9;
        }
        v4 = (char)a3 + 4;
        if ( (unsigned __int8)(a3 - 48) <= 9u )
        {
LABEL_9:
          sub_A17B10(a1, v4, 6);
          return;
        }
        if ( (_BYTE)a3 == 46 )
        {
          v4 = 62;
          goto LABEL_9;
        }
        if ( (_BYTE)a3 == 95 )
        {
          v4 = 63;
          goto LABEL_9;
        }
LABEL_19:
        BUG();
      }
      sub_A17B10(a1, (char)a3 - 97, 6);
      break;
    case 1:
      if ( *(_QWORD *)a2 )
        sub_A17B10(a1, a3, *(_QWORD *)a2);
      break;
    default:
      goto LABEL_19;
  }
}
