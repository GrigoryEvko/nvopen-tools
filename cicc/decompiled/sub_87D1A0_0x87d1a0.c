// Function: sub_87D1A0
// Address: 0x87d1a0
//
__int64 __fastcall sub_87D1A0(__int64 a1, char *a2)
{
  char v2; // al
  __int64 v3; // r8

  v2 = *(_BYTE *)(a1 + 80);
  switch ( v2 )
  {
    case 2:
    case 8:
    case 12:
      goto LABEL_2;
    case 3:
    case 4:
    case 5:
    case 6:
      v3 = *(_QWORD *)(a1 + 88);
      v2 = 6;
      goto LABEL_3;
    case 7:
    case 9:
      if ( v2 == 7 )
      {
LABEL_2:
        v3 = *(_QWORD *)(a1 + 88);
      }
      else
      {
        if ( v2 != 9 )
          sub_721090();
        v3 = *(_QWORD *)(a1 + 88);
        v2 = 7;
      }
      goto LABEL_3;
    case 10:
    case 11:
      v3 = *(_QWORD *)(a1 + 88);
      v2 = 11;
      goto LABEL_3;
    case 19:
    case 20:
    case 21:
    case 22:
      v3 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 104LL);
      v2 = 59;
      goto LABEL_3;
    case 23:
      v3 = *(_QWORD *)(a1 + 88);
      v2 = 28;
LABEL_3:
      if ( !v3 )
        v2 = 0;
      break;
    default:
      v3 = 0;
      v2 = 0;
      break;
  }
  if ( a2 )
    *a2 = v2;
  return v3;
}
