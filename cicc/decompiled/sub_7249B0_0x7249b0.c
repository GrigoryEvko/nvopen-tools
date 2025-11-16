// Function: sub_7249B0
// Address: 0x7249b0
//
void __fastcall sub_7249B0(__int64 a1, char a2)
{
  *(_BYTE *)(a1 + 177) &= 0xC0u;
  *(_BYTE *)(a1 + 176) = a2;
  switch ( a2 )
  {
    case 0:
    case 1:
    case 4:
    case 12:
      *(_QWORD *)(a1 + 184) = 0;
      break;
    case 2:
      return;
    case 3:
      *(_QWORD *)(a1 + 184) = 0;
      *(_QWORD *)(a1 + 192) = 0;
      *(_BYTE *)(a1 + 200) = 0;
      break;
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
      *(_BYTE *)(a1 + 200) &= ~1u;
      *(_QWORD *)(a1 + 184) = 0;
      *(_QWORD *)(a1 + 192) = 0;
      break;
    case 11:
      *(_QWORD *)(a1 + 184) = 0;
      *(_QWORD *)(a1 + 192) = 0;
      break;
    case 13:
      *(_BYTE *)(a1 + 192) &= ~1u;
      *(_QWORD *)(a1 + 184) = 0;
      break;
    default:
      sub_721090();
  }
}
