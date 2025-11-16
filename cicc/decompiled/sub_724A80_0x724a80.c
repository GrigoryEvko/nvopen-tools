// Function: sub_724A80
// Address: 0x724a80
//
void __fastcall sub_724A80(__int64 a1, char a2)
{
  _OWORD *v2; // rax

  *(_BYTE *)(a1 + 173) = a2;
  switch ( a2 )
  {
    case 0:
    case 14:
      return;
    case 1:
      sub_620D80((_WORD *)(a1 + 176), 0);
      break;
    case 2:
      *(_BYTE *)(a1 + 196) &= ~1u;
      *(_QWORD *)(a1 + 176) = 0;
      *(_QWORD *)(a1 + 184) = 0;
      *(_DWORD *)(a1 + 192) = -1;
      break;
    case 3:
    case 5:
      *(_OWORD *)(a1 + 176) = 0;
      break;
    case 4:
      v2 = sub_7247C0(32);
      *(_QWORD *)(a1 + 176) = v2;
      *v2 = 0;
      v2[1] = 0;
      break;
    case 6:
      *(_BYTE *)(a1 + 176) = 1;
      *(_QWORD *)(a1 + 184) = 0;
      *(_QWORD *)(a1 + 192) = 0;
      *(_QWORD *)(a1 + 200) = 0;
      break;
    case 7:
      *(_BYTE *)(a1 + 192) &= 0xFCu;
      *(_QWORD *)(a1 + 176) = 0;
      *(_QWORD *)(a1 + 184) = 0;
      *(_QWORD *)(a1 + 200) = 0;
      break;
    case 8:
    case 9:
      *(_QWORD *)(a1 + 176) = 0;
      *(_QWORD *)(a1 + 184) = 0;
      break;
    case 10:
      *(_BYTE *)(a1 + 192) &= ~1u;
      *(_QWORD *)(a1 + 176) = 0;
      *(_QWORD *)(a1 + 184) = 0;
      *(_QWORD *)(a1 + 200) = 0;
      break;
    case 11:
      *(_QWORD *)(a1 + 176) = 0;
      *(_QWORD *)(a1 + 184) = 0;
      *(_BYTE *)(a1 + 192) = 0;
      break;
    case 12:
      sub_7249B0(a1, 0);
      break;
    case 13:
      *(_BYTE *)(a1 + 176) &= 0xF8u;
      *(_QWORD *)(a1 + 184) = 0;
      break;
    case 15:
      *(_BYTE *)(a1 + 176) = 0;
      *(_QWORD *)(a1 + 184) = 0;
      *(_DWORD *)(a1 + 192) = 0;
      break;
    default:
      sub_721090();
  }
}
