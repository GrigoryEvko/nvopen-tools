// Function: sub_7264E0
// Address: 0x7264e0
//
void __fastcall sub_7264E0(__int64 a1, int a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rax
  _BYTE *v4; // rax
  __int16 v5; // dx

  *(_BYTE *)(a1 + 24) = a2;
  *(_QWORD *)(a1 + 8) = 0;
  switch ( a2 )
  {
    case 0:
    case 16:
    case 19:
      return;
    case 1:
      *(_BYTE *)(a1 + 60) &= 0xF8u;
      *(_DWORD *)(a1 + 56) = 5496;
      *(_QWORD *)(a1 + 64) = 0;
      *(_QWORD *)(a1 + 72) = 0;
      break;
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 10:
    case 22:
    case 26:
    case 28:
    case 29:
    case 32:
    case 33:
      *(_QWORD *)(a1 + 56) = 0;
      *(_QWORD *)(a1 + 64) = 0;
      break;
    case 7:
      v4 = sub_7246D0(56);
      *(_QWORD *)(a1 + 56) = v4;
      v5 = *(_WORD *)v4;
      *((_QWORD *)v4 + 1) = 0;
      *((_QWORD *)v4 + 2) = 0;
      *((_QWORD *)v4 + 3) = 0;
      *(_WORD *)v4 = v5 & 0xFC00 | 1;
      *((_QWORD *)v4 + 4) = 0;
      *((_QWORD *)v4 + 5) = 0;
      *((_QWORD *)v4 + 6) = 0;
      break;
    case 8:
      v3 = sub_7246D0(24);
      *(_QWORD *)(a1 + 56) = v3;
      *v3 = 0;
      v3[1] = 0;
      v3[2] = 0;
      break;
    case 9:
      v2 = sub_7246D0(32);
      *(_QWORD *)(a1 + 56) = v2;
      *v2 = 0;
      v2[1] = 0;
      v2[2] = 0;
      v2[3] = 0;
      break;
    case 11:
    case 27:
    case 34:
      *(_BYTE *)(a1 + 64) &= ~1u;
      *(_QWORD *)(a1 + 56) = 0;
      break;
    case 12:
    case 13:
    case 14:
    case 15:
      *(_QWORD *)(a1 + 64) = 0;
      *(_WORD *)(a1 + 56) = 1;
      break;
    case 17:
    case 18:
    case 21:
    case 24:
    case 25:
    case 31:
    case 35:
    case 37:
      *(_QWORD *)(a1 + 56) = 0;
      break;
    case 20:
      *(_QWORD *)(a1 + 56) = 0;
      *(_QWORD *)(a1 + 64) = 0;
      *(_QWORD *)(a1 + 72) = 0;
      break;
    case 23:
      *(_BYTE *)(a1 + 56) = 117;
      *(_QWORD *)(a1 + 64) = 0;
      break;
    case 30:
      *(_BYTE *)(a1 + 66) &= ~1u;
      *(_QWORD *)(a1 + 56) = 0;
      *(_WORD *)(a1 + 64) = 0;
      break;
    case 36:
      *(_QWORD *)(a1 + 56) = 0;
      *(_DWORD *)(a1 + 64) = 0;
      break;
    default:
      sub_721090();
  }
}
