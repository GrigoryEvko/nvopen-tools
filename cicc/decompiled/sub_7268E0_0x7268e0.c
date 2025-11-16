// Function: sub_7268E0
// Address: 0x7268e0
//
void __fastcall sub_7268E0(__int64 a1, char a2)
{
  _BYTE *v2; // rax
  __int64 v3; // rdx
  _BYTE *v4; // rax
  _BYTE *v5; // rax
  __int64 v6; // rdx
  _BYTE *v7; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rax

  *(_BYTE *)(a1 + 40) = a2;
  *(_QWORD *)(a1 + 48) = 0;
  switch ( a2 )
  {
    case 0:
    case 5:
    case 8:
    case 9:
    case 12:
    case 17:
    case 18:
    case 21:
    case 25:
      *(_QWORD *)(a1 + 72) = 0;
      break;
    case 1:
    case 3:
    case 4:
      *(_QWORD *)(a1 + 80) = 0;
      *(_QWORD *)(a1 + 72) = 0;
      *(_QWORD *)(a1 + 88) = *(_QWORD *)&dword_4F077C8;
      break;
    case 2:
      v2 = sub_7246D0(32);
      *(_QWORD *)v2 = 0;
      v3 = *(_QWORD *)&dword_4F077C8;
      v2[24] &= 0xFCu;
      *((_QWORD *)v2 + 1) = 0;
      *((_QWORD *)v2 + 2) = v3;
      *(_QWORD *)(a1 + 72) = v2;
      break;
    case 6:
    case 7:
    case 15:
      *(_QWORD *)(a1 + 72) = 0;
      *(_QWORD *)(a1 + 80) = 0;
      break;
    case 10:
    case 23:
    case 24:
      return;
    case 11:
      *(_QWORD *)(a1 + 72) = 0;
      v5 = sub_7246D0(32);
      *(_QWORD *)(a1 + 80) = v5;
      v6 = *(_QWORD *)&dword_4F077C8;
      *((_QWORD *)v5 + 1) = 0;
      *((_QWORD *)v5 + 2) = 0;
      *(_QWORD *)v5 = v6;
      v5[24] = v5[24] & 0xF8 | 1;
      break;
    case 13:
      *(_QWORD *)(a1 + 72) = 0;
      v8 = sub_7246D0(24);
      *(_QWORD *)(a1 + 80) = v8;
      *v8 = 0;
      v8[1] = 0;
      v8[2] = 0;
      break;
    case 14:
      *(_QWORD *)(a1 + 72) = 0;
      v7 = sub_7246D0(80);
      *(_QWORD *)(a1 + 80) = v7;
      *(_QWORD *)v7 = 0;
      v7[72] &= ~1u;
      *((_QWORD *)v7 + 1) = 0;
      *((_QWORD *)v7 + 2) = 0;
      *((_QWORD *)v7 + 3) = 0;
      *((_QWORD *)v7 + 4) = 0;
      *((_QWORD *)v7 + 5) = 0;
      *((_QWORD *)v7 + 6) = 0;
      *((_QWORD *)v7 + 7) = 0;
      *((_QWORD *)v7 + 8) = 0;
      break;
    case 16:
      *(_QWORD *)(a1 + 72) = 0;
      v9 = sub_7246D0(24);
      *v9 = 0;
      v9[1] = 0;
      v9[2] = 0;
      *(_QWORD *)(a1 + 80) = v9;
      break;
    case 19:
      v4 = sub_7246D0(32);
      *(_QWORD *)(a1 + 72) = v4;
      *v4 = 0;
      *((_QWORD *)v4 + 1) = 0;
      *((_QWORD *)v4 + 2) = 0;
      *((_QWORD *)v4 + 3) = 0;
      break;
    case 20:
      *(_BYTE *)(a1 + 80) &= ~1u;
      *(_QWORD *)(a1 + 72) = 0;
      break;
    case 22:
      *(_BYTE *)(a1 + 72) = 0;
      *(_QWORD *)(a1 + 80) = 0;
      break;
    default:
      sub_721090();
  }
}
