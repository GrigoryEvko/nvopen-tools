// Function: sub_87E690
// Address: 0x87e690
//
void __fastcall sub_87E690(__int64 a1, char a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  bool v4; // zf
  unsigned __int64 v5; // rcx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r12

  *(_BYTE *)(a1 + 80) = a2;
  switch ( a2 )
  {
    case 0:
      *(_BYTE *)(a1 + 90) &= ~1u;
      *(_WORD *)(a1 + 88) = 0;
      *(_DWORD *)(a1 + 92) = 0;
      break;
    case 1:
    case 2:
    case 18:
      *(_QWORD *)(a1 + 88) = 0;
      break;
    case 3:
      *(_QWORD *)(a1 + 88) = 0;
      *(_QWORD *)(a1 + 96) = 0;
      *(_BYTE *)(a1 + 104) = 0;
      break;
    case 4:
    case 5:
      *(_QWORD *)(a1 + 88) = 0;
      v2 = sub_823970(344);
      *(_QWORD *)(a1 + 96) = v2;
      *(_QWORD *)v2 = 0;
      v3 = *(_QWORD *)&dword_4F077C8;
      *(_DWORD *)(v2 + 182) &= 0xFFFE0000;
      *(_QWORD *)(v2 + 88) = v3;
      *(_QWORD *)(v2 + 96) = 0xFFFFFFFFLL;
      v4 = dword_4F077C4 == 2;
      *(_QWORD *)(v2 + 8) = 0;
      v5 = *(_QWORD *)(v2 + 176) & 0xFFFF000000000000LL;
      *(_QWORD *)(v2 + 16) = 0;
      *(_QWORD *)(v2 + 24) = 0;
      *(_QWORD *)(v2 + 32) = 0;
      *(_QWORD *)(v2 + 40) = 0;
      *(_QWORD *)(v2 + 48) = 0;
      *(_QWORD *)(v2 + 56) = 0;
      *(_QWORD *)(v2 + 64) = 0;
      *(_QWORD *)(v2 + 72) = 0;
      *(_QWORD *)(v2 + 80) = 0;
      *(_QWORD *)(v2 + 104) = 0;
      *(_DWORD *)(v2 + 112) = 0;
      *(_QWORD *)(v2 + 120) = 0;
      *(_QWORD *)(v2 + 128) = 0;
      *(_QWORD *)(v2 + 136) = 0;
      *(_QWORD *)(v2 + 144) = 0;
      *(_QWORD *)(v2 + 152) = 0;
      *(_QWORD *)(v2 + 160) = 0;
      *(_QWORD *)(v2 + 168) = 0;
      *(_QWORD *)(v2 + 176) = v5 | ((unsigned __int64)!v4 << 22) | 0x800000000000LL;
      sub_85FFE0(v2 + 192);
      break;
    case 6:
      *(_QWORD *)(a1 + 88) = 0;
      v7 = sub_823970(56);
      *(_QWORD *)(a1 + 96) = v7;
      *(_QWORD *)v7 = 0;
      v8 = *(_QWORD *)&dword_4F077C8;
      *(_BYTE *)(v7 + 48) &= ~1u;
      *(_QWORD *)(v7 + 8) = 0;
      *(_QWORD *)(v7 + 16) = 0;
      *(_QWORD *)(v7 + 24) = 0;
      *(_QWORD *)(v7 + 32) = 0;
      *(_QWORD *)(v7 + 40) = v8;
      break;
    case 7:
    case 9:
      *(_QWORD *)(a1 + 88) = 0;
      *(_QWORD *)(a1 + 96) = 0;
      *(_QWORD *)(a1 + 104) = 0;
      break;
    case 8:
      *(_QWORD *)(a1 + 88) = 0;
      *(_QWORD *)(a1 + 96) = 0;
      v10 = sub_823970(32);
      *(_BYTE *)(v10 + 28) &= 0xF0u;
      *(_DWORD *)v10 = 0;
      *(_QWORD *)(v10 + 8) = 0;
      *(_QWORD *)(v10 + 16) = 0;
      *(_DWORD *)(v10 + 24) = 0;
      *(_QWORD *)(a1 + 104) = v10;
      break;
    case 10:
    case 11:
      *(_BYTE *)(a1 + 104) &= ~1u;
      *(_QWORD *)(a1 + 88) = 0;
      *(_QWORD *)(a1 + 96) = 0;
      break;
    case 12:
      *(_QWORD *)(a1 + 88) = 0;
      *(_QWORD *)(a1 + 96) = 0;
      break;
    case 13:
      return;
    case 14:
    case 15:
      v6 = sub_823970(24);
      *(_QWORD *)(a1 + 88) = v6;
      *(_QWORD *)v6 = 0;
      *(_QWORD *)(v6 + 8) = 0;
      *(_BYTE *)(v6 + 16) = 0;
      break;
    case 16:
      v9 = (_QWORD *)sub_823970(24);
      *v9 = 0;
      v9[1] = 0;
      v9[2] = 0;
      *(_QWORD *)(a1 + 88) = v9;
      *(_BYTE *)(a1 + 96) &= 0xC0u;
      break;
    case 17:
      *(_QWORD *)(a1 + 88) = 0;
      *(_BYTE *)(a1 + 96) = 0;
      break;
    case 19:
    case 20:
    case 21:
    case 22:
      *(_QWORD *)(a1 + 88) = sub_87E420(a2);
      break;
    case 23:
      *(_QWORD *)(a1 + 88) = 0;
      v11 = sub_823970(208);
      *(_BYTE *)(v11 + 200) &= 0xFCu;
      v12 = v11;
      *(_QWORD *)(v11 + 152) = 0;
      *(_QWORD *)(v11 + 160) = 0;
      *(_DWORD *)(v11 + 168) = 0;
      *(_QWORD *)(v11 + 176) = 0;
      *(_QWORD *)(v11 + 184) = 0;
      *(_QWORD *)(v11 + 192) = 0;
      sub_85FFE0(v11);
      *(_QWORD *)(a1 + 96) = v12;
      *(_QWORD *)(v12 + 160) = a1;
      break;
    case 24:
      *(_BYTE *)(a1 + 96) &= 0xF8u;
      *(_QWORD *)(a1 + 88) = 0;
      break;
    case 25:
      *(_BYTE *)(a1 + 104) &= 0xFCu;
      *(_QWORD *)(a1 + 88) = 0;
      *(_QWORD *)(a1 + 96) = 0;
      break;
    default:
      sub_721090();
  }
}
