// Function: sub_725570
// Address: 0x725570
//
void __fastcall sub_725570(__int64 a1, char a2)
{
  _QWORD *v2; // r13
  _QWORD *v3; // rax
  __m128i v4; // xmm0
  _QWORD *v5; // rax
  char v6; // dl
  __int64 v7; // rcx
  _QWORD *v8; // rax
  int v9; // eax
  _QWORD *v10; // rax
  __int64 v11; // rdx

  *(_BYTE *)(a1 + 140) = a2;
  switch ( a2 )
  {
    case 0:
    case 1:
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
      return;
    case 2:
      v9 = *(_DWORD *)(a1 + 160);
      *(_QWORD *)(a1 + 168) = 0;
      *(_DWORD *)(a1 + 160) = v9 & 0xF8000000 | 5;
      v10 = sub_7247C0(32);
      *(_BYTE *)v10 &= 0xF8u;
      v11 = *(_QWORD *)&dword_4F077C8;
      v10[1] = 0;
      v10[3] = 0;
      v10[2] = v11;
      *(_QWORD *)(a1 + 176) = v10;
      break;
    case 3:
    case 4:
    case 5:
      *(_BYTE *)(a1 + 160) = 2;
      break;
    case 6:
      *(_BYTE *)(a1 + 168) &= 0xFCu;
      *(_QWORD *)(a1 + 160) = 0;
      break;
    case 7:
      *(_QWORD *)(a1 + 160) = 0;
      v5 = sub_7247C0(64);
      *(_QWORD *)(a1 + 176) = 0;
      *(_QWORD *)(a1 + 168) = v5;
      *v5 = 0;
      v6 = unk_4F06CF8;
      *((_WORD *)v5 + 8) &= 0xF000u;
      v5[1] = 0;
      *((_DWORD *)v5 + 7) = 0;
      v5[4] = 0;
      v7 = v5[2] & 0xFE0000000FFFLL;
      v5[5] = 0;
      v5[2] = v7 | ((unsigned __int64)(v6 & 7) << 12) | 0xFFFF000000000000LL;
      *((_WORD *)v5 + 12) = 0;
      v5[6] = 0;
      v5[7] = 0;
      break;
    case 8:
      *(_WORD *)(a1 + 168) &= 0x8000u;
      *(_QWORD *)(a1 + 160) = 0;
      *(_QWORD *)(a1 + 176) = 0;
      *(_QWORD *)(a1 + 184) = 0;
      break;
    case 9:
    case 10:
    case 11:
      *(_QWORD *)(a1 + 176) &= 0xFFFFFC0000000000LL;
      *(_QWORD *)(a1 + 160) = 0;
      *(_DWORD *)(a1 + 184) = 0;
      v2 = sub_7247C0(264);
      sub_725420((__int64)v2);
      *(_QWORD *)(a1 + 168) = v2;
      *((_BYTE *)v2 + 108) = a2;
      break;
    case 12:
      *(_QWORD *)(a1 + 160) = 0;
      v3 = sub_7247C0(72);
      *v3 = 0;
      v3[1] = 0;
      v4 = _mm_loadu_si128((const __m128i *)&unk_4F07370);
      v3[3] = 0;
      v3[2] = 0;
      v3[4] = 0;
      v3[5] = 0;
      *((_DWORD *)v3 + 12) = -1;
      *(__m128i *)((char *)v3 + 52) = v4;
      *(_QWORD *)(a1 + 168) = v3;
      *(_DWORD *)(a1 + 184) &= 0xFC040000;
      *(_QWORD *)(a1 + 176) = 0;
      *(_QWORD *)(a1 + 128) = 0;
      *(_DWORD *)(a1 + 136) = 1;
      break;
    case 13:
      *(_QWORD *)(a1 + 160) = 0;
      *(_QWORD *)(a1 + 168) = 0;
      break;
    case 14:
      *(_WORD *)(a1 + 160) &= 0xE000u;
      v8 = sub_7247C0(40);
      *v8 = 0;
      v8[1] = 0;
      v8[2] = 0;
      v8[4] = 0;
      *(_QWORD *)(a1 + 168) = v8;
      v8[3] = 0;
      break;
    case 15:
      *(_WORD *)(a1 + 176) &= 0xFEu;
      *(_QWORD *)(a1 + 160) = 0;
      *(_QWORD *)(a1 + 168) = 0;
      break;
    case 16:
      *(_QWORD *)(a1 + 160) = 0;
      *(_BYTE *)(a1 + 168) = 0;
      break;
    default:
      sub_721090();
  }
}
