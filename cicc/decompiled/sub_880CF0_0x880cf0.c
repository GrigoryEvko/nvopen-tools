// Function: sub_880CF0
// Address: 0x880cf0
//
_QWORD *__fastcall sub_880CF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rcx
  char v5; // bl
  _QWORD *v6; // r13
  __int64 v7; // rax
  int v8; // edx
  __int64 v10; // [rsp+8h] [rbp-38h]

  switch ( *(_BYTE *)(a1 + 80) )
  {
    case 4:
    case 5:
      v4 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
      break;
    case 6:
      v4 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v4 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v4 = *(_QWORD *)(a1 + 88);
      break;
    default:
      BUG();
  }
  v10 = v4;
  v5 = 8 * !(*(_BYTE *)(v4 + 160) >> 7) + 1;
  v6 = sub_87EBB0(((*(_BYTE *)(a1 + 81) & 0x10) == 0) + 10, *(_QWORD *)a1, (_QWORD *)(a1 + 48));
  v7 = sub_880C60();
  *(_QWORD *)(v7 + 32) = a1;
  *(_QWORD *)(v7 + 24) = v6;
  *(_QWORD *)(v7 + 56) = v10;
  v8 = *(_DWORD *)(a1 + 40);
  v6[12] = v7;
  LOBYTE(v7) = *((_BYTE *)v6 + 81);
  v6[11] = a2;
  *((_DWORD *)v6 + 10) = v8;
  *((_BYTE *)v6 + 81) = *(_BYTE *)(a1 + 81) & 0x10 | v7 & 0xEF;
  v6[8] = *(_QWORD *)(a1 + 64);
  *(_QWORD *)(a2 + 240) = sub_896D70(a1, a3, 0);
  *(_BYTE *)(a2 + 195) = *(_BYTE *)(a2 + 195) & 0xF6 | v5;
  return v6;
}
