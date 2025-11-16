// Function: sub_65BC40
// Address: 0x65bc40
//
__int64 __fastcall sub_65BC40(__int64 a1, char *a2, _QWORD *a3, __int64 a4)
{
  __int64 v6; // r15
  int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rbx
  _QWORD *v23; // [rsp+10h] [rbp-50h]
  __int64 v25[7]; // [rsp+28h] [rbp-38h] BYREF

  v6 = *(_QWORD *)(a1 + 304);
  v7 = *(_DWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C);
  *(_BYTE *)(a1 + 125) |= 0x10u;
  sub_7B8B50(qword_4F04C68, a2, a3, a4);
  sub_87E3B0(a2);
  sub_627530(a1, 0x100000u, v25, a2, 0, 0, 0, 0, 0, 0, 0, 0, 1, a4);
  v8 = v25[0];
  if ( *(_BYTE *)(v25[0] + 140) == 7 )
  {
    *(_QWORD *)(v25[0] + 160) = *(_QWORD *)(a1 + 272);
    *(_QWORD *)(*(_QWORD *)(v8 + 168) + 40LL) = v6;
    *(_BYTE *)(*(_QWORD *)(v8 + 168) + 21LL) |= 1u;
  }
  *(_QWORD *)(a1 + 288) = v8;
  *(_BYTE *)(a1 + 122) = ((a2[64] & 4) != 0) | *(_BYTE *)(a1 + 122) & 0xFE;
  v9 = *(_QWORD *)(*(_QWORD *)(v6 + 168) + 32LL);
  sub_878710(v9, a3);
  a3[1] = *(_QWORD *)(a1 + 32);
  if ( *(_DWORD *)(v9 + 40) != v7 )
  {
    sub_6854C0(2881, a1 + 32, v9);
    if ( (*(_BYTE *)(*(_QWORD *)(v9 + 88) + 265LL) & 1) == 0 )
      goto LABEL_6;
LABEL_21:
    sub_6854C0(3284, a1 + 32, v9);
    goto LABEL_6;
  }
  if ( (*(_BYTE *)(v9 + 81) & 0x10) != 0
    && (unsigned __int8)sub_87D550(v9) != (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 5) & 3) )
  {
    sub_6854C0(2895, a1 + 32, v9);
  }
  if ( (*(_BYTE *)(*(_QWORD *)(v9 + 88) + 265LL) & 1) != 0 )
    goto LABEL_21;
LABEL_6:
  sub_65BB50(a1, v9);
  v10 = *a3;
  v23 = a3 + 1;
  v11 = sub_87EBB0(11, *a3);
  *(_QWORD *)a1 = v11;
  v12 = v11;
  *(_DWORD *)(v11 + 40) = v7;
  v14 = sub_725FD0(11, v10, v13);
  v15 = *(_QWORD *)(a1 + 288);
  *(_QWORD *)(v14 + 152) = v15;
  *(_QWORD *)(v14 + 264) = v15;
  sub_725ED0(v14, 7);
  *(_QWORD *)(v14 + 176) = *(_QWORD *)(*(_QWORD *)(v9 + 88) + 104LL);
  *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v14 + 152) + 168LL) + 8LL) = v14;
  *(_BYTE *)(v14 + 88) = *(_BYTE *)(v14 + 88) & 0x8F | 0x20;
  sub_729470(v14, a4);
  *(_QWORD *)(v12 + 88) = v14;
  sub_877D80(v14, v12);
  sub_877F10(v14, v12);
  sub_7362F0(v14, 0xFFFFFFFFLL);
  sub_8756F0(3, v12, v23, *(_QWORD *)(a1 + 352));
  v16 = *(_QWORD *)(a1 + 352);
  v17 = dword_4F04C64;
  if ( v16 && !*(_BYTE *)(v16 + 16) )
  {
    v18 = *(__int64 **)(v16 + 8);
    if ( v18 )
      v19 = *v18;
    else
      v19 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 328);
    *(_QWORD *)(a1 + 352) = v19;
  }
  if ( (_DWORD)v17 == -1
    || (v20 = qword_4F04C68[0] + 776 * v17, (*(_BYTE *)(v20 + 7) & 1) == 0)
    || dword_4F04C44 == -1 && (*(_BYTE *)(v20 + 6) & 2) == 0 )
  {
    if ( (a2[65] & 8) == 0 )
      sub_87E280(a2 + 8);
  }
  sub_65BAF0(a1, v14);
  switch ( *(_BYTE *)(v9 + 80) )
  {
    case 4:
    case 5:
      v21 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 80LL);
      break;
    case 6:
      v21 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v21 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v21 = *(_QWORD *)(v9 + 88);
      break;
    default:
      sub_87F0B0(v12, 216);
      MEMORY[0x10A] &= ~0x80u;
      BUG();
  }
  sub_87F0B0(v12, v21 + 216);
  *(_BYTE *)(v21 + 266) |= 0x80u;
  return sub_854980(v12, 0);
}
