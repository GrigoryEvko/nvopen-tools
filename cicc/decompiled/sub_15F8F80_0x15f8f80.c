// Function: sub_15F8F80
// Address: 0x15f8f80
//
__int64 __fastcall sub_15F8F80(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        unsigned int a6,
        __int16 a7,
        char a8,
        __int64 a9)
{
  __int64 v11; // rsi
  unsigned __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int16 v15; // ax

  sub_15F1EA0(a1, a2, 30, a1 - 24, 1, a9);
  if ( *(_QWORD *)(a1 - 24) )
  {
    v11 = *(_QWORD *)(a1 - 16);
    v12 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v12 = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = *(_QWORD *)(v11 + 16) & 3LL | v12;
  }
  *(_QWORD *)(a1 - 24) = a3;
  if ( a3 )
  {
    v13 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(a1 - 16) = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = (a1 - 16) | *(_QWORD *)(v13 + 16) & 3LL;
    v14 = *(_QWORD *)(a1 - 8);
    *(_QWORD *)(a3 + 8) = a1 - 24;
    *(_QWORD *)(a1 - 8) = (a3 + 8) | v14 & 3;
  }
  *(_WORD *)(a1 + 18) = a5 | *(_WORD *)(a1 + 18) & 0xFFFE;
  sub_15F8F50(a1, a6);
  v15 = *(_WORD *)(a1 + 18);
  *(_BYTE *)(a1 + 56) = a8;
  *(_WORD *)(a1 + 18) = v15 & 0xFC7F | (a7 << 7);
  nullsub_556();
  return sub_164B780(a1, a4);
}
