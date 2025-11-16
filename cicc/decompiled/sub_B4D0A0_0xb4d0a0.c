// Function: sub_B4D0A0
// Address: 0xb4d0a0
//
__int64 __fastcall sub_B4D0A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        char a5,
        char a6,
        __int16 a7,
        char a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v13; // rax
  __int64 v14; // rax
  __int16 v15; // ax

  sub_B44260(a1, a2, 32, 1u, a9, a10);
  if ( *(_QWORD *)(a1 - 32) )
  {
    v13 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a3;
  if ( a3 )
  {
    v14 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a1 - 24) = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = a1 - 32;
  }
  v15 = *(_WORD *)(a1 + 2);
  *(_BYTE *)(a1 + 72) = a8;
  *(_WORD *)(a1 + 2) = (a7 << 7) | (2 * a6) & 0x7F | a5 & 1 | v15 & 0xFC00;
  nullsub_64();
  return sub_BD6B50(a1, a4);
}
