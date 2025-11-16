// Function: sub_B4D260
// Address: 0xb4d260
//
void __fastcall sub_B4D260(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        char a4,
        char a5,
        __int16 a6,
        char a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int16 v17; // r9

  v11 = sub_BD5C60(a2, a2);
  v12 = sub_BCB120(v11);
  sub_B44260(a1, v12, 33, 2u, a8, a9);
  if ( *(_QWORD *)(a1 - 64) )
  {
    v13 = *(_QWORD *)(a1 - 56);
    **(_QWORD **)(a1 - 48) = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = *(_QWORD *)(a1 - 48);
  }
  *(_QWORD *)(a1 - 64) = a2;
  if ( a2 )
  {
    v14 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(a1 - 56) = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = a1 - 56;
    *(_QWORD *)(a1 - 48) = a2 + 16;
    *(_QWORD *)(a2 + 16) = a1 - 64;
  }
  if ( *(_QWORD *)(a1 - 32) )
  {
    v15 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a3;
  if ( a3 )
  {
    v16 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a1 - 24) = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = a1 - 32;
  }
  v17 = *(_WORD *)(a1 + 2);
  *(_BYTE *)(a1 + 72) = a7;
  *(_WORD *)(a1 + 2) = (a6 << 7) | (2 * a5) & 0x7F | a4 & 1 | v17 & 0xFC00;
  nullsub_65();
}
