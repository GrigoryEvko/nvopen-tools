// Function: sub_15F9480
// Address: 0x15f9480
//
void __fastcall sub_15F9480(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 a4,
        unsigned int a5,
        __int16 a6,
        char a7,
        __int64 a8)
{
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  __int16 v18; // r9

  v9 = sub_16498A0(a2);
  v10 = sub_1643270(v9);
  sub_15F1EA0(a1, v10, 31, a1 - 48, 2, a8);
  if ( *(_QWORD *)(a1 - 48) )
  {
    v11 = *(_QWORD *)(a1 - 40);
    v12 = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v12 = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = *(_QWORD *)(v11 + 16) & 3LL | v12;
  }
  *(_QWORD *)(a1 - 48) = a2;
  if ( a2 )
  {
    v13 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a1 - 40) = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = (a1 - 40) | *(_QWORD *)(v13 + 16) & 3LL;
    v14 = *(_QWORD *)(a1 - 32);
    *(_QWORD *)(a2 + 8) = a1 - 48;
    *(_QWORD *)(a1 - 32) = (a2 + 8) | v14 & 3;
  }
  if ( *(_QWORD *)(a1 - 24) )
  {
    v15 = *(_QWORD *)(a1 - 16);
    v16 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v16 = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = *(_QWORD *)(v15 + 16) & 3LL | v16;
  }
  *(_QWORD *)(a1 - 24) = a3;
  if ( a3 )
  {
    v17 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(a1 - 16) = v17;
    if ( v17 )
      *(_QWORD *)(v17 + 16) = (a1 - 16) | *(_QWORD *)(v17 + 16) & 3LL;
    *(_QWORD *)(a1 - 8) = (a3 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
    *(_QWORD *)(a3 + 8) = a1 - 24;
  }
  *(_WORD *)(a1 + 18) = a4 | *(_WORD *)(a1 + 18) & 0xFFFE;
  sub_15F9450(a1, a5);
  v18 = *(_WORD *)(a1 + 18);
  *(_BYTE *)(a1 + 56) = a7;
  *(_WORD *)(a1 + 18) = (a6 << 7) | v18 & 0xFC7F;
  nullsub_557();
}
