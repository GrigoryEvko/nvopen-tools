// Function: sub_15FEC10
// Address: 0x15fec10
//
__int64 __fastcall sub_15FEC10(
        __int64 a1,
        __int64 a2,
        int a3,
        __int16 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v10; // rcx
  unsigned __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // rax

  sub_15F1EA0(a1, a2, a3, a1 - 48, 2, a8);
  if ( *(_QWORD *)(a1 - 48) )
  {
    v10 = *(_QWORD *)(a1 - 40);
    v11 = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v11 = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = *(_QWORD *)(v10 + 16) & 3LL | v11;
  }
  *(_QWORD *)(a1 - 48) = a5;
  if ( a5 )
  {
    v12 = *(_QWORD *)(a5 + 8);
    *(_QWORD *)(a1 - 40) = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = (a1 - 40) | *(_QWORD *)(v12 + 16) & 3LL;
    v13 = *(_QWORD *)(a1 - 32);
    *(_QWORD *)(a5 + 8) = a1 - 48;
    *(_QWORD *)(a1 - 32) = (a5 + 8) | v13 & 3;
  }
  if ( *(_QWORD *)(a1 - 24) )
  {
    v14 = *(_QWORD *)(a1 - 16);
    v15 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v15 = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(v14 + 16) & 3LL | v15;
  }
  *(_QWORD *)(a1 - 24) = a6;
  if ( a6 )
  {
    v16 = *(_QWORD *)(a6 + 8);
    *(_QWORD *)(a1 - 16) = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = (a1 - 16) | *(_QWORD *)(v16 + 16) & 3LL;
    *(_QWORD *)(a1 - 8) = (a6 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
    *(_QWORD *)(a6 + 8) = a1 - 24;
  }
  *(_WORD *)(a1 + 18) = a4 | *(_WORD *)(a1 + 18) & 0x8000;
  return sub_164B780(a1, a7);
}
