// Function: sub_15F9860
// Address: 0x15f9860
//
__int64 __fastcall sub_15F9860(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int16 a5, unsigned int a6, char a7)
{
  __int64 result; // rax
  __int64 v8; // r11
  unsigned __int64 v9; // r9
  __int64 v10; // r9
  __int64 v11; // r9
  unsigned __int64 v12; // rsi
  __int64 v13; // rsi
  __int64 v14; // rsi
  unsigned __int64 v15; // rdx
  __int64 v16; // rdx
  __int16 v17; // r9
  __int16 v18; // r9

  result = a6;
  if ( *(_QWORD *)(a1 - 72) )
  {
    v8 = *(_QWORD *)(a1 - 64);
    v9 = *(_QWORD *)(a1 - 56) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v9 = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = *(_QWORD *)(v8 + 16) & 3LL | v9;
  }
  *(_QWORD *)(a1 - 72) = a2;
  if ( a2 )
  {
    v10 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a1 - 64) = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = (a1 - 64) | *(_QWORD *)(v10 + 16) & 3LL;
    *(_QWORD *)(a1 - 56) = (a2 + 8) | *(_QWORD *)(a1 - 56) & 3LL;
    *(_QWORD *)(a2 + 8) = a1 - 72;
  }
  if ( *(_QWORD *)(a1 - 48) )
  {
    v11 = *(_QWORD *)(a1 - 40);
    v12 = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v12 = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = *(_QWORD *)(v11 + 16) & 3LL | v12;
  }
  *(_QWORD *)(a1 - 48) = a3;
  if ( a3 )
  {
    v13 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(a1 - 40) = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = (a1 - 40) | *(_QWORD *)(v13 + 16) & 3LL;
    *(_QWORD *)(a1 - 32) = (a3 + 8) | *(_QWORD *)(a1 - 32) & 3LL;
    *(_QWORD *)(a3 + 8) = a1 - 48;
  }
  if ( *(_QWORD *)(a1 - 24) )
  {
    v14 = *(_QWORD *)(a1 - 16);
    v15 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v15 = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(v14 + 16) & 3LL | v15;
  }
  *(_QWORD *)(a1 - 24) = a4;
  if ( a4 )
  {
    v16 = *(_QWORD *)(a4 + 8);
    *(_QWORD *)(a1 - 16) = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = (a1 - 16) | *(_QWORD *)(v16 + 16) & 3LL;
    *(_QWORD *)(a1 - 8) = (a4 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
    *(_QWORD *)(a4 + 8) = a1 - 24;
  }
  v17 = *(_WORD *)(a1 + 18);
  *(_BYTE *)(a1 + 56) = a7;
  v18 = (4 * a5) | v17 & 0xFFE3;
  LOBYTE(v18) = v18 & 0x1F;
  *(_WORD *)(a1 + 18) = (32 * result) | v18;
  return result;
}
