// Function: sub_B4D470
// Address: 0xb4d470
//
__int64 __fastcall sub_B4D470(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, __int16 a6, __int16 a7, char a8)
{
  __int64 v9; // r8
  __int64 v10; // r8
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int16 v15; // r8
  __int64 result; // rax
  __int16 v17; // r8

  if ( *(_QWORD *)(a1 - 96) )
  {
    v9 = *(_QWORD *)(a1 - 88);
    **(_QWORD **)(a1 - 80) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = *(_QWORD *)(a1 - 80);
  }
  *(_QWORD *)(a1 - 96) = a2;
  if ( a2 )
  {
    v10 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(a1 - 88) = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = a1 - 88;
    *(_QWORD *)(a1 - 80) = a2 + 16;
    *(_QWORD *)(a2 + 16) = a1 - 96;
  }
  if ( *(_QWORD *)(a1 - 64) )
  {
    v11 = *(_QWORD *)(a1 - 56);
    **(_QWORD **)(a1 - 48) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = *(_QWORD *)(a1 - 48);
  }
  *(_QWORD *)(a1 - 64) = a3;
  if ( a3 )
  {
    v12 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a1 - 56) = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = a1 - 56;
    *(_QWORD *)(a1 - 48) = a3 + 16;
    *(_QWORD *)(a3 + 16) = a1 - 64;
  }
  if ( *(_QWORD *)(a1 - 32) )
  {
    v13 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a4;
  if ( a4 )
  {
    v14 = *(_QWORD *)(a4 + 16);
    *(_QWORD *)(a1 - 24) = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a4 + 16;
    *(_QWORD *)(a4 + 16) = a1 - 32;
  }
  v15 = *(_WORD *)(a1 + 2);
  result = (unsigned int)(a5 << 8);
  *(_BYTE *)(a1 + 72) = a8;
  v17 = (4 * a6) | v15 & 0xFFE3;
  LOBYTE(v17) = v17 & 0x1F;
  *(_WORD *)(a1 + 2) = result | ((32 * a7) | v17) & 0xC0FF;
  return result;
}
