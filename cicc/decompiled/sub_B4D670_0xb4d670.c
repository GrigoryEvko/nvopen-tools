// Function: sub_B4D670
// Address: 0xb4d670
//
__int64 __fastcall sub_B4D670(__int64 a1, __int16 a2, __int64 a3, __int64 a4, int a5, __int16 a6, char a7)
{
  __int64 v8; // r8
  __int64 v9; // r8
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int16 v12; // r8
  __int64 result; // rax

  if ( *(_QWORD *)(a1 - 64) )
  {
    v8 = *(_QWORD *)(a1 - 56);
    **(_QWORD **)(a1 - 48) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = *(_QWORD *)(a1 - 48);
  }
  *(_QWORD *)(a1 - 64) = a3;
  if ( a3 )
  {
    v9 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a1 - 56) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = a1 - 56;
    *(_QWORD *)(a1 - 48) = a3 + 16;
    *(_QWORD *)(a3 + 16) = a1 - 64;
  }
  if ( *(_QWORD *)(a1 - 32) )
  {
    v10 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a4;
  if ( a4 )
  {
    v11 = *(_QWORD *)(a4 + 16);
    *(_QWORD *)(a1 - 24) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a4 + 16;
    *(_QWORD *)(a4 + 16) = a1 - 32;
  }
  v12 = *(_WORD *)(a1 + 2);
  result = (unsigned int)(a5 << 9);
  *(_BYTE *)(a1 + 72) = a7;
  *(_WORD *)(a1 + 2) = result | (2 * a6) & 0x81FF | (16 * a2) & 0x81FF | v12 & 0x8001;
  return result;
}
