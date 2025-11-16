// Function: sub_15F9B00
// Address: 0x15f9b00
//
__int64 __fastcall sub_15F9B00(__int64 a1, __int16 a2, __int64 a3, __int64 a4, int a5, char a6)
{
  __int64 v7; // r10
  unsigned __int64 v8; // r8
  __int64 v9; // r8
  __int64 v10; // r8
  unsigned __int64 v11; // rdx
  __int64 v12; // rdx
  __int16 v13; // r8
  __int64 result; // rax

  if ( *(_QWORD *)(a1 - 48) )
  {
    v7 = *(_QWORD *)(a1 - 40);
    v8 = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v8 = v7;
    if ( v7 )
      *(_QWORD *)(v7 + 16) = *(_QWORD *)(v7 + 16) & 3LL | v8;
  }
  *(_QWORD *)(a1 - 48) = a3;
  if ( a3 )
  {
    v9 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(a1 - 40) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = (a1 - 40) | *(_QWORD *)(v9 + 16) & 3LL;
    *(_QWORD *)(a1 - 32) = (a3 + 8) | *(_QWORD *)(a1 - 32) & 3LL;
    *(_QWORD *)(a3 + 8) = a1 - 48;
  }
  if ( *(_QWORD *)(a1 - 24) )
  {
    v10 = *(_QWORD *)(a1 - 16);
    v11 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v11 = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = *(_QWORD *)(v10 + 16) & 3LL | v11;
  }
  *(_QWORD *)(a1 - 24) = a4;
  if ( a4 )
  {
    v12 = *(_QWORD *)(a4 + 8);
    *(_QWORD *)(a1 - 16) = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = (a1 - 16) | *(_QWORD *)(v12 + 16) & 3LL;
    *(_QWORD *)(a1 - 8) = (a4 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
    *(_QWORD *)(a4 + 8) = a1 - 24;
  }
  v13 = *(_WORD *)(a1 + 18);
  result = (unsigned int)(4 * a5);
  *(_BYTE *)(a1 + 56) = a6;
  *(_WORD *)(a1 + 18) = result | (32 * a2) | v13 & 0x8003;
  return result;
}
