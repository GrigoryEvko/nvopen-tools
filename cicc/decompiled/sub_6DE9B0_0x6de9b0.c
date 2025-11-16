// Function: sub_6DE9B0
// Address: 0x6de9b0
//
__int64 __fastcall sub_6DE9B0(int a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  char v6; // cl
  char v7; // si
  char v8; // cl
  unsigned __int8 v9; // cl
  char v10; // al
  char v11; // al
  char v12; // si
  char v13; // si
  char v14; // al
  char v15; // si
  char v16; // al

  result = *(_BYTE *)(a3 + 17) & 0xBF | *(_BYTE *)(a2 + 17) & 0x40u;
  *(_BYTE *)(a3 + 17) = *(_BYTE *)(a3 + 17) & 0xBF | *(_BYTE *)(a2 + 17) & 0x40;
  *(_QWORD *)(a3 + 112) = *(_QWORD *)(a2 + 112);
  *(_QWORD *)(a3 + 120) = *(_QWORD *)(a2 + 120);
  v6 = (*(_BYTE *)(a3 + 20) | *(_BYTE *)(a2 + 20)) & 0x80 | *(_BYTE *)(a3 + 20) & 0x7F;
  v7 = *(_BYTE *)(a3 + 21);
  *(_BYTE *)(a3 + 20) = v6;
  v8 = v7 & 0xF7 | (v7 | *(_BYTE *)(a2 + 21)) & 8;
  *(_BYTE *)(a3 + 21) = v8;
  v9 = (v8 | *(_BYTE *)(a2 + 21)) & 0x10 | v8 & 0xEF;
  *(_BYTE *)(a3 + 21) = v9;
  if ( a1 )
  {
    v10 = *(_BYTE *)(a2 + 17) & 1 | result & 0xFE;
    *(_BYTE *)(a3 + 17) = v10;
    v11 = *(_BYTE *)(a2 + 17) & 2 | v10 & 0xFD;
    *(_BYTE *)(a3 + 17) = v11;
    v12 = *(_BYTE *)(a3 + 18);
    *(_BYTE *)(a3 + 17) = *(_BYTE *)(a2 + 17) & 0x20 | v11 & 0xDF;
    v13 = *(_BYTE *)(a2 + 18) & 0x10 | v12 & 0xEF;
    *(_BYTE *)(a3 + 18) = v13;
    v14 = v13 & 0xBF | (v13 | *(_BYTE *)(a2 + 18)) & 0x40;
    *(_BYTE *)(a3 + 18) = v14;
    v15 = *(_BYTE *)(a3 + 19);
    *(_BYTE *)(a3 + 18) = (v14 | *(_BYTE *)(a2 + 18)) & 0x80 | v14 & 0x7F;
    v16 = v15 & 0xFD | (v15 | *(_BYTE *)(a2 + 19)) & 2;
    *(_BYTE *)(a3 + 19) = v16;
    *(_BYTE *)(a3 + 19) = (v16 | *(_BYTE *)(a2 + 19)) & 4 | v16 & 0xFB;
    result = (v9 | *(_BYTE *)(a2 + 21)) & 4;
    *(_BYTE *)(a3 + 21) = result | v9 & 0xFB;
  }
  return result;
}
