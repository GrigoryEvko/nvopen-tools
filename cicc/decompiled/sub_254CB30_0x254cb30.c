// Function: sub_254CB30
// Address: 0x254cb30
//
char __fastcall sub_254CB30(__int64 a1)
{
  __int64 v1; // r13
  __int16 v2; // r12
  __int16 v3; // ax
  char v4; // cl
  char v5; // dl
  char result; // al

  v1 = sub_25096F0((_QWORD *)(a1 + 72));
  v2 = sub_B2D9D0(v1);
  v3 = sub_B2DAA0(v1);
  v4 = v3;
  v5 = HIBYTE(v3);
  if ( v3 == -1 )
  {
    v4 = v2;
    v5 = HIBYTE(v2);
  }
  *(_WORD *)(a1 + 96) = v2;
  result = HIBYTE(v2);
  *(_BYTE *)(a1 + 98) = v4;
  *(_BYTE *)(a1 + 99) = v5;
  if ( HIBYTE(v2) != 3 )
  {
    result = v5 != 3;
    if ( v5 != 3 && v4 != 3 && *(_BYTE *)(a1 + 96) != 3 )
      *(_BYTE *)(a1 + 100) = 1;
  }
  return result;
}
