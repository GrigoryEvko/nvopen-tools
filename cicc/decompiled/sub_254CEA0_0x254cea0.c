// Function: sub_254CEA0
// Address: 0x254cea0
//
char __fastcall sub_254CEA0(__int64 a1, __int64 a2)
{
  __m128i *v2; // r13
  unsigned __int64 v3; // rax
  char v4; // al
  char v5; // dl
  char v6; // al
  unsigned __int8 *v7; // rax
  char result; // al
  int v9[9]; // [rsp+Ch] [rbp-24h] BYREF

  v2 = (__m128i *)(a1 + 72);
  v3 = sub_250C680((__int64 *)(a1 + 72));
  if ( !v3 )
    goto LABEL_5;
  if ( (unsigned __int8)sub_B2D680(v3) )
  {
    v5 = *(_BYTE *)(a1 + 96) & 0xFC | 2;
    *(_BYTE *)(a1 + 96) = v5;
    v4 = v5 | *(_BYTE *)(a1 + 97) & 0xFE | 2;
  }
  else
  {
    v4 = *(_BYTE *)(a1 + 97);
    v5 = *(_BYTE *)(a1 + 96);
  }
  *(_BYTE *)(a1 + 97) = v5 | v4 & 3;
  v9[0] = 81;
  v6 = sub_2516400(a2, v2, (__int64)v9, 1, 1, 0);
  sub_2535850(a2, v2, a1 + 88, v6);
  v7 = sub_250CBE0(v2->m128i_i64, (__int64)v2);
  result = sub_B2FC80((__int64)v7);
  if ( result )
  {
LABEL_5:
    result = *(_BYTE *)(a1 + 96);
    *(_BYTE *)(a1 + 97) = result;
  }
  return result;
}
