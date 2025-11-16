// Function: sub_746E90
// Address: 0x746e90
//
__int64 __fastcall sub_746E90(__int64 a1, _BYTE *a2)
{
  int v2; // r14d
  char v4; // al
  __int64 i; // rbx
  __int64 *v6; // rsi
  char v7; // al
  __int64 *v8; // rsi

  v2 = 0;
  v4 = *(_BYTE *)(a1 + 140);
  for ( i = a1; v4 == 12; v4 = *(_BYTE *)(i + 140) )
  {
    v6 = *(__int64 **)(i + 104);
    if ( v6 && sub_736C60(51, v6) )
      v2 = 1;
    i = *(_QWORD *)(i + 160);
  }
  if ( v4 == 5 )
  {
    v7 = *(_BYTE *)(i + 160);
    if ( (unsigned __int8)(v7 - 7) <= 1u )
    {
      if ( v2 )
      {
LABEL_12:
        *a2 = v7;
        *(_BYTE *)(i + 160) = 2;
        return i;
      }
      v8 = *(__int64 **)(i + 104);
      if ( v8 && sub_736C60(51, v8) )
      {
        v7 = *(_BYTE *)(i + 160);
        goto LABEL_12;
      }
    }
  }
  return 0;
}
