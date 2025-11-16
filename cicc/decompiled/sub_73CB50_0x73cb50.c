// Function: sub_73CB50
// Address: 0x73cb50
//
__m128i *__fastcall sub_73CB50(__int64 a1, int a2)
{
  char v4; // al
  const __m128i *v5; // rdi
  unsigned __int8 v7; // dl
  char v8; // si

  v4 = *(_BYTE *)(a1 + 144);
  v5 = *(const __m128i **)(a1 + 120);
  if ( dword_4F077C0 && (v4 & 4) != 0 && dword_4F06BA0 == 8 )
  {
    v7 = *(_BYTE *)(a1 + 137);
    if ( v7 == 32 )
    {
      v8 = 3;
    }
    else if ( v7 <= 0x20u )
    {
      if ( v7 == 8 )
      {
        v8 = 1;
      }
      else
      {
        if ( v7 != 16 )
          goto LABEL_4;
        v8 = 2;
      }
    }
    else if ( v7 == 64 )
    {
      v8 = 4;
    }
    else
    {
      if ( v7 != 0x80 )
        goto LABEL_4;
      v8 = 5;
    }
    v5 = (const __m128i *)sub_5CFCE0((__int64)v5, v8, 0);
    v4 = *(_BYTE *)(a1 + 144);
  }
LABEL_4:
  if ( (v4 & 0x20) != 0 )
    a2 &= ~1u;
  return sub_73C570(v5, a2);
}
