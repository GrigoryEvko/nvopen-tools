// Function: sub_82EAF0
// Address: 0x82eaf0
//
__m128i *__fastcall sub_82EAF0(__int64 a1, __int64 a2, int a3)
{
  __int64 *v3; // rbx
  __m128i *v4; // rdi
  __int64 v5; // rax
  unsigned __int16 v6; // si
  __int64 v8; // rax
  __int64 v9; // rax

  for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  v3 = *(__int64 **)(a1 + 168);
  if ( a2 && *(_BYTE *)(a2 + 80) == 16 && ((*(_BYTE *)(a2 + 96) & 4) != 0 || a3) )
  {
    v4 = *(__m128i **)(a2 + 64);
    if ( !v4 )
      goto LABEL_15;
LABEL_6:
    if ( (*((_BYTE *)v3 + 18) & 0x7F) != 0 )
    {
      v4 = sub_73C570(v4, *((_BYTE *)v3 + 18) & 0x7F);
      if ( (*((_BYTE *)v3 + 19) & 0xC0) != 0x80 )
      {
LABEL_8:
        v5 = sub_72D600(v4);
        v6 = *((_WORD *)v3 + 9);
        v4 = (__m128i *)v5;
        if ( (v6 & 0x3F80) == 0 )
          return v4;
        return sub_73C570(v4, (v6 >> 7) & 0x7F);
      }
    }
    else if ( (*((_BYTE *)v3 + 19) & 0xC0) != 0x80 )
    {
      goto LABEL_8;
    }
    v8 = sub_72D6A0(v4);
    v6 = *((_WORD *)v3 + 9);
    v4 = (__m128i *)v8;
    if ( (v6 & 0x3F80) == 0 )
      return v4;
    return sub_73C570(v4, (v6 >> 7) & 0x7F);
  }
  v4 = (__m128i *)v3[5];
  if ( v4 )
    goto LABEL_6;
LABEL_15:
  v9 = *v3;
  if ( !*v3 || (*(_BYTE *)(v9 + 35) & 1) == 0 )
    return v4;
  return *(__m128i **)(v9 + 8);
}
