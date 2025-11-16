// Function: sub_73C570
// Address: 0x73c570
//
__m128i *__fastcall sub_73C570(const __m128i *a1, int a2)
{
  __int64 v2; // r12
  int v3; // ebx
  int v4; // r15d
  int v5; // r13d
  __m128i *v6; // r13
  __int64 *v8; // rax
  _QWORD *v9; // rcx
  char v10; // al

  v2 = (__int64)a1;
  v3 = a2;
  v4 = sub_8D3410(a1);
  if ( v4 )
  {
    v4 = 1;
    v2 = sub_8D40F0(a1);
  }
  v5 = 0;
  if ( (*(_BYTE *)(v2 + 140) & 0xFB) == 8 )
  {
    v5 = sub_8D4C10(v2, dword_4F077C4 != 2);
    v3 = ~v5 & a2;
  }
  if ( !v3 )
    return (__m128i *)a1;
  if ( (unsigned int)sub_8D32E0(v2) )
  {
    v3 &= 4u;
    if ( !v3 )
      return (__m128i *)a1;
  }
  else if ( (v3 & 0x70) != 0 && (v5 & 0x70) != 0 )
  {
    v3 &= 0xFFFFFF8F;
    goto LABEL_24;
  }
  if ( v5 )
  {
LABEL_24:
    while ( *(_BYTE *)(v2 + 140) == 12 && !*(_QWORD *)(v2 + 8) )
    {
      v10 = *(_BYTE *)(v2 + 185);
      v2 = *(_QWORD *)(v2 + 160);
      v3 |= v10 & 0x7F;
    }
  }
  v8 = *(__int64 **)(v2 + 120);
  if ( v8 )
  {
    v9 = 0;
    while ( 1 )
    {
      if ( !*((_BYTE *)v8 + 16) )
      {
        v6 = (__m128i *)v8[1];
        if ( v3 == (v6[11].m128i_i8[9] & 0x7F) )
          break;
      }
      v9 = v8;
      if ( !*v8 )
        goto LABEL_29;
      v8 = (__int64 *)*v8;
    }
    if ( v9 )
    {
      *v9 = *v8;
      *v8 = *(_QWORD *)(v2 + 120);
      *(_QWORD *)(v2 + 120) = v8;
    }
  }
  else
  {
LABEL_29:
    v6 = (__m128i *)sub_7259C0(12);
    v6[10].m128i_i64[0] = v2;
    v6[11].m128i_i8[9] = v6[11].m128i_i8[9] & 0x80 | v3 & 0x7F;
    sub_728520((_QWORD *)v2, 0, (__int64)v6);
  }
  if ( v4 )
  {
    v6 = sub_73C420(a1, (__int64)v6);
    sub_728520(v6, 4, (__int64)a1);
  }
  return v6;
}
