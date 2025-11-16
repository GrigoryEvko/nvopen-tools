// Function: sub_8399C0
// Address: 0x8399c0
//
char __fastcall sub_8399C0(__int64 a1, const __m128i *a2, int a3, __m128i *a4, __int64 a5, __int64 a6)
{
  __int64 i; // r14
  __int64 v10; // r15
  __int64 v11; // rsi
  char result; // al
  __int64 v13; // rdi
  int v14; // r14d
  __m128i *v15; // rax
  __m128i *v16; // rax

  for ( i = a1; *(_BYTE *)(a5 + 140) == 12; a5 = *(_QWORD *)(a5 + 160) )
    ;
  v10 = *(_QWORD *)(a5 + 168);
  if ( a1 )
    a2 = *(const __m128i **)a1;
  if ( !a3 )
  {
    if ( (unsigned int)sub_8D3A70(a2) )
    {
      if ( (*(_BYTE *)(v10 + 19) & 0xC0) == 0 )
        goto LABEL_22;
    }
    else
    {
      sub_8D3D40(a2);
      if ( (*(_BYTE *)(v10 + 19) & 0xC0) == 0 )
        goto LABEL_22;
    }
    if ( a1 )
    {
      if ( *(_BYTE *)(a1 + 17) != 1 || sub_6ED0A0(a1) )
      {
        if ( !(unsigned int)sub_8D4D20(a4) )
          goto LABEL_35;
      }
      else if ( (unsigned int)sub_8D3110(a4) )
      {
        goto LABEL_35;
      }
      v11 = 0;
      goto LABEL_16;
    }
    if ( !(unsigned int)sub_8D4D20(a4) )
      goto LABEL_35;
LABEL_21:
    v11 = (__int64)a2;
    goto LABEL_9;
  }
  if ( !(unsigned int)sub_8D2EF0(a2) )
  {
    if ( (unsigned int)sub_8DD3B0(a2) )
    {
      v11 = dword_4D03B80;
      if ( (*(_BYTE *)(v10 + 19) & 0xC0) == 0 )
        goto LABEL_9;
LABEL_8:
      if ( !(unsigned int)sub_8D3110(a4) )
        goto LABEL_9;
LABEL_35:
      sub_82D850(a6);
      *(_DWORD *)(a6 + 8) = 7;
      goto LABEL_17;
    }
    if ( (*(_BYTE *)(v10 + 19) & 0xC0) != 0 )
    {
      if ( (unsigned int)sub_8D3110(a4) )
        goto LABEL_35;
      goto LABEL_21;
    }
LABEL_22:
    v11 = (__int64)a2;
    i = 0;
    goto LABEL_16;
  }
  v11 = sub_8D46C0(a2);
  if ( (*(_BYTE *)(v10 + 19) & 0xC0) != 0 )
    goto LABEL_8;
LABEL_9:
  a2 = (const __m128i *)v11;
  i = 0;
LABEL_16:
  sub_838020(i, v11, a4, 0, 0, 0, (__m128i *)a6);
  if ( *(_DWORD *)(a6 + 8) == 7 )
  {
    if ( unk_4D0494C )
    {
      if ( (unsigned int)sub_8D32E0(a4) )
      {
        v13 = sub_8D46C0(a4);
        if ( ((*(_BYTE *)(v13 + 140) & 0xFB) != 8 || (sub_8D4C10(v13, dword_4F077C4 != 2) & 1) == 0)
          && (a2[8].m128i_i8[12] & 0xFB) == 8 )
        {
          v14 = sub_8D4C10(a2, dword_4F077C4 != 2);
          if ( (v14 & 1) != 0 )
          {
            v15 = sub_73D4C0(a2, dword_4F077C4 == 2);
            v16 = sub_73C570(v15, v14 & 0xFFFFFFFE);
            sub_838020(0, (__int64)v16, a4, 0, 0, 0, (__m128i *)a6);
            if ( *(_DWORD *)(a6 + 8) != 7 )
              *(_WORD *)(a6 + 13) = 257;
          }
        }
      }
    }
  }
LABEL_17:
  *(_BYTE *)(a6 + 15) = 1;
  result = *(_BYTE *)(v10 + 19) >> 6;
  *(_BYTE *)(a6 + 20) = result;
  return result;
}
