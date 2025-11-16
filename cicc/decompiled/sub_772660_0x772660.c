// Function: sub_772660
// Address: 0x772660
//
__int64 __fastcall sub_772660(__int64 a1, __int64 a2, __int64 a3, char **a4, __m128i *a5)
{
  char *v5; // rax
  char v6; // dl
  __int64 v8; // rcx
  unsigned __int8 v9; // dl
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 i; // rax
  __int64 v13; // rdx

  v5 = *a4;
  v6 = **a4;
  if ( v6 != 13 )
    goto LABEL_2;
  v8 = *((_QWORD *)v5 + 1);
  v9 = *(_BYTE *)(v8 + 24);
  if ( v9 == 4 )
  {
    *v5 = 8;
    v13 = *(_QWORD *)(v8 + 56);
    *((_QWORD *)v5 + 1) = v13;
LABEL_25:
    if ( (*(_BYTE *)(v13 - 8) & 1) == 0 )
      goto LABEL_4;
    *((_DWORD *)v5 + 4) = 0;
    v6 = *v5;
LABEL_2:
    if ( v6 != 7 )
    {
      if ( v6 != 11 )
      {
LABEL_4:
        *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
        return 1;
      }
      goto LABEL_22;
    }
    goto LABEL_11;
  }
  if ( v9 > 4u )
  {
    if ( v9 == 20 )
    {
      *v5 = 11;
      v11 = *(_QWORD *)(v8 + 56);
      *((_QWORD *)v5 + 1) = v11;
      if ( (*(_BYTE *)(v11 - 8) & 1) == 0 )
      {
        if ( (*(_BYTE *)(v11 + 89) & 4) == 0 )
          goto LABEL_4;
        goto LABEL_17;
      }
      *((_DWORD *)v5 + 4) = 0;
LABEL_22:
      v11 = *((_QWORD *)v5 + 1);
      if ( (*(_BYTE *)(v11 + 89) & 4) == 0 )
        goto LABEL_4;
LABEL_17:
      for ( i = *(_QWORD *)(v11 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      if ( *(_QWORD *)(*(_QWORD *)(i + 168) + 40LL) )
        goto LABEL_4;
      goto LABEL_13;
    }
LABEL_27:
    if ( (*(_BYTE *)(v8 - 8) & 1) != 0 )
      *((_DWORD *)v5 + 4) = 0;
    goto LABEL_4;
  }
  if ( v9 == 2 )
  {
    *v5 = 2;
    v13 = *(_QWORD *)(v8 + 56);
    *((_QWORD *)v5 + 1) = v13;
    goto LABEL_25;
  }
  if ( v9 != 3 )
    goto LABEL_27;
  *v5 = 7;
  v10 = *(_QWORD *)(v8 + 56);
  *((_QWORD *)v5 + 1) = v10;
  if ( (*(_BYTE *)(v10 - 8) & 1) == 0 )
    goto LABEL_12;
  *((_DWORD *)v5 + 4) = 0;
LABEL_11:
  v10 = *((_QWORD *)v5 + 1);
LABEL_12:
  if ( (*(_BYTE *)(v10 + 89) & 4) == 0 )
    goto LABEL_4;
LABEL_13:
  *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08280);
  return 1;
}
