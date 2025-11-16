// Function: sub_773D00
// Address: 0x773d00
//
__int64 __fastcall sub_773D00(__int64 a1, __int64 a2, __int64 a3, char **a4, __m128i *a5)
{
  char *v5; // rax
  char v6; // cl
  __int64 v7; // rdx
  __int64 v9; // rsi
  unsigned __int8 v10; // cl
  __int64 v11; // rcx
  __m128i *v12; // [rsp+8h] [rbp-18h]

  v5 = *a4;
  v6 = **a4;
  if ( v6 != 13 )
    goto LABEL_2;
  v9 = *((_QWORD *)v5 + 1);
  v10 = *(_BYTE *)(v9 + 24);
  if ( v10 == 4 )
  {
    *v5 = 8;
    v11 = *(_QWORD *)(v9 + 56);
    *((_QWORD *)v5 + 1) = v11;
    goto LABEL_11;
  }
  if ( v10 <= 4u )
  {
    if ( v10 == 2 )
    {
      *v5 = 2;
      v11 = *(_QWORD *)(v9 + 56);
      *((_QWORD *)v5 + 1) = v11;
      goto LABEL_11;
    }
    if ( v10 == 3 )
    {
      *v5 = 7;
      v11 = *(_QWORD *)(v9 + 56);
      *((_QWORD *)v5 + 1) = v11;
LABEL_11:
      if ( (*(_BYTE *)(v11 - 8) & 1) == 0 )
        goto LABEL_12;
      *((_DWORD *)v5 + 4) = 0;
      v6 = *v5;
LABEL_2:
      if ( v6 == 11 )
      {
LABEL_3:
        v7 = *((_QWORD *)v5 + 1);
        goto LABEL_4;
      }
LABEL_12:
      if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
      {
        v12 = a5;
        sub_6855B0(0xD2Fu, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
        sub_770D30(a1);
        a5 = v12;
      }
      goto LABEL_14;
    }
LABEL_19:
    if ( (*(_BYTE *)(v9 - 8) & 1) != 0 )
      *((_DWORD *)v5 + 4) = 0;
    goto LABEL_12;
  }
  if ( v10 != 20 )
    goto LABEL_19;
  *v5 = 11;
  v7 = *(_QWORD *)(v9 + 56);
  *((_QWORD *)v5 + 1) = v7;
  if ( (*(_BYTE *)(v7 - 8) & 1) != 0 )
  {
    *((_DWORD *)v5 + 4) = 0;
    goto LABEL_3;
  }
LABEL_4:
  if ( (*(_DWORD *)(v7 + 192) & 0x18000) != 0 )
  {
    *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08280);
    return 1;
  }
LABEL_14:
  *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
  return 1;
}
