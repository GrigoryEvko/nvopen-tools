// Function: sub_773E50
// Address: 0x773e50
//
__int64 __fastcall sub_773E50(__int64 a1, __int64 a2, __int64 a3, char **a4, __m128i *a5)
{
  char *v5; // rax
  char v6; // cl
  __int64 v7; // rdx
  __int64 v9; // rcx
  __int64 v10; // rsi
  unsigned __int8 v11; // cl
  __m128i *v12; // [rsp+8h] [rbp-18h]

  v5 = *a4;
  v6 = **a4;
  if ( v6 != 13 )
    goto LABEL_2;
  v10 = *((_QWORD *)v5 + 1);
  v11 = *(_BYTE *)(v10 + 24);
  if ( v11 == 4 )
  {
    *v5 = 8;
    v9 = *(_QWORD *)(v10 + 56);
    *((_QWORD *)v5 + 1) = v9;
    goto LABEL_9;
  }
  if ( v11 > 4u )
  {
    if ( v11 != 20 )
      goto LABEL_21;
    *v5 = 11;
    v9 = *(_QWORD *)(v10 + 56);
    *((_QWORD *)v5 + 1) = v9;
LABEL_9:
    if ( (*(_BYTE *)(v9 - 8) & 1) == 0 )
      goto LABEL_10;
    *((_DWORD *)v5 + 4) = 0;
    v6 = *v5;
LABEL_2:
    if ( v6 == 7 )
      goto LABEL_3;
LABEL_10:
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      v12 = a5;
      sub_6855B0(0xD2Fu, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      a5 = v12;
    }
    goto LABEL_12;
  }
  if ( v11 == 2 )
  {
    *v5 = 2;
    v9 = *(_QWORD *)(v10 + 56);
    *((_QWORD *)v5 + 1) = v9;
    goto LABEL_9;
  }
  if ( v11 != 3 )
  {
LABEL_21:
    if ( (*(_BYTE *)(v10 - 8) & 1) != 0 )
      *((_DWORD *)v5 + 4) = 0;
    goto LABEL_10;
  }
  *v5 = 7;
  v7 = *(_QWORD *)(v10 + 56);
  *((_QWORD *)v5 + 1) = v7;
  if ( (*(_BYTE *)(v7 - 8) & 1) == 0 )
    goto LABEL_4;
  *((_DWORD *)v5 + 4) = 0;
LABEL_3:
  v7 = *((_QWORD *)v5 + 1);
LABEL_4:
  if ( (*(_BYTE *)(v7 + 176) & 8) == 0 && *(_BYTE *)(v7 + 136) <= 2u )
  {
    *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08280);
    return 1;
  }
LABEL_12:
  *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
  return 1;
}
