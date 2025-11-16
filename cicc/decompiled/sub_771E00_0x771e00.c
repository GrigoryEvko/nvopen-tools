// Function: sub_771E00
// Address: 0x771e00
//
__int64 __fastcall sub_771E00(__int64 a1, __int64 a2, __int64 a3, char **a4, __m128i *a5)
{
  char *v5; // rax
  char v6; // dl
  __int64 v8; // rcx
  unsigned __int8 v9; // dl
  __int64 v10; // rdx
  __int64 v11; // rdx

  v5 = *a4;
  v6 = **a4;
  if ( v6 != 13 )
    goto LABEL_2;
  v8 = *((_QWORD *)v5 + 1);
  v9 = *(_BYTE *)(v8 + 24);
  if ( v9 == 4 )
  {
    *v5 = 8;
    v11 = *(_QWORD *)(v8 + 56);
    *((_QWORD *)v5 + 1) = v11;
    if ( (*(_BYTE *)(v11 - 8) & 1) != 0 )
      *((_DWORD *)v5 + 4) = 0;
    goto LABEL_3;
  }
  if ( v9 > 4u )
  {
    if ( v9 == 20 )
    {
      *v5 = 11;
      v10 = *(_QWORD *)(v8 + 56);
      *((_QWORD *)v5 + 1) = v10;
      goto LABEL_9;
    }
LABEL_14:
    if ( (*(_BYTE *)(v8 - 8) & 1) != 0 )
      *((_DWORD *)v5 + 4) = 0;
    goto LABEL_10;
  }
  if ( v9 == 2 )
  {
    *v5 = 2;
    v10 = *(_QWORD *)(v8 + 56);
    *((_QWORD *)v5 + 1) = v10;
    goto LABEL_9;
  }
  if ( v9 != 3 )
    goto LABEL_14;
  *v5 = 7;
  v10 = *(_QWORD *)(v8 + 56);
  *((_QWORD *)v5 + 1) = v10;
LABEL_9:
  if ( (*(_BYTE *)(v10 - 8) & 1) == 0 )
    goto LABEL_10;
  *((_DWORD *)v5 + 4) = 0;
  v6 = *v5;
LABEL_2:
  if ( v6 == 8 )
  {
LABEL_3:
    *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08280);
    return 1;
  }
LABEL_10:
  *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
  return 1;
}
