// Function: sub_772080
// Address: 0x772080
//
__int64 __fastcall sub_772080(__int64 a1, __int64 a2, __int64 a3, char **a4, __m128i *a5)
{
  char *v5; // rax
  char v6; // dl
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned __int8 v10; // dl
  __int64 v11; // rdx

  v5 = *a4;
  v6 = **a4;
  if ( v6 != 13 )
    goto LABEL_2;
  v9 = *((_QWORD *)v5 + 1);
  v10 = *(_BYTE *)(v9 + 24);
  if ( v10 != 4 )
  {
    if ( v10 > 4u )
    {
      if ( v10 == 20 )
      {
        *v5 = 11;
        v11 = *(_QWORD *)(v9 + 56);
        *((_QWORD *)v5 + 1) = v11;
        goto LABEL_14;
      }
    }
    else
    {
      if ( v10 == 2 )
      {
        *v5 = 2;
        v11 = *(_QWORD *)(v9 + 56);
        *((_QWORD *)v5 + 1) = v11;
        goto LABEL_14;
      }
      if ( v10 == 3 )
      {
        *v5 = 7;
        v11 = *(_QWORD *)(v9 + 56);
        *((_QWORD *)v5 + 1) = v11;
LABEL_14:
        if ( (*(_BYTE *)(v11 - 8) & 1) == 0 )
          goto LABEL_3;
        *((_DWORD *)v5 + 4) = 0;
        v6 = *v5;
LABEL_2:
        if ( v6 != 8 )
        {
LABEL_3:
          *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
          return 1;
        }
        goto LABEL_6;
      }
    }
    if ( (*(_BYTE *)(v9 - 8) & 1) != 0 )
      *((_DWORD *)v5 + 4) = 0;
    goto LABEL_3;
  }
  *v5 = 8;
  v8 = *(_QWORD *)(v9 + 56);
  *((_QWORD *)v5 + 1) = v8;
  if ( (*(_BYTE *)(v8 - 8) & 1) == 0 )
    goto LABEL_7;
  *((_DWORD *)v5 + 4) = 0;
LABEL_6:
  v8 = *((_QWORD *)v5 + 1);
LABEL_7:
  if ( (*(_BYTE *)(v8 + 144) & 4) == 0 )
    goto LABEL_3;
  *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08280);
  return 1;
}
