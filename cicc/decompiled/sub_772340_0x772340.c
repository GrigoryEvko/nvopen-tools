// Function: sub_772340
// Address: 0x772340
//
__int64 __fastcall sub_772340(__int64 a1, __int64 a2, __int64 a3, char **a4, __m128i *a5)
{
  char *v5; // rax
  char v6; // dl
  __int64 v8; // rcx
  char v9; // dl

  v5 = *a4;
  v6 = **a4;
  if ( v6 != 48 )
  {
    if ( v6 == 6 )
    {
LABEL_3:
      *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08280);
      return 1;
    }
    goto LABEL_5;
  }
  v8 = *((_QWORD *)v5 + 1);
  v9 = *(_BYTE *)(v8 + 8);
  if ( v9 == 1 )
  {
    *v5 = 2;
    *((_QWORD *)v5 + 1) = *(_QWORD *)(v8 + 32);
LABEL_5:
    *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
    return 1;
  }
  if ( v9 != 2 )
  {
    if ( v9 )
      sub_721090();
    *v5 = 6;
    *((_QWORD *)v5 + 1) = *(_QWORD *)(v8 + 32);
    goto LABEL_3;
  }
  *v5 = 59;
  *((_QWORD *)v5 + 1) = *(_QWORD *)(v8 + 32);
  *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
  return 1;
}
