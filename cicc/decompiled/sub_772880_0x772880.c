// Function: sub_772880
// Address: 0x772880
//
__int64 __fastcall sub_772880(__int64 a1, __int64 a2, __int64 a3, char **a4, __m128i *a5)
{
  char *v5; // rax
  char v6; // dl
  __int64 v8; // rdx
  __int64 v9; // rcx
  char v10; // dl

  v5 = *a4;
  v6 = **a4;
  if ( v6 == 48 )
  {
    v9 = *((_QWORD *)v5 + 1);
    v10 = *(_BYTE *)(v9 + 8);
    if ( v10 == 1 )
    {
      *v5 = 2;
      *((_QWORD *)v5 + 1) = *(_QWORD *)(v9 + 32);
      goto LABEL_3;
    }
    if ( v10 != 2 )
    {
      if ( v10 )
        sub_721090();
      *v5 = 6;
      *((_QWORD *)v5 + 1) = *(_QWORD *)(v9 + 32);
      goto LABEL_3;
    }
    *v5 = 59;
    v8 = *(_QWORD *)(v9 + 32);
    *((_QWORD *)v5 + 1) = v8;
  }
  else
  {
    if ( v6 != 59 )
      goto LABEL_3;
    v8 = *((_QWORD *)v5 + 1);
  }
  if ( *(_BYTE *)(v8 + 120) == 1 && (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v8 + 192) + 140LL) - 9) <= 2u )
  {
    *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08280);
    return 1;
  }
LABEL_3:
  *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
  return 1;
}
