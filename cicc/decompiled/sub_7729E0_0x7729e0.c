// Function: sub_7729E0
// Address: 0x7729e0
//
__int64 __fastcall sub_7729E0(__int64 a1, __int64 a2, __int64 a3, char **a4, __m128i *a5)
{
  char *v5; // rax
  char v6; // dl
  __int64 v8; // rdx
  __int64 v9; // rcx
  char v10; // dl

  v5 = *a4;
  v6 = **a4;
  switch ( v6 )
  {
    case 48:
      v9 = *((_QWORD *)v5 + 1);
      v10 = *(_BYTE *)(v9 + 8);
      if ( v10 == 1 )
      {
        *v5 = 2;
        *((_QWORD *)v5 + 1) = *(_QWORD *)(v9 + 32);
        break;
      }
      if ( v10 == 2 )
      {
        *v5 = 59;
        *((_QWORD *)v5 + 1) = *(_QWORD *)(v9 + 32);
        break;
      }
      if ( v10 )
        sub_721090();
      *v5 = 6;
      v8 = *(_QWORD *)(v9 + 32);
      *((_QWORD *)v5 + 1) = v8;
      if ( *(_BYTE *)(v8 + 140) != 12 )
        break;
LABEL_12:
      if ( *(_QWORD *)(v8 + 8) )
        goto LABEL_4;
      break;
    case 6:
      v8 = *((_QWORD *)v5 + 1);
      if ( *(_BYTE *)(v8 + 140) != 12 )
        break;
      goto LABEL_12;
    case 28:
LABEL_4:
      *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08280);
      return 1;
  }
  *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
  return 1;
}
