// Function: sub_7727C0
// Address: 0x7727c0
//
__int64 __fastcall sub_7727C0(__int64 a1, __int64 a2, __int64 a3, char **a4, __m128i *a5)
{
  char *v6; // rax
  char v7; // dl
  __int64 v9; // rcx
  char v10; // dl
  __int64 v11; // rdi

  v6 = *a4;
  v7 = **a4;
  if ( v7 == 48 )
  {
    v9 = *((_QWORD *)v6 + 1);
    v10 = *(_BYTE *)(v9 + 8);
    if ( v10 == 1 )
    {
      *v6 = 2;
      *((_QWORD *)v6 + 1) = *(_QWORD *)(v9 + 32);
      goto LABEL_3;
    }
    if ( v10 == 2 )
    {
      *v6 = 59;
      *((_QWORD *)v6 + 1) = *(_QWORD *)(v9 + 32);
      goto LABEL_3;
    }
    if ( v10 )
      sub_721090();
    *v6 = 6;
    v11 = *(_QWORD *)(v9 + 32);
    *((_QWORD *)v6 + 1) = v11;
  }
  else
  {
    if ( v7 != 6 )
    {
LABEL_3:
      *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
      return 1;
    }
    v11 = *((_QWORD *)v6 + 1);
  }
  if ( !(unsigned int)sub_8D23B0(v11) )
    goto LABEL_3;
  *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08280);
  return 1;
}
