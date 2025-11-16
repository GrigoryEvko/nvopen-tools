// Function: sub_772D00
// Address: 0x772d00
//
__int64 __fastcall sub_772D00(__int64 a1, __int64 a2, __int64 a3, char **a4, __m128i *a5)
{
  char *v6; // rbx
  char v7; // al
  __int64 v9; // rdx
  char v10; // al
  _QWORD *v11; // rax
  __int64 v12; // rdi
  _QWORD v13[3]; // [rsp+8h] [rbp-18h] BYREF

  v6 = *a4;
  v7 = **a4;
  if ( v7 == 48 )
  {
    v9 = *((_QWORD *)v6 + 1);
    v10 = *(_BYTE *)(v9 + 8);
    if ( v10 != 1 )
    {
      if ( v10 == 2 )
      {
        *v6 = 59;
        *((_QWORD *)v6 + 1) = *(_QWORD *)(v9 + 32);
      }
      else
      {
        if ( v10 )
          sub_721090();
        *v6 = 6;
        *((_QWORD *)v6 + 1) = *(_QWORD *)(v9 + 32);
      }
      goto LABEL_4;
    }
    *v6 = 2;
    *((_QWORD *)v6 + 1) = *(_QWORD *)(v9 + 32);
  }
  else if ( v7 != 2 )
  {
    if ( v7 != 13 )
    {
LABEL_4:
      *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
      return 1;
    }
    v11 = sub_724DC0();
    v12 = *((_QWORD *)v6 + 1);
    v13[0] = v11;
    if ( !(unsigned int)sub_716120(v12, (__int64)v11) )
    {
      sub_724E30((__int64)v13);
      goto LABEL_4;
    }
    sub_724E30((__int64)v13);
  }
  *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08280);
  return 1;
}
