// Function: sub_854430
// Address: 0x854430
//
__m128i *sub_854430()
{
  __m128i *result; // rax
  __m128i *v1; // rbx
  unsigned __int8 v2; // di
  unsigned int v3; // esi
  __m128i *v4; // r12
  __int64 v5; // r13
  unsigned int v6; // eax
  void (__fastcall *v7)(__m128i *); // rax
  __int64 v8; // rdx
  __m128i **v9; // rdx

  sub_853C60(0);
  result = (__m128i *)&qword_4D03E88;
  v1 = (__m128i *)qword_4D03E88;
  for ( qword_4D03E88 = 0; v1; result = (__m128i *)sub_853F90(v4) )
  {
    while ( 1 )
    {
      v4 = v1;
      v1 = (__m128i *)v1->m128i_i64[0];
      v5 = v4->m128i_i64[1];
      v6 = *(_DWORD *)(v5 + 12);
      if ( v6 == 3 )
        break;
      if ( v6 <= 3 )
      {
        if ( v6 == 1 )
        {
          v2 = *(_BYTE *)(v5 + 19);
          if ( v2 != 3 )
          {
            v3 = 608;
            if ( (*(_BYTE *)(v5 + 17) & 3) != 3 )
              v3 = 607 - (*(_BYTE *)(v5 + 17) & 1);
            sub_684AA0(v2, v3, (__m128i *)v4[3].m128i_i32);
          }
          goto LABEL_7;
        }
        if ( v6 != 2 )
          goto LABEL_25;
        if ( (*(_BYTE *)(v5 + 17) & 8) != 0 )
LABEL_24:
          sub_8543B0(v4, 0, 0);
LABEL_13:
        v7 = (void (__fastcall *)(__m128i *))off_4A51EC0[*(unsigned __int8 *)(v5 + 16)];
        if ( v7 )
          v7(v4);
        goto LABEL_7;
      }
      if ( v6 != 4 )
LABEL_25:
        sub_721090();
      v8 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      result = *(__m128i **)(v8 + 432);
      if ( result )
      {
        do
        {
          v9 = (__m128i **)result;
          result = (__m128i *)result->m128i_i64[0];
        }
        while ( result );
        *v9 = v4;
      }
      else
      {
        *(_QWORD *)(v8 + 432) = v4;
      }
      v4->m128i_i64[0] = 0;
      if ( !v1 )
        return result;
    }
    if ( (v4[4].m128i_i8[8] & 8) == 0 )
    {
      if ( (*(_BYTE *)(v5 + 17) & 8) != 0 )
        goto LABEL_24;
      goto LABEL_13;
    }
LABEL_7:
    ;
  }
  return result;
}
