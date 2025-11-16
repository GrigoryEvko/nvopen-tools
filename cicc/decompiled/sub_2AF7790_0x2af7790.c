// Function: sub_2AF7790
// Address: 0x2af7790
//
void __fastcall sub_2AF7790(_QWORD *src, _QWORD *a2, const __m128i *a3)
{
  _QWORD *i; // r12
  __int64 v6; // rcx
  char *v7; // rax
  _QWORD *v8; // rdi
  _QWORD *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // [rsp-80h] [rbp-80h]
  __m128i v12; // [rsp-78h] [rbp-78h]
  __m128i v13; // [rsp-58h] [rbp-58h] BYREF
  __int64 v14; // [rsp-48h] [rbp-48h]
  __int64 v15; // [rsp-40h] [rbp-40h]

  if ( src != a2 )
  {
    for ( i = src + 1; a2 != i; *src = v6 )
    {
      while ( 1 )
      {
        v7 = (char *)a3->m128i_i64[0];
        v8 = (_QWORD *)(a3[1].m128i_i64[1] + a3->m128i_i64[1]);
        if ( (a3->m128i_i64[0] & 1) != 0 )
          v7 = *(char **)&v7[*v8 - 1];
        if ( ((unsigned __int8 (__fastcall *)(_QWORD *, __int64, _QWORD, _QWORD))v7)(v8, a3[1].m128i_i64[0], *i, *src) )
          break;
        v9 = i;
        v10 = a3[1].m128i_i64[1];
        ++i;
        v12 = _mm_loadu_si128(a3);
        v14 = a3[1].m128i_i64[0];
        v15 = v10;
        v13 = v12;
        sub_2AF7720(v9, (char **)&v13);
        if ( a2 == i )
          return;
      }
      v6 = *i;
      if ( src != i )
      {
        v11 = *i;
        memmove(src + 1, src, (char *)i - (char *)src);
        v6 = v11;
      }
      ++i;
    }
  }
}
