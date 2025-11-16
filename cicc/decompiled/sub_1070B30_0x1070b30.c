// Function: sub_1070B30
// Address: 0x1070b30
//
void __fastcall sub_1070B30(char *src, const __m128i *a2)
{
  const __m128i *v3; // rbx
  __int64 v4; // r8
  __int64 v5; // r14
  char v6; // r15
  const __m128i *v7; // rdi
  __int64 v8; // [rsp-40h] [rbp-40h]

  if ( src != (char *)a2 )
  {
    v3 = (const __m128i *)(src + 24);
    if ( a2 != (const __m128i *)(src + 24) )
    {
      do
      {
        while ( !sub_10704F0((__int64)v3, (__int64)src) )
        {
          v7 = v3;
          v3 = (const __m128i *)((char *)v3 + 24);
          sub_1070780(v7);
          if ( a2 == v3 )
            return;
        }
        v4 = v3->m128i_i64[0];
        v5 = v3->m128i_i64[1];
        v6 = v3[1].m128i_i8[0];
        if ( src != (char *)v3 )
        {
          v8 = v3->m128i_i64[0];
          memmove(src + 24, src, (char *)v3 - src);
          v4 = v8;
        }
        v3 = (const __m128i *)((char *)v3 + 24);
        *(_QWORD *)src = v4;
        *((_QWORD *)src + 1) = v5;
        src[16] = v6;
      }
      while ( a2 != v3 );
    }
  }
}
